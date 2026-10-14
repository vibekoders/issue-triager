# triager-evals

Datasets and the eval harness for the triage strategies (see `PLAN.md`).
Run every command from the repo root through devenv, which puts `uv` on the
PATH:

```
devenv shell -- bash -c 'cd evals && uv run <command> ...'
```

Paths given to the commands may be relative to the repo root (`datasets/...`,
`configs/...`) even though uv runs in `evals/`.

## Datasets

```
uv run fetch-issues --mode eligible --target 3000 --skip-recent-days 60
uv run fetch-issues --mode corpus
uv run fetch-issues --mode backfill    # only for a dump fetched before renames/duplicate sides were
uv run normalize-issues                # data/normalized/kubernetes.jsonl and the search corpus
uv run build-dataset --repo kubernetes --split dev --limit 200 --stratify kind --build-date 2026-10-14
```

(and the same `build-dataset` with `--split test --limit 600`, and with
`--split dev|test --limit 20|500` for the experiment sets). Each dataset records its build
parameters and the source's sha256 in its `# dataset-metadata:` line.

What a case shows is the issue as filed: the title before the first rename
(triagers often retitle with the sig, the kind or the diagnosis), the body
from its creation snapshot in `userContentEdits`, and no slash-command lines
(`/kind`, `/sig`, `/committee security-response`, `/wg`, misspellings like
`/sig-network`, ...). An issue that was edited but whose creation snapshot is
gone (mostly security placeholders later filled in with the advisory) is
marked `original_body_unknown` and left out of the datasets; pass
`--keep-unknown-body` to keep it. Duplicate targets only count when a
comment states them (not "is this a duplicate of #N?", unless the label or
state reason confirms it) and when they were filed before the issue, since
the search tools can't see later ones.

## Running evals

Prerequisites: Restate (`docker compose up -d restate`), an Ollama with the
models the config names, and the service (`cargo run -p triage-service -- serve`,
or without Rust `docker compose --profile app up -d --build` with
`SERVICE_URL=http://service:9080` in `.env`).
`triage-eval` registers `$SERVICE_URL` (default
`http://host.docker.internal:9080`) with Restate before it starts; pass
`--no-register` to use whatever deployment is registered already.

```
uv run triage-eval --dataset datasets/kubernetes-dev-200.yaml --strategy Heuristic --no-judge
uv run triage-eval --dataset datasets/kubernetes-dev-20.yaml --strategy SingleShot \
    --config configs/single-shot-qwen3-1.7b.yaml --limit 5 --judge-model ollama:qwen3:4b
uv run triage-eval --dataset datasets/kubernetes-dev-200.yaml --strategy Workflow \
    --config configs/workflow-tev1-none.yaml --config configs/workflow-tev1-escalate.yaml --no-judge
```

| Option | Meaning |
|---|---|
| `--strategy` | `Heuristic`, `SingleShot`, `Workflow`, `Agent`, `Ensemble`, `DraftVerify` |
| `--config FILE` | the strategy config (YAML, what goes in `TriageRequest.config`). Repeat it for a sweep: one experiment per config. Optional for `Heuristic`. Examples for every strategy are in `configs/`. |
| `--judge-model` | model for the LLM judge, addressed like the service (`openai:`, `anthropic:`, `gemini:`, `azure:`, `ollama:`; other prefixes go to pydantic-ai as they are) (default `$JUDGE_MODEL`, else `ollama:qwen3:4b`; Ollama is reached at `$OLLAMA_BASE_URL`, default `http://localhost:11434/v1`). See "The judge's model family" below |
| `--no-judge` | skip the judge; `quality` and `suggested_reply` are then not scored and drop out of the composite |
| `--run-id ID` | default: a UTC timestamp. Re-running with the same ID reuses Restate's stored results (no model calls, no cost) |
| `--repeats N` | run each case N times (separate workflow keys), for consistency numbers |
| `--limit N` | only the first N cases (datasets are ordered so every prefix stays stratified by kind) |
| `--label TEXT` | the experiment label (default: the config file's name) |
| `--max-concurrency N` | default 2, for a local Ollama |

Each `--config` is checked against its strategy's schema in `schema/`, with
unknown keys rejected, before anything runs: the Rust config structs ignore
keys they don't know, so a typo (`escalaton:`) would otherwise silently run
the defaults under a label that says otherwise.

Each case goes to `POST /{Strategy}/{run_id}:{config_hash}:{case_digest}:{repeat}/run`
on the Restate ingress (`$RESTATE_INGRESS_URL`, default `http://localhost:8080`);
a 409 means the key already ran, and the stored result is read through
`/restate/workflow/{Strategy}/{key}/attach`. Responses are checked against
`schema/triage-response.schema.json` (a mismatch fails the case) and their
`config_hash` against the harness's own (a mismatch is a loud warning).

Every experiment is named `{Strategy}@{version} {config_hash} {label}` and
goes to Logfire when `LOGFIRE_TOKEN` is set (the service's spans join the
case's trace through `traceparent`). It is also saved to
`evals/results/<run_id>/<Strategy>-<config_hash>.json`, with every case's full
output, scores, per-field coverage, usage metrics and the experiment metadata
(git SHA, config, dataset hash, suite ID, judge model). The file format is
documented in `triager_evals/results.py`. `evals/results/` is gitignored; add a
experiment's files with `git add -f` if you want them in the repo.

### What is scored

| Field | Evaluator | Score |
|---|---|---|
| kind | `KindMatches` | in the issue's `kind/*` labels; macro-F1 in compare-runs |
| area | `AreaMatches` | in the issue's `sig/*` labels |
| severity | `SeverityMatches` | exact `priority/*`, plus distance in levels |
| is_regression, security_sensitive | `RegressionMatches`, `SecurityMatches` | exact per case; at run level balanced accuracy (see below), plus precision/recall |
| enough_info | `EnoughInfoMatches` | `reproducibility.enough_info` vs needs-information / fixed |
| duplicates | `DuplicateRecall` | recall@5 of the labeled duplicate target |
| quality | `QualityJudge` | the judge rates the issue on triage-core's rubric; score 1 − abs difference / 4 |
| suggested_reply | `ReplyJudge` (`LLMJudge`) | the judge's rubric score |
| – | `SanityChecks` | hard errors (spam that was later fixed, a duplicate filed after the issue, an area that isn't a repo label, ...) |

An evaluator records nothing when the case has no expected answer or the
strategy didn't produce the field. The judge's tokens and cost are recorded
per case under `judge_usage`, never in the strategy's metrics.

The composite score is the weighted mean of the per-field run-level scores
with the weights in `evals/suite.yaml`; a field with an expected answer that
the strategy didn't produce counts as 0. A field's run-level score is the mean
of its per-case scores, except for `is_regression` and `security_sensitive`:
their positives are rare (2-3%), so plain accuracy would reward always
answering false. Their run-level score is balanced accuracy, the mean of the
true-positive and true-negative rates (unattempted cases count as wrong in the
composite), so "always false" scores 50 instead of 98. If the cases hold only
one class it is that class's rate alone, flagged with ⚑ in the reports
(dev-20 and test-20 have no positives for either field). Results files
written before this change (`evaluators_version` 1) keep plain accuracy when
read; re-score them (below) to compare them with new runs. The suite ID hashes the dataset file,
`EVALUATORS_VERSION`, `RUBRIC_VERSION`, the judge model and the weights; only
compare runs within one suite.

### Re-scoring stored outputs

```
uv run triage-eval --rescore results/<run>/<file>.json [--dataset D] [--run-id ID] [--judge-model M | --no-judge]
```

runs a results file's stored outputs through the current evaluators without
calling the service, and rewrites the file (or writes a new one under
`--run-id`). `--dataset` may be a subset of the stored run's dataset, such as
test-20 from test-600. The judge runs again if the file was judged (or if
`--judge-model` is given). The outputs' provenance (git SHA, start time) is
kept and the rescoring is recorded under `metadata.rescored`. Use it after
an evaluator change: re-running with the same `--run-id` also re-scores, but
only while Restate still has the outputs (workflow results are kept for one
day by default); after that the run calls the models again, with whatever
code the service runs now.

## Experiments

`evals/experiments/` holds the PLAN.md experiments, one file per experiment
(`<experiment id>.yaml`: its steps, each one triage-eval run and one
pydantic-evals experiment in Logfire, plus what it compares), and
`profiles.yaml` with the model profiles and shared settings. The format, the
step keys and how to add your own experiment are in
[experiments/README.md](experiments/README.md). `run-experiments` loads the
directory as one spec, runs the steps through the same code as triage-eval
and writes the reports. `--list` shows them:

| Experiment (formerly) | Steps |
|---|---|
| `heuristic-base-rates` (m1) | `heuristic`, `heuristic-base-rates`, each on dev and test (`heuristic@dev`, `heuristic@test`, ...) |
| `single-shot-model-size` (m2) | `single-shot-llm`, `single-shot-llm-large` |
| `workflow-escalation` (m3) | `workflow-decision-only`, `workflow-esc-0.8`, `workflow-esc-0.6`, `workflow-esc-0.9`, `workflow-decision-alt` |
| `agent-tools` (m4) | `agent-no-tools`, `agent-tools` |
| `ensemble-voters` (m5) | `ensemble-1-voter`, `ensemble-vote`, `ensemble-judge` |
| `escalation-coordination` (m6) | `draft-verify` (compared with `workflow-esc-0.8`) |
| `final-comparison` (m7) | `final-heuristic`, `final-heuristic-base-rates`, `final-single-shot`, `final-workflow-decision-only`, `final-workflow-esc-0.8`, `final-agent-tools`, `final-ensemble-vote`, `final-draft-verify` |

The first six experiments run on the dev split, `final-comparison` on the
test split.

Steps run in dependency order: a step runs after the steps it derives from,
takes its config from or has as baseline, and after the other experiments'
steps that its experiment compares with; otherwise in file name order, then
file order. Steps of different experiments can therefore interleave
(`final-heuristic` runs right after `heuristic@test`); each experiment's
report is written when its last step is done. `--list` shows the order.
`--file experiments/agent-tools.yaml` loads only that file (with
`profiles.yaml` next to it); references to steps in other files are then an
error.

```
uv run run-experiments --list                         # experiments, steps, labels with the resolved models
uv run run-experiments --dry-run                      # the plan for 20 issues, with estimated model calls and time
uv run run-experiments                                # everything on the 20-issue sets, in order
uv run run-experiments --size 500                     # everything on the 500-issue sets
uv run run-experiments --only workflow-escalation     # one experiment (repeatable)
uv run run-experiments --step workflow-esc-0.8        # one step (repeatable; id, or id@split such as heuristic@test)
uv run run-experiments --report-only                  # only rewrite the reports from existing results
```

### Sizes

`--size 20` (default) or `--size 500`. A step names a split (`dev` or
`test`), not a file; its dataset is `datasets/kubernetes-{split}-{size}.yaml`
(`defaults.dataset`; a step may set its own `dataset:`). The sets are built
with `build-dataset --stratify kind --build-date 2026-10-14 --split dev|test
--limit 20|500`, from the same source, so each 20-issue set is the first 20
cases of its 500-issue set (a test checks this). The test split has 580
cases, so test-500 is most of it.

### Run ids

Run ids are composed, `{step id}-{split}-{size}-{profile}`, e.g.
`workflow-esc-0.8-dev-20-openai`, so sizes and profiles never collide. Results
go to `evals/results/<run id>/<Strategy>-<config_hash>.json` as with
triage-eval. The config hash is computed after the model roles are filled in
and is part of both the Restate workflow key and the file name, so a run with
another model (a `--model` override under the same profile, say) never reuses
stored outputs. A step is only skipped when the file for its exact config
hash exists.

### Model roles and profiles

Configs are inline in the experiment files, with placeholders for model roles:

```yaml
config: {decision_model: "{decision}", escalation: {model: "{llm}", thresholds: {accept: 0.8}}}
label: workflow {decision} + escalation {llm}, accept 0.8
```

| Role | Used for |
|---|---|
| `llm` | the cheap/default LLM: single_shot, escalation, agent, the verifier, the ensemble's first voter |
| `llm_large` | the larger LLM in `single-shot-model-size`, the ensemble's second voter and judge aggregator |
| `decision` | the decision model (System One) |
| `decision_alt` | the decision model `workflow-decision-alt` compares with |
| `judge` | the eval judge (pydantic-ai side, addressed like the service: `openai:`, `anthropic:`, `gemini:`, `azure:`, `ollama:`; see `judge.py`) |

Labels take placeholders too, so Logfire's experiment names show the models.
After substitution every config is checked against
`schema/<strategy>-config.schema.json` (unknown keys rejected) before
anything runs. The files in `configs/` are no longer used by the experiments;
they are still handy for one-off `triage-eval` runs.

Profiles (in `experiments/profiles.yaml`) set every role:

| Profile | llm | llm_large | decision | decision_alt | judge | concurrency |
|---|---|---|---|---|---|---|
| `openai` (default) | `openai:gpt-6-luna` | `openai:gpt-6-sol` | `typesafe:jev-latest` | `ollama:tev1:0.8b-q8_0` | `openai:gpt-6-sol` | 8 |
| `anthropic` | `anthropic:claude-haiku-4-5` | `anthropic:claude-sonnet-5` | `typesafe:jev-latest` | `ollama:tev1:0.8b-q8_0` | `anthropic:claude-sonnet-5` | 8 |
| `gemini` | `gemini:gemini-3.5-flash-lite` | `gemini:gemini-3.1-pro-preview` | `typesafe:jev-latest` | `ollama:tev1:0.8b-q8_0` | `gemini:gemini-3.1-pro-preview` | 8 |
| `azure` | `azure:gpt-6-luna` | `azure:gpt-6-sol` | `typesafe:jev-latest` | `ollama:tev1:0.8b-q8_0` | `azure:gpt-6-sol` | 8 |
| `local` | `ollama:qwen3:1.7b` | `ollama:qwen3:4b` | `ollama:tev1:0.8b-q8_0` | `ollama:tev1:4b` | `ollama:qwen3:4b` | 1 |

Each hosted profile needs its provider's key (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or `AZURE_OPENAI_RESOURCE` and
`AZURE_OPENAI_API_KEY`; the service and the judge read the same ones) and
`TYPESAFE_API_KEY` for Jev. Their `decision_alt` is the local tev1, used only
by `workflow-decision-alt`: without an Ollama, skip that step or pass
`--model decision_alt=...`. `azure` is the profile RESULTS.md was produced
with (then named `hosted`, so its run ids end in `-hosted`): the 20-issue
version took about 45 minutes and $3.60. `local` takes about 10 hours for
size 20 on a 4-CPU machine without a GPU, judged by Qwen like the strategies
(see "The judge's model family" below). A step whose config or judge uses an
Ollama model runs one case at a time (`defaults.ollama_max_concurrency`),
whatever the profile.

Resolution, highest first:

1. `--model ROLE=ADDRESS` (repeatable)
2. the env `EXPERIMENT_MODEL_<ROLE>` (`EXPERIMENT_MODEL_LLM`,
   `EXPERIMENT_MODEL_JUDGE`, ...); `JUDGE_MODEL` is an alias for the judge,
   below `EXPERIMENT_MODEL_JUDGE`
3. the profile: `--profile NAME`, else `$EXPERIMENT_PROFILE`, else
   `default_profile`

Every run prints the resolved roles and where each came from, and records
them (with the profile, size, split, experiment and step id) in each results
file's metadata and in the Logfire experiment's metadata. devenv loads
`.env`, so a `JUDGE_MODEL=` left there overrides every profile's judge; the
printout shows `(env JUDGE_MODEL)` when it does.

```
uv run run-experiments --profile local --size 20
uv run run-experiments --model judge=azure:gpt-6-luna                 # a cheaper judge
uv run run-experiments --model llm=azure:gpt-6-sol --only agent-tools # another LLM, same profile
EXPERIMENT_MODEL_DECISION=ollama:tev1:4b uv run run-experiments --only workflow-escalation
uv run run-experiments --no-judge                                     # no judge at all
```

Steps are judged unless `--no-judge` is given (or a step sets `judge:
false`). Without the judge, `quality` and `suggested_reply` drop out of the
composite and the suite changes, so judged and unjudged runs are not compared
with each other.

### Options

| Option | Meaning |
|---|---|
| `--size 20\|500` | dataset size (default 20) |
| `--profile NAME` | model profile (default `$EXPERIMENT_PROFILE`, else `default_profile`) |
| `--model ROLE=ADDRESS` | override a role; repeatable |
| `--only ID` | only this experiment; repeatable |
| `--step ID` | only this step (`heuristic` selects both splits, `heuristic@test` one); repeatable. The experiment's report still lists its other steps from existing results |
| `--list` | the experiments and every step in run order; no service needed |
| `--dry-run` | the resolved roles and, per step, the run id, dataset, concurrency and action (skip, rescore, derive, run, blocked, optional) with estimated model calls and time |
| `--force` | re-run steps whose results exist, through the service (Restate returns the outputs it still has, so those cost nothing) |
| `--report-only` | write the reports from the existing results, run nothing |
| `--no-judge` | skip the eval judge |
| `--no-register` | don't register `$SERVICE_URL` with Restate first (registering is the default) |
| `--max-concurrency N` | cases in flight per step, over the profile, the step and the Ollama cap |
| `--file PATH` | the experiments directory (default `evals/experiments/`) or one experiment file |

What happens to a step:

- **skip**: its results file exists and was scored with the current
  evaluators and suite.
- **rescore**: its results file exists but was scored with older evaluators
  or another suite (another judge, for example); it is re-scored offline from
  the stored outputs (the judge runs again).
- **derive**: the step has a `source` with results; that run's stored outputs
  are re-scored under this step's run id. `final-heuristic` and
  `final-heuristic-base-rates` derive from `heuristic@test` and
  `heuristic-base-rates@test` (the same test set), so they cost nothing but
  the judge.
- **run**: a new run through the Restate ingress.
- **blocked**: the dataset is missing, the service is not reachable, or the
  step pins a strategy `version` the service doesn't report. The runner moves
  on, and the report lists the step as blocked.
- **optional**: skipped unless `--include-optional` (no step is optional now).

`final-single-shot` takes the config of whichever `single-shot-model-size`
step scored higher (`config_from_best`).

Reports go to `evals/results/experiments/{size}-{profile}/`, so the 20- and
500-issue reports and the profiles don't overwrite each other:

- `<experiment id>.md` after each experiment: the resolved models, the steps
  with their status, config hash, composite, cost and latency; a before →
  after table (each step against its `baseline`, with per-case wins and a
  sign test); and the full compare-runs report of each comparison, with
  `<id>-<n>-<title>.csv` and `-pareto.svg` next to it.
- `overview.md` at the end: the Pareto frontier (table, marginal cost per
  point, `overview-pareto.svg`) across the final-comparison runs, a line per
  experiment, and a per-strategy table of every experiment run.

Results from before the composed run ids (the `h2m*` and `m1-*` runs that
RESULTS.md reports) are in `evals/results/archive-before-profiles/`.

### Comparing in Logfire

With `LOGFIRE_TOKEN` set (and `LOGFIRE_ENVIRONMENT`, which becomes the
`deployment.environment.name` of both the harness and the Rust service), each
step is one pydantic-evals experiment, named
`{Strategy}@{version} {config_hash} {label}` (the label shows the models),
with the step's metadata (profile, models, size, split, experiment and step
id, config, suite id, ...). The dataset name is the dataset file's
(`kubernetes-dev-20`, `kubernetes-test-500`, ...), so all runs on one dataset
are grouped together for side-by-side comparison. Per case, the strategy's
usage is recorded as the metrics `input_tokens`, `output_tokens`, `cost`
(USD), `latency_ms`, `llm_calls`, `decision_calls` and `tool_calls`. The
judge's calls show up as their own pydantic-ai spans with their token usage,
but never in the strategy's `cost`. The harness instruments httpx before
any client exists, so each case's request carries `traceparent` and the Rust
service's spans (model calls, tools) appear inside the case's trace. Logfire
is flushed before the process exits, also after an error. Restate's own
traces go to Logfire through `docker-compose.yaml`.

To compare, open the dataset (say `kubernetes-test-20`) in Logfire's evals
view and select the experiments, e.g. the `final-comparison` steps of one
profile, or the same step under two profiles. Each case links to its trace.

### Differences from PLAN.md

- **m3 (`workflow-escalation`), Jev vs. Clef:** the hosted profiles compare
  Jev with the local tev1:0.8b; there is no Cloudflare key here for Clef.
  `configs/workflow-clef.yaml` is ready for the original comparison once
  `CLOUDFLARE_ACCOUNT_ID`/`CLOUDFLARE_API_TOKEN` are set, e.g.
  `--model decision_alt=cloudflare:...`, or:

  ```
  uv run triage-eval --dataset datasets/kubernetes-dev-20.yaml --strategy Workflow \
      --config configs/workflow-jev.yaml --config configs/workflow-clef.yaml --no-judge
  ```

- **m4 (`agent-tools`), agent without tools → with tools:** PLAN.md compares
  no tools with `search_issues` alone. The `agent-tools` step leaves `tools`
  unset and so gets all five tools (`search_issues`, `get_issue`,
  `list_labels`, `get_issue_templates`, `get_releases`). Isolating search
  needs a step with `tools: [search_issues]`.

## Comparing runs

```
uv run compare-runs results/<runA>/<file>.json results/<runB>/ [--baseline FILE] [--format md|json] [--out FILE]
```

Arguments are results files or directories of them. The default baseline is
the first file. The Markdown report has:

- a per-run table: cases, composite, kind macro-F1, cost per issue, tokens,
  calls, latency (mean/p50/p95), sanity failures, judge cost;
- per-field accuracy over attempted cases with Wilson 95% intervals and
  coverage (attempted/expected), plus severity distance, precision/recall for
  the boolean fields, and repeat consistency (kind, area) when repeats > 1;
- a head-to-head of every run against the baseline on the shared
  (case, repeat) pairs: only-A, only-B, both, neither per field, with an exact
  McNemar p-value (an unattempted field counts as wrong here);
- the Pareto frontier of cost per issue against composite, and the marginal
  cost per composite point between neighbouring frontier runs;
- calibration: reliability bins and ECE for every field with a confidence.

It also writes the per-run table as CSV and a small SVG Pareto chart (next to
`--out`, or `evals/results/compare.csv` and `compare-pareto.svg`).

### The judge's model family

PLAN.md wants the judge from a different model family than the strategies,
so it doesn't favour its own family's output. The default judge,
`ollama:qwen3:4b`, breaks that: the chat strategies run qwen3:1.7b/4b, and
`configs/single-shot-qwen3-4b.yaml` and the ensemble's judge aggregator use
the very same model. It is the default only because every model on the local
Ollama is Qwen-based (qwen3, and tev1/nimble on Qwen 3.5). Judged scores
(`quality`, `reply`) made with it may carry self-preference bias; the only
judged experiment step is the optional 5-case one in m2. For judged results
anyone will rely on, pull a model of another family and pass it, e.g.
`--judge-model ollama:llama3.2:3b` or `ollama:gemma3:4b` (or a hosted model;
set `JUDGE_MODEL` for every run). The judge model is part of the suite ID,
so runs judged by different models are never compared as one suite.

### Checking the judge

Hand-label issue quality in `evals/gold/quality.yaml` (about 50 cases, on the
same rubric), then:

```
uv run compare-runs judge-agreement results/<run>/<file>.json [--gold gold/quality.yaml]
```

reports the judge's (and the strategy's) exact and within-one agreement, MAE
and weighted kappa against the hand labels.

## Tests

```
uv run pytest
```

`tests/test_harness.py` covers the config hash (pinned against triage-core),
case digests, the evaluators including skips, the statistics (Wilson,
McNemar, Pareto, macro-F1, balanced accuracy, ECE), the composite with
unattempted fields, and validates every file in `configs/` against
`schema/*-config.schema.json` (with unknown keys rejected).
`tests/test_experiments.py` checks `experiments/` for both sizes and
profiles (references, datasets, composed run ids, placeholders filled in,
configs valid against their schemas), profile and model-role resolution and
its precedence, the dataset prefixes (dev-20 ⊂ dev-500, test-20 ⊂ test-500 ⊂
test-600), the call estimates and concurrency, step selection, the step
planning (skip only for the exact config hash, rescore, derive, run, blocked,
optional, the version guard), offline rescoring and deriving with the run
metadata, the reports, `--list`/`--dry-run`, and the loader: merging the
files, cross-file references, duplicate ids, the run order (dependencies in
a later-named file, cycles), a single `--file`, missing references and the
"Adding your own experiment" example.

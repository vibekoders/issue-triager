# Issue triager: plan

A GitHub issue triager built several ways, so the approaches can be compared on
quality, cost and latency. Each approach is a Restate workflow in a Rust
service. A Python harness runs them over datasets with pydantic-evals, and
everything is traced to Logfire.

What it demonstrates: start with something simple, run the eval, make one
change, run the eval again, compare the two, repeat.

## Goals

- Several triage strategies, from no AI to agents, all with the same input and
  output.
- A reproducible eval loop: versioned datasets, versioned strategy configs,
  results in Logfire experiments.
- An honest answer to "which approach gives the best quality for the cost?",
  using a Pareto frontier rather than one made-up ratio.

Not goals: a production deployment, posting results back to GitHub.

## Stack

| Part | Choice |
|---|---|
| Service | Rust, native (tokio), upstream `restate-sdk`, run with `cargo run -- serve` |
| Durable execution | Restate server in docker compose |
| LLM and decision-model calls | `allama` crates: `allama`, `allama-derive`, `allama-providers` (chat, using `provider:model` addresses), `allama-systemone` (Jev, Clef) |
| Tracing | Logfire: the `logfire` crate in Rust, `logfire` in Python, Restate server traces through OTLP |
| Evals | Python, `pydantic-evals`, run with `uv` |
| Dev environment | devenv (Rust, Python, uv) |

Every model is configured with a `provider:model` address, so the provider is
a config choice (Workers AI over REST, Anthropic, OpenAI, Ollama and so on),
not a code change.

## Repository layout

```
Cargo.toml                 workspace
crates/
  triage-core/             request, output and config types, Usage, cost table
  triage-service/          Restate workflows, strategies, tools, telemetry, `serve` binary
evals/                     uv project
  pyproject.toml
  triager_evals/
    fetch.py               GitHub GraphQL → data/raw (resumable)
    normalize.py           raw → data/normalized/<repo>.jsonl (leak removal, expected answers)
    build_dataset.py       normalized → datasets/*.yaml (split, stratify, limit)
    run.py                 pydantic-evals runner (calls Restate ingress)
    evaluators.py          per-field evaluators, LLM judge
    compare.py             paired comparison, intervals, Pareto table
  repos/kubernetes.yaml    label → field mapping
datasets/                  committed pydantic-evals YAML datasets (small)
data/                      gitignored: raw dumps, normalized records, search index
docker-compose.yaml        Restate, traces → Logfire
```

The JSON schema for the output types is generated from Rust with `schemars`.
The Python harness checks responses against it, so the two sides can't drift
apart.

## Triage contract

### Input

```jsonc
{
  "config": { /* strategy-specific, see below */ },
  "issue": {
    "number": 123456,
    "title": "...",
    "body": "...",                 // slash commands removed, see Normalization
    "author_association": "NONE",
    "created_at": "2025-03-01T12:00:00Z"
  },
  "repo": {
    "name": "kubernetes/kubernetes",
    "labels": [{ "name": "sig/node", "description": "..." }],
    "latest_release": "v1.32.2",   // the latest release when the issue was created
    "as_of": "2025-03-01T12:00:00Z"
  }
}
```

Every strategy gets exactly the same input.

### Output (`Triage`)

Each field holds `{ value, confidence?, evidence? }`. A field a strategy can't
produce is left out (`null`), not guessed.

| Field | Values |
|---|---|
| `kind` | bug, feature, question, docs, cleanup, failing_test, flake, other, spam |
| `area` | one of the repo's `sig/*` labels |
| `severity` | critical, important_soon, important_longterm, backlog |
| `is_regression` | bool |
| `security_sensitive` | bool |
| `quality` | 1–5 against a written rubric |
| `reproducibility` | checklist: version, environment, steps, expected/actual, minimal repro, logs; plus `missing[]` and `enough_info: bool` |
| `outdated_version` | bool, the reported version against `repo.latest_release` |
| `scope` | single, multiple |
| `duplicates` | ranked candidate issue numbers |
| `suggested_reply` | text |

Every output also has:

- `usage`: input/output tokens, LLM, decision and tool call counts, wall time,
  cost in USD.
- `version`: the strategy's code version.
- `config_hash`: a hash of the config that produced the output.

## Strategies

| # | Strategy | How it works | Config |
|---|---|---|---|
| 1 | `heuristic` | regex and parsing only (template sections, code blocks, stack traces, version strings, keywords). The no-AI baseline. | none |
| 2 | `single_shot` | one LLM call with structured output | `model` |
| 3 | `workflow` | one decision-model call answering every typed field over the issue JSON → per-field routing in plain Rust on confidence → targeted LLM calls or a sub-agent only for low-confidence fields → optional reply | `decision_model` (jev/clef), `escalation: none \| {model, thresholds}`, `reply_model?` |
| 4 | `agent` | tool-using agent loop, each turn and tool call in its own `ctx.run` | `model`, `decision_tool?` |
| 5 | `ensemble` | several voters (LLMs and/or a decision model), aggregated by weighted vote in code or by an LLM judge | `voters[]`, `aggregator` |
| 6 | `draft_verify` | the decision model's draft is stored in workflow state and readable early through a shared `preliminary()` handler; then one agent checks only the low-confidence fields | `decision_model`, `verifier_model`, `thresholds` |

`workflow` with `escalation: none` is the decision-model-only setup. The key
comparison is `workflow` against `draft_verify`: escalation coordinated by the
workflow versus escalation coordinated by an agent.

### Agent tools

All tools read the local snapshot in `data/`, never live GitHub. That keeps
runs reproducible and avoids leakage, because a live search would find issues
filed after the one being triaged.

- `search_issues(query)`: full-text search (SQLite FTS5) over the repo's issue
  corpus, limited to issues created before `repo.as_of`. Returns title, number
  and a snippet; never labels or close reasons.
- `get_issue(number)`: same `as_of` limit, same fields.
- `list_labels()`, `get_issue_templates()`, `get_releases()`.

### Restate layout

- One Restate workflow per strategy (`Heuristic`, `SingleShot`, `Workflow`,
  `Agent`, `Ensemble`, `DraftVerify`), each with a `run` handler.
- Workflow key: `{run_id}:{config_hash}:{case_digest}:{repeat}`. The config hash
  is required: without it, a re-run with a changed config but the same run ID
  would get the stored result from the old config.
- Restate's journal means a retried or resumed run never repeats a model call,
  and never pays for it twice.

## Telemetry

Carried over from the earlier prototype:

- The harness sends `traceparent` (`logfire.instrument_httpx()`). The handler's
  root span is parented on it, so the Restate and Rust spans land in the eval
  case's trace.
- Calls made through `ctx` forward the trace headers in a fixed order, because
  Restate checks headers on replay.
- Spans that are counted (model calls, tool calls) are only created inside
  `ctx.run`, so they are recorded once even when Restate replays a handler.
- `ReplayAwareFilter` on both the Logfire layer and the console layer.
- Model-call spans use the `gen_ai.*` semantic-convention attributes plus
  `llm.cost_usd`. Integer attributes are recorded as `i64`, because Logfire
  exports unsigned integers as strings.

## Datasets

The main dataset is kubernetes/kubernetes: it is large and carefully labeled,
and its label set maps almost one-to-one onto the triage fields. More repos can
be added later with their own `repos/<repo>.yaml` label mapping.

### Size: 3000 eligible issues

Fetch the newest issues first, until 3000 pass the eligibility filters. Skip
issues created in the last 60 days, because their triage may not be settled
yet.

Why 3000:

- The test split (20%) has about 600 issues. That's enough for experiment
  comparisons with usable confidence intervals, and enough for rare kinds:
  around 3% of issues are `kind/documentation`, so roughly 90 in total.
- The dev split (about 2400) is far more than one iteration needs. Runs use
  `--limit`, so a couple of hundred is the normal size, increased only when
  intervals overlap too much.
- At current k8s volume this reaches back roughly a year or a year and a half.
  Most of that is recent enough that models are less likely to have seen it
  during training.

Separately, the search corpus for `search_issues` covers **all** k8s issues,
with only number, title, body, `created_at` and state. Without it, duplicate
detection can't find older originals. This is a lighter fetch than the
eligible set.

### What to capture per issue

Inputs (go to strategies):

- number, title, body (original version from `userContentEdits` when
  available), `createdAt`, `authorAssociation`

Ground truth and metadata (never sent to strategies):

| Data | Source | Used for |
|---|---|---|
| Final labels | `labels` | `kind`, `area`, `severity`, `is_regression`, `security_sensitive`, reproducibility |
| Label history | `LabeledEvent` / `UnlabeledEvent` with actor and time | who chose each label (author through a body command, or a triager later); whether `kind` changed over time (an ambiguous case) |
| Close state | `state`, `stateReason` (COMPLETED, NOT_PLANNED, DUPLICATE, REOPENED), `closedAt` | invalid / won't fix / duplicate, as below |
| Who closed it | `ClosedEvent.actor`, `ClosedEvent.closer` (a PR or commit) | closed by a human, by a bot (stale), or by a fix |
| Duplicate target | `MarkedAsDuplicateEvent.canonical`; "duplicate of #N" in `/close` comments as a fallback | `duplicates` |
| Fixing PRs | `closedByPullRequestsReferences`, `ClosedEvent.closer` | confirmed, actionable bug; repro sufficiency |
| Cross-references | `CrossReferencedEvent` | extra duplicate candidates |
| Transfers | `TransferredEvent` | issue moved to another repo (often a wrong-repo report) |
| Lock | `locked`, `activeLockReason` (spam, off-topic, too heated) | spam / invalid |
| Milestone | `milestone` (for example `v1.33`) | severity signal |
| Engagement | comment count, participants, thumbs-up reactions | difficulty and interest; filtering |
| Early maintainer comments | the first 5 comments not by the author | spotting "please provide more info" requests; available to the judge, never to strategies |
| Template | which issue template the body follows (bug report form, feature request, failing test, flake) | heuristic features; quality |

### Kubernetes label mapping (`repos/kubernetes.yaml`)

| Field | Labels |
|---|---|
| `kind` | `kind/bug` → bug, `kind/feature` → feature, `kind/support` → question, `kind/documentation` → docs, `kind/cleanup` → cleanup, `kind/failing-test` → failing_test, `kind/flake` → flake. `kind/api-change` and `kind/deprecation` are extra tags, not kinds. |
| `area` | `sig/*` |
| `severity` | `priority/critical-urgent`, `priority/important-soon`, `priority/important-longterm`, `priority/backlog`; `priority/awaiting-more-evidence` → no answer |
| `is_regression` | `kind/regression` |
| `security_sensitive` | `area/security`, `official-cve-feed`, `committee/security-response` |
| reproducibility | `triage/needs-information`, `triage/not-reproducible` → not enough info; a fixing PR → enough info |
| `duplicates` | `triage/duplicate`, `stateReason: DUPLICATE`, the duplicate event |
| invalid / not a bug | `triage/support` (redirected to support channels), `triage/unresolved`, `NOT_PLANNED` closed by a human, locked as spam |
| triage status | `needs-triage`, `needs-sig`, `needs-kind`, `triage/accepted` |
| lifecycle | `lifecycle/stale`, `lifecycle/rotten`, `lifecycle/frozen` |
| extra | `good first issue`, `help wanted` |

The exact label names are checked against the repo's label list when the
mapping is written.

### Close outcome

`normalize.py` derives one `close_outcome` per issue:

| Outcome | Rule |
|---|---|
| `fixed` | closed by a PR or commit, or a fixing PR is linked |
| `duplicate` | `stateReason: DUPLICATE`, `triage/duplicate`, or a duplicate event |
| `invalid` | `triage/support`, locked as spam or off-topic, or `NOT_PLANNED` closed by a human with no fix |
| `needs_info_timeout` | closed while `triage/needs-information` was set |
| `stale` | closed by the triage bot after `lifecycle/rotten` |
| `completed_other` | closed by a human, `COMPLETED`, none of the above |
| `open` | still open |

It is metadata, not a triage field. It lets results be broken down (for
example "accuracy on issues that were really fixed") and gives extra checks
(an issue predicted as spam that was later fixed by a PR is a clear error).

### Normalization and leak removal

- **Remove slash commands from the body.** The k8s templates ask authors to
  write `/kind bug`, `/sig node` and so on, and the bot turns those lines into
  labels. Lines starting with `/kind`, `/sig`, `/area`, `/priority`,
  `/triage`, `/assign`, `/cc` and `/label` are removed. Without this,
  `heuristic` v1 would score near 100% on `kind` by reading the answer.
- **Record label provenance.** A label that came from a command in the body is
  the author's choice; one added later is a triager's. That allows a
  `triager_verified` subset for stricter evaluation.
- **Snapshot repo context as of `created_at`**: the latest release at that
  point, and the label list.
- **Eligibility:**
  - Exclude issues still carrying `needs-triage`, or with no `kind/*` label.
  - Exclude issues transferred out of the repo.
  - Keep stale-closed issues, but mark them in metadata.
- Fields with no label are left out of the expected answers, not filled with a
  default. An issue without a `priority/*` label simply isn't scored on
  severity.

### Building datasets

```
uv run build-dataset --repo kubernetes --split dev --limit 200 --stratify kind
```

- The dev/test split is based on a hash of the issue number (80/20), so the
  splits never overlap and stay stable when the dump is re-fetched.
- Inside a split, issues are ordered by the same hash, so a larger limit always
  contains the smaller one: `--limit 400` includes every case from
  `--limit 200`. Runs at different sizes stay comparable.
- Stratified by `kind`, so rare kinds appear in small datasets.
- Output: `datasets/kubernetes-dev-200.yaml` (pydantic-evals format,
  committed). Its metadata records the filter parameters and the hash of the
  source dump.
- `run.py --limit N` can cut a dataset down further for a quick check.

Tune on `dev`. Report experiment results on `test`.

### Fetching

- GitHub GraphQL with `GITHUB_TOKEN`, 100 issues per page, newest first.
  Only issues are returned, so the PRs that dominate the k8s repo aren't
  downloaded.
- Resumable: the cursor is saved after every page and raw pages are appended to
  `data/raw/kubernetes/*.jsonl`.
- Timeline events are fetched in the same query where possible, and in
  follow-up queries for issues with long timelines.
- GraphQL allows 5000 points per hour, so the 3000 eligible issues take a few
  minutes to an hour depending on how much timeline data comes with them. The
  lighter corpus fetch for all issues takes longer and only needs to run once.

## Eval harness (pydantic-evals)

### Running

```
uv run triage-eval --dataset datasets/kubernetes-dev-200.yaml \
  --strategy workflow --config configs/workflow-jev.yaml \
  --judge-model anthropic:claude-sonnet-5 --run-id ... --repeats 1 --label "jev only"
```

- The task posts each case to `/{Strategy}/{key}/run` on the Restate ingress
  and, on `409 Conflict`, reads the stored result through `attach`, as in the
  prototype.
- Usage the Rust service returns is recorded with `increment_eval_metric`
  (`input_tokens`, `output_tokens`, `cost`, `latency_ms`, `llm_calls`,
  `decision_calls`, `tool_calls`).
- Experiment name: `{strategy}@{version} {config_hash} {label}`. Metadata:
  `git_sha`, the full config, `dataset_hash`, `suite_id`.

### Evaluators

Code-based, one per field with expected answers:

| Evaluator | Metric |
|---|---|
| `KindMatches` | accuracy, plus macro-F1 across the run |
| `AreaMatches` | accuracy (exact `sig/*`) |
| `SeverityMatches` | accuracy, plus distance (off by one level is better than off by three) |
| `RegressionMatches`, `SecurityMatches` | precision and recall across the run (rare classes) |
| `EnoughInfoMatches` | accuracy against needs-information / fixed |
| `DuplicateRecall` | recall@k of the duplicate target |
| `SanityChecks` | hard errors, such as predicting spam for an issue later fixed by a PR |

An evaluator skips a case when the expected answer is missing or the strategy
didn't produce the field. Skips are counted separately, so coverage is visible:
the heuristic isn't penalized for having no reply, but it doesn't look better
by answering less either.

LLM judge (`LLMJudge`), only where there are no labels: `quality` and
`suggested_reply`. The model is configurable (`--judge-model` / `JUDGE_MODEL`,
a pydantic-ai model string), and should be from a different family than the
strategies under test. The judge model and rubric version are part of the
suite ID, because scores from different judges can't be compared.

To check the judge: hand-label about 50 issues for quality, and report how
often the judge agrees with the hand labels.

### Suite ID

A hash of: dataset hash, evaluator versions, rubric version, judge model,
composite weights. Runs are only compared within the same suite. Changing any
of these starts a new suite instead of looking like a regression.

## Comparing cost and quality

1. **Composite score:** a weighted mean of the per-field scores. The weights
   are fixed before the first run and stored with the suite. Per-field scores
   are always reported too.
2. **Pareto frontier:** cost per issue (x) against composite score (y). A
   strategy is only worth considering if no other one is both cheaper and
   better.
3. **Marginal cost along the frontier:** for example "agent costs +$0.004 per
   issue for +3 points over workflow". The demo shows the trade-off; whether
   it's worth paying depends on the budget.
4. **Paired comparisons:** every strategy sees the same cases, so `compare.py`
   counts the cases where two runs disagree (head-to-head) and reports a
   paired test. This needs far fewer cases than comparing two accuracies.
5. **Wilson 95% intervals** on every rate.
6. **Also reported:** latency, run-to-run consistency across repeats, and
   calibration (reliability curves for Jev, Clef, LLM self-reported confidence
   and ensemble agreement).

Cost is in USD, from each provider's prices in a table in `triage-core`. The
judge's cost is reported separately, never added to a strategy's cost.

### Hypotheses (written before the first run)

- `workflow`, `agent` or `ensemble` will be on the Pareto frontier.
- `workflow` with no escalation will be close to `single_shot` on typed fields
  for a fraction of the cost.
- `agent` will win clearly on duplicates and lose on cost.
- Jev's confidence will be better calibrated than LLM self-reported
  confidence.

If the results disagree, that's a finding, not a failure.

## Experiments

Each experiment ends with an eval run and a comparison against the previous
one.

| # | Experiment | Change to compare | What it shows |
|---|---|---|---|
| 0 | Foundation | workspace, devenv (Rust, Python, uv), docker compose, telemetry, fetcher, normalizer, dataset builder, harness with a stub strategy | the pipeline works end to end |
| 1 | `heuristic` v1 → v2 | for example detecting template sections, mapping keywords to `sig/*` | the eval loop itself, at $0 |
| 2 | `single_shot` small model → large model | config only | first points on the cost/quality chart; the LLM judge comes in with `suggested_reply` |
| 3 | `workflow` without escalation → with escalation → threshold sweep; Jev vs. Clef | config | a cost/quality curve within one strategy; calibration |
| 4 | `agent` without tools → with `search_issues` | tool use | where agents pay off (duplicates) |
| 5 | `ensemble` | 3 voters vs. 1; vote vs. judge | buying accuracy with more calls |
| 6 | `draft_verify` | against `workflow` with escalation | workflow-coordinated vs. agent-coordinated escalation |
| 7 | Final run on `test` | – | the Pareto chart across all strategies |

### Experiment 0 in detail

1. devenv: add Python and uv next to Rust; `.env` for `LOGFIRE_TOKEN`,
   `GITHUB_TOKEN` and provider keys.
2. `docker-compose.yaml` with Restate, reaching the service at
   `host.docker.internal:9080`, traces to Logfire.
3. Cargo workspace with `triage-core` (types, configs, schema generation) and
   `triage-service` (`serve`, telemetry, a stub `Heuristic` workflow).
4. `fetch.py` and `normalize.py` for Kubernetes; `repos/kubernetes.yaml`
   checked against the real label list.
5. `build_dataset.py`; first committed dataset `kubernetes-dev-200.yaml`.
6. `run.py`, `evaluators.py` and `compare.py`, run against the stub, results
   visible in Logfire.

## Open questions

- Default models: **Ollama for now**, for chat and for decision models (Nimble,
  Tev1 through `allama-systemone`). Which Ollama models exactly is decided in
  experiment 0, based on the hardware. Hosted models (and a hosted judge) can be
  swapped in later through config. Local calls cost $0, so they need shadow
  prices (tokens priced at a comparable hosted model's rates) for the cost axis
  to mean anything, and local latency depends on the hardware.
- Access to Jev (early access, needs an API key) and Clef over REST through
  `allama-systemone`: to be confirmed in experiment 3.

## Deviations

Deliberate departures from this plan, as built:

- **Experiment 3:** no Jev or Clef keys were available (the open question
  above), so the decision-model comparison is local tev1:4b vs tev1:0.8b,
  and the Jev-calibration hypothesis is checked with tev1's confidence.
  `configs/workflow-jev.yaml` and `configs/workflow-clef.yaml` run the
  original comparison once keys exist.
- **Experiment 4:** the "with tools" run gives the agent all five tools, not
  `search_issues` alone, so it measures the full toolset.
- **Judge family:** the default judge is `ollama:qwen3:4b`, the strategies'
  own family, because the local Ollama has only Qwen-based models. Judged
  scores may favour Qwen output; pass another family's model for judged
  results that matter (`evals/README.md`, "The judge's model family").
- **Issues whose body as filed is lost** (edited, creation snapshot deleted;
  mostly security placeholders later filled in with the advisory) are kept
  in the normalized data with `original_body_unknown` but left out of the
  datasets (`build-dataset --keep-unknown-body` keeps them). That makes the
  test split 580 cases rather than 589.
- **Titles:** cases use the title as filed (before the first rename), like
  the body. In the search corpus only the eligible issues have that history;
  the other issues keep their current title.
- **Models:** the experiments in RESULTS.md ran on hosted models: GPT-6 Luna
  (strategies) and Sol (m2's larger model, the ensemble judge, the eval judge)
  on Azure OpenAI, and TypeSafe Jev, which answers the Jev/Clef question for
  Jev. Local Ollama runs were too slow on the available CPU-only machine
  (`run-experiments --profile local` still runs them).
- **Experiment 1:** heuristic v3 vs v3 with base rates, because the leak fixes
  changed the dataset inputs after the original v1 → v2 run and that code is
  gone (archived results in `evals/results/archive-pre-leak-fixes/`).


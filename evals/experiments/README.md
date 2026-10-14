# Experiments

The experiments from PLAN.md, for `run-experiments`
(`triager_evals/experiments.py`). One file per experiment, plus `profiles.yaml`:

| File | What |
|---|---|
| `profiles.yaml` | the model profiles (`openai`, `anthropic`, `gemini`, `azure`, `local`), `default_profile`, and `defaults` (dataset pattern, sizes, concurrency, seconds per call for estimates) |
| `heuristic-base-rates.yaml` | heuristic v3 → v3 with base rates (formerly m1) |
| `single-shot-model-size.yaml` | single_shot llm → llm_large (m2) |
| `workflow-escalation.yaml` | workflow escalation, threshold sweep, decision vs decision_alt (m3) |
| `agent-tools.yaml` | agent without → with tools (m4) |
| `ensemble-voters.yaml` | ensemble 1 voter → 3 voters (vote) → 3 voters (judge) (m5) |
| `escalation-coordination.yaml` | draft_verify vs workflow with escalation (m6) |
| `final-comparison.yaml` | every strategy on the test split (m7), and the `overview` (the Pareto frontier in `overview.md`) |

```
uv run run-experiments --list                      # experiments, and every step in run order
uv run run-experiments --dry-run                   # the plan for 20 issues, default profile
uv run run-experiments --size 500                  # everything on the 500-issue sets
uv run run-experiments --only workflow-escalation  # one experiment (the file name)
uv run run-experiments --step workflow-esc-0.8     # one step
uv run run-experiments --file experiments/agent-tools.yaml --dry-run   # only the experiments in one file
uv run run-experiments --profile local --model judge=azure:gpt-6-luna
```

## How the files are loaded

`run-experiments` reads `profiles.yaml` and every other `*.yaml` in this
directory as one spec (`--file` takes this directory, the default, or a
single experiment file; a single file takes its profiles from
`profiles.yaml` next to it, else from this directory's).

- An experiment's id is its file name without `.yaml`; an `id:` in the file
  is optional and must equal it. `--only` takes that id.
- Step ids are unique across all files; a duplicate step or experiment id is
  an error that names both files.
- References (`baseline`, `source`, `config_from_best`, `compare` runs, the
  `overview`) work across files. A reference to a step that isn't in the
  loaded files is an error (for example `escalation-coordination.yaml` on
  its own: it compares with `workflow-escalation`'s steps).
- Run order: a step runs after the steps it derives from (`source`), takes
  its config from (`config_from_best`) or has as `baseline`, and after the
  other experiments' steps that its experiment compares with. Otherwise
  steps run in file name order, then in their order within the file. So steps
  of different experiments can interleave: `final-heuristic` runs right after
  `heuristic@test`, and `draft-verify` right after `workflow-esc-0.8`. A
  dependency cycle is an error. `--list` shows the order. Each experiment's
  report is written once its last step is done.

Run ids, config hashes and report paths don't depend on which file a step is
in, so moving a step to another file keeps its results.

## An experiment file

```yaml
# a short comment on what it shows
title: agent without tools → with tools     # id: optional, = the file name
shows: where agents pay off
steps:
  - id: agent-no-tools
    strategy: Agent
    split: dev
    config: {model: "{llm}", max_turns: 8, tools: []}
    label: agent {llm}, no tools
  - ...
compare:
  - title: tool use
    baseline: agent-no-tools
    runs: [agent-no-tools, agent-tools]
```

Each experiment is a list of steps and a compare section. A step is one
triage-eval run (one pydantic-evals experiment in Logfire):

| Key | Meaning |
|---|---|
| `id` | the step id, unique across all files; comparisons, baselines and `--step` refer to it. The run id is composed from it: `{id}-{split}-{size}-{profile}`, e.g. `workflow-esc-0.8-dev-20-openai` |
| `strategy` | `Heuristic`, `SingleShot`, `Workflow`, `Agent`, `Ensemble` or `DraftVerify` |
| `split` | `dev`, `test`, or a list (`[dev, test]`) for one run per split. The dataset is `defaults.dataset` with `{split}` and the `--size` filled in. A step with several splits is referred to as `id@split` (`heuristic@test`); `--step heuristic` selects all of its splits, and a bare reference from a step with a split picks the same split |
| `dataset` | optional: an explicit dataset file instead (`{split}`/`{size}` allowed) |
| `config` | the strategy config, inline. Strings may hold model-role placeholders (`{llm}`, `{llm_large}`, `{decision}`, `{decision_alt}`), filled in from the profile; the result is checked against `schema/<strategy>-config.schema.json` (unknown keys rejected). Its config hash (part of the Restate workflow key and the results file name) is computed after the substitution, so another model never reuses stored results. A string is read as a config file |
| `label` | the run's label (part of its Logfire experiment name); may hold placeholders too, so the experiment name shows the models |
| `judge` | `true` (default) or `false`. The judge model is always the `judge` role; `--no-judge` turns the judge off everywhere |
| `baseline` | the step this one is a change from, for the "before → after" table of the report |
| `version` | optional: the strategy code version the step is about. A step that has to call the service is blocked while `Info/versions` reports another version |
| `source` | optional: derive this step offline from another step's stored outputs, re-scored on this step's dataset (whose cases must be in the source run). Falls back to a normal run |
| `config_from_best` | optional: use the config of whichever of these steps has the highest composite (ties: the cheaper); `config` is the fallback |
| `optional` | `true`: only with `--include-optional` |
| `max_concurrency`, `calls_per_case` | optional overrides (see `defaults` in `profiles.yaml`) |

`compare` entries have a `title`, a `baseline` and `runs` (step ids). An
`overview` (`title`, `runs`) may appear once, in an experiment file or in
`profiles.yaml`; it is in `final-comparison.yaml`.

A step whose results file (for the exact config hash) exists is skipped; if
it was scored with older evaluators or another suite (another judge, say), it
is re-scored offline from the file. `--force` re-runs it through the
service, where Restate returns the stored outputs for keys it still has.

## Model roles

| Role | Used for |
|---|---|
| `llm` | the cheap/default LLM: single_shot, escalation, agent, verifier |
| `llm_large` | the larger LLM: `single-shot-model-size`'s comparison, the ensemble's second voter and judge aggregator |
| `decision` | the decision model (System One) |
| `decision_alt` | the decision model `workflow-escalation` compares with |
| `judge` | the eval judge (pydantic-ai side: `azure:...`, `ollama:...`, ...) |

Resolution, highest first: `--model ROLE=ADDRESS`, the env
`EXPERIMENT_MODEL_<ROLE>` (`JUDGE_MODEL` is an alias for the judge), then the
profile (`--profile`, else `$EXPERIMENT_PROFILE`, else `default_profile`).

## Adding your own experiment

Say you want to compare the agent on two specific models. Put a new file
here, e.g. `agent-models.yaml`:

```yaml
# The agent with tools on Luna vs Sol, whatever the profile says.
title: agent Luna → Sol
shows: what the larger model buys the agent
steps:
  - id: agent-luna
    strategy: Agent
    split: dev
    config: {model: azure:gpt-6-luna, max_turns: 8}
    label: agent azure:gpt-6-luna
  - id: agent-sol
    strategy: Agent
    split: dev
    config: {model: azure:gpt-6-sol, max_turns: 8}
    label: agent azure:gpt-6-sol
    baseline: agent-luna
  - id: agent-profile-llm
    strategy: Agent
    split: dev
    config: {model: "{llm}", max_turns: 8}
    label: agent {llm}
    baseline: agent-luna
compare:
  - title: model
    baseline: agent-luna
    runs: [agent-luna, agent-sol, agent-profile-llm]
```

Then `uv run run-experiments --only agent-models --dry-run` and without
`--dry-run` to run it (the report lands in
`evals/results/experiments/{size}-{profile}/agent-models.md`).

- Literal addresses (`azure:gpt-6-luna`) are used as written: `--profile`,
  `EXPERIMENT_MODEL_LLM` and `--model llm=...` don't change them.
- Placeholders (`"{llm}"`) follow the profile and the overrides:
  `agent-profile-llm` is Luna under `openai` and `azure`, `ollama:qwen3:1.7b` under
  `local`, and whatever `--model llm=...` says.
- The judge is always the `judge` role, for literal and placeholder steps
  alike (`--model judge=azure:gpt-6-luna` for a cheaper one, `--no-judge` for
  none).
- Step ids must not clash with any other file's (`agent-tools` is taken), and
  the run ids still carry the profile name (`agent-luna-dev-20-openai`), even
  though that step's model doesn't depend on it.

# Issue triager

A GitHub issue triager built six different ways, so the approaches can be
compared on quality, cost and latency. Each approach is a durable
[Restate](https://restate.dev) workflow in a Rust service; a Python harness
runs them over real Kubernetes issues with
[pydantic-evals](https://ai.pydantic.dev/evals/), and every run lands in
[Logfire](https://logfire.pydantic.dev) as an experiment you can compare.

The point is the loop: start simple, measure, change one thing, measure
again, compare. The walkthrough below does exactly that in about 15 minutes.
The design is in [PLAN.md](PLAN.md), our own results in [RESULTS.md](RESULTS.md).

## Strategies

All six take the same input (an issue as it was filed, plus the repo's labels
and latest release) and return the same [`Triage`](crates/triage-core/src/triage.rs):
kind, area (`sig/*`), severity, regression, security, quality, a
reproducibility checklist, outdated version, scope, duplicates and a
suggested reply, each with a confidence.

| Service | How it works |
|---|---|
| `Heuristic` | Regexes, template sections and keyword tables. No models; the baseline. |
| `SingleShot` | One LLM call with structured output. |
| `Workflow` | One decision-model call ([TypeSafe Jev](https://typesafe.ai)) answers every typed field at once; targeted LLM calls only for the fields it's unsure about. |
| `Agent` | A tool-using agent over a local snapshot of the repo's issues (search, get issue, labels, templates, releases). |
| `Ensemble` | Several LLM and decision-model voters, aggregated by a confidence-weighted vote or an LLM judge. |
| `DraftVerify` | A decision-model draft (readable early), then one agent verifies only the low-confidence fields. |

## Walkthrough

### 0. Prerequisites

- Docker (with compose).
- Either [devenv](https://devenv.sh) (`devenv shell` gives you Rust, Python and
  uv), or just [uv](https://docs.astral.sh/uv/) for the Python harness, with
  the service running in Docker. Rust is only needed to work on the service.

### 1. Choose a provider

```sh
cp .env.example .env
```

Pick a profile in `.env` and set its key. Every model the experiments use
comes from that profile ([evals/experiments/profiles.yaml](evals/experiments/profiles.yaml)):

| `EXPERIMENT_PROFILE` | Cheap model (`llm`) | Large model (`llm_large`, judge) | Key |
|---|---|---|---|
| `openai` (default) | GPT-6 Luna | GPT-6 Sol | `OPENAI_API_KEY` |
| `anthropic` | Claude Haiku 4.5 | Claude Sonnet 5 | `ANTHROPIC_API_KEY` |
| `gemini` | Gemini 3.5 Flash-Lite | Gemini 3.1 Pro | `GEMINI_API_KEY` |
| `azure` | GPT-6 Luna deployment | GPT-6 Sol deployment | `AZURE_OPENAI_RESOURCE`, `AZURE_OPENAI_API_KEY` |
| `local` | qwen3:1.7b on Ollama | qwen3:4b on Ollama | none |

> [!WARNING]
> **Hosted models cost money.** The walkthrough below costs well under $1 on
> the `openai` profile (most of it is the judge), a few dollars on Claude or
> Gemini. Running *every* experiment costs about $1.50–7 on 20 issues and
> $36–170 on 500, depending on the provider and judge.
> [docs/costs.md](docs/costs.md) has the full table, including time.
> `run-experiments --dry-run` shows the plan before anything is spent, and
> `--model judge=<cheap model>` cuts the biggest line item.

`local` costs nothing but is slow without a GPU (minutes per issue for the
LLM strategies). The GPT-6 models have been run end to end (through the
`azure` profile); `openai`, `anthropic` and `gemini` are configured and
tested offline.

For step 6 you'll also need a [TypeSafe](https://typesafe.ai) key
(`TYPESAFE_API_KEY`) for the Jev decision model.

### 2. Logfire

Set `LOGFIRE_TOKEN` in `.env` (a write token from your Logfire project). Each
eval run then shows up as an **experiment** in Logfire's Experiments tab, and
each case links to its full trace: the harness, Restate, the Rust service's
model calls (with tokens and cost) and the judge.

Without a token everything still works, with results in the console and in
`evals/results/`. To send traces to another OpenTelemetry backend instead,
see [docs/telemetry.md](docs/telemetry.md).

### 3. Start Restate and the service

```sh
docker compose --profile app up -d --build
```

This starts the Restate server and builds and starts the triage service. Tell
the harness where Restate can reach the service, in `.env`:

```sh
SERVICE_URL=http://service:9080
```

Working on the service yourself? Start only Restate (`docker compose up -d`)
and run the service from source with `cargo run -- serve` (it listens on
:9080; keep the default `SERVICE_URL`).

Restate's UI is at <http://localhost:9070>. Keep it open: every triage is a
durable workflow there, and you can see each model call in its journal.

### 4. The harness

From the `evals/` directory (the harness reads `../.env` itself):

```sh
cd evals
uv run run-experiments --list
```

This shows the resolved models and every experiment. Nothing runs yet.

### 5. A baseline: the heuristic

```sh
uv run run-experiments --step heuristic@dev
```

No model calls (only the judge, which rates quality): keyword rules and the
issue templates, on the 20-issue dev set. Expect a composite score in the low
60s. The composite is 0–100, a weighted score over kind, area, severity,
reproducibility and more, where a field a strategy doesn't answer counts as 0.
With 20 issues, differences of a few points are noise.

The console prints a summary line per run, and the report goes to
`evals/results/experiments/20-<profile>/`. In Logfire it's an experiment named
`Heuristic@3 … heuristic v3` on the dataset `kubernetes-dev-20`.

### 6. We can do better

**One LLM call**, with the cheap and then the large model:

```sh
uv run run-experiments --step single-shot-llm --step single-shot-llm-large
```

Expect a jump to the mid-70s for a fraction of a cent per issue with the
cheap model, and a few more points for roughly 15× the cost with the large
one.

**A decision model** (needs `TYPESAFE_API_KEY`): Jev answers every typed field
in one call, in about 0.2 seconds for a hundredth of a cent per issue. Then
let an LLM re-check only the fields Jev is unsure about:

```sh
uv run run-experiments --step workflow-decision-only --step workflow-esc-0.8
```

**Compare in Logfire.** Open the Experiments tab, pick the dataset
`kubernetes-dev-20`, and select the runs: scores per field, cost, tokens and
latency side by side, down to single cases and their traces.

Re-running a step is free: results are kept by Restate and in
`evals/results/`, so only new configurations call the models.

### 7. Next

- **More experiments:** agents with and without tools, ensembles, the threshold
  sweep, and a final comparison of everything on the test split:
  `uv run run-experiments --only <experiment>` or, for all of them,
  `uv run run-experiments` (20 issues) and `--size 500` (see the costs first).
  The agent strategies search a local copy of the repo's issues; fetch it
  first (needs a GitHub token, about 40 minutes): see
  [evals/README.md](evals/README.md#datasets).
- **Change the models:** `--model llm=anthropic:claude-haiku-4-5`,
  `EXPERIMENT_MODEL_JUDGE=…` in `.env`, or another `--profile`.
- **Write your own experiment:** add a YAML file to
  [evals/experiments/](evals/experiments/README.md), e.g. one strategy on
  several models.
- **Results we got:** [RESULTS.md](RESULTS.md).

### Troubleshooting

- **`connection refused` on :8080 or :9070.** Restate isn't running:
  `docker compose --profile app up -d`.
- **Every case fails with a 404 or "service not found".** The service isn't
  registered. The harness registers `$SERVICE_URL` on start; check that it
  matches how the service runs (`http://service:9080` in Docker,
  `http://host.docker.internal:9080` from source).
- **401/403 from a provider.** The profile's key is missing in `.env`. The
  service container reads `.env` at start, so restart it after changing keys:
  `docker compose --profile app up -d`.
- **The `workflow-decision-alt` step hangs or fails.** It compares Jev with a
  local model on Ollama. Start it with `docker compose --profile local up -d`
  and `docker compose exec ollama ollama pull tev1:0.8b-q8_0`, or skip the step.

Stop everything with `docker compose --profile app --profile local down`.

## Layout

| Path | What |
|---|---|
| [crates/triage-core](crates/triage-core) | The contract: request, response, configs, usage, prices, config hash; generates [schema/](schema/) |
| [crates/triage-service](crates/triage-service) | The Restate service: strategies, models, tools, telemetry ([README](crates/triage-service/README.md)) |
| [evals](evals) | Data pipeline, eval harness, comparison, experiments ([README](evals/README.md)) |
| [evals/experiments](evals/experiments) | One YAML file per experiment, plus the model profiles |
| [configs](configs) | Strategy configs for one-off `triage-eval` runs |
| [datasets](datasets) | Committed pydantic-evals datasets (Kubernetes dev/test splits: 20, 200, 500, 580 issues) |
| [docs](docs) | [Costs and timing](docs/costs.md), [telemetry](docs/telemetry.md) |
| `data/` | Gitignored: raw GitHub dumps, normalized issues, search corpus |

## Tests

```sh
cargo test --workspace
cargo clippy --workspace -- -D warnings
(cd evals && uv run pytest -q)
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

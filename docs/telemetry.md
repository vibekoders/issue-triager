# Tracing and experiments

Three components emit traces: the Python harness, the Rust service and the
Restate server. By default they all go to [Logfire](https://logfire.pydantic.dev),
which also has the Experiments view for comparing runs. Any other
OpenTelemetry backend (Jaeger, Tempo, Honeycomb, an otel-collector, ...)
works too, with the settings below. You get the traces there, but not the
experiments UI.

## What each component sends

| Component | `service.name` | What | Configured by |
|---|---|---|---|
| Harness (`triage-eval`, `run-experiments`) | `triage-evals` | One span per pydantic-evals experiment (`evaluate {name}`), one per case (`case: {case_name}`) with the task (`execute {task}`) and each evaluator. The httpx request to Restate's ingress (`POST`) carries `traceparent`. The judge's pydantic-ai calls are separate spans with their own token usage, which never counts towards the strategy's `cost` metric. | `logfire.configure(...)` in [evals/triager_evals/run.py](../evals/triager_evals/run.py) (`setup_logfire`), plus `instrument_httpx` and `instrument_pydantic_ai` |
| Rust service | `triage-service` | `triage.run`, the handler span (`strategy`, `strategy.version`, `config_hash`, `eval.key`, `trace.linked`). Inside it: `chat` spans (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens` / `output_tokens`, `gen_ai.response.finish_reasons`, `llm.step`, `llm.calls`, `llm.latency_ms`, `llm.cost_usd`), `decision` spans (the same, plus `llm.questions`) and `tool` spans (`tool.name`, `tool.arguments`, `tool.is_error`, `tool.latency_ms`). Log events become log spans. Events from a replayed journal are dropped. | [crates/triage-service/src/telemetry.rs](../crates/triage-service/src/telemetry.rs) (the logfire crate 0.13), [trace.rs](../crates/triage-service/src/trace.rs) |
| Restate server | `Restate` | Ingress and invocation spans (`ingress Heuristic/{key}/run`, ...). | `RESTATE_TRACING_ENDPOINT` and `RESTATE_TRACING_HEADERS__*` in [docker-compose.yaml](../docker-compose.yaml) |

**How one case's trace hangs together.** The harness instruments httpx
before any client exists, so the POST to `/{Strategy}/{key}/run` carries the
case's `traceparent`. The Restate server continues that trace with its ingress
span. Restate stores the invocation's headers, and the service parents
`triage.run` on the stored `traceparent`, not on the SDK's per-attempt span. So
a retried or resumed invocation lands in the same trace, and every model and
tool span of the workflow sits under the harness's case span.
`trace.linked=false` on `triage.run` means no `traceparent` arrived, for
example on a plain `curl`. Restate doesn't put its own span context into the
headers, so `triage.run` sits beside Restate's spans in the trace, not under
them.

## Logfire (the default)

In `.env` (devenv loads it; docker compose reads it for the Restate server):

```sh
LOGFIRE_TOKEN=pylf_v1_eu_...      # a write token of your Logfire project
LOGFIRE_ENVIRONMENT=dev           # optional: deployment.environment.name for the harness and the service
LOGFIRE_REGION=eu                 # eu or us; only the Restate server needs it (it can't read the region from the token)
```

- **Harness**: `send_to_logfire="if-token-present"`, so it sends only when
  `LOGFIRE_TOKEN` is set. It flushes before exiting, also after an error.
- **Rust service**: sends only when `LOGFIRE_TOKEN` is set and not empty
  (`SendToLogfire::No` otherwise, so an empty `LOGFIRE_TOKEN=` from
  `.env.example` doesn't make it export without credentials). The region
  comes from the token. Restart the service after changing `.env`.
- **Restate server**: compose sets `RESTATE_TRACING_ENDPOINT` to
  `otlp+https://logfire-${LOGFIRE_REGION}.pydantic.dev/v1/traces` and the
  `Authorization` header to `LOGFIRE_TOKEN`. Run `docker compose up -d restate`
  again after changing them. Without a token the server still tries Logfire and
  is rejected. That is harmless, but you can point it elsewhere, see below.

**Where to compare.** Each `run-experiments` step (or `triage-eval --config`)
is one pydantic-evals experiment named `{Strategy}@{version} {config_hash}
{label}`, with the run's metadata (profile, models, size, split, experiment and
step id, config, suite id). The dataset name is the dataset file's
(`kubernetes-dev-20`, `kubernetes-test-500`, ...). Open the dataset in
Logfire's **Experiments** view (evals) and select the experiments to compare,
e.g. the `final-comparison` steps of one profile, or one step under two
profiles. Compare only experiments on the same dataset and suite (the same
judge setting). Per case, the metrics `input_tokens`, `output_tokens`, `cost`
(USD), `latency_ms`, `llm_calls`, `decision_calls` and `tool_calls` are the
strategy's. Each case links to its trace, with the service's model and tool
spans inside.

## Any OpenTelemetry backend

What each component supports, from its source (logfire 5.2.0 for Python in
`.devenv/state/venv`, the logfire crate 0.13.1 and opentelemetry-otlp 0.32
for Rust, Restate 1.7.13):

| Component | Generic OTLP | Protocol | Headers |
|---|---|---|---|
| Harness (Python logfire) | **Yes, via the standard variables.** `logfire.configure()` adds an OTLP exporter whenever `OTEL_EXPORTER_OTLP_ENDPOINT` (or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is set. It works **in addition to** Logfire when `LOGFIRE_TOKEN` is set, and on its own when it isn't. Metrics and logs go to the same endpoint unless `OTEL_METRICS_EXPORTER=none` / `OTEL_LOGS_EXPORTER=none`. | HTTP/protobuf only (`opentelemetry-exporter-otlp-proto-http`); the path `/v1/traces` is appended | `OTEL_EXPORTER_OTLP_HEADERS` (`key=value,key2=value2`) |
| Rust service (logfire crate 0.13) | **Not through `OTEL_EXPORTER_OTLP_ENDPOINT`**: the crate never reads it, and with no token `telemetry.rs` sets `SendToLogfire::No`, which builds no exporter at all. **Workaround:** the crate reads `LOGFIRE_BASE_URL` and uses it instead of Logfire's URL. So `LOGFIRE_TOKEN=<anything non-empty>` plus `LOGFIRE_BASE_URL=http://collector:4318` exports OTLP there. It replaces Logfire, so it can't send to both. | `OTEL_EXPORTER_OTLP_PROTOCOL` / `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` are read, but only `http/protobuf` is compiled in (the default). `grpc` needs the crate's `export-grpc` feature and fails at startup without it. `/v1/traces` (and `/v1/logs`) is appended to the base URL | Only `Authorization: Bearer <LOGFIRE_TOKEN>`; `OTEL_EXPORTER_OTLP_HEADERS` is ignored. A backend that wants another header needs a collector in between |
| Restate server | **Yes**: `RESTATE_TRACING_ENDPOINT`. Compose takes it from the environment / `.env` and falls back to Logfire. | `http://host:4317` is OTLP/gRPC; `otlp+http://host:4318/v1/traces` or `otlp+https://...` is OTLP/HTTP | `RESTATE_TRACING_HEADERS__<NAME>=value` (compose sets `__AUTHORIZATION` from `LOGFIRE_TOKEN`) |

Logfire shows span names as message templates, so a generic backend lists the
harness's spans as `case: {case_name}`, `evaluator: {evaluator_name}` and so on.
The rendered message is in the `logfire.msg` attribute.

**What you lose.** pydantic-evals' Experiments view (side-by-side experiments
per dataset, per-case scores and metrics) is Logfire-specific. With another
backend you get the traces (each case with its Restate, handler, model and
tool spans, tokens and `llm.cost_usd` as attributes) but no experiments UI. The
comparison then comes from the results files: `evals/results/<run id>/*.json`,
the `run-experiments` reports in `evals/results/experiments/{size}-{profile}/`
and `compare-runs`.

### Worked example: a local Jaeger

**Tested** on 2026-10-14 with `jaegertracing/jaeger:latest` (v2.22.0), a
throwaway Restate 1.7.13 container, the service binary and
`triage-eval --strategy Heuristic --limit 2 --no-judge`. Jaeger listed the services
`triage-evals`, `Restate` and `triage-service`, and each case was one trace with
harness, Restate ingress and `triage.run` spans. Model spans were not
exercised (the heuristic makes no calls), but they use the same exporter.

```sh
# Jaeger: UI on :16686, OTLP gRPC on :4317, OTLP HTTP on :4318
docker run -d --name jaeger -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/jaeger:latest
```

`.env` (Restate server and harness):

```sh
LOGFIRE_TOKEN=                                     # no Logfire for the harness
# Restate server, from inside its container (compose maps host.docker.internal to the host)
RESTATE_TRACING_ENDPOINT=http://host.docker.internal:4317
# Harness: OTLP/HTTP to Jaeger; Jaeger takes traces only, so turn off metrics and logs
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_METRICS_EXPORTER=none
OTEL_LOGS_EXPORTER=none
```

```sh
docker compose up -d restate        # re-create it with the new endpoint

# The service: give the Rust exporter a dummy token and Jaeger as its "Logfire" URL.
# Set these on the command, not in .env: the Python logfire also reads LOGFIRE_BASE_URL.
LOGFIRE_TOKEN=local LOGFIRE_BASE_URL=http://localhost:4318 cargo run -- serve

cd evals && uv run triage-eval --dataset datasets/kubernetes-dev-20.yaml --strategy Heuristic --no-judge --limit 2
# open http://localhost:16686 and pick the service triage-evals
```

Without `OTEL_METRICS_EXPORTER=none` the harness prints `Failed to export
metrics batch code: 404`, because Jaeger has no OTLP metrics receiver. The
traces still arrive. For a backend that needs credentials, run an
otel-collector locally (receivers `otlp` on 4317/4318, an `otlphttp` or `otlp`
exporter with the backend's headers) and point all three components at it.
The collector is also how you send to Logfire and another backend at once
from the Rust service.

To go back to Logfire, remove the `RESTATE_TRACING_ENDPOINT` and `OTEL_*`
lines, set `LOGFIRE_TOKEN`, recreate the Restate container, and start the
service without `LOGFIRE_BASE_URL`.

### Follow-up: generic OTLP for the Rust service without the workaround

A small change in `telemetry.rs` would make the service honor the standard
variables, alongside Logfire or on its own. Not done here, and untested:

1. Add `opentelemetry-otlp = "0.32"` with its default features (`http-proto`,
   `reqwest-blocking-client`) to the service. Its HTTP exporter reads
   `OTEL_EXPORTER_OTLP_ENDPOINT` / `_TRACES_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`
   and the protocol itself when no endpoint is set in code.
2. In `init`, when `OTEL_EXPORTER_OTLP_ENDPOINT` or
   `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is set and not empty, add the exporter
   as an extra span processor before `finish()`:

   ```rust
   let mut config = logfire::configure().local() /* ...as now... */;
   if otlp_endpoint_set() {
       let exporter = opentelemetry_otlp::SpanExporter::builder().with_http().build()?;
       config = config.with_additional_span_processor(
           opentelemetry_sdk::trace::BatchSpanProcessor::builder(exporter).build(),
       );
   }
   ```

   The blocking reqwest client matters: the SDK's `BatchSpanProcessor` exports
   from its own thread, outside a tokio runtime. That is why the
   `logfire::exporters::span_exporter` helper (async reqwest, run on logfire's
   own runtime) is not the right building block here.
3. Document the variables in `crates/triage-service/README.md`. gRPC would
   need the `grpc-tonic` feature as well.

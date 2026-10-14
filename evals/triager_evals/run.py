"""Runs a triage strategy over a dataset as a pydantic-evals experiment.

    triage-eval --dataset datasets/kubernetes-dev-200.yaml --strategy Workflow \\
        --config configs/workflow-tev1-escalate.yaml [--config other.yaml ...] \\
        [--judge-model ollama:qwen3:4b | --no-judge] [--run-id ID] [--repeats N] \\
        [--limit N] [--label TEXT] [--max-concurrency 2] [--no-register]

Each `--config` is one experiment (a config sweep runs them one after the
other). The strategies run in the Rust service; this harness posts each case
to `/{Strategy}/{key}/run` on the Restate ingress, and on 409 (the key already
ran) reads the stored result through `attach`. The key is
`{run_id}:{config_hash}:{case_digest}:{repeat}`, so re-running with the same
`--run-id` gets Restate's stored results instead of calling the models again,
while a changed config or case gets a new key.

Each experiment shows up in Logfire (when LOGFIRE_TOKEN is set) and is saved
to `evals/results/<run_id>/<Strategy>-<config_hash>.json`, which is what
`compare-runs` reads.

    triage-eval --rescore evals/results/<run_id>/<file>.json [--dataset D] [--run-id ID] \
        [--judge-model M | --no-judge]

re-scores a results file with the current evaluators without calling the
service: the stored outputs go through the same task and evaluators, and the
file is rewritten (or, with another --run-id or --dataset, a new one is
written; a dataset whose cases are a subset of the stored run's works too).
This is what to use once the outputs have aged out of Restate (workflow
results are kept for a day by default) or the service runs another version.
The judge runs again when the rescored file is judged (default: as stored).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import logfire
import yaml
from pydantic_evals import Case, Dataset, increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import ReportEvaluator, ReportEvaluatorContext
from pydantic_evals.reporting import EvaluationReport
from pydantic_evals.reporting.analyses import ConfusionMatrix, ReportAnalysis, ScalarResult

from . import compare, results
from .build_dataset import TriageInput, load_dataset
from .env import load_repo_env
from .evaluators import EVALUATORS_VERSION, code_evaluators
from .judge import DEFAULT_JUDGE_MODEL, JudgeLedger, judge_evaluators
from .suite import (
    CONFIG_SCHEMAS,
    RESULTS_DIR,
    ROOT,
    RUBRIC_VERSION,
    SCHEMA_DIR,
    Suite,
    case_digest,
    config_errors,
    config_hash,
    file_sha256,
    load_suite,
    suite_id,
    workflow_key,
)

INGRESS_URL = os.environ.get("RESTATE_INGRESS_URL", "http://localhost:8080")
ADMIN_URL = os.environ.get("RESTATE_ADMIN_URL", "http://localhost:9070")
SERVICE_URL = os.environ.get("SERVICE_URL", "http://host.docker.internal:9080")
STRATEGIES = ("Heuristic", "SingleShot", "Workflow", "Agent", "Ensemble", "DraftVerify")
METRICS = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cost": "cost_usd",
    "latency_ms": "latency_ms",
    "llm_calls": "llm_calls",
    "decision_calls": "decision_calls",
    "tool_calls": "tool_calls",
}
JUDGE_EVALUATOR_FIELDS = {"QualityJudge": "quality", "ReplyJudge": "reply"}


def response_validator() -> jsonschema.protocols.Validator:
    schema = json.loads((SCHEMA_DIR / "triage-response.schema.json").read_text())
    cls = jsonschema.validators.validator_for(schema)
    return cls(schema)


def git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted", True


def resolve(path: Path) -> Path:
    """Paths are taken relative to the working directory, else to the repo root
    (uv runs from evals/, while datasets/ and configs/ are at the root)."""
    if path.is_absolute() or path.exists():
        return path
    return ROOT / path if (ROOT / path).exists() else path


def check_config(strategy: str, data: Any, where: str) -> dict[str, Any]:
    """`data` if it is a valid config for `strategy` (strict schema, unknown keys rejected)."""
    if not isinstance(data, dict):
        raise SystemExit(f"{where}: a config must be a mapping, got {type(data).__name__}")
    if errors := config_errors(strategy, data):
        raise SystemExit(
            f"{where}: not a valid {strategy} config (schema/{CONFIG_SCHEMAS[strategy]}.schema.json):\n  "
            + "\n  ".join(errors)
        )
    return data


def load_config(path: Path, strategy: str | None = None) -> dict[str, Any]:
    """A config file; with `strategy`, checked against that strategy's strict schema."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: a config must be a YAML mapping, got {type(data).__name__}")
    if strategy is not None:
        check_config(strategy, data, str(path))
    return data


def setup_logfire() -> None:
    """Logfire for the harness: experiments, case spans, httpx and the judge's pydantic-ai calls.

    Sends only when LOGFIRE_TOKEN is set; the environment comes from
    LOGFIRE_ENVIRONMENT. httpx is instrumented before any client exists, so
    every request to the Restate ingress carries `traceparent` and the Rust
    service's spans land in the case's trace. pydantic-ai is instrumented so
    the judge's calls show up as spans with their own token usage; they run in
    the evaluators, outside the task span, so they never count towards the
    strategy's `cost` metric (the judge's cost is recorded separately).
    Call `shutdown_logfire` before exiting.

    The console shows only errors (the harness prints its own progress), but
    keeps the project link, which logfire prints once it has checked the token.
    """
    logfire.configure(
        service_name="triage-evals",
        send_to_logfire="if-token-present",
        environment=os.environ.get("LOGFIRE_ENVIRONMENT") or None,
        console=logfire.ConsoleOptions(min_log_level="error", show_project_link=True),
    )
    logfire.instrument_httpx()
    logfire.instrument_pydantic_ai()


def shutdown_logfire() -> None:
    """Exports what is still buffered (the last experiment), also after an error."""
    try:
        logfire.force_flush()
        logfire.shutdown()
    except Exception as e:  # noqa: BLE001 - never hide the run's own outcome
        print(f"warning: flushing Logfire failed: {e!r}", file=sys.stderr)


@dataclass
class Started:
    """What the task knows about one run of a case; matched back to report rows by workflow key."""

    case: str
    repeat: int
    key: str
    error: str | None = None


# (workflow key, request body) -> (TriageResponse JSON, whether it was read back through attach)
Fetch = Callable[[str, dict[str, Any]], Awaitable[tuple[dict[str, Any], bool]]]


def restate_fetch(http: httpx.AsyncClient, strategy: str, config: dict[str, Any]) -> Fetch:
    """Posts the case to the strategy's workflow; on 409 (the key already ran) reads the stored result."""

    async def fetch(key: str, body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        attached = False
        resp = await http.post(f"/{strategy}/{key}/run", json={"config": config, **body})
        if resp.status_code == httpx.codes.CONFLICT:
            attached = True
            resp = await http.get(f"/restate/workflow/{strategy}/{key}/attach")
        if resp.is_error:
            raise RuntimeError(f"{strategy} returned {resp.status_code}: {resp.text[:500]}")
        return resp.json(), attached

    return fetch


def stored_cases(run: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """A results file's case records by (case digest, repeat), parsed from their workflow keys."""
    out = {}
    for c in run["cases"]:
        key = c.get("workflow_key")
        if not key or key.count(":") < 3:
            continue
        _, _, digest, repeat = key.rsplit(":", 3)
        out[(digest, int(repeat))] = c
    return out


def stored_key_run_id(run: dict[str, Any]) -> str:
    """The run ID in a results file's workflow keys.

    For a derived or re-keyed file it is the run that called the service,
    not `metadata.run_id`; re-scoring keeps it so the keys still name the
    Restate invocations.
    """
    for c in run["cases"]:
        key = c.get("workflow_key")
        if key and key.count(":") >= 3:
            return key.rsplit(":", 3)[0]
    return run["metadata"]["run_id"]


def stored_fetch(stored: dict[tuple[str, int], dict[str, Any]]) -> Fetch:
    """Returns the outputs a results file recorded, instead of calling the service."""

    async def fetch(key: str, body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        _, _, digest, repeat = key.rsplit(":", 3)
        rec = stored.get((digest, int(repeat)))
        if rec is None:
            raise LookupError(f"the stored run has no output for case {digest} repeat {repeat}")
        if rec.get("output") is None:
            raise RuntimeError(f"stored run failed: {rec.get('error')}")
        return rec["output"], False

    return fetch


@dataclass
class TaskState:
    hash_mismatches: list[str] = field(default_factory=list)
    started: dict[str, Started] = field(default_factory=dict)
    versions: set[str] = field(default_factory=set)
    attached: int = 0


def strategy_task(
    fetch: Fetch,
    run_id: str,
    config: dict[str, Any],
    names_by_digest: dict[str, str],
    state: TaskState,
):
    """Returns the task that runs one case through `fetch` (the strategy in Restate, or stored outputs).

    Each repeat of a case gets its own workflow key, numbered in the order the
    repeats start; the keys are the same on every re-run with the same run ID.
    """
    cfg_hash = config_hash(config)
    validator = response_validator()
    seen: dict[str, int] = defaultdict(lambda: -1)

    async def run(inputs: TriageInput) -> dict[str, Any]:
        body = inputs.model_dump(mode="json")
        digest = case_digest(body)
        seen[digest] += 1
        key = workflow_key(run_id, cfg_hash, digest, seen[digest])
        started = state.started[key] = Started(names_by_digest.get(digest, digest), seen[digest], key)
        set_eval_attribute("workflow_key", key)
        set_eval_attribute("repeat", seen[digest])
        try:
            out, attached = await fetch(key, body)
            state.attached += attached
            errors = sorted(validator.iter_errors(out), key=lambda e: list(e.path))
            if errors:
                where = "/".join(map(str, errors[0].path)) or "<root>"
                raise ValueError(f"response does not match triage-response.schema.json at {where}: {errors[0].message}")
        except Exception as e:
            started.error = str(e)
            raise

        if out["config_hash"] != cfg_hash:
            state.hash_mismatches.append(key)
            msg = f"config_hash mismatch for {key}: harness {cfg_hash}, service {out['config_hash']}"
            logfire.warn(msg)
            print(f"WARNING: {msg}", file=sys.stderr)
        state.versions.add(out["version"])

        # pydantic-evals only collects usage from spans in this process; the
        # strategy runs in the Rust service, so record what it reports.
        usage = out["usage"]
        for metric, name in METRICS.items():
            increment_eval_metric(metric, usage[name])
        set_eval_attribute("strategy_version", out["version"])
        set_eval_attribute("config_hash", out["config_hash"])
        return out

    return run


async def register_deployment() -> None:
    async with httpx.AsyncClient(base_url=ADMIN_URL, timeout=30) as admin:
        try:
            resp = await admin.post("/deployments", json={"uri": SERVICE_URL, "force": True})
        except httpx.HTTPError as e:
            raise SystemExit(f"registering {SERVICE_URL} with {ADMIN_URL} failed: {e!r}") from e
        if resp.is_error:
            raise SystemExit(f"registering {SERVICE_URL} with {ADMIN_URL} failed: {resp.status_code} {resp.text}")


async def strategy_versions(http: httpx.AsyncClient) -> dict[str, str]:
    """The `Info/versions` handler's map of strategy -> code version, or {} if it isn't there."""
    try:
        # The handler takes no input, and the ingress rejects a body for it.
        resp = await http.post("/Info/versions", timeout=30)
    except httpx.HTTPError:
        return {}
    if resp.is_error:
        return {}
    try:
        data = resp.json()
    except ValueError:
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def report_records(
    report: EvaluationReport, state: TaskState, ledger: JudgeLedger, judged: bool, weights: dict[str, float]
) -> list[dict[str, Any]]:
    """Turns the report's rows into results-file case records."""
    records = []
    by_key = state.started
    for rc in report.cases:
        key = rc.attributes.get("workflow_key")
        started = by_key.get(key)
        scores: dict[str, Any] = {}
        reasons: dict[str, str] = {}
        for group in (rc.scores, rc.assertions, rc.labels):
            for name, res in group.items():
                scores[name] = res.value
                if res.reason:
                    reasons[name] = res.reason
        failures = [
            {
                "evaluator": f.name,
                "field": JUDGE_EVALUATOR_FIELDS.get(f.name),
                "error": f.error_message,
            }
            for f in rc.evaluator_failures
        ]
        judge = ledger.by_key.get(key)
        records.append(
            results.case_record(
                name=rc.source_case_name or results.base_case_name(rc.name),
                repeat=started.repeat if started else rc.attributes.get("repeat"),
                workflow_key=key,
                expected=rc.expected_output,
                output=rc.output,
                error=None,
                scores=scores,
                reasons=reasons,
                metrics=rc.metrics,
                duration=rc.task_duration,
                judged=judged,
                weights=weights,
                judge_usage=asdict(judge) if judge and judge.requests else None,
                evaluator_failures=failures,
                trace_id=rc.trace_id,
            )
        )
    # Failures carry no attributes; match them to the task's own record of
    # which keys failed, per case, in order.
    failed = defaultdict(list)
    for s in by_key.values():
        if s.error is not None:
            failed[s.case].append(s)
    for f in report.failures:
        name = f.source_case_name or results.base_case_name(f.name)
        s = failed[name].pop(0) if failed[name] else None
        records.append(
            results.case_record(
                name=name,
                repeat=s.repeat if s else None,
                workflow_key=s.key if s else None,
                expected=f.expected_output,
                output=None,
                error=f.error_message,
                scores={},
                reasons={},
                metrics={},
                duration=None,
                judged=judged,
                weights=weights,
                trace_id=f.trace_id,
            )
        )
    return records


@dataclass
class RunSummary(ReportEvaluator[Any, Any, Any]):
    """Run-level numbers for the Logfire experiment: composite, kind macro-F1, cost per issue."""

    judged: bool = False
    weights: dict[str, float] = field(default_factory=dict)

    def evaluate(self, ctx: ReportEvaluatorContext[Any, Any, Any]) -> list[ReportAnalysis]:
        records = report_records(ctx.report, TaskState(), JudgeLedger(), self.judged, self.weights)
        run = {"metadata": {"weights": self.weights}, "cases": records}
        stats = compare.run_stats(run)
        out: list[ReportAnalysis] = [
            ScalarResult(title="composite", value=round(stats["composite"] or 0.0, 4)),
            ScalarResult(title="cost per issue", value=stats["cost_mean"] or 0.0, unit="USD"),
        ]
        if stats["kind_macro_f1"] is not None:
            out.append(ScalarResult(title="kind macro-F1", value=round(stats["kind_macro_f1"], 4)))
        labels, matrix = compare.kind_confusion(records)
        if labels:
            out.append(ConfusionMatrix(title="kind", class_labels=labels, matrix=matrix))
        return out


async def evaluate_config(
    *,
    http: httpx.AsyncClient | None,
    args: argparse.Namespace,
    config_path: Path,
    cases: list[Case],
    versions: dict[str, str],
    suite: Suite,
    dataset_hash: str,
    config: dict[str, Any] | None = None,
    stored: dict[tuple[str, int], dict[str, Any]] | None = None,
    key_run_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> Path:
    """Runs one experiment and saves its results file.

    With `stored`, the outputs come from a results file (see `rescore`)
    instead of the service; `key_run_id` is then the stored run's ID, so the
    workflow keys still name the Restate invocations that made the outputs.
    `run_metadata` goes into both the Logfire experiment's metadata and the
    results file; `extra_metadata` (provenance of rescored outputs) only into
    the results file.
    """
    config = load_config(config_path, args.strategy) if config is None else config
    cfg_hash = config_hash(config)
    judge_model = None if args.no_judge else args.judge_model
    judged = judge_model is not None
    weights = suite.weights_for(judged)
    sid = suite_id(dataset_hash, judge_model, suite)
    version = versions.get(args.strategy, "?")
    git_sha, git_dirty = git_state()
    label = args.label or config_path.stem
    campaign = (run_metadata or {}).get("campaign")
    prefix = f"[{campaign}] " if campaign else ""
    experiment = f"{prefix}{args.strategy}@{version} {cfg_hash} {label}".strip()

    ledger = JudgeLedger()
    evaluators = code_evaluators(suite.duplicates_k)
    if judged:
        evaluators += judge_evaluators(judge_model, ledger)
    dataset = Dataset[TriageInput, Any, Any](
        name=args.dataset.stem,
        cases=cases,
        evaluators=evaluators,
        report_evaluators=[RunSummary(judged=judged, weights=weights)],
    )
    names_by_digest = {case_digest(c.inputs.model_dump(mode="json")): c.name for c in cases}
    state = TaskState()
    metadata: dict[str, Any] = {
        "experiment": experiment,
        "run_id": args.run_id,
        "strategy": args.strategy,
        "version": version,
        "config": config,
        "config_file": str(config_path),
        "config_hash": cfg_hash,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "dataset": str(args.dataset),
        "dataset_hash": dataset_hash,
        "suite_id": sid,
        "evaluators_version": EVALUATORS_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "judge_model": judge_model,
        "weights": weights,
        "duplicates_k": suite.duplicates_k,
        "repeats": args.repeats,
        "limit": args.limit,
        "label": label,
    }
    if run_metadata:
        metadata |= run_metadata
    started_at = datetime.now(UTC)
    print(f"\n=== {experiment}: {len(cases)} cases x {args.repeats} repeat(s), suite {sid}", flush=True)
    if stored is not None:
        fetch = stored_fetch(stored)
    else:
        assert http is not None
        fetch = restate_fetch(http, args.strategy, config)
    report = await dataset.evaluate(
        strategy_task(fetch, key_run_id or args.run_id, config, names_by_digest, state),
        name=experiment,
        task_name=args.strategy,
        max_concurrency=args.max_concurrency,
        repeat=args.repeats,
        metadata=metadata,
    )

    if version == "?" and state.versions:
        version = min(state.versions)
        metadata["version"] = version
        metadata["experiment"] = f"{prefix}{args.strategy}@{version} {cfg_hash} {label}".strip()
    if len(state.versions) > 1:
        print(f"WARNING: responses carry several versions: {sorted(state.versions)}", file=sys.stderr)
    total = ledger.total()
    metadata |= {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "attached": state.attached,
        "config_hash_mismatches": len(state.hash_mismatches),
        "judge_usage": asdict(total) if total.requests else None,
    }
    if extra_metadata:
        metadata |= extra_metadata
    records = report_records(report, state, ledger, judged, weights)
    if stored is not None:
        # keep what the original run measured, not the replay's timing
        for rec in records:
            key = rec.get("workflow_key")
            if key and key.count(":") >= 3:
                _, _, digest, repeat = key.rsplit(":", 3)
                src = stored.get((digest, int(repeat)))
                if src is not None:
                    rec["duration"] = src.get("duration")
                    rec["trace_id"] = src.get("trace_id") or rec["trace_id"]
    path = results.results_path(args.results_dir, args.run_id, args.strategy, cfg_hash)
    results.save(path, metadata, records)

    report.print(include_input=False, include_output=False, include_averages=True)
    run = results.load(path)
    print(compare.short_summary(run))
    if state.hash_mismatches:
        print(
            f"WARNING: {len(state.hash_mismatches)} responses had a config_hash other than {cfg_hash}; "
            "the harness and the service disagree on the config hash",
            file=sys.stderr,
        )
    print(f"results: {path}")
    return path


async def run(
    args: argparse.Namespace,
    configs: list[tuple[Path, dict[str, Any]]] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> list[Path]:
    """One experiment per config. `configs` gives them inline as (name, config),
    instead of `args.config`'s files; they are checked against the schema too."""
    if configs is None:
        # fail before the first experiment, not halfway
        configs = [(p, load_config(p, args.strategy)) for p in args.config]
    else:
        configs = [(p, check_config(args.strategy, c, str(p))) for p, c in configs]
    if args.register:
        await register_deployment()
    dataset = load_dataset(args.dataset)
    cases = dataset.cases[: args.limit] if args.limit else dataset.cases
    dataset_hash = file_sha256(args.dataset)
    suite = load_suite()
    paths = []
    async with httpx.AsyncClient(base_url=INGRESS_URL, timeout=None) as http:
        versions = await strategy_versions(http)
        if not versions:
            print("note: Info/versions is not available; the version comes from the responses", file=sys.stderr)
        for config_path, config in configs:
            paths.append(
                await evaluate_config(
                    http=http,
                    args=args,
                    config_path=config_path,
                    cases=cases,
                    versions=versions,
                    suite=suite,
                    dataset_hash=dataset_hash,
                    config=config,
                    run_metadata=run_metadata,
                )
            )
    return paths


async def rescore(
    source: Path,
    *,
    dataset: Path | None = None,
    run_id: str | None = None,
    judge_model: str | None | bool = True,
    limit: int | None = None,
    label: str | None = None,
    max_concurrency: int = 4,
    results_dir: Path = RESULTS_DIR,
    run_metadata: dict[str, Any] | None = None,
) -> Path:
    """Re-scores a results file's stored outputs with the current evaluators; no service calls.

    `dataset` defaults to the stored run's (its cases must all be in the stored
    run, so a prefix such as test-20 of test-600 works). `run_id` defaults to
    the stored run's, which rewrites that file in place. `judge_model`: True
    keeps the stored run's judge (or none), None turns it off, a string sets
    it. The judge, when on, is called again.
    """
    src = results.load(source)
    meta = src["metadata"]
    dataset = resolve(dataset or Path(meta["dataset"]))
    if judge_model is True:
        judge_model = meta.get("judge_model")
    args = argparse.Namespace(
        dataset=dataset,
        strategy=meta["strategy"],
        run_id=run_id or meta["run_id"],
        no_judge=judge_model is None,
        judge_model=judge_model,
        repeats=meta.get("repeats") or 1,
        limit=limit,
        label=label or meta.get("label"),
        max_concurrency=max_concurrency,
        results_dir=results_dir,
    )
    ds = load_dataset(dataset)
    cases = ds.cases[:limit] if limit else ds.cases
    stored = stored_cases(src)
    missing = [c.name for c in cases if (case_digest(c.inputs.model_dump(mode="json")), 0) not in stored]
    if missing:
        raise SystemExit(
            f"{source}: no stored output for {len(missing)} of {len(cases)} cases of {dataset.name} "
            f"(first: {missing[0]}); rescoring needs a dataset whose cases the stored run covers"
        )
    now = datetime.now(UTC).isoformat()
    git_sha, git_dirty = git_state()
    extra = {
        # the outputs are the stored run's: keep its provenance
        "git_sha": meta.get("git_sha"),
        "git_dirty": meta.get("git_dirty"),
        "started_at": meta.get("started_at"),
        "finished_at": meta.get("finished_at"),
        "attached": meta.get("attached"),
        "config_file": meta.get("config_file"),
        "rescored": {
            "at": now,
            "from": str(source),
            "from_run_id": meta["run_id"],
            "from_dataset": meta.get("dataset"),
            "from_evaluators_version": meta.get("evaluators_version"),
            "from_suite_id": meta.get("suite_id"),
            "git_sha": git_sha,
            "git_dirty": git_dirty,
        },
    }
    if meta.get("rescored"):
        extra["rescored"]["first"] = meta["rescored"].get("first") or {
            k: meta["rescored"].get(k) for k in ("from", "from_run_id", "from_evaluators_version")
        }
    print(f"rescoring {source} ({len(cases)} cases of {dataset.name}) with evaluators v{EVALUATORS_VERSION}")
    return await evaluate_config(
        http=None,
        args=args,
        config_path=Path(meta.get("config_file") or "config.yaml"),
        cases=cases,
        versions={meta["strategy"]: str(meta.get("version", "?"))},
        suite=load_suite(),
        dataset_hash=file_sha256(dataset),
        config=meta.get("config") or {},
        stored=stored,
        key_run_id=stored_key_run_id(src),
        extra_metadata=extra,
        run_metadata=run_metadata,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="triage-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rescore", type=Path, help="re-score this results file's stored outputs (no service calls)")
    p.add_argument("--dataset", type=Path)
    p.add_argument("--strategy", choices=STRATEGIES)
    p.add_argument("--config", type=Path, action="append", help="strategy config YAML (repeatable)")
    p.add_argument(
        "--judge-model",
        default=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        help="pydantic-ai model for the LLM judge (default: $JUDGE_MODEL or %(default)s)",
    )
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge (quality, suggested_reply)")
    p.add_argument("--run-id", default=None, help="reuse to get Restate's stored results (default: a timestamp)")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="only the first N cases")
    p.add_argument("--label", default=None, help="experiment label (default: the config file's name)")
    p.add_argument("--max-concurrency", type=int, default=2)
    p.add_argument("--register", action=argparse.BooleanOptionalAction, default=True, help="register SERVICE_URL")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv)
    if args.rescore:
        args.rescore = resolve(args.rescore)
        if args.dataset:
            args.dataset = resolve(args.dataset)
        if args.run_id and (":" in args.run_id or "/" in args.run_id):
            p.error("--run-id must not contain ':' or '/'")
        explicit_judge = any(a == "--judge-model" or a.startswith("--judge-model=") for a in (argv or sys.argv[1:]))
        args.rescore_judge = None if args.no_judge else (args.judge_model if explicit_judge else True)
        return args
    if not args.dataset or not args.strategy:
        p.error("--dataset and --strategy are required (unless --rescore)")
    if not args.config:
        args.config = [ROOT / "configs" / "heuristic.yaml"] if args.strategy == "Heuristic" else None
        if args.config is None:
            p.error("--config is required for every strategy but Heuristic")
    args.dataset = resolve(args.dataset)
    args.config = [resolve(c) for c in args.config]
    args.run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if ":" in args.run_id or "/" in args.run_id:
        p.error("--run-id must not contain ':' or '/'")
    return args


def main(argv: list[str] | None = None) -> None:
    load_repo_env()
    args = parse_args(argv)
    setup_logfire()
    try:
        _main(args)
    finally:
        shutdown_logfire()


def _main(args: argparse.Namespace) -> None:
    start = time.monotonic()
    if args.rescore:
        path = asyncio.run(
            rescore(
                args.rescore,
                dataset=args.dataset,
                run_id=args.run_id,
                judge_model=args.rescore_judge,
                limit=args.limit,
                label=args.label,
                max_concurrency=args.max_concurrency,
                results_dir=args.results_dir,
            )
        )
        print(f"\nrescored ({time.monotonic() - start:.0f}s): {path}")
        return
    paths = asyncio.run(run(args))
    elapsed = time.monotonic() - start
    print(f"\nrun id: {args.run_id} ({elapsed:.0f}s); compare with: compare-runs {' '.join(map(str, paths))}")

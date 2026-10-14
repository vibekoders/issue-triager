"""Runs the PLAN.md experiments from `evals/experiments/` and writes their reports.

    run-experiments [--size 20|500] [--profile NAME] [--model ROLE=ADDRESS ...]
        [--only EXPERIMENT ...] [--step STEP ...] [--list] [--dry-run] [--force]
        [--report-only] [--no-judge] [--no-register] [--max-concurrency N]

Each step is one triage-eval run on `datasets/kubernetes-{split}-{size}.yaml`,
with its inline config's model-role placeholders ({llm}, {llm_large},
{decision}, {decision_alt}; {judge} for the eval judge) filled in from the
profile, `EXPERIMENT_MODEL_<ROLE>` and `--model`. Its run id is
`{step id}-{split}-{size}-{profile}`.

The experiments directory (`profiles.yaml` plus one file per experiment; see
its README) is loaded as one spec. Steps run in dependency order (after the
steps they derive from, take their config from, have as baseline, or whose
experiments compare with them; otherwise file name order, then file order)
through the same code as triage-eval (`run.run`, or `run.rescore` for offline
re-scoring):

- a step whose results file (for its exact config hash) exists is skipped,
  or re-scored offline from that file when it was scored with older
  evaluators or another suite (another judge, say);
- `--force` runs it through the service again, where Restate returns the
  outputs it still has stored (409 -> attach) instead of calling the models;
- a step is blocked, and the runner moves on, when its dataset is missing,
  the service is unreachable, or it pins a strategy version the service
  doesn't report.

When an experiment's last step is done, `evals/results/experiments/{size}-{profile}/<id>.md`
gets the step table, the before -> after table and the compare-runs report of
every comparison (with CSV and SVG next to it). At the end, `overview.md` gets
the Pareto frontier across the final runs and a per-strategy summary.

`--dry-run` prints the plan with an estimate of the model calls and the time
each step would take (from `defaults.seconds_per_call`); `--list` the
experiments and every step in run order.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import os
import re
import sys
import time
import traceback
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from . import compare, results, run
from .build_dataset import read_dataset_metadata
from .env import load_repo_env
from .evaluators import EVALUATORS_VERSION
from .suite import EVALS_DIR, RESULTS_DIR, ROOT, config_hash, file_sha256, load_suite, suite_id

EXPERIMENTS_DIR = EVALS_DIR / "experiments"
PROFILES_FILE = "profiles.yaml"
PROFILES_KEYS = {"profiles", "default_profile", "defaults", "overview"}
EXPERIMENT_KEYS = {"id", "title", "shows", "steps", "compare", "overview"}
DEFAULT_SIZE = 20
DEFAULT_DATASET = "datasets/kubernetes-{split}-{size}.yaml"
ROLES = ("llm", "llm_large", "decision", "decision_alt", "judge")
PROFILE_SETTINGS = {"max_concurrency"}
STEP_KEYS = {
    "id", "strategy", "split", "dataset", "config", "limit", "label", "judge", "baseline", "version", "source",
    "config_from_best", "optional", "max_concurrency", "calls_per_case",
}  # fmt: skip
SPLITS = ("dev", "test")
PLACEHOLDER = re.compile(r"\{([A-Za-z_]+)\}")
CAMPAIGN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


# --- model roles and profiles ---------------------------------------------------------


@dataclass
class Models:
    """The resolved model roles, and where each came from."""

    profile: str
    roles: dict[str, str]
    sources: dict[str, str]
    max_concurrency: int | None = None
    profile_source: str = "default_profile"


def resolve_models(
    data: Mapping[str, Any],
    profile: str | None = None,
    overrides: Iterable[str] = (),
    env: Mapping[str, str] | None = None,
) -> Models:
    """Roles from (highest first) `overrides` (ROLE=ADDRESS), the env
    EXPERIMENT_MODEL_<ROLE> (JUDGE_MODEL is an alias for the judge), and the
    profile: `profile`, else $EXPERIMENT_PROFILE, else `default_profile`."""
    env = os.environ if env is None else env
    profiles = data.get("profiles") or {}
    if not profiles:
        raise SystemExit("experiments: the file defines no profiles")
    if profile:
        name, how = profile, "--profile"
    elif env.get("EXPERIMENT_PROFILE"):
        name, how = env["EXPERIMENT_PROFILE"], "EXPERIMENT_PROFILE"
    else:
        name, how = data.get("default_profile") or next(iter(profiles)), "default_profile"
    if name not in profiles:
        raise SystemExit(f"profile {name!r} (from {how}) is not defined; have {', '.join(profiles)}")
    prof = dict(profiles[name] or {})
    unknown = set(prof) - set(ROLES) - PROFILE_SETTINGS
    if unknown:
        raise SystemExit(f"experiments: profile {name!r} has unknown keys {sorted(unknown)}")
    roles: dict[str, str] = {}
    sources: dict[str, str] = {}
    for role in ROLES:
        if role in prof:
            roles[role], sources[role] = str(prof[role]), f"profile {name}"
        for var in ([f"EXPERIMENT_MODEL_{role.upper()}"] + (["JUDGE_MODEL"] if role == "judge" else []))[::-1]:
            # reversed: EXPERIMENT_MODEL_JUDGE wins over the JUDGE_MODEL alias
            if env.get(var):
                roles[role], sources[role] = env[var], f"env {var}"
    for item in overrides:
        role, sep, address = item.partition("=")
        role = role.strip().lower().replace("-", "_")
        if not sep or not address.strip():
            raise SystemExit(f"--model {item!r}: expected ROLE=ADDRESS, e.g. judge=azure:gpt-6-luna")
        if role not in ROLES:
            raise SystemExit(f"--model {item!r}: unknown role {role!r}; roles are {', '.join(ROLES)}")
        roles[role], sources[role] = address.strip(), "--model"
    missing = [r for r in ROLES if r not in roles]
    if missing:
        raise SystemExit(f"profile {name!r} does not set {', '.join(missing)} (and no override does)")
    for role, address in roles.items():
        if ":" not in address:
            raise SystemExit(f"model for {role} ({sources[role]}): {address!r} is not a provider:model address")
    conc = prof.get("max_concurrency")
    return Models(name, roles, sources, int(conc) if conc is not None else None, how)


def substitute(value: Any, values: Mapping[str, Any], where: str) -> Any:
    """`value` with every `{name}` in its strings replaced from `values` (deep copy)."""
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            if m.group(1) not in values:
                raise SystemExit(f"{where}: unknown placeholder {m.group(0)} (have {', '.join(sorted(values))})")
            return str(values[m.group(1)])

        return PLACEHOLDER.sub(repl, value)
    if isinstance(value, Mapping):
        return {k: substitute(v, values, where) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, values, where) for v in value]
    return copy.deepcopy(value)


def compose_run_id(step_id: str, split: str, size: int, profile: str, campaign: str | None = None) -> str:
    run_id = f"{step_id}-{split}-{size}-{profile}"
    return f"{campaign}-{run_id}" if campaign else run_id


def dataset_path(pattern: str, split: str, size: int) -> Path:
    return run.resolve(Path(substitute(pattern, {"split": split, "size": size}, "dataset")))


def uses_ollama(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("ollama:")
    if isinstance(value, Mapping):
        return any(uses_ollama(v) for v in value.values())
    if isinstance(value, list):
        return any(uses_ollama(v) for v in value)
    return False


# --- the experiments file ------------------------------------------------------------


@dataclass
class Step:
    key: str  # what references use: the step id, or id@split for a step with several splits
    strategy: str
    config: dict[str, Any]
    dataset: Path
    label: str
    experiment: str
    id: str = ""
    split: str = ""
    run_id: str = ""
    judge: str | None = None
    limit: int | None = None
    baseline: str | None = None
    version: str | None = None
    source: str | None = None
    config_from_best: list[str] = field(default_factory=list)
    optional: bool = False
    max_concurrency: int | None = None
    calls_per_case: float | None = None
    file: Path | None = None  # the experiment file that defines it

    def __post_init__(self) -> None:
        self.id = self.id or self.key
        self.run_id = self.run_id or self.key


@dataclass
class Comparison:
    title: str
    baseline: str
    runs: list[str]


@dataclass
class Experiment:
    id: str
    title: str
    shows: str
    steps: list[Step]
    compare: list[Comparison]


@dataclass
class Spec:
    defaults: dict[str, Any]
    experiments: list[Experiment]
    overview: Comparison | None
    steps: dict[str, Step]
    models: Models | None = None
    size: int = DEFAULT_SIZE
    campaign: str | None = None  # --campaign: groups runs, and gives them their own run ids
    path: Path = EXPERIMENTS_DIR
    order: list[str] = field(default_factory=list)  # step keys in run order
    files: dict[str, Path] = field(default_factory=dict)  # experiment id -> its file

    def step(self, ref: str) -> Step:
        try:
            return self.steps[ref]
        except KeyError:
            raise SystemExit(f"experiments: unknown step {ref!r}") from None

    def ordered(self, steps: Iterable[Step] | None = None) -> list[Step]:
        """`steps` (default: all) in run order."""
        keys = set(self.steps) if steps is None else {s.key for s in steps}
        return [self.steps[k] for k in self.order if k in keys]


def experiment_files(path: Path) -> tuple[Path, list[Path]]:
    """The profiles file and the experiment files to load from `path`, a directory or one experiment file.

    A single file takes its profiles from `profiles.yaml` next to it, else from
    the default directory's.
    """
    path = run.resolve(path)
    if path.is_dir():
        files = sorted(p for p in path.glob("*.yaml") if p.name != PROFILES_FILE)
        profiles = path / PROFILES_FILE
    elif path.is_file():
        if path.name == PROFILES_FILE:
            raise SystemExit(f"--file {path}: that is the profiles file; pass an experiment file or the directory")
        files = [path]
        profiles = path.parent / PROFILES_FILE
        if not profiles.exists():
            profiles = EXPERIMENTS_DIR / PROFILES_FILE
    else:
        raise SystemExit(f"--file {path}: no such file or directory")
    if not profiles.exists():
        raise SystemExit(f"experiments: no {PROFILES_FILE} for {path}")
    if not files:
        raise SystemExit(f"experiments: no experiment files in {path}")
    return profiles, files


def _read_yaml(path: Path, allowed: set[str]) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{rel(path)}: expected a mapping at the top level")
    unknown = set(data) - allowed
    if unknown:
        raise SystemExit(f"{rel(path)}: unknown keys {sorted(unknown)} (allowed: {', '.join(sorted(allowed))})")
    return data


def run_order(steps: dict[str, Step], deps: dict[str, set[str]], rank: dict[str, tuple[int, int]]) -> list[str]:
    """Step keys, each after the steps it depends on; ties in (file, position in file) order."""
    import heapq

    waiting = {k: set(d) for k, d in deps.items()}
    users: dict[str, list[str]] = {k: [] for k in steps}
    for k, d in deps.items():
        for dep in d:
            users[dep].append(k)
    ready = [(rank[k], k) for k, d in waiting.items() if not d]
    heapq.heapify(ready)
    order = []
    while ready:
        _, k = heapq.heappop(ready)
        order.append(k)
        for u in users[k]:
            waiting[u].discard(k)
            if not waiting[u]:
                heapq.heappush(ready, (rank[u], u))
    if len(order) < len(steps):
        stuck = sorted((k for k in steps if k not in order), key=lambda k: rank[k])
        detail = "; ".join(f"{k} needs {', '.join(sorted(waiting[k]))}" for k in stuck)
        raise SystemExit(f"experiments: dependency cycle among steps: {detail}")
    return order


def load_spec(
    path: Path = EXPERIMENTS_DIR,
    *,
    size: int = DEFAULT_SIZE,
    profile: str | None = None,
    models: Iterable[str] = (),
    env: Mapping[str, str] | None = None,
    no_judge: bool = False,
    campaign: str | None = None,
) -> Spec:
    """The experiments directory (profiles.yaml plus one file per experiment) or a
    single experiment file, as one spec resolved for one size and one set of model roles."""
    if campaign is not None and not CAMPAIGN.fullmatch(campaign):
        raise SystemExit(f"--campaign {campaign!r}: use letters, digits, '.', '_' and '-' only")
    profiles_path, files = experiment_files(path)
    pdata = _read_yaml(profiles_path, PROFILES_KEYS)
    defaults = pdata.get("defaults") or {}
    sizes = [int(s) for s in defaults.get("sizes") or [size]]
    if size not in sizes:
        raise SystemExit(f"--size {size}: not one of the sizes in {profiles_path.name} ({', '.join(map(str, sizes))})")
    resolved = resolve_models(pdata, profile, models, env)
    roles = resolved.roles
    pattern = defaults.get("dataset", DEFAULT_DATASET)
    loaded = ", ".join(rel(f) for f in files)

    # first pass: experiments, steps and their keys (a step with several splits gets one key per split)
    exps: list[tuple[str, Path, dict[str, Any]]] = []
    exp_files: dict[str, Path] = {}
    raw_steps: list[tuple[str, Path, int, dict[str, Any], list[str]]] = []
    ids: dict[str, list[str]] = {}
    id_files: dict[str, Path] = {}
    overview_raw: tuple[Path, dict[str, Any]] | None = None
    if pdata.get("overview"):
        overview_raw = (profiles_path, pdata["overview"])
    for fi, f in enumerate(files):
        m = _read_yaml(f, EXPERIMENT_KEYS)
        exp_id = str(m.get("id") or f.stem)
        if exp_id != f.stem:
            raise SystemExit(f"{rel(f)}: id {exp_id!r} must equal the file name ({f.stem!r}), or be left out")
        if exp_id in exp_files:
            raise SystemExit(f"experiments: duplicate experiment {exp_id!r} in {rel(exp_files[exp_id])} and {rel(f)}")
        exp_files[exp_id] = f
        exps.append((exp_id, f, m))
        if m.get("overview"):
            if overview_raw is not None:
                raise SystemExit(f"experiments: overview defined in both {rel(overview_raw[0])} and {rel(f)}")
            overview_raw = (f, m["overview"])
        for pos, raw in enumerate(m.get("steps") or []):
            sid = raw.get("id")
            if not sid:
                raise SystemExit(f"{rel(f)}: step {pos + 1} has no id")
            unknown = set(raw) - STEP_KEYS
            if unknown:
                raise SystemExit(f"{rel(f)}: step {sid!r} has unknown keys {sorted(unknown)}")
            if any(c in sid for c in ":/@"):
                raise SystemExit(f"{rel(f)}: step id {sid!r} must not contain ':', '/' or '@'")
            if sid in ids:
                raise SystemExit(f"experiments: duplicate step {sid!r} in {rel(id_files[sid])} and {rel(f)}")
            splits = raw.get("split")
            splits = [splits] if isinstance(splits, str) else list(splits or [])
            if not splits or any(s not in SPLITS for s in splits):
                raise SystemExit(f"{rel(f)}: step {sid!r}: split must be dev, test or a list of them")
            ids[sid], id_files[sid] = splits, f
            raw_steps.append((exp_id, f, fi * 10_000 + pos, raw, splits))

    def key_of(sid: str, split: str) -> str:
        return sid if len(ids[sid]) == 1 else f"{sid}@{split}"

    def ref_key(ref: str, where: str, split: str | None = None) -> str:
        sid, _, at = ref.partition("@")
        if sid not in ids:
            raise SystemExit(f"{where}: unknown step {ref!r}; it is not defined in the loaded files ({loaded})")
        if at:
            if at not in ids[sid]:
                raise SystemExit(f"{where}: step {sid!r} has no split {at!r}")
            return key_of(sid, at)
        if len(ids[sid]) == 1:
            return sid
        if split in ids[sid]:
            return key_of(sid, split)
        raise SystemExit(f"{where}: {ref!r} has several splits; say {ref}@{ids[sid][0]}")

    steps: dict[str, Step] = {}
    rank: dict[str, tuple[int, int]] = {}
    by_experiment: dict[str, list[Step]] = {}
    for exp_id, f, order_key, raw, splits in raw_steps:
        sid = raw["id"]
        if raw.get("strategy") not in run.STRATEGIES:
            raise SystemExit(f"{rel(f)}: step {sid!r}: unknown strategy {raw.get('strategy')!r}")
        for si, split in enumerate(splits):
            key = key_of(sid, split)
            where = f"{rel(f)}: step {key}"
            raw_config = raw.get("config", {})
            if isinstance(raw_config, str):
                raw_config = run.load_config(run.resolve(Path(raw_config)))
            config = run.check_config(raw["strategy"], substitute(raw_config or {}, roles, where), where)
            judge = raw.get("judge", True)
            if not isinstance(judge, bool):
                raise SystemExit(f"{where}: judge must be true or false (the model is the judge role)")
            judge = roles["judge"] if judge and not no_judge else None
            run_id = compose_run_id(sid, split, size, resolved.profile, campaign)
            if any(c in run_id for c in ":/"):
                raise SystemExit(f"{where}: run id {run_id!r} must not contain ':' or '/'")
            step = Step(
                key=key,
                id=sid,
                split=split,
                run_id=run_id,
                strategy=raw["strategy"],
                config=config,
                dataset=dataset_path(raw.get("dataset") or pattern, split, size),
                label=substitute(raw.get("label") or sid, roles, where),
                experiment=exp_id,
                judge=judge,
                limit=raw.get("limit"),
                baseline=ref_key(raw["baseline"], where, split) if raw.get("baseline") else None,
                version=None if raw.get("version") is None else str(raw["version"]),
                source=ref_key(raw["source"], where, split) if raw.get("source") else None,
                config_from_best=[ref_key(r, where, split) for r in raw.get("config_from_best") or []],
                optional=bool(raw.get("optional", False)),
                max_concurrency=raw.get("max_concurrency"),
                calls_per_case=raw.get("calls_per_case"),
                file=f,
            )
            steps[key] = step
            rank[key] = (order_key, si)
            by_experiment.setdefault(exp_id, []).append(step)

    experiments = []
    for exp_id, f, m in exps:
        ms_steps = by_experiment.get(exp_id, [])
        where = f"{rel(f)}: compare"
        comparisons = [
            Comparison(
                title=c.get("title") or f"comparison {i}",
                baseline=ref_key(c["baseline"], where),
                runs=[ref_key(r, where) for r in c["runs"]],
            )
            for i, c in enumerate(m.get("compare") or [], 1)
        ]
        if not comparisons and ms_steps:
            comparisons = [Comparison("all steps", ms_steps[0].key, [s.key for s in ms_steps])]
        experiments.append(
            Experiment(
                id=exp_id, title=m.get("title", exp_id), shows=m.get("shows", ""), steps=ms_steps, compare=comparisons
            )
        )
    overview = None
    if overview_raw:
        of, ov = overview_raw
        runs = [ref_key(r, f"{rel(of)}: overview") for r in ov["runs"]]
        base = ref_key(ov["baseline"], f"{rel(of)}: overview") if ov.get("baseline") else runs[0]
        overview = Comparison(ov.get("title", "overview"), base, runs)

    # Run order: a step runs after the steps it derives from, takes its config
    # from, or has as baseline, and after the other experiments' steps its own
    # experiment compares it with.
    deps: dict[str, set[str]] = {k: set() for k in steps}
    for m in experiments:
        own = {s.key for s in m.steps}
        external = {r for c in m.compare for r in [c.baseline, *c.runs] if r not in own}
        for s in m.steps:
            deps[s.key] |= {r for r in [s.baseline, s.source, *s.config_from_best] if r} | external
            deps[s.key].discard(s.key)
    order = run_order(steps, deps, rank)
    return Spec(
        defaults=defaults, experiments=experiments, overview=overview, steps=steps, models=resolved, size=size,
        campaign=campaign, path=run.resolve(path), order=order, files=exp_files,
    )  # fmt: skip


def report_dir(spec: Spec, results_dir: Path) -> Path:
    profile = spec.models.profile if spec.models else "default"
    name = f"{spec.size}-{profile}"
    return results_dir / "experiments" / (f"{spec.campaign}-{name}" if spec.campaign else name)


def models_lines(models: Models) -> list[str]:
    lines = [f"profile: {models.profile} (from {models.profile_source})"]
    width = max(len(r) for r in ROLES)
    for role in ROLES:
        lines.append(f"  {role:<{width}}  {models.roles[role]:<28} ({models.sources[role]})")
    return lines


def run_metadata(spec: Spec, step: Step) -> dict[str, Any]:
    """What the results file and the Logfire experiment record about the step and its models."""
    meta: dict[str, Any] = {
        "experiment_id": step.experiment,
        "step_id": step.id,
        "step": step.key,
        "split": step.split,
        "size": spec.size,
    }
    if spec.campaign:
        meta["campaign"] = spec.campaign
    if spec.models:
        meta |= {"profile": spec.models.profile, "models": dict(spec.models.roles)}
        meta["model_sources"] = dict(spec.models.sources)
    return meta


# --- where results are, and what a step needs -----------------------------------------


@dataclass
class Context:
    spec: Spec
    results_dir: Path
    versions: dict[str, str]
    force: bool = False
    include_optional: bool = False
    max_concurrency: int | None = None  # --max-concurrency: overrides everything
    _suite: Any = None
    _hashes: dict[Path, str] = field(default_factory=dict)

    def dataset_hash(self, path: Path) -> str:
        if path not in self._hashes:
            self._hashes[path] = file_sha256(path)
        return self._hashes[path]

    def suite(self) -> Any:
        if self._suite is None:
            self._suite = load_suite()
        return self._suite


def results_file(ctx: Context, step: Step, config: dict[str, Any] | None = None) -> Path:
    """The step's results file for its exact config hash (it may not exist yet)."""
    config = chosen_config(ctx, step)[0] if config is None else config
    return results.results_path(ctx.results_dir, step.run_id, step.strategy, config_hash(config))


def existing(ctx: Context, ref: str) -> Path | None:
    path = results_file(ctx, ctx.spec.step(ref))
    return path if path.exists() else None


def load_stats(path: Path) -> dict[str, Any]:
    r = results.load(path)
    s = compare.run_stats(r)
    s["metadata"] = r["metadata"]
    return s


def chosen_config(ctx: Context, step: Step) -> tuple[dict[str, Any], str | None]:
    """The config to use and, for config_from_best, why."""
    if not step.config_from_best:
        return step.config, None
    best = None
    for ref in step.config_from_best:
        cand = ctx.spec.step(ref)
        path = results_file(ctx, cand, cand.config)
        if not path.exists():
            continue
        st = load_stats(path)
        key = (st["composite"] or 0.0, -(st["cost_mean"] or 0.0))
        if best is None or key > best[0]:
            best = (key, cand, st)
    if best is None:
        return step.config, f"no results yet for {', '.join(step.config_from_best)}; default config"
    _, cand, st = best
    return (
        cand.config,
        f"best of {', '.join(step.config_from_best)}: {cand.key} ({compare.num(100 * (st['composite'] or 0))})",
    )


def case_count(step: Step) -> int | None:
    if not step.dataset.exists():
        return None
    n = read_dataset_metadata(step.dataset).get("cases")
    if n is None:
        return None
    return min(n, step.limit) if step.limit else n


def step_concurrency(ctx: Context, step: Step, config: dict[str, Any] | None = None) -> int:
    """Cases in flight at once: --max-concurrency, else the step's, else the profile's
    (the heuristic's own default), at most `ollama_max_concurrency` when the step or
    its judge uses Ollama."""
    if ctx.max_concurrency:
        return ctx.max_concurrency
    if step.max_concurrency:
        return int(step.max_concurrency)
    d = ctx.spec.defaults
    profile_conc = ctx.spec.models.max_concurrency if ctx.spec.models else None
    if step.strategy == "Heuristic":
        conc = int(d.get("heuristic_max_concurrency", 8))
    else:
        conc = int(profile_conc or d.get("max_concurrency", 1))
    if uses_ollama(step.config if config is None else config) or uses_ollama(step.judge or ""):
        conc = min(conc, int(d.get("ollama_max_concurrency", 1)))
    return max(conc, 1)


def up_to_date(ctx: Context, step: Step, path: Path) -> tuple[bool, str]:
    meta = results.load(path)["metadata"]
    want_suite = suite_id(ctx.dataset_hash(step.dataset), step.judge, ctx.suite())
    if str(meta.get("evaluators_version")) != EVALUATORS_VERSION:
        return False, f"scored with evaluators v{meta.get('evaluators_version')}, now v{EVALUATORS_VERSION}"
    if meta.get("suite_id") != want_suite:
        return False, f"suite {meta.get('suite_id')} != {want_suite} (dataset, judge or weights changed)"
    if meta.get("limit") != step.limit:
        return False, f"limit {meta.get('limit')} != {step.limit}"
    return True, "results up to date"


# --- estimates ----------------------------------------------------------------------


def _escalated_fields(accept: float) -> int:
    """A rough guess of how many of the ~8 escalatable fields fall below `accept`.

    From the live runs with tev1:0.8b: at 0.8, 6 fields were escalated (kind,
    area, severity, is_regression, quality, scope).
    """
    if accept <= 0.6:
        return 3
    if accept <= 0.8:
        return 6
    return 7


DECISION_CALLS = 3  # tev1:0.8b's context fits the draft only in 3 calls
AGENT_CALLS = 5  # about 4 turns plus the final extraction
AGENT_NOTOOLS_CALLS = 1  # `tools: []`: a single structured call


def calls_per_case(step: Step, config: dict[str, Any]) -> list[tuple[str, float]]:
    """Estimated (model, calls) per case for a strategy config, judge excluded."""
    s = step.strategy
    calls: list[tuple[str, float]] = []
    if s == "SingleShot":
        calls.append((config["model"], 1))
    elif s == "Workflow":
        calls.append((config["decision_model"], DECISION_CALLS))
        esc = config.get("escalation")
        if esc:
            accept = float((esc.get("thresholds") or {}).get("accept", 0.8))
            calls.append((esc["model"], _escalated_fields(accept)))
            if esc.get("sub_agent"):
                calls.append((esc["model"], 3))
        if config.get("reply_model"):
            calls.append((config["reply_model"], 1))
    elif s == "Agent":
        no_tools = "tools" in config and not config["tools"]
        calls.append((config["model"], AGENT_NOTOOLS_CALLS if no_tools else AGENT_CALLS))
        if config.get("decision_tool"):
            calls.append((config["decision_tool"], DECISION_CALLS))
    elif s == "Ensemble":
        for v in config.get("voters") or []:
            calls.append((v["model"], DECISION_CALLS if v.get("kind") == "decision" else 1))
        agg = config.get("aggregator") or {}
        if agg.get("kind") == "judge":
            calls.append((agg["model"], 1))
    elif s == "DraftVerify":
        calls.append((config["decision_model"], DECISION_CALLS))
        calls.append((config["verifier_model"], 4))
    if step.calls_per_case is not None and calls:
        calls = [(calls[0][0], float(step.calls_per_case))]
    return calls


def judge_calls(step: Step) -> list[tuple[str, float]]:
    return [(step.judge, 2)] if step.judge else []  # QualityJudge + ReplyJudge


def seconds_for(ctx: Context, calls: list[tuple[str, float]], cases: int, concurrency: int = 1) -> float:
    table = ctx.spec.defaults.get("seconds_per_call") or {}
    default = float(table.get("default", 30))
    return sum(n * cases * float(table.get(model, default)) for model, n in calls) / max(concurrency, 1)


# --- planning -----------------------------------------------------------------------


@dataclass
class StepPlan:
    step: Step
    action: str  # skip | rescore | derive | run | blocked | optional
    reason: str
    config: dict[str, Any]
    path: Path | None
    cases: int | None
    calls: float = 0.0  # model calls this action makes (strategy + judge)
    seconds: float = 0.0
    full_calls: float = 0.0  # what a full run would make (for blocked/optional steps)
    full_seconds: float = 0.0
    source_path: Path | None = None
    concurrency: int = 1


def plan_step(ctx: Context, step: Step) -> StepPlan:
    config, why = chosen_config(ctx, step)
    cases = case_count(step)
    path = results_file(ctx, step, config)
    conc = step_concurrency(ctx, step, config)

    def mk(action: str, reason: str, **kw: Any) -> StepPlan:
        return StepPlan(
            step, action, reason if not why else f"{reason}; {why}", config, path, cases, concurrency=conc, **kw
        )

    if not step.dataset.exists():
        return mk("blocked", f"dataset {rel(step.dataset)} is missing")
    strategy_calls = calls_per_case(step, config)
    full = strategy_calls + judge_calls(step)
    full_calls = sum(n for _, n in full) * (cases or 0)
    full_seconds = seconds_for(ctx, full, cases or 0, conc)
    judge_only = sum(n for _, n in judge_calls(step)) * (cases or 0)
    judge_seconds = seconds_for(ctx, judge_calls(step), cases or 0, conc)
    est = {"full_calls": full_calls, "full_seconds": full_seconds}

    if step.optional and not ctx.include_optional:
        return mk("optional", "optional step (pass --include-optional)", **est)
    if path.exists() and not ctx.force:
        ok, reason = up_to_date(ctx, step, path)
        if ok:
            return mk("skip", reason, **est)
        return mk(
            "rescore",
            f"{reason}; re-scored offline from the stored outputs",
            calls=judge_only,
            seconds=judge_seconds,
            **est,
        )
    if step.source:
        src = existing(ctx, step.source)
        if src is not None:
            return mk(
                "derive",
                f"stored outputs of {step.source}, re-scored on {step.dataset.name}",
                calls=judge_only,
                seconds=judge_seconds,
                source_path=src,
                **est,
            )
    live = ctx.versions.get(step.strategy)
    if step.version is not None and live != step.version:
        have = f"service has v{live}" if live else "service unreachable or has no Info/versions"
        return mk("blocked", f"needs {step.strategy} v{step.version}, {have}", **est)
    if not ctx.versions:
        return mk("blocked", "the Restate ingress / service is not reachable", **est)
    if step.source:
        reason = f"new run ({step.source} has no results to derive from yet)"
    else:
        reason = "--force: through the service (stored outputs come back via attach)" if path.exists() else "new run"
    return mk("run", reason, calls=full_calls, seconds=full_seconds, **est)


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def rel(path: Path | None) -> str:
    if path is None:
        return "–"
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def ds_name(step: Step) -> str:
    return step.dataset.stem.removeprefix("kubernetes-") + (f" (first {step.limit})" if step.limit else "")


def print_list(spec: Spec) -> None:
    """The experiments, and every step in run order."""
    print(f"{rel(spec.path)}: size {spec.size}")
    for line in models_lines(spec.models) if spec.models else []:
        print(line)
    print("\nexperiments:")
    for m in spec.experiments:
        print(f"  {m.id:<26} {m.title}  ({rel(spec.files.get(m.id))})")
        if m.shows:
            print(f"  {'':<26} shows: {m.shows}")
    print("\nsteps in run order:")
    for i, s in enumerate(spec.ordered(), 1):
        print(f"  {i:>2}. {s.key:<30} {s.experiment:<24} {s.strategy:<11} {ds_name(s):<9} {s.label}")


def print_plan(ctx: Context, steps: list[Step]) -> None:
    """The plan for `steps`, in run order; a heading whenever the experiment changes."""
    spec = ctx.spec
    titles = {m.id: m.title for m in spec.experiments}
    live = ", ".join(f"{k} {v}" for k, v in sorted(ctx.versions.items())) or "unreachable"
    print(f"size {spec.size}; service versions: {live}; evaluators v{EVALUATORS_VERSION}")
    print(f"results in {rel(ctx.results_dir)}; reports in {rel(report_dir(spec, ctx.results_dir))}")
    total_calls = total_s = pending_calls = pending_s = 0.0
    per_exp: dict[str, list[float]] = {}
    current = None
    for step in steps:
        if step.experiment != current:
            current = step.experiment
            print(f"\n{current}: {titles[current]}")
        p = plan_step(ctx, step)
        acc = per_exp.setdefault(step.experiment, [0.0, 0.0])
        acc[0] += p.calls
        acc[1] += p.seconds
        total_calls += p.calls
        total_s += p.seconds
        if p.action in ("blocked", "optional"):
            pending_calls += p.full_calls
            pending_s += p.full_seconds
        cost = f"{p.calls:.0f} calls, ~{fmt_duration(p.seconds)}"
        if p.action in ("blocked", "optional", "skip") and p.full_calls:
            cost += f" (full run: {p.full_calls:.0f} calls, ~{fmt_duration(p.full_seconds)})"
        ds = f"{step.dataset.stem.removeprefix('kubernetes-')}({p.cases if p.cases is not None else '?'})"
        judge = f"judge={step.judge}" if step.judge else "no judge"
        print(f"  {p.action:<8} {step.run_id:<44} {step.strategy:<11} {ds:<12} x{p.concurrency:<2} {judge}")
        print(f"           {step.label} [config {config_hash(p.config)}]")
        print(f"           {cost}; {p.reason}")
    print("\nper experiment:")
    for m in spec.experiments:
        if m.id in per_exp:
            calls, secs = per_exp[m.id]
            print(f"  -> {m.id}: {calls:.0f} model calls now, ~{fmt_duration(secs)}")
    print(
        f"\ntotal: {total_calls:.0f} model calls now (~{fmt_duration(total_s)}); "
        f"blocked/optional steps would add {pending_calls:.0f} (~{fmt_duration(pending_s)})"
    )
    print(
        "estimates use defaults.seconds_per_call divided by the step's concurrency (xN); escalation, agent "
        "turns and decision-draft splits are rough guesses (see experiments.calls_per_case)"
    )


# --- execution ----------------------------------------------------------------------


@dataclass
class Outcome:
    step: Step
    status: str
    detail: str
    path: Path | None


def eval_args(ctx: Context, step: Step, concurrency: int) -> argparse.Namespace:
    """triage-eval's arguments for a step (its config is passed inline to `run.run`)."""
    return argparse.Namespace(
        dataset=step.dataset,
        strategy=step.strategy,
        config=[],
        judge_model=step.judge,
        no_judge=step.judge is None,
        run_id=step.run_id,
        repeats=1,
        limit=step.limit,
        label=step.label,
        max_concurrency=concurrency,
        register=False,  # run-experiments registers once, before the first step
        results_dir=ctx.results_dir,
    )


async def execute_step(ctx: Context, p: StepPlan) -> Outcome:
    step = p.step
    if p.action in ("skip", "blocked", "optional"):
        return Outcome(step, p.action, p.reason, p.path if p.path and p.path.exists() else None)
    print(f"\n### {step.experiment} {step.run_id}: {p.action} ({p.reason})", flush=True)
    start = time.monotonic()
    meta = run_metadata(ctx.spec, step)
    try:
        if p.action == "rescore":
            assert p.path is not None
            path = await run.rescore(
                p.path, dataset=step.dataset, judge_model=step.judge, limit=step.limit, label=step.label,
                max_concurrency=p.concurrency, results_dir=ctx.results_dir, run_metadata=meta,
            )  # fmt: skip
        elif p.action == "derive":
            assert p.source_path is not None
            path = await run.rescore(
                p.source_path, dataset=step.dataset, run_id=step.run_id, judge_model=step.judge, limit=step.limit,
                label=step.label, max_concurrency=p.concurrency, results_dir=ctx.results_dir, run_metadata=meta,
            )  # fmt: skip
        else:
            source = Path(f"{rel(step.file or ctx.spec.path)}#{step.key}")
            paths = await run.run(eval_args(ctx, step, p.concurrency), configs=[(source, p.config)], run_metadata=meta)
            path = paths[0]
    except (Exception, SystemExit) as e:  # noqa: BLE001 - a failed step must not stop the others
        traceback.print_exc()
        return Outcome(step, "failed", f"{type(e).__name__}: {e}", None)
    detail = f"{p.reason} ({fmt_duration(time.monotonic() - start)})"
    rmeta = results.load(path)["metadata"]
    if step.version is not None and str(rmeta.get("version")) != step.version:
        detail += f"; WARNING: outputs are v{rmeta.get('version')}, the step is about v{step.version}"
    status = {"run": "ran", "rescore": "rescored", "derive": "derived"}[p.action]
    return Outcome(step, status, detail, path)


# --- reports ------------------------------------------------------------------------


def demote(markdown: str, levels: int = 1) -> str:
    return re.sub(r"^(#+) ", lambda m: "#" * (len(m.group(1)) + levels) + " ", markdown, flags=re.MULTILINE)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "compare"


def pts(x: float | None) -> str:
    return compare.num(None if x is None else 100 * x)


def comparison_section(
    ctx: Context, comp: Comparison, out_dir: Path, file_stem: str, heading: str = "##"
) -> tuple[list[str], dict[str, Any] | None]:
    """compare-runs over the comparison's runs that have results; the Markdown lines and the report."""
    lines = [f"{heading} {comp.title}", ""]
    available = [(ref, existing(ctx, ref)) for ref in comp.runs]
    have = [(ref, p) for ref, p in available if p is not None]
    missing = [ref for ref, p in available if p is None]
    if not have:
        lines += [f"No results yet for {', '.join(comp.runs)}.", ""]
        return lines, None
    base_path = existing(ctx, comp.baseline)
    note = ""
    if base_path is None:
        base_path = have[0][1]
        note = f"baseline {comp.baseline} has no results yet; using {have[0][0]} instead. "
    report = compare.build_report([p for _, p in have], base_path)
    csv_path = out_dir / f"{file_stem}.csv"
    svg_path = out_dir / f"{file_stem}-pareto.svg"
    compare.write_csv_svg(report["stats"], report["front"], csv_path, svg_path)
    tags = ", ".join(f"R{i} = {Path(r['path']).parent.name}" for i, r in enumerate(report["runs"], 1))
    lines.append(f"{note}Baseline R{report['base_i'] + 1}. {tags}.")
    if missing:
        lines.append(f"Not run yet: {', '.join(missing)}.")
    lines += ["", f"Files: [{csv_path.name}]({csv_path.name}), [{svg_path.name}]({svg_path.name})", ""]
    lines.append(demote(report["markdown"], len(heading)).replace(f"{'#' * (len(heading) + 1)} Run comparison\n", ""))
    return lines, report


def step_row(o: Outcome) -> list[Any]:
    s = o.step
    composite = cost = p50 = "–"
    cfg = config_hash(s.config)
    if o.path is not None:
        st = load_stats(o.path)
        composite, cost, p50 = pts(st["composite"]), compare.usd(st["cost_mean"]), compare.num(st["latency_p50"], ".0f")
        cfg = st["metadata"].get("config_hash") or cfg
    return [
        f"`{s.run_id}`", s.strategy, s.label, f"`{cfg}`", ds_name(s), s.judge or "off", f"**{o.status}**: {o.detail}",
        composite, cost, p50,
    ]  # fmt: skip


def before_after_rows(ctx: Context, steps: list[Step]) -> list[list[Any]]:
    rows = []
    for s in steps:
        if not s.baseline:
            continue
        a, b = existing(ctx, s.baseline), existing(ctx, s.key)
        if a is None or b is None:
            rows.append([f"`{s.baseline}`", f"`{s.key}`", "–", "–", "–", "–", "not both run yet"])
            continue
        ra, rb = results.load(a), results.load(b)
        sa, sb = compare.run_stats(ra), compare.run_stats(rb)
        h = compare.head_to_head(ra, rb)["composite"]
        same = ra["metadata"].get("suite_id") == rb["metadata"].get("suite_id")
        delta = f"{100 * ((sb['composite'] or 0) - (sa['composite'] or 0)):+.1f}" if same else "– (different suites)"
        rows.append(
            [
                f"`{s.baseline}`",
                f"`{s.key}`",
                f"{pts(sa['composite'])} → {pts(sb['composite'])}",
                delta,
                f"{compare.usd(sa['cost_mean'])} → {compare.usd(sb['cost_mean'])}",
                f"{h['b_better']}/{h['a_better']}/{h['ties']}",
                compare.pval(h["p"]),
            ]
        )
    return rows


def header_lines(ctx: Context) -> list[str]:
    spec = ctx.spec
    lines = [
        (
            f"Written by `run-experiments` at {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}; evaluators "
            f"v{EVALUATORS_VERSION}; size {spec.size}. Composite in points (0–100). Results files: "
            "`evals/results/<run id>/`."
        ),
        "",
    ]
    if spec.models:
        lines += [
            f"Models (profile `{spec.models.profile}`): "
            + ", ".join(f"{r} `{spec.models.roles[r]}`" for r in ROLES)
            + ".",
            "",
        ]
    return lines


def write_experiment_report(ctx: Context, m: Experiment, outcomes: list[Outcome], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {m.id}: {m.title}",
        "",
        f"What it shows: {m.shows}." if m.shows else "",
        "",
        *header_lines(ctx),
        "## Steps",
        "",
        compare.md_table(
            [
                "run id",
                "strategy",
                "label",
                "config",
                "dataset",
                "judge",
                "status",
                "composite",
                "cost/issue",
                "latency p50 ms",
            ],
            [step_row(o) for o in outcomes],
        ),
        "",
    ]
    ba = before_after_rows(ctx, m.steps)
    if ba:
        lines += [
            "## Before → after",
            "",
            "per-case wins: cases where the after run's composite is higher / lower / equal; p = exact sign test.",
            "",
            compare.md_table(["before", "after", "composite", "Δ points", "cost/issue", "per-case wins", "p"], ba),
            "",
        ]
    for i, comp in enumerate(m.compare, 1):
        section, _ = comparison_section(ctx, comp, out_dir, f"{m.id}-{i}-{slug(comp.title)}")
        lines += section
    path = out_dir / f"{m.id}.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def write_overview(ctx: Context, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = ctx.spec
    lines = [
        "# Experiments overview",
        "",
        *header_lines(ctx),
        "Cost is the strategy's shadow-priced cost per issue.",
        "",
    ]
    if spec.overview:
        ov = spec.overview
        lines += [f"## Pareto frontier: {ov.title}", ""]
        have = [(ref, p) for ref in ov.runs if (p := existing(ctx, ref)) is not None]
        missing = [ref for ref in ov.runs if existing(ctx, ref) is None]
        if have:
            report = compare.build_report([p for _, p in have], None)
            stats, front = report["stats"], report["front"]
            svg = out_dir / "overview-pareto.svg"
            compare.write_csv_svg(stats, front, out_dir / "overview.csv", svg)
            suites = {r["metadata"].get("suite_id") for r in report["runs"]}
            if len(suites) > 1:
                lines += [f"**Different suites** ({', '.join(sorted(map(str, suites)))}): not comparable.", ""]
            rows = []
            for i, (s, r) in enumerate(zip(stats, report["runs"], strict=True), 1):
                tag = f"R{i}"
                status = "**frontier**" if tag in front["frontier"] else f"dominated by {front['dominated'].get(tag)}"
                run_id = r["metadata"].get("run_id")
                rows.append(
                    [tag, f"`{run_id}`", s["label"], compare.usd(s["cost_mean"]), pts(s["composite"]),
                     pts(s["kind_macro_f1"]), compare.num(s["latency_p50"], ".0f"), status]
                )  # fmt: skip
            lines += [
                compare.md_table(
                    [
                        "run",
                        "run id",
                        "strategy@version config label",
                        "cost/issue",
                        "composite",
                        "kind macro-F1",
                        "latency p50 ms",
                        "Pareto",
                    ],
                    rows,
                ),
                "",
            ]
            for st in front["steps"]:
                cpp = st["cost_per_point"]
                lines.append(
                    f"- {st['from']} → {st['to']}: +{compare.usd(st['extra_cost'])} per issue for "
                    f"+{st['extra_points']:.1f} points"
                    + (f" ({compare.usd(cpp)} per point)" if cpp is not None else "")
                )
            if not front["steps"]:
                lines.append("- one point on the frontier, no trade-off to price")
            lines += ["", f"![Pareto chart]({svg.name})", ""]
            last = next((m for m in spec.experiments if any(s.key in ov.runs for s in m.steps)), None)
            if last and (out_dir / f"{last.id}.md").exists():
                lines += [f"Per-field results and head-to-heads: [{last.id}.md]({last.id}.md).", ""]
        if missing:
            lines += [f"Not run yet: {', '.join(missing)}.", ""]
        if not have:
            lines += ["No final runs yet.", ""]

    lines += ["## Experiments", ""]
    for m in spec.experiments:
        report = out_dir / f"{m.id}.md"
        lines.append(
            f"- [{m.id}]({report.name}): {m.title}" if report.exists() else f"- {m.id}: {m.title} (no report yet)"
        )
        for row in before_after_rows(ctx, m.steps):
            if row[2] != "–":
                lines.append(f"  - {row[1]} vs {row[0]}: composite {row[2]} ({row[3]}), cost {row[4]}")
    lines.append("")

    lines += [
        "## Per strategy",
        "",
        (
            "Every experiment run that has results, grouped by strategy. Composites are only comparable between "
            "runs on the same dataset and suite (same judge setting); ★ marks the best composite per strategy and "
            "dataset."
        ),
        "",
    ]
    by_strategy: dict[str, list[tuple[Step, dict[str, Any]]]] = {}
    for m in spec.experiments:
        for s in m.steps:
            p = existing(ctx, s.key)
            if p is not None:
                by_strategy.setdefault(s.strategy, []).append((s, load_stats(p)))
    if not by_strategy:
        lines.append("No runs yet.")
    for strategy in run.STRATEGIES:
        entries = by_strategy.get(strategy)
        if not entries:
            continue
        best: dict[tuple[str, Any], float] = {}
        for s, st in entries:
            key = (s.dataset.name, st["suite_id"])
            best[key] = max(best.get(key, -1.0), st["composite"] or 0.0)
        rows = []
        for s, st in sorted(entries, key=lambda e: (e[0].dataset.name, -(e[1]["composite"] or 0))):
            star = (
                " ★" if (st["composite"] or 0.0) == best[(s.dataset.name, st["suite_id"])] and len(entries) > 1 else ""
            )
            calls = "/".join(compare.num(st[f"{k}_calls_mean"], ".1f") for k in ("llm", "decision", "tool"))
            rows.append(
                [s.experiment, f"`{s.run_id}`", st["label"], ds_name(s), s.judge or "off", f"{pts(st['composite'])}{star}",
                 pts(st["kind_macro_f1"]), compare.usd(st["cost_mean"]), calls, compare.num(st["latency_p50"], ".0f")]
            )  # fmt: skip
        n_versions = Counter(str(st["metadata"].get("version")) for _, st in entries)
        lines += [
            f"### {strategy}",
            "",
            f"{len(entries)} run(s); code versions {', '.join(f'v{v}' for v in sorted(n_versions))}.",
            "",
            compare.md_table(
                [
                    "experiment",
                    "run id",
                    "strategy@version config label",
                    "dataset",
                    "judge",
                    "composite",
                    "kind macro-F1",
                    "cost/issue",
                    "calls llm/dec/tool",
                    "latency p50 ms",
                ],
                rows,
            ),
            "",
        ]
    path = out_dir / "overview.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


# --- CLI ------------------------------------------------------------------------------


async def live_versions() -> dict[str, str]:
    async with httpx.AsyncClient(base_url=run.INGRESS_URL, timeout=30) as http:
        return await run.strategy_versions(http)


def select(spec: Spec, only: list[str] | None, steps: list[str] | None) -> list[tuple[Experiment, list[Step]]]:
    """The experiments to work on and, in each, the steps to run (all, unless --step names some)."""
    ids = [m.id for m in spec.experiments]
    for o in only or []:
        if o not in ids:
            raise SystemExit(f"--only {o}: no such experiment (have {', '.join(ids)})")
    wanted: set[str] = set()
    for ref in steps or []:
        match = [s.key for s in spec.steps.values() if ref in (s.key, s.id)]
        if not match:
            raise SystemExit(f"--step {ref}: no such step (see --list)")
        wanted.update(match)
    out = []
    for m in spec.experiments:
        if only and m.id not in only:
            continue
        chosen = [s for s in m.steps if not wanted or s.key in wanted]
        if chosen:
            out.append((m, chosen))
    return out


async def main_async(args: argparse.Namespace) -> int:
    spec = load_spec(
        args.file, size=args.size, profile=args.profile, models=args.model or [], no_judge=args.no_judge,
        campaign=args.campaign,
    )  # fmt: skip
    if args.list:
        print_list(spec)
        return 0
    selected = select(spec, args.only, args.step)
    assert spec.models is not None
    for line in models_lines(spec.models):
        print(line)
    if args.no_judge:
        print("  (--no-judge: the judge is off)")
    if args.register and not (args.dry_run or args.report_only):
        print(f"registering {run.SERVICE_URL} with {run.ADMIN_URL}", flush=True)
        await run.register_deployment()
    ctx = Context(
        spec=spec,
        results_dir=args.results_dir,
        versions=await live_versions(),
        force=args.force,
        include_optional=args.include_optional,
        max_concurrency=args.max_concurrency,
    )
    sequence = spec.ordered(s for _, chosen in selected for s in chosen)
    if args.dry_run:
        print_plan(ctx, sequence)
        return 0
    out_dir = report_dir(spec, args.results_dir)
    failed = 0
    # Steps run in run order (across experiments); an experiment's report is
    # written once its last selected step is done. Its other steps are listed
    # from existing results.
    outcomes: dict[str, dict[str, Outcome]] = {}
    pending: dict[str, int] = {}
    for m, chosen in selected:
        outcomes[m.id] = {}
        for step in m.steps:
            if args.report_only or step not in chosen:
                p = existing(ctx, step.key)
                why = "--report-only" if args.report_only else "not selected"
                outcomes[m.id][step.key] = Outcome(step, "exists" if p else "not run", why, p)
        pending[m.id] = 0 if args.report_only else len(chosen)
    by_id = {m.id: m for m, _ in selected}

    def report(exp_id: str) -> None:
        m = by_id[exp_id]
        path = write_experiment_report(ctx, m, [outcomes[exp_id][s.key] for s in m.steps], out_dir)
        print(f"[{exp_id}] report: {rel(path)}", flush=True)

    for exp_id, n in pending.items():
        if n == 0:
            report(exp_id)
    for step in [] if args.report_only else sequence:
        ctx.versions = await live_versions() or ctx.versions
        o = await execute_step(ctx, plan_step(ctx, step))
        failed += o.status == "failed"
        print(f"[{step.experiment}] {step.run_id}: {o.status} — {o.detail}", flush=True)
        outcomes[step.experiment][step.key] = o
        pending[step.experiment] -= 1
        if pending[step.experiment] == 0:
            report(step.experiment)
    path = write_overview(ctx, out_dir)
    print(f"overview: {rel(path)}")
    return 1 if failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run-experiments", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--size", type=int, default=DEFAULT_SIZE, help="dataset size: 20 or 500 (default: %(default)s)")
    p.add_argument("--profile", help="model profile (default: $EXPERIMENT_PROFILE, else the file's default_profile)")
    p.add_argument(
        "--model",
        action="append",
        metavar="ROLE=ADDRESS",
        help=f"override a model role (repeatable); roles: {', '.join(ROLES)}",
    )
    p.add_argument("--only", action="append", metavar="EXPERIMENT", help="run only this experiment (repeatable)")
    p.add_argument("--step", action="append", metavar="STEP", help="run only this step (repeatable; id or id@split)")
    p.add_argument(
        "--campaign",
        metavar="NAME",
        help="group the runs under NAME: it prefixes run ids (so they are fresh runs, not earlier results), "
        "the Logfire experiment names and the report directory, and is in their metadata",
    )
    p.add_argument("--list", action="store_true", help="list the experiments and their steps")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the estimated model calls")
    p.add_argument("--force", action="store_true", help="re-run steps whose results exist, through the service")
    p.add_argument("--report-only", action="store_true", help="only (re)write the reports from existing results")
    p.add_argument("--no-judge", action="store_true", help="skip the eval judge (quality, suggested_reply)")
    p.add_argument(
        "--register",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="register $SERVICE_URL with Restate before running (default: on)",
    )
    p.add_argument("--max-concurrency", type=int, help="cases in flight per step (default: profile/step settings)")
    p.add_argument("--include-optional", action="store_true", help="also run steps marked optional")
    p.add_argument(
        "--file",
        type=Path,
        default=EXPERIMENTS_DIR,
        help="the experiments directory, or one experiment file (default: %(default)s)",
    )
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    load_repo_env()
    args = parse_args(argv)
    if args.list or args.dry_run:
        sys.exit(asyncio.run(main_async(args)))
    run.setup_logfire()
    try:
        code = asyncio.run(main_async(args))
    finally:
        run.shutdown_logfire()
    sys.exit(code)

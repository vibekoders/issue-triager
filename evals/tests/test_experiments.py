"""Tests for the experiment runner: the experiments file, model roles and profiles,
planning, offline rescoring, reports, the CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from triager_evals import experiments as ms
from triager_evals import results, run
from triager_evals.build_dataset import load_dataset, read_dataset_metadata
from triager_evals.evaluators import EVALUATORS_VERSION
from triager_evals.suite import ROOT, case_digest, config_hash, workflow_key

DATASETS = ROOT / "datasets"
DEV20 = DATASETS / "kubernetes-dev-20.yaml"
EXPERIMENT_IDS = [
    "heuristic-base-rates",
    "single-shot-model-size",
    "workflow-escalation",
    "agent-tools",
    "ensemble-voters",
    "escalation-coordination",
    "final-comparison",
    "agentic-sol",
    "final-extras",
]
AZURE = {
    "llm": "azure:gpt-6-luna",
    "llm_large": "azure:gpt-6-sol",
    "decision": "typesafe:jev-latest",
    "decision_alt": "ollama:tev1:0.8b-q8_0",
    "judge": "azure:gpt-6-sol",
}

PROFILES = ["openai", "anthropic", "gemini", "azure", "local"]
# The hosted profiles: one provider for the LLMs and the judge, Jev for decisions,
# the local tev1 only for the decision-model comparison.
HOSTED_PROFILES = {
    "openai": ("openai:gpt-6-luna", "openai:gpt-6-sol", "openai:gpt-6-sol"),
    "anthropic": ("anthropic:claude-haiku-4-5", "anthropic:claude-sonnet-5", "anthropic:claude-sonnet-5"),
    "gemini": ("gemini:gemini-3.5-flash-lite", "gemini:gemini-3.1-pro-preview", "gemini:gemini-3.1-pro-preview"),
    "azure": ("azure:gpt-6-luna", "azure:gpt-6-sol", "azure:gpt-6-sol"),
}


@pytest.fixture(autouse=True)
def clean_model_env(monkeypatch):
    """The real .env (loaded by devenv) may set JUDGE_MODEL or EXPERIMENT_*; tests start without them."""
    import os

    for var in list(os.environ):
        if var == "JUDGE_MODEL" or var.startswith("EXPERIMENT_"):
            monkeypatch.delenv(var)


# --- the experiments file ------------------------------------------------------------


@pytest.mark.parametrize("size", [20, 500])
@pytest.mark.parametrize("profile", PROFILES)
def test_experiments_file_loads_and_references_resolve(size, profile):
    spec = ms.load_spec(size=size, profile=profile, env={})
    assert [m.id for m in spec.experiments] == sorted(EXPERIMENT_IDS)  # file name order
    assert spec.files["agent-tools"] == ms.EXPERIMENTS_DIR / "agent-tools.yaml"
    assert spec.models is not None and spec.models.profile == profile
    for step in spec.steps.values():
        assert step.dataset == DATASETS / f"kubernetes-{step.split}-{size}.yaml", step.key
        assert step.dataset.exists(), step.dataset
        assert step.run_id == f"{step.id}-{step.split}-{size}-{profile}"
        assert ":" not in step.run_id and "/" not in step.run_id
        # every placeholder was filled in, and the config is valid for its strategy
        assert not ms.PLACEHOLDER.search(json.dumps(step.config) + step.label), step.key
        assert run.check_config(step.strategy, step.config, step.key) == step.config
        assert step.judge == spec.models.roles["judge"]
    exps = {m.id: m for m in spec.experiments}
    assert {s.split for s in exps["final-comparison"].steps} == {"test"}
    assert all(s.split == "dev" for e in EXPERIMENT_IDS[1:6] for s in exps[e].steps)
    assert {s.key for s in exps["heuristic-base-rates"].steps} == {
        "heuristic@dev", "heuristic@test", "heuristic-base-rates@dev", "heuristic-base-rates@test",
    }  # fmt: skip
    assert spec.overview is not None and all(r.startswith("final-") for r in spec.overview.runs)
    # a multi-split baseline resolves to the same split
    assert spec.step("heuristic-base-rates@test").baseline == "heuristic@test"
    assert spec.step("final-heuristic").source == "heuristic@test"


def test_step_ids_are_unique_and_run_ids_never_collide():
    seen: set[str] = set()
    for size in (20, 500):
        for profile in PROFILES:
            spec = ms.load_spec(size=size, profile=profile, env={})
            ids = [s.run_id for s in spec.steps.values()]
            assert len(ids) == len(set(ids))
            assert not seen & set(ids)
            seen |= set(ids)


def test_profiles_fill_in_the_roles():
    azure = ms.load_spec(profile="azure", env={})
    assert azure.models.roles == AZURE and azure.models.max_concurrency == 8
    assert azure.step("single-shot-llm").config == {"model": "azure:gpt-6-luna"}
    assert azure.step("single-shot-llm").label == "single_shot azure:gpt-6-luna"
    assert azure.step("ensemble-judge").config == {
        "voters": [
            {"kind": "llm", "model": "azure:gpt-6-luna"},
            {"kind": "llm", "model": "azure:gpt-6-sol"},
            {"kind": "decision", "model": "typesafe:jev-latest"},
        ],
        "aggregator": {"kind": "judge", "model": "azure:gpt-6-sol"},
    }
    local = ms.load_spec(profile="local", env={})
    assert local.step("workflow-decision-alt").config == {"decision_model": "ollama:tev1:4b"}
    assert local.step("workflow-esc-0.8").config == {
        "decision_model": "ollama:tev1:0.8b-q8_0",
        "escalation": {"model": "ollama:qwen3:1.7b", "thresholds": {"accept": 0.8}},
    }
    # another model -> another config hash -> another results file and Restate key
    a, b = azure.step("single-shot-llm"), local.step("single-shot-llm")
    assert config_hash(a.config) != config_hash(b.config)


@pytest.mark.parametrize("profile", HOSTED_PROFILES)
def test_hosted_profiles(profile):
    llm, llm_large, judge = HOSTED_PROFILES[profile]
    spec = ms.load_spec(profile=profile, env={})
    assert spec.models is not None and spec.models.max_concurrency == 8
    assert spec.models.roles == {
        "llm": llm,
        "llm_large": llm_large,
        "decision": "typesafe:jev-latest",
        "decision_alt": "ollama:tev1:0.8b-q8_0",
        "judge": judge,
    }
    ctx = ms.Context(spec=spec, results_dir=Path("/nonexistent"), versions={})
    assert ms.step_concurrency(ctx, spec.step("single-shot-llm")) == 8
    assert ms.step_concurrency(ctx, spec.step("workflow-decision-alt")) == 1  # local tev1 on Ollama


def test_profile_models_are_priced():
    """Every model a profile names has a price in triage-core's table (the judge's cost reads it too)."""
    from triager_evals.suite import price_cost

    for name, profile in DATA["profiles"].items():
        for role, address in profile.items():
            if role != "max_concurrency":
                assert price_cost(address, 1_000_000, 0) is not None, (name, role, address)
    assert price_cost("openai:gpt-6-luna", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert price_cost("gemini:gemini-3.1-pro-preview", 1_000_000, 1_000_000) == pytest.approx(14.0)


# --- model resolution ----------------------------------------------------------------

DATA = yaml.safe_load((ms.EXPERIMENTS_DIR / "profiles.yaml").read_text())


def test_profile_selection_precedence():
    assert ms.resolve_models(DATA, env={}).profile == "openai"
    assert ms.resolve_models(DATA, env={}).profile_source == "default_profile"
    m = ms.resolve_models(DATA, env={"EXPERIMENT_PROFILE": "local"})
    assert (m.profile, m.profile_source) == ("local", "EXPERIMENT_PROFILE")
    m = ms.resolve_models(DATA, "azure", env={"EXPERIMENT_PROFILE": "local"})
    assert (m.profile, m.profile_source) == ("azure", "--profile")
    with pytest.raises(SystemExit, match="profile 'nope'"):
        ms.resolve_models(DATA, "nope", env={})


def test_role_precedence_cli_over_env_over_profile():
    env = {"EXPERIMENT_MODEL_LLM": "openai:env-llm", "JUDGE_MODEL": "ollama:alias-judge"}
    m = ms.resolve_models(DATA, "azure", env=env)
    assert m.roles["llm"] == "openai:env-llm" and m.sources["llm"] == "env EXPERIMENT_MODEL_LLM"
    assert m.roles["judge"] == "ollama:alias-judge" and m.sources["judge"] == "env JUDGE_MODEL"
    assert m.roles["decision"] == AZURE["decision"] and m.sources["decision"] == "profile azure"
    # EXPERIMENT_MODEL_JUDGE wins over the JUDGE_MODEL alias
    m = ms.resolve_models(DATA, "azure", env=env | {"EXPERIMENT_MODEL_JUDGE": "azure:gpt-6-luna"})
    assert m.roles["judge"] == "azure:gpt-6-luna" and m.sources["judge"] == "env EXPERIMENT_MODEL_JUDGE"
    # --model wins over everything
    m = ms.resolve_models(DATA, "azure", ["llm=anthropic:cli", "JUDGE=ollama:cli-judge"], env=env)
    assert m.roles["llm"] == "anthropic:cli" and m.sources["llm"] == "--model"
    assert m.roles["judge"] == "ollama:cli-judge"


@pytest.mark.parametrize(
    ("override", "error"),
    [("bogus=a:b", "unknown role"), ("llm", "ROLE=ADDRESS"), ("llm=", "ROLE=ADDRESS"), ("llm=nocolon", "provider")],
)
def test_bad_model_overrides(override, error):
    with pytest.raises(SystemExit, match=error):
        ms.resolve_models(DATA, None, [override], env={})


def test_overrides_reach_configs_labels_and_judge():
    spec = ms.load_spec(profile="azure", models=["llm=ollama:qwen3:1.7b", "judge=azure:gpt-6-luna"], env={})
    step = spec.step("workflow-esc-0.8")
    assert step.config["escalation"]["model"] == "ollama:qwen3:1.7b"
    assert "escalation ollama:qwen3:1.7b" in step.label
    assert step.judge == "azure:gpt-6-luna"
    assert step.run_id == "workflow-esc-0.8-dev-20-azure"
    assert all(s.judge is None for s in ms.load_spec(no_judge=True, env={}).steps.values())


def test_substitute():
    roles = {"llm": "a:b", "judge": "c:d"}
    out = ms.substitute({"m": "{llm}", "l": ["x {judge} y", 3], "n": None}, roles, "here")
    assert out == {"m": "a:b", "l": ["x c:d y", 3], "n": None}
    with pytest.raises(SystemExit, match=r"unknown placeholder \{lmm\}"):
        ms.substitute("{lmm}", roles, "here")


def test_compose_run_id_and_dataset_by_size():
    assert ms.compose_run_id("workflow-esc-0.8", "dev", 20, "azure") == "workflow-esc-0.8-dev-20-azure"
    assert ms.dataset_path(ms.DEFAULT_DATASET, "test", 500) == DATASETS / "kubernetes-test-500.yaml"
    with pytest.raises(SystemExit, match="--size 7"):
        ms.load_spec(size=7, env={})


def test_campaign_prefixes_run_ids_report_dir_and_metadata(tmp_path):
    spec = ms.load_spec(size=500, profile="azure", env={}, campaign="kube-500")
    step = spec.step("final-agent-tools")
    assert step.run_id == "kube-500-final-agent-tools-test-500-azure"
    assert ms.report_dir(spec, tmp_path) == tmp_path / "experiments" / "kube-500-500-azure"
    assert ms.run_metadata(spec, step)["campaign"] == "kube-500"
    plain = ms.load_spec(size=500, profile="azure", env={})
    assert "campaign" not in ms.run_metadata(plain, plain.step("final-agent-tools"))
    with pytest.raises(SystemExit, match="--campaign"):
        ms.load_spec(profile="azure", env={}, campaign="a/b")


# --- the datasets ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("small", "big"),
    [("dev-20", "dev-500"), ("test-20", "test-500"), ("test-20", "test-600"), ("test-500", "test-600")],
)
def test_smaller_datasets_are_prefixes(small, big):
    a = load_dataset(DATASETS / f"kubernetes-{small}.yaml")
    b = load_dataset(DATASETS / f"kubernetes-{big}.yaml")
    n = len(a.cases)
    assert [c.name for c in a.cases] == [c.name for c in b.cases[:n]]
    digest = lambda c: case_digest(c.inputs.model_dump(mode="json"))
    assert [digest(c) for c in a.cases] == [digest(c) for c in b.cases[:n]]
    # built from the same source with the same flags; only the limit differs
    meta = [read_dataset_metadata(DATASETS / f"kubernetes-{name}.yaml") for name in (small, big)]
    for m in meta:
        del m["cases"], m["limit"]
    assert meta[0] == meta[1]


def test_500_sets_have_500_cases():
    for split in ("dev", "test"):
        assert read_dataset_metadata(DATASETS / f"kubernetes-{split}-500.yaml")["cases"] == 500


def test_calls_estimates():
    step = ms.Step("r", "Workflow", {}, DEV20, "l", "m")
    none = ms.calls_per_case(step, {"decision_model": "ollama:tev1:0.8b-q8_0"})
    assert none == [("ollama:tev1:0.8b-q8_0", 3)]
    esc = ms.calls_per_case(
        step,
        {"decision_model": "d", "escalation": {"model": "e", "thresholds": {"accept": 0.6}}, "reply_model": "e"},
    )
    assert esc == [("d", 3), ("e", 3), ("e", 1)]
    ens = ms.Step("r", "Ensemble", {}, DEV20, "l", "m", judge="ollama:qwen3:4b")
    cfg = {
        "voters": [{"kind": "llm", "model": "a"}, {"kind": "decision", "model": "b"}],
        "aggregator": {"kind": "judge", "model": "j"},
    }
    assert ms.calls_per_case(ens, cfg) == [("a", 1), ("b", 3), ("j", 1)]
    assert ms.judge_calls(ens) == [("ollama:qwen3:4b", 2)]
    agent = ms.Step("r", "Agent", {}, DEV20, "l", "m")
    assert ms.calls_per_case(agent, {"model": "a", "tools": []}) == [("a", 1)]
    assert ms.calls_per_case(agent, {"model": "a"}) == [("a", 5)]


def test_concurrency_respects_ollama_and_cli():
    azure = ms.load_spec(profile="azure", env={})
    ctx = ms.Context(spec=azure, results_dir=Path("/nonexistent"), versions={})
    assert ms.step_concurrency(ctx, azure.step("single-shot-llm")) == 8
    assert ms.step_concurrency(ctx, azure.step("workflow-decision-alt")) == 1  # local tev1 on Ollama
    assert ms.step_concurrency(ctx, azure.step("heuristic@dev")) == 8
    ctx.max_concurrency = 3
    assert ms.step_concurrency(ctx, azure.step("workflow-decision-alt")) == 3
    local = ms.load_spec(profile="local", env={})
    ctx = ms.Context(spec=local, results_dir=Path("/nonexistent"), versions={})
    assert ms.step_concurrency(ctx, local.step("heuristic@dev")) == 1  # the Ollama judge
    no_judge = ms.load_spec(profile="local", no_judge=True, env={})
    ctx = ms.Context(spec=no_judge, results_dir=Path("/nonexistent"), versions={})
    assert ms.step_concurrency(ctx, no_judge.step("heuristic@dev")) == 8


def test_select_steps_and_experiments():
    spec = ms.load_spec(env={})
    sel = ms.select(spec, None, ["heuristic"])
    assert [(m.id, [s.key for s in steps]) for m, steps in sel] == [
        ("heuristic-base-rates", ["heuristic@dev", "heuristic@test"])
    ]
    sel = ms.select(spec, None, ["heuristic@test", "workflow-esc-0.8"])
    assert [[s.key for s in steps] for _, steps in sel] == [["heuristic@test"], ["workflow-esc-0.8"]]
    sel = ms.select(spec, ["agent-tools"], None)
    assert [m.id for m, _ in sel] == ["agent-tools"] and len(sel[0][1]) == 2
    assert len(ms.select(spec, None, None)) == len(EXPERIMENT_IDS)
    with pytest.raises(SystemExit, match="--only m3"):
        ms.select(spec, ["m3"], None)
    with pytest.raises(SystemExit, match="--step nope"):
        ms.select(spec, None, ["nope"])


# --- planning and running against a scratch results directory ---------------------------

PROFILE = {"llm": "ollama:a", "llm_large": "ollama:b", "decision": "x:d", "decision_alt": "x:e", "judge": "ollama:j"}


def rid(step_id: str) -> str:
    return f"{step_id}-dev-20-t"


def triage_response(kind: str = "bug", version: str = "1", cfg_hash: str = "x") -> dict[str, Any]:
    usage = {k: 0 for k in ("input_tokens", "output_tokens", "llm_calls", "decision_calls", "tool_calls", "latency_ms")}
    return {
        "triage": {"kind": {"value": kind, "confidence": 0.7}, "is_regression": {"value": False}},
        "usage": usage | {"cost_usd": 0.0},
        "strategy": "Heuristic",
        "version": version,
        "config_hash": cfg_hash,
    }


def stored_run(
    results_dir: Path, run_id: str, n: int, kind: str = "bug", evaluators_version: str = "1", config=None
) -> Path:
    """A results file shaped like triage-eval's, for the first n dev-20 cases."""
    config = config or {}
    cfg_hash = config_hash(config)
    cases = []
    for c in load_dataset(DEV20).cases[:n]:
        key = workflow_key(run_id, cfg_hash, case_digest(c.inputs.model_dump(mode="json")), 0)
        cases.append(
            results.case_record(
                name=c.name, repeat=0, workflow_key=key, expected=c.expected_output,
                output=triage_response(kind, cfg_hash=cfg_hash), error=None, scores={}, reasons={},
                metrics={"cost": 0.0}, duration=0.5, judged=False, weights={"kind": 3},
            )
        )  # fmt: skip
    meta = {
        "experiment": f"Heuristic@1 {cfg_hash} t", "run_id": run_id, "strategy": "Heuristic", "version": "1",
        "config": config, "config_file": "x", "config_hash": cfg_hash,
        "dataset": str(DEV20), "dataset_hash": "old", "suite_id": "old", "evaluators_version": evaluators_version,
        "judge_model": None, "weights": {"kind": 3}, "repeats": 1, "limit": None, "label": "t", "attached": 0,
        "git_sha": "abc", "started_at": "then",
    }  # fmt: skip
    path = results.results_path(results_dir, run_id, "Heuristic", cfg_hash)
    results.save(path, meta, cases)
    return path


STEP_BASE = {"strategy": "Heuristic", "config": {}, "split": "dev", "dataset": str(DEV20), "judge": False}


def write_dir(
    root: Path, experiments: dict[str, dict[str, Any]], overview: dict[str, Any] | None = None, profiles: bool = True
) -> Path:
    """An experiments directory: profiles.yaml (profile "t") and one file per experiment."""
    root.mkdir(parents=True, exist_ok=True)
    if profiles:
        data: dict[str, Any] = {
            "profiles": {"t": PROFILE},
            "default_profile": "t",
            "defaults": {"seconds_per_call": {"default": 10}, "sizes": [20]},
        }
        if overview:
            data["overview"] = overview
        (root / "profiles.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    for name, exp in experiments.items():
        exp = {"title": name} | exp
        exp["steps"] = [STEP_BASE | s for s in exp.get("steps", [])]
        (root / f"{name}.yaml").write_text(yaml.safe_dump(exp), encoding="utf-8")
    return root


def write_spec(tmp_path: Path, steps: list[dict[str, Any]], compare: list[dict[str, Any]] | None = None) -> Path:
    exp = {"title": "test", "steps": steps, "compare": compare or []}
    return write_dir(tmp_path / "experiments", {"mx": exp}, overview={"title": "final", "runs": [steps[-1]["id"]]})


def test_inline_configs_are_checked_against_the_schema(tmp_path):
    path = write_spec(tmp_path, [{"id": "bad", "strategy": "SingleShot", "config": {"model": "{llm}", "modle": 1}}])
    with pytest.raises(SystemExit, match="not a valid SingleShot config"):
        ms.load_spec(path, env={})
    path = write_spec(tmp_path, [{"id": "ok", "strategy": "SingleShot", "config": {"model": "{llm}"}}])
    assert ms.load_spec(path, env={}).step("ok").config == {"model": "ollama:a"}
    path = write_spec(tmp_path, [{"id": "a"}, {"id": "b", "baseline": "nope"}])
    with pytest.raises(SystemExit, match="unknown step 'nope'"):
        ms.load_spec(path, env={})
    path = write_spec(tmp_path, [{"id": "a", "split": ["dev", "test"]}], [{"baseline": "a", "runs": ["a"]}])
    with pytest.raises(SystemExit, match="several splits"):
        ms.load_spec(path, env={})


def test_plan_actions(tmp_path):
    results_dir = tmp_path / "results"
    stored_run(results_dir, rid("old-v1"), 20)
    spec = ms.load_spec(
        write_spec(
            tmp_path,
            [
                {"id": "old-v1", "version": "1"},
                {"id": "new-v2", "version": "2", "baseline": "old-v1"},
                {"id": "derived", "source": "old-v1", "limit": 5},
                {"id": "fresh"},
                {"id": "slow", "optional": True},
                {"id": "nodata", "dataset": "datasets/does-not-exist-{size}.yaml"},
            ],
        ),
        env={},
    )
    ctx = ms.Context(spec=spec, results_dir=results_dir, versions={"Heuristic": "1"})
    actions = {s.key: ms.plan_step(ctx, s) for m in spec.experiments for s in m.steps}
    assert actions["old-v1"].action == "rescore" and "evaluators v1" in actions["old-v1"].reason
    assert actions["new-v2"].action == "blocked" and "needs Heuristic v2" in actions["new-v2"].reason
    assert actions["derived"].action == "derive"
    assert actions["fresh"].action == "run"
    assert actions["slow"].action == "optional"
    assert actions["nodata"].action == "blocked" and "missing" in actions["nodata"].reason
    # a v1 step must not be re-run through a service that runs v2
    ctx.force = True
    ctx.versions = {"Heuristic": "2"}
    assert ms.plan_step(ctx, spec.step("old-v1")).action == "blocked"
    assert ms.plan_step(ctx, spec.step("new-v2")).action == "run"
    ctx.versions = {}
    assert ms.plan_step(ctx, spec.step("fresh")).action == "blocked"


def test_skip_needs_the_exact_config_hash(tmp_path):
    """Results in the run id's directory for another config (another model) don't count."""
    results_dir = tmp_path / "results"
    stored_run(results_dir, rid("s"), 20, evaluators_version=EVALUATORS_VERSION)
    spec = ms.load_spec(write_spec(tmp_path, [{"id": "s", "config": {"base_rates": True}}]), env={})
    ctx = ms.Context(spec=spec, results_dir=results_dir, versions={"Heuristic": "1"})
    p = ms.plan_step(ctx, spec.step("s"))
    assert p.action == "run"
    assert p.path == results.results_path(results_dir, rid("s"), "Heuristic", config_hash({"base_rates": True}))
    assert ms.existing(ctx, "s") is None


def test_rescore_and_derive_offline_then_reports(tmp_path):
    results_dir = tmp_path / "results"
    src = stored_run(results_dir, rid("old-v1"), 20)
    spec = ms.load_spec(
        write_spec(
            tmp_path,
            [
                {"id": "old-v1", "version": "1"},
                {"id": "derived", "source": "old-v1", "limit": 5, "baseline": "old-v1"},
            ],
            compare=[{"title": "both", "baseline": "old-v1", "runs": ["old-v1", "derived"]}],
        ),
        env={},
    )
    ctx = ms.Context(spec=spec, results_dir=results_dir, versions={})  # no service needed
    outcomes = [asyncio.run(ms.execute_step(ctx, ms.plan_step(ctx, s))) for s in spec.experiments[0].steps]
    assert [o.status for o in outcomes] == ["rescored", "derived"], [o.detail for o in outcomes]

    rescored = results.load(src)
    assert rescored["metadata"]["evaluators_version"] == EVALUATORS_VERSION
    assert rescored["metadata"]["rescored"]["from_evaluators_version"] == "1"
    assert rescored["metadata"]["git_sha"] == "abc"  # provenance of the outputs is kept
    assert rescored["metadata"]["models"] == PROFILE and rescored["metadata"]["profile"] == "t"
    assert rescored["metadata"]["experiment_id"] == "mx" and rescored["metadata"]["step_id"] == "old-v1"
    assert len(rescored["cases"]) == 20 and all(c["duration"] == 0.5 for c in rescored["cases"])
    kinds_right = sum(c["scores"].get("kind") is True for c in rescored["cases"])
    assert kinds_right == sum("bug" in c["expected"]["kind"] for c in rescored["cases"])

    derived = results.load(outcomes[1].path)
    assert derived["metadata"]["run_id"] == rid("derived") and derived["metadata"]["limit"] == 5
    assert derived["metadata"]["size"] == 20 and derived["metadata"]["split"] == "dev"
    assert len(derived["cases"]) == 5
    # workflow keys still name the Restate invocations that made the outputs
    assert all(c["workflow_key"].startswith(rid("old-v1") + ":") for c in derived["cases"])
    # ... also after the derived file is re-scored in place (evaluators or suite changed)
    again = asyncio.run(run.rescore(outcomes[1].path, dataset=DEV20, limit=5, results_dir=results_dir))
    assert again == outcomes[1].path
    rekeyed = results.load(again)
    assert rekeyed["metadata"]["run_id"] == rid("derived")
    assert all(c["workflow_key"].startswith(rid("old-v1") + ":") for c in rekeyed["cases"])
    # now up to date: skipped
    assert ms.plan_step(ctx, spec.step("old-v1")).action == "skip"

    out_dir = ms.report_dir(spec, results_dir)
    assert out_dir == results_dir / "experiments" / "20-t"
    report = ms.write_experiment_report(ctx, spec.experiments[0], outcomes, out_dir).read_text()
    assert "# mx: test" in report and "## Before → after" in report and "## both" in report
    assert "#### Per field" in report and "Head-to-head" in report and "profile `t`" in report
    assert (out_dir / "mx-1-both.csv").exists() and (out_dir / "mx-1-both-pareto.svg").exists()
    overview = ms.write_overview(ctx, out_dir).read_text()
    assert "## Pareto frontier: final" in overview and "### Heuristic" in overview
    assert (out_dir / "overview-pareto.svg").exists()


def test_rescore_refuses_cases_the_stored_run_lacks(tmp_path):
    src = stored_run(tmp_path, "partial", 3)
    with pytest.raises(SystemExit, match="no stored output"):
        asyncio.run(run.rescore(src, dataset=DEV20, run_id="x", results_dir=tmp_path))


def test_config_from_best_picks_higher_composite(tmp_path):
    results_dir = tmp_path / "results"
    cfg_a, cfg_b = {}, {"base_rates": True}
    stored_run(results_dir, rid("a"), 20, kind="feature", config=cfg_a)
    stored_run(results_dir, rid("b"), 20, kind="bug", config=cfg_b)
    for r, cfg in ((rid("a"), cfg_a), (rid("b"), cfg_b)):
        asyncio.run(
            run.rescore(results.results_path(results_dir, r, "Heuristic", config_hash(cfg)), results_dir=results_dir)
        )
    spec = ms.load_spec(
        write_spec(
            tmp_path,
            [
                {"id": "a", "config": cfg_a},
                {"id": "b", "config": cfg_b},
                {"id": "final", "config_from_best": ["a", "b"]},
            ],
        ),
        env={},
    )
    ctx = ms.Context(spec=spec, results_dir=results_dir, versions={"Heuristic": "1"})
    config, why = ms.chosen_config(ctx, spec.step("final"))
    assert config == cfg_b and why is not None and "b (" in why


# --- CLI ------------------------------------------------------------------------------


def test_cli_list(capsys):
    asyncio.run(ms.main_async(ms.parse_args(["--list", "--size", "500", "--profile", "local"])))
    out = capsys.readouterr().out
    assert "size 500" in out and "profile: local (from --profile)" in out
    for exp in EXPERIMENT_IDS:
        assert f"  {exp} " in out
    assert "heuristic@test" in out and "test-500" in out and "single_shot ollama:qwen3:1.7b" in out
    order = [line.split()[1] for line in out.split("steps in run order:")[1].strip().splitlines()]
    assert order == [s.key for s in ms.load_spec(size=500, profile="local", env={}).ordered()]
    assert len(order) == 32


def test_cli_dry_run(capsys, monkeypatch, tmp_path):
    async def no_service() -> dict[str, str]:
        return {}

    monkeypatch.setattr(ms, "live_versions", no_service)
    argv = ["--dry-run", "--results-dir", str(tmp_path), "--model", "judge=azure:gpt-6-luna", "--step", "heuristic"]
    assert asyncio.run(ms.main_async(ms.parse_args(argv))) == 0
    out = capsys.readouterr().out
    assert "judge         azure:gpt-6-luna" in out and "(--model)" in out
    assert "heuristic-dev-20-openai" in out and "heuristic-test-20-openai" in out
    assert "single-shot" not in out  # only the selected step
    assert "blocked" in out and "service" in out  # nothing reachable here
    assert "reports in" in out and "20-openai" in out
    # nothing ran: no results, no reports
    assert not any(tmp_path.iterdir())


def test_cli_defaults():
    args = ms.parse_args([])
    assert args.size == 20 and args.register is True and args.profile is None and not args.no_judge
    assert ms.parse_args(["--no-register"]).register is False


# --- loading the directory ----------------------------------------------------------------


def test_real_run_order_respects_dependencies():
    spec = ms.load_spec(env={})
    pos = {k: i for i, k in enumerate(spec.order)}
    assert sorted(pos) == sorted(spec.steps)
    for s in spec.steps.values():
        for dep in [s.baseline, s.source, *s.config_from_best]:
            if dep:
                assert pos[dep] < pos[s.key], (dep, s.key)
    assert pos["heuristic@test"] < pos["final-heuristic"]
    assert pos["single-shot-llm-large"] < pos["final-single-shot"]
    # escalation-coordination compares with workflow-decision-only, so runs after it
    assert pos["workflow-decision-only"] < pos["draft-verify"] and pos["workflow-esc-0.8"] < pos["draft-verify"]
    # no dependencies: file name order, then order within the file
    assert spec.order[:2] == ["agent-no-tools", "agent-tools"]


def test_files_merge_into_one_spec_with_cross_file_references(tmp_path):
    root = write_dir(
        tmp_path / "exp",
        {
            # "a" sorts first but depends on "b"
            "a": {
                "steps": [{"id": "a1", "baseline": "b2"}, {"id": "a2"}],
                "compare": [{"baseline": "a1", "runs": ["a1", "b1"]}],
            },
            "b": {"steps": [{"id": "b1"}, {"id": "b2", "baseline": "b1"}]},
            "c": {"id": "c", "steps": [{"id": "c1", "source": "a2"}]},
        },
        overview={"title": "o", "runs": ["c1", "a1"]},
    )
    spec = ms.load_spec(root, env={})
    assert [m.id for m in spec.experiments] == ["a", "b", "c"]
    assert spec.files == {"a": root / "a.yaml", "b": root / "b.yaml", "c": root / "c.yaml"}
    assert spec.step("a1").baseline == "b2" and spec.step("c1").source == "a2"
    assert spec.overview is not None and spec.overview.runs == ["c1", "a1"]
    # a's steps wait for b1 (experiment a compares with it), a1 also for b2 (its baseline),
    # c1 for a2 (its source); otherwise file order
    assert spec.order == ["b1", "a2", "b2", "a1", "c1"]
    assert spec.step("a1").file == root / "a.yaml"
    # reports still go per experiment
    assert [s.key for s in spec.experiments[0].steps] == ["a1", "a2"]


def test_duplicate_step_ids_across_files(tmp_path):
    root = write_dir(tmp_path / "exp", {"a": {"steps": [{"id": "x"}]}, "b": {"steps": [{"id": "x"}]}})
    with pytest.raises(SystemExit, match=r"duplicate step 'x' in .*a\.yaml and .*b\.yaml"):
        ms.load_spec(root, env={})


def test_experiment_id_must_match_the_file_name(tmp_path):
    root = write_dir(tmp_path / "exp", {"a": {"id": "b", "steps": [{"id": "x"}]}})
    with pytest.raises(SystemExit, match="must equal the file name"):
        ms.load_spec(root, env={})


def test_overview_only_once(tmp_path):
    root = write_dir(
        tmp_path / "exp",
        {"a": {"steps": [{"id": "x"}], "overview": {"runs": ["x"]}}},
        overview={"title": "o", "runs": ["x"]},
    )
    with pytest.raises(SystemExit, match="overview defined in both"):
        ms.load_spec(root, env={})


def test_dependency_cycle_is_an_error(tmp_path):
    root = write_dir(
        tmp_path / "exp",
        {"a": {"steps": [{"id": "a1", "baseline": "b1"}]}, "b": {"steps": [{"id": "b1", "source": "a1"}]}},
    )
    with pytest.raises(SystemExit, match="dependency cycle.*a1 needs b1.*b1 needs a1"):
        ms.load_spec(root, env={})


def test_single_file_and_missing_references(tmp_path):
    root = write_dir(
        tmp_path / "exp",
        {"a": {"steps": [{"id": "a1", "baseline": "b1"}]}, "b": {"steps": [{"id": "b1"}, {"id": "b2"}]}},
    )
    spec = ms.load_spec(root / "b.yaml", env={})  # profiles.yaml next to it
    assert [m.id for m in spec.experiments] == ["b"] and spec.order == ["b1", "b2"]
    assert spec.models is not None and spec.models.profile == "t"
    with pytest.raises(SystemExit, match=r"unknown step 'b1'; it is not defined in the loaded files \(.*a\.yaml\)"):
        ms.load_spec(root / "a.yaml", env={})
    with pytest.raises(SystemExit, match="profiles file"):
        ms.load_spec(root / "profiles.yaml", env={})
    # no profiles.yaml next to it: the default directory's (default profile openai)
    lone = write_dir(tmp_path / "lone", {"b": {"steps": [{"id": "b1"}]}}, profiles=False)
    spec = ms.load_spec(lone / "b.yaml", env={})
    assert spec.models is not None and spec.models.profile == "openai"
    assert spec.step("b1").run_id == "b1-dev-20-openai"


def test_real_single_file():
    spec = ms.load_spec(ms.EXPERIMENTS_DIR / "agent-tools.yaml", env={})
    assert [s.key for s in spec.ordered()] == ["agent-no-tools", "agent-tools"]
    assert spec.step("agent-tools").run_id == ms.load_spec(env={}).step("agent-tools").run_id
    with pytest.raises(SystemExit, match="not defined in the loaded files"):
        ms.load_spec(ms.EXPERIMENTS_DIR / "escalation-coordination.yaml", env={})


def test_judge_must_be_a_boolean(tmp_path):
    root = write_dir(tmp_path / "exp", {"a": {"steps": [{"id": "x", "judge": "azure:gpt-6-luna"}]}})
    with pytest.raises(SystemExit, match="judge must be true or false"):
        ms.load_spec(root, env={})


def test_readme_example_literal_models_ignore_profiles(tmp_path):
    """The "Adding your own experiment" example in evals/experiments/README.md."""
    import re
    import shutil

    readme = (ms.EXPERIMENTS_DIR / "README.md").read_text()
    example = re.search(r"`agent-models.yaml`:\n\n```yaml\n(.*?)```", readme, re.DOTALL)
    assert example is not None
    root = tmp_path / "exp"
    root.mkdir()
    shutil.copy(ms.EXPERIMENTS_DIR / "profiles.yaml", root / "profiles.yaml")
    (root / "agent-models.yaml").write_text(example.group(1))
    for profile, models in [("azure", []), ("local", []), ("azure", ["llm=openai:x"])]:
        spec = ms.load_spec(root, profile=profile, models=models, env={})
        assert spec.step("agent-luna").config["model"] == "azure:gpt-6-luna"
        assert spec.step("agent-sol").config["model"] == "azure:gpt-6-sol"
        assert spec.step("agent-profile-llm").config["model"] == spec.models.roles["llm"]
        assert spec.step("agent-luna").judge == spec.models.roles["judge"]

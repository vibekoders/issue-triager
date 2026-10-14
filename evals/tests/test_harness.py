"""Tests for the eval harness: hashes, evaluators, statistics, configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml
from pydantic_evals import Case, Dataset

from triager_evals import compare, results
from triager_evals.build_dataset import Expected, load_dataset
from triager_evals.evaluators import code_evaluators, recall_at_k, sanity_problems, severity_distance
from triager_evals.suite import (
    QUALITY_RUBRIC,
    ROOT,
    SCHEMA_DIR,
    Suite,
    case_digest,
    composite,
    config_errors,
    config_hash,
    load_suite,
    parse_rust_rubric,
    price_cost,
    strict_schema,
    suite_id,
    workflow_key,
)

CONFIGS = ROOT / "configs"
SCHEMA_BY_PREFIX = {
    "heuristic": "heuristic-config",
    "single-shot": "single-shot-config",
    "workflow": "workflow-config",
    "agent": "agent-config",
    "ensemble": "ensemble-config",
    "draft-verify": "draft-verify-config",
}


# --- hashes and suite -----------------------------------------------------------


def test_config_hash_matches_rust_pinned_value():
    # crates/triage-core/src/hash.rs: PYTHON_HASH
    assert config_hash({"model": "ollama:qwen3:4b"}) == "aa413933d122"
    # the service's responses seen live (report-rust_service.md)
    assert config_hash({}) == "44136fa355b3"
    assert config_hash({"model": "ollama:qwen3:1.7b"}) == "fcb198213b35"


def test_config_hash_ignores_key_order_and_keeps_unicode():
    a = {"b": 1, "a": {"y": 0.8, "x": "é"}}
    b = {"a": {"x": "é", "y": 0.8}, "b": 1}
    assert config_hash(a) == config_hash(b)
    assert config_hash({"x": "é"}) != config_hash({"x": "\\u00e9"})


def test_case_digest_is_stable():
    ds = load_dataset(ROOT / "datasets" / "kubernetes-dev-20.yaml")
    first = ds.cases[0].inputs.model_dump(mode="json")
    again = load_dataset(ROOT / "datasets" / "kubernetes-dev-20.yaml").cases[0].inputs.model_dump(mode="json")
    assert case_digest(first) == case_digest(again)
    assert case_digest(dict(reversed(list(first.items())))) == case_digest(first)
    digests = {case_digest(c.inputs.model_dump(mode="json")) for c in ds.cases}
    assert len(digests) == len(ds.cases)
    assert workflow_key("r1", "abc", "def", 2) == "r1:abc:def:2"


def test_rubric_matches_rust():
    assert parse_rust_rubric() == QUALITY_RUBRIC


def test_judge_cost_uses_rust_price_table():
    assert price_cost("ollama:qwen3:4b", 1_000_000, 1_000_000) == pytest.approx(0.06 + 0.25)
    assert price_cost("ollama:mistral", 1_000_000, 0) == pytest.approx(0.10)
    assert price_cost("nosuch:model", 10, 10) is None


def test_suite_id_changes_with_judge_and_weights():
    suite = load_suite()
    assert suite.weights["kind"] == 3
    a = suite_id("d1", None, suite)
    assert a == suite_id("d1", None, suite)
    assert a != suite_id("d1", "ollama:qwen3:4b", suite)
    assert a != suite_id("d2", None, suite)
    other = Suite(weights={**suite.weights, "kind": 4}, duplicates_k=suite.duplicates_k)
    assert a != suite_id("d1", None, other)
    # without a judge the judged fields' weights don't matter
    no_reply = Suite(weights={**suite.weights, "reply": 7}, duplicates_k=suite.duplicates_k)
    assert a == suite_id("d1", None, no_reply)


def test_composite_counts_unattempted_as_zero():
    weights = {"kind": 3, "area": 2, "severity": 1}
    assert composite({"kind": 1.0, "area": 1.0}, weights) == pytest.approx(1.0)
    # severity expected but not attempted: 0, so answering less doesn't help
    assert composite({"kind": 1.0, "area": 1.0, "severity": 0.0}, weights) == pytest.approx(5 / 6)
    assert composite({}, weights) is None

    fields = {
        "kind": {"expected": True, "attempted": True, "score": 1.0},
        "area": {"expected": True, "attempted": False, "score": None},
        "severity": {"expected": False, "attempted": True, "score": None},
    }
    assert results.composite_of(fields, weights) == pytest.approx(3 / 5)


# --- evaluators -------------------------------------------------------------------

REPO = {
    "name": "kubernetes/kubernetes",
    "labels": [{"name": "sig/node"}, {"name": "sig/network"}, {"name": "kind/bug"}],
    "latest_release": "v1.32.0",
    "as_of": "2025-03-01T00:00:00Z",
}


def response(**triage: Any) -> dict[str, Any]:
    return {
        "triage": {k: (v if isinstance(v, dict) and "value" in v else {"value": v}) for k, v in triage.items()},
        "usage": {},
        "strategy": "Test",
        "version": "1",
        "config_hash": "x",
    }


def run_evaluators(cases: list[tuple[str, dict, Expected, dict]]) -> dict[str, dict[str, Any]]:
    """Runs the code evaluators through pydantic-evals on fixed outputs; name -> evaluation values."""
    outputs = {}
    ds_cases = []
    for name, output, expected, metadata in cases:
        number = int(name.rsplit("-", 1)[1])
        inputs = {"issue": {"number": number, "title": name, "body": ""}, "repo": REPO}
        outputs[name] = output
        ds_cases.append(Case(name=name, inputs=inputs, expected_output=expected, metadata=metadata))
    ds = Dataset[dict, Any, dict](name="t", cases=ds_cases, evaluators=code_evaluators(5))

    async def task(inputs: dict) -> dict:
        return outputs[inputs["issue"]["title"]]

    report = ds.evaluate_sync(task, progress=False)
    out = {}
    for rc in report.cases:
        vals = {}
        for group in (rc.scores, rc.assertions, rc.labels):
            vals |= {k: v.value for k, v in group.items()}
        out[rc.name] = vals
    return out


def test_evaluators_score_and_skip():
    exp = Expected(
        kind=["bug", "flake"],
        area=["sig/node"],
        severity="important_soon",
        is_regression=True,
        security_sensitive=False,
        enough_info=None,
        duplicate_of=[50],
        close_outcome="fixed",
    )
    full = response(
        kind="flake",
        area="sig/network",
        severity="backlog",
        is_regression=True,
        security_sensitive=False,
        reproducibility={"value": {"enough_info": True}},
        duplicates=[7, 50],
    )
    partial = response(kind="bug")
    got = run_evaluators([("k8s-100", full, exp, {}), ("k8s-101", partial, exp, {})])

    a = got["k8s-100"]
    assert a["kind"] is True  # flake is one of the two kinds
    assert a["area"] is False
    assert a["severity"] is False and a["severity_distance"] == 2
    assert a["is_regression"] is True and a["security_sensitive"] is True
    assert "enough_info" not in a  # no expected answer: skipped, not failed
    assert a["duplicates"] == 1.0
    assert a["sanity"] is True
    assert a["coverage"] == 1.0

    b = got["k8s-101"]
    assert b["kind"] is True
    for skipped in ("area", "severity", "severity_distance", "is_regression", "duplicates", "enough_info"):
        assert skipped not in b  # not produced: skipped
    assert b["coverage"] == pytest.approx(1 / 6)


def test_sanity_checks():
    inputs = {"issue": {"number": 100}, "repo": REPO}
    spam = {"kind": {"value": "spam"}}
    assert sanity_problems(spam, {"close_outcome": "fixed"}, inputs, {}) == ["predicted spam, but the issue was fixed"]
    question = {"kind": {"value": "question"}}
    assert sanity_problems(question, {"close_outcome": "open"}, inputs, {"fixing_prs": [5]})
    assert sanity_problems({"kind": {"value": "bug"}}, {"close_outcome": "fixed"}, inputs, {"fixing_prs": [5]}) == []
    dupes = {"duplicates": {"value": [100, 120]}}
    assert len(sanity_problems(dupes, {}, inputs, {})) == 2
    bad = {"area": {"value": "sig/nope"}, "quality": {"value": 9}, "kind": {"value": "bug", "confidence": 1.5}}
    assert len(sanity_problems(bad, {}, inputs, {})) == 3


def test_recall_and_distance():
    assert recall_at_k([1, 2, 3, 4, 5, 6], [6], 5) == 0.0
    assert recall_at_k([6, 1], [6, 9], 5) == 0.5
    assert severity_distance("critical", "backlog") == 3


# --- statistics ---------------------------------------------------------------------


def test_wilson_interval():
    lo, hi = compare.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)
    assert compare.wilson(0, 0) == (0.0, 1.0)
    lo, hi = compare.wilson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(0.1611, abs=1e-4)


def test_mcnemar_exact():
    assert compare.mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert compare.mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert compare.mcnemar_exact(3, 3) == 1.0
    assert compare.mcnemar_exact(0, 0) == 1.0


def test_pareto_frontier_and_marginal_cost():
    front = compare.pareto(
        [("heuristic", 0.0, 0.40), ("single", 0.001, 0.60), ("bad", 0.002, 0.55), ("agent", 0.005, 0.70)]
    )
    assert front["frontier"] == ["heuristic", "single", "agent"]
    assert front["dominated"] == {"bad": "single"}
    step = front["steps"][1]
    assert step["extra_points"] == pytest.approx(10.0)
    assert step["cost_per_point"] == pytest.approx(0.0004)
    # equal points: neither dominates the other
    assert compare.pareto([("a", 0.0, 0.5), ("b", 0.0, 0.5)])["dominated"] == {}


def test_macro_f1_and_precision_recall():
    pairs = [("bug", ["bug"]), ("bug", ["feature"]), ("feature", ["feature"]), ("flake", ["bug", "flake"])]
    # bug: tp1 fp1 fn0 -> 2/3; feature: tp1 fn1 -> 2/3; flake: tp1 -> 1
    assert compare.macro_f1(pairs) == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)
    p, r, pp, ap = compare.precision_recall([(True, True), (True, False), (False, True), (False, False)])
    assert (p, r, pp, ap) == (0.5, 0.5, 2, 2)


def test_balanced_accuracy():
    # 1 of 2 positives, 3 of 4 negatives: (0.5 + 0.75) / 2, while plain accuracy is 4/6
    ba = compare.balanced_accuracy(
        [(True, True), (False, True), (False, False), (False, False), (False, False), (True, False)]
    )
    assert ba["value"] == pytest.approx(0.625)
    assert (ba["tp"], ba["pos"], ba["tn"], ba["neg"], ba["one_class"]) == (1, 2, 3, 4, None)
    # always answering false: plain accuracy 98%, balanced 50%
    always_false = [(False, True)] * 2 + [(False, False)] * 98
    assert compare.balanced_accuracy(always_false)["value"] == pytest.approx(0.5)
    # unattempted (None) is wrong for either class
    assert compare.balanced_accuracy([(None, True), (True, True), (None, False), (False, False)])["value"] == 0.5
    # one class only: the rate on that class, flagged
    only_neg = compare.balanced_accuracy([(False, False), (True, False), (False, False), (False, False)])
    assert only_neg["value"] == pytest.approx(0.75) and only_neg["one_class"] == "negative" and only_neg["tpr"] is None
    assert compare.balanced_accuracy([(True, True)])["one_class"] == "positive"
    assert compare.balanced_accuracy([]) is None


def bool_record(name: str, predicted: bool | None, truth: bool) -> dict[str, Any]:
    output = response(kind="bug", is_regression=predicted) if predicted is not None else response(kind="bug")
    scores = {"kind": True} | ({"is_regression": predicted == truth} if predicted is not None else {})
    return results.case_record(
        name=name,
        repeat=0,
        workflow_key=f"r:x:{name}:0",
        expected=Expected(kind=["bug"], is_regression=truth, close_outcome="open"),
        output=output,
        error=None,
        scores=scores,
        reasons={},
        metrics={"cost": 0.0},
        duration=0.0,
        judged=False,
        weights={"kind": 3, "is_regression": 1},
    )


def test_run_stats_scores_booleans_by_balanced_accuracy(tmp_path):
    # 1 positive (missed), 3 negatives (one unattempted): TPR 0, TNR 2/3
    cases = [
        bool_record("c1", False, True),
        bool_record("c2", False, False),
        bool_record("c3", False, False),
        bool_record("c4", None, False),
    ]
    meta = {"strategy": "S", "version": "1", "config_hash": "h", "weights": {"kind": 3, "is_regression": 1}}
    v2 = tmp_path / "v2.json"
    results.save(v2, meta | {"evaluators_version": "2"}, cases)
    st = compare.run_stats(results.load(v2))
    reg = st["fields"]["is_regression"]
    assert reg["accuracy"] == pytest.approx(2 / 3)  # plain, over attempted: still reported
    assert reg["balanced"]["value"] == pytest.approx(0.5)  # over attempted: (0 + 2/2) / 2
    assert reg["effective"] == pytest.approx(1 / 3)  # unattempted wrong: (0 + 2/3) / 2
    assert st["composite"] == pytest.approx((3 * 1.0 + 1 / 3) / 4)
    assert (reg["precision"], reg["recall"]) == (None, 0.0)
    cell = compare.field_cell(reg)
    assert cell.startswith("bal. 50.0%") and "TPR 0/1" in cell and "TNR 2/2" in cell
    header, rows = compare.csv_rows([st])
    assert rows[0][header.index("is_regression_balanced")] == pytest.approx(0.5)

    # files scored with EVALUATORS_VERSION 1 keep plain accuracy
    v1 = tmp_path / "v1.json"
    results.save(v1, meta | {"evaluators_version": "1"}, cases)
    st1 = compare.run_stats(results.load(v1))
    assert "balanced" not in st1["fields"]["is_regression"]
    assert st1["fields"]["is_regression"]["effective"] == pytest.approx(2 / 4)

    # only negatives: falls back to TNR and says so
    neg = tmp_path / "neg.json"
    results.save(neg, meta | {"evaluators_version": "2"}, cases[1:])
    cell = compare.field_cell(compare.run_stats(results.load(neg))["fields"]["is_regression"])
    assert "only negative cases" in cell and "TPR" not in cell


def test_reliability_and_ece():
    bins, ece = compare.reliability([(0.95, True), (0.95, False), (0.15, False), (0.15, False)])
    assert [b["n"] for b in bins] == [2, 2]
    assert ece == pytest.approx(0.5 * abs(0.15 - 0.0) + 0.5 * abs(0.95 - 0.5))


def test_judge_agreement_kappa():
    perfect = compare.agreement([(1, 1), (3, 3), (5, 5)])
    assert perfect["exact"] == 1.0 and perfect["kappa"] == pytest.approx(1.0)
    off = compare.agreement([(1, 2), (3, 4), (5, 4)])
    assert off["within_one"] == 1.0 and off["mae"] == 1.0


def record(name: str, repeat: int, kind: str | None, expected_kind: str, cost: float, **extra: Any) -> dict[str, Any]:
    output = response(kind=kind) if kind else response()
    scores = {"kind": kind == expected_kind} if kind else {}
    return results.case_record(
        name=name,
        repeat=repeat,
        workflow_key=f"r:{name}:{repeat}",
        expected=Expected(kind=[expected_kind], close_outcome="open"),
        output=output,
        error=None,
        scores=scores,
        reasons={},
        metrics={"cost": cost, "latency_ms": 10},
        duration=0.01,
        judged=False,
        weights={"kind": 3, "area": 2},
        **extra,
    )


def make_run(tmp: Path, name: str, cases: list[dict[str, Any]]) -> Path:
    path = tmp / f"{name}.json"
    results.save(
        path, {"strategy": name, "version": "1", "config_hash": name, "weights": {"kind": 3}, "repeats": 2}, cases
    )
    return path


def test_results_round_trip_stats_and_head_to_head(tmp_path):
    a = make_run(
        tmp_path,
        "A",
        [record("c1", 0, "bug", "bug", 0.0), record("c1", 1, "bug", "bug", 0.0), record("c2", 0, "bug", "feature", 0.0),
         record("c2", 1, None, "feature", 0.0)],
    )  # fmt: skip
    b = make_run(
        tmp_path,
        "B",
        [record("c1", 0, "bug", "bug", 0.01), record("c1", 1, "bug", "bug", 0.01),
         record("c2", 0, "feature", "feature", 0.01), record("c2", 1, "bug", "feature", 0.01)],
    )  # fmt: skip
    run_a, run_b = results.load(a), results.load(b)
    sa, sb = compare.run_stats(run_a), compare.run_stats(run_b)
    assert sa["fields"]["kind"]["attempted"] == 3 and sa["fields"]["kind"]["expected"] == 4
    assert sa["fields"]["kind"]["accuracy"] == pytest.approx(2 / 3)
    assert sa["composite"] == pytest.approx(2 / 4)  # the unattempted case counts as 0
    assert sb["composite"] == pytest.approx(3 / 4)
    assert sa["consistency"]["kind"] == 0.5  # c2's repeats disagree (bug vs nothing)
    assert sb["cost_mean"] == pytest.approx(0.01)

    h = compare.head_to_head(run_a, run_b)
    assert h["pairs"] == 4
    assert h["fields"]["kind"] == {"n": 4, "only_a": 0, "only_b": 1, "both": 2, "neither": 1, "p": 1.0}

    out = tmp_path / "cmp.md"
    compare.main([str(a), str(b), "--baseline", str(a), "--out", str(out)])
    text = out.read_text()
    assert "Head-to-head" in text and "Pareto" in text
    assert (tmp_path / "cmp.csv").exists() and (tmp_path / "cmp-pareto.svg").exists()


# --- configs --------------------------------------------------------------------------


def schema_for(path: Path) -> str:
    for prefix, schema in sorted(SCHEMA_BY_PREFIX.items(), key=lambda kv: -len(kv[0])):
        if path.stem.startswith(prefix):
            return schema
    raise AssertionError(f"no schema for {path.name}")


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_config_files_validate(path: Path):
    config = yaml.safe_load(path.read_text()) or {}
    schema = json.loads((SCHEMA_DIR / f"{schema_for(path)}.schema.json").read_text())
    jsonschema.validate(config, strict_schema(schema))


def test_every_strategy_has_an_example_config():
    stems = [p.stem for p in CONFIGS.glob("*.yaml")]
    for prefix in SCHEMA_BY_PREFIX:
        assert any(s.startswith(prefix) for s in stems), prefix


def test_strict_schema_rejects_typos():
    schema = json.loads((SCHEMA_DIR / "single-shot-config.schema.json").read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"model": "ollama:x", "temprature": 0.2}, strict_schema(schema))


def test_triage_eval_rejects_a_config_typo(tmp_path):
    from triager_evals import run

    bad = tmp_path / "workflow-typo.yaml"
    bad.write_text("decision_model: ollama:tev1:0.8b-q8_0\nescalaton:\n  model: ollama:qwen3:1.7b\n")
    assert any("escalaton" in e for e in config_errors("Workflow", yaml.safe_load(bad.read_text())))
    with pytest.raises(SystemExit, match="escalaton"):
        run.load_config(bad, "Workflow")
    assert run.load_config(bad) is not None  # without a strategy: only parsed
    assert config_errors("Workflow", yaml.safe_load((CONFIGS / "workflow-tev1-none.yaml").read_text())) == []

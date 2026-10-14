"""Compares eval runs from their results files.

    compare-runs evals/results/<run>/<file>.json [more files or directories ...] \\
        [--baseline FILE] [--format md|json] [--out FILE] [--csv FILE] [--svg FILE]
    compare-runs judge-agreement [--gold evals/gold/quality.yaml] FILES...

Reports, per run: coverage and accuracy per field (Wilson 95% intervals),
kind macro-F1, the composite score, cost, tokens, calls, latency and, with
repeats, how consistent the repeats are. Then a paired head-to-head of every
run against the baseline on the same (case, repeat) pairs with an exact
McNemar test, the Pareto frontier of cost per issue against composite, and
calibration (reliability bins and ECE) for every field with a confidence.

A field the strategy didn't attempt counts as wrong in the composite and in
the head-to-head (so answering less never looks better), but not in the
per-field accuracy, which is over attempted cases and always shown with its
coverage.

`is_regression` and `security_sensitive` (evaluators.BALANCED_FIELDS) are
scored at run level by balanced accuracy, the mean of the true-positive and
true-negative rates, because their positives are rare and plain accuracy
rewards always answering false. With only one class among the cases it falls
back to the rate on that class, flagged in the report. Results files written
with EVALUATORS_VERSION 1 keep plain accuracy, so an old file reads as it was
scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from . import results
from .evaluators import BALANCED_FIELDS, EVALUATORS_VERSION, FIELDS
from .suite import EVALS_DIR, RESULTS_DIR, composite

GOLD_FILE = EVALS_DIR / "gold" / "quality.yaml"
BOOL_FIELDS = ("is_regression", "security_sensitive", "enough_info")
CONSISTENCY_FIELDS = ("kind", "area")


# --- statistics --------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (95% by default)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value against p = 0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Exact McNemar test: a binomial test on the discordant pairs."""
    return binom_two_sided(only_a, only_a + only_b)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in 0..100."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def macro_f1(pairs: Iterable[tuple[str, Sequence[str]]]) -> float | None:
    """Macro-F1 over (predicted, expected labels) pairs.

    Some issues have two kinds; a prediction matching either counts as right,
    and a wrong one is charged to the first expected kind.
    """
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    any_pair = False
    for pred, expected in pairs:
        any_pair = True
        truth = pred if pred in expected else expected[0]
        if pred == truth:
            tp[truth] += 1
        else:
            fp[pred] += 1
            fn[truth] += 1
    if not any_pair:
        return None
    classes = set(tp) | set(fp) | set(fn)
    f1s = []
    for c in classes:
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append(2 * tp[c] / denom if denom else 0.0)
    return sum(f1s) / len(f1s)


def precision_recall(pairs: Iterable[tuple[bool, bool]]) -> tuple[float | None, float | None, int, int]:
    """(precision, recall, predicted positives, actual positives) for the True class."""
    tp = fp = fn = 0
    for pred, truth in pairs:
        tp += pred and truth
        fp += pred and not truth
        fn += truth and not pred
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return precision, recall, tp + fp, tp + fn


def balanced_accuracy(pairs: Iterable[tuple[bool | None, bool]]) -> dict[str, Any] | None:
    """Balanced accuracy over (predicted, truth) pairs; a predicted None counts as wrong.

    The mean of the true-positive rate and the true-negative rate. When only
    one class is present, the rate on that class (`one_class` names it), since
    the other rate is undefined. None without any pairs.
    """
    tp = tn = pos = neg = 0
    for pred, truth in pairs:
        if truth:
            pos += 1
            tp += pred is True
        else:
            neg += 1
            tn += pred is False
    if not pos and not neg:
        return None
    tpr = tp / pos if pos else None
    tnr = tn / neg if neg else None
    rates = [r for r in (tpr, tnr) if r is not None]
    return {
        "value": sum(rates) / len(rates),
        "tpr": tpr,
        "tnr": tnr,
        "tp": tp,
        "tn": tn,
        "pos": pos,
        "neg": neg,
        "one_class": None if pos and neg else ("negative" if neg else "positive"),
    }


def uses_balanced(metadata: Mapping[str, Any]) -> bool:
    """Whether a run's boolean fields are scored by balanced accuracy (EVALUATORS_VERSION >= 2)."""
    version = str(metadata.get("evaluators_version") or EVALUATORS_VERSION)
    try:
        return int(version) >= 2
    except ValueError:
        return True


def reliability(pairs: Sequence[tuple[float, bool]], bins: int = 10) -> tuple[list[dict[str, Any]], float | None]:
    """Reliability bins (equal width over 0..1) and the expected calibration error."""
    if not pairs:
        return [], None
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for conf, correct in pairs:
        buckets[min(bins - 1, max(0, int(conf * bins)))].append((conf, correct))
    rows, ece = [], 0.0
    for i, b in enumerate(buckets):
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(ok for _, ok in b) / len(b)
        ece += len(b) / len(pairs) * abs(acc - conf)
        rows.append({"bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}", "n": len(b), "confidence": conf, "accuracy": acc})
    return rows, ece


def pareto(points: Sequence[tuple[str, float, float]]) -> dict[str, Any]:
    """Pareto frontier over (name, cost, score): lower cost and higher score are better.

    Returns the frontier (sorted by cost), the dominated names with who
    dominates them, and the marginal cost per composite point between
    adjacent frontier points.
    """
    dominated: dict[str, str] = {}
    for name, cost, score in points:
        for other, c2, s2 in points:
            if other != name and c2 <= cost and s2 >= score and (c2 < cost or s2 > score):
                dominated[name] = other
                break
    frontier = sorted((p for p in points if p[0] not in dominated), key=lambda p: (p[1], -p[2]))
    steps = []
    for (a, ca, sa), (b, cb, sb) in zip(frontier, frontier[1:], strict=False):
        points_gained = (sb - sa) * 100
        steps.append(
            {
                "from": a,
                "to": b,
                "extra_cost": cb - ca,
                "extra_points": points_gained,
                "cost_per_point": (cb - ca) / points_gained if points_gained > 0 else None,
            }
        )
    return {"frontier": [p[0] for p in frontier], "dominated": dominated, "steps": steps}


# --- per-run numbers -----------------------------------------------------------


def correct(field: str, f: Mapping[str, Any], case: Mapping[str, Any]) -> bool:
    """Whether a field counts as right for the paired test and calibration.

    Unattempted is wrong. Duplicates: the target is in the top k. Quality:
    exact agreement with the judge. Reply: the judge's pass.
    """
    if not f["attempted"] or f["score"] is None:
        return False
    if field == "quality":
        return bool(case["scores"].get("quality_agree"))
    if field == "reply":
        return bool(case["scores"].get("reply_pass", f["score"] >= 0.5))
    if field == "duplicates":
        return f["score"] > 0
    return f["score"] >= 1.0


def run_label(run: Mapping[str, Any]) -> str:
    m = run["metadata"]
    return f"{m.get('strategy', '?')}@{m.get('version', '?')} {m.get('config_hash', '')} {m.get('label') or ''}".strip()


def kind_confusion(cases: Iterable[Mapping[str, Any]]) -> tuple[list[str], list[list[int]]]:
    """Rows: expected kind (first label, or the predicted one when it matches any); columns: predicted."""
    pairs = []
    for c in cases:
        f = c["fields"]["kind"]
        if f["expected"] and f["attempted"]:
            expected = c["expected"]["kind"]
            pairs.append((expected[0] if f["value"] not in expected else f["value"], f["value"]))
    labels = sorted({x for p in pairs for x in p})
    idx = {k: i for i, k in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for truth, pred in pairs:
        matrix[idx[truth]][idx[pred]] += 1
    return labels, matrix


def mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def run_stats(run: Mapping[str, Any]) -> dict[str, Any]:
    cases = run["cases"]
    weights = run["metadata"].get("weights") or {}
    balanced = uses_balanced(run["metadata"])
    ok = [c for c in cases if c["error"] is None]
    per_field: dict[str, dict[str, Any]] = {}
    effective: dict[str, float] = {}
    for spec in FIELDS:
        exp = [c for c in cases if c["fields"][spec.name]["expected"]]
        att = [c for c in exp if c["fields"][spec.name]["attempted"] and c["fields"][spec.name]["score"] is not None]
        scores = [c["fields"][spec.name]["score"] for c in att]
        k = sum(correct(spec.name, c["fields"][spec.name], c) for c in att)
        st: dict[str, Any] = {
            "expected": len(exp),
            "attempted": len(att),
            "coverage": len(att) / len(exp) if exp else None,
            "correct": k,
            "accuracy": k / len(att) if att else None,
            "ci": wilson(k, len(att)) if att else None,
            "mean_score": mean(scores),
            # unattempted counts as 0: what the composite uses
            "effective": sum(scores) / len(exp) if exp else None,
        }
        if exp:
            effective[spec.name] = st["effective"]
        if spec.name == "kind":
            st["macro_f1"] = macro_f1((c["fields"]["kind"]["value"], c["expected"]["kind"]) for c in att)
        if spec.name == "severity":
            st["mean_distance"] = mean(
                [c["scores"]["severity_distance"] for c in att if "severity_distance" in c["scores"]]
            )
        if spec.name in BOOL_FIELDS:
            key = spec.name
            p, r, pp, ap = precision_recall((bool(c["fields"][key]["value"]), bool(c["expected"][key])) for c in att)
            st |= {"precision": p, "recall": r, "predicted_pos": pp, "actual_pos": ap}
        if balanced and spec.name in BALANCED_FIELDS:
            key = spec.name
            attempted_ids = {id(c) for c in att}
            # over attempted cases (what the summary shows) and with unattempted
            # cases counted as wrong (what the composite uses)
            st["balanced"] = balanced_accuracy((c["fields"][key]["value"], bool(c["expected"][key])) for c in att)
            st["balanced_effective"] = balanced_accuracy(
                (c["fields"][key]["value"] if id(c) in attempted_ids else None, bool(c["expected"][key])) for c in exp
            )
            if st["balanced_effective"] is not None:
                st["effective"] = st["balanced_effective"]["value"]
                effective[spec.name] = st["effective"]
        if spec.name == "quality":
            st["mean_abs_error"] = mean(
                [c["scores"]["quality_abs_error"] for c in att if "quality_abs_error" in c["scores"]]
            )
        per_field[spec.name] = st

    def metric(name: str) -> list[float]:
        return [c["metrics"].get(name, 0) for c in ok]

    latency = metric("latency_ms")
    durations = [c["duration"] * 1000 for c in ok if c.get("duration") is not None]
    sanity_fail = [c for c in ok if c["scores"].get("sanity") is False]
    judge = [c["judge_usage"] for c in cases if c.get("judge_usage")]
    stats = {
        "label": run_label(run),
        "path": run.get("path"),
        "suite_id": run["metadata"].get("suite_id"),
        "cases": len(cases),
        "errors": len(cases) - len(ok),
        "repeats": run["metadata"].get("repeats", 1),
        "fields": per_field,
        "composite": composite(effective, weights),
        "kind_macro_f1": per_field["kind"].get("macro_f1"),
        "cost_mean": mean(metric("cost")),
        "cost_total": sum(metric("cost")),
        "input_tokens_mean": mean(metric("input_tokens")),
        "output_tokens_mean": mean(metric("output_tokens")),
        "llm_calls_mean": mean(metric("llm_calls")),
        "decision_calls_mean": mean(metric("decision_calls")),
        "tool_calls_mean": mean(metric("tool_calls")),
        "latency_mean": mean(latency),
        "latency_p50": percentile(latency, 50),
        "latency_p95": percentile(latency, 95),
        "wall_ms_mean": mean(durations),
        "sanity_failures": len(sanity_fail),
        "judge_cost": sum(u.get("cost_usd", 0.0) for u in judge) if judge else None,
        "judge_tokens": sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in judge) if judge else None,
        "consistency": consistency(cases),
        "calibration": calibration(cases),
    }
    return stats


def consistency(cases: Iterable[Mapping[str, Any]]) -> dict[str, float | None] | None:
    """Per field: the share of cases where every repeat gave the same answer (cases with 2+ repeats)."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for c in cases:
        groups[c["name"]].append(c)
    multi = [g for g in groups.values() if len(g) > 1]
    if not multi:
        return None
    out = {}
    for field in CONSISTENCY_FIELDS:
        same = [len({json.dumps(c["fields"][field]["value"]) for c in g}) == 1 for g in multi]
        out[field] = sum(same) / len(same)
    return out


def calibration(cases: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    cases = list(cases)
    for spec in FIELDS:
        pairs = [
            (float(c["fields"][spec.name]["confidence"]), correct(spec.name, c["fields"][spec.name], c))
            for c in cases
            if c["fields"][spec.name]["expected"]
            and c["fields"][spec.name]["attempted"]
            and c["fields"][spec.name]["score"] is not None
            and c["fields"][spec.name]["confidence"] is not None
        ]
        if pairs:
            bins, ece = reliability(pairs)
            out[spec.name] = {"n": len(pairs), "ece": ece, "bins": bins}
    return out


def head_to_head(baseline: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Paired counts on the (case, repeat) pairs both runs have."""

    def index(run: Mapping[str, Any]) -> dict[tuple[str, Any], Mapping[str, Any]]:
        return {(c["name"], c["repeat"]): c for c in run["cases"]}

    a_idx, b_idx = index(baseline), index(other)
    shared = sorted(set(a_idx) & set(b_idx), key=str)
    fields = {}
    for spec in FIELDS:
        counts = Counter()
        for key in shared:
            ca, cb = a_idx[key], b_idx[key]
            fa, fb = ca["fields"][spec.name], cb["fields"][spec.name]
            if not (fa["expected"] and fb["expected"]):
                continue
            ra, rb = correct(spec.name, fa, ca), correct(spec.name, fb, cb)
            counts[("both" if ra and rb else "only_a" if ra else "only_b" if rb else "neither")] += 1
        n = sum(counts.values())
        if n:
            fields[spec.name] = {
                "n": n,
                "only_a": counts["only_a"],
                "only_b": counts["only_b"],
                "both": counts["both"],
                "neither": counts["neither"],
                "p": mcnemar_exact(counts["only_a"], counts["only_b"]),
            }
    a_better = b_better = ties = 0
    for key in shared:
        x, y = a_idx[key].get("composite"), b_idx[key].get("composite")
        if x is None or y is None:
            continue
        if abs(x - y) < 1e-12:
            ties += 1
        elif x > y:
            a_better += 1
        else:
            b_better += 1
    return {
        "pairs": len(shared),
        "fields": fields,
        "composite": {
            "a_better": a_better,
            "b_better": b_better,
            "ties": ties,
            "p": binom_two_sided(a_better, a_better + b_better),
        },
    }


# --- rendering -----------------------------------------------------------------


def pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def num(x: float | None, fmt: str = ".1f") -> str:
    return "–" if x is None else format(x, fmt)


def usd(x: float | None) -> str:
    if x is None:
        return "–"
    return f"${x:.6f}" if x < 0.01 else f"${x:.4f}"


def pval(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def rate_cell(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{100 * lo:.0f}–{100 * hi:.0f}]"


def field_cell(st: Mapping[str, Any]) -> str:
    if not st["expected"]:
        return "–"
    if not st["attempted"]:
        return f"not attempted (0/{st['expected']})"
    ba = st.get("balanced")
    if ba is not None:
        rates = []
        if ba["pos"]:
            rates.append(f"TPR {rate_cell(ba['tp'], ba['pos'])}")
        if ba["neg"]:
            rates.append(f"TNR {rate_cell(ba['tn'], ba['neg'])}")
        alone = "TNR" if ba["one_class"] == "negative" else "TPR"
        flag = f" ⚑ only {ba['one_class']} cases: {alone} alone" if ba["one_class"] else ""
        return f"bal. {pct(ba['value'])} ({', '.join(rates)}){flag} · {st['attempted']}/{st['expected']}"
    lo, hi = st["ci"]
    return f"{pct(st['accuracy'])} [{100 * lo:.0f}–{100 * hi:.0f}] · {st['attempted']}/{st['expected']}"


def overview_rows(stats: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    rows = []
    for i, s in enumerate(stats, 1):
        rows.append(
            [
                f"R{i}",
                s["label"],
                f"{s['cases'] - s['errors']}/{s['cases']}",
                num(s["composite"] and 100 * s["composite"]),
                num(s["kind_macro_f1"] and 100 * s["kind_macro_f1"]),
                usd(s["cost_mean"]),
                f"{num(s['input_tokens_mean'], '.0f')}/{num(s['output_tokens_mean'], '.0f')}",
                "/".join(num(s[f"{k}_calls_mean"], ".2f") for k in ("llm", "decision", "tool")),
                f"{num(s['latency_mean'], '.0f')}/{num(s['latency_p50'], '.0f')}/{num(s['latency_p95'], '.0f')}",
                s["sanity_failures"],
                usd(s["judge_cost"]) if s["judge_cost"] is not None else "–",
            ]
        )
    return rows


OVERVIEW_HEADER = [
    "run",
    "strategy@version config label",
    "ok/cases",
    "composite",
    "kind macro-F1",
    "cost/issue",
    "tokens in/out",
    "calls llm/dec/tool",
    "latency ms mean/p50/p95",
    "sanity fails",
    "judge cost (total)",
]


def extras(s: Mapping[str, Any]) -> list[str]:
    f = s["fields"]
    out = []
    if f["severity"].get("mean_distance") is not None:
        out.append(f"severity mean distance {f['severity']['mean_distance']:.2f}")
    for name in BOOL_FIELDS:
        st = f[name]
        if st["attempted"]:
            plain = f", plain accuracy {pct(st['accuracy'])}" if st.get("balanced") is not None else ""
            out.append(
                f"{name} P {pct(st['precision'], 0)} / R {pct(st['recall'], 0)} "
                f"(predicted {st['predicted_pos']}, actual {st['actual_pos']}){plain}"
            )
    if f["quality"].get("mean_abs_error") is not None:
        out.append(f"quality |strategy − judge| {f['quality']['mean_abs_error']:.2f}")
    if s["consistency"]:
        out.append("repeat consistency " + ", ".join(f"{k} {pct(v)}" for k, v in s["consistency"].items()))
    return out


def render_markdown(
    stats: list[dict[str, Any]],
    h2h: list[tuple[int, dict[str, Any]]],
    front: dict[str, Any],
    base_i: int,
    notes: list[str],
) -> str:
    out = ["# Run comparison", ""]
    out += [f"- {n}" for n in notes]
    if notes:
        out.append("")
    out += ["## Runs", "", md_table(OVERVIEW_HEADER, overview_rows(stats)), ""]
    out.append(
        "composite and macro-F1 in points (0–100); cost is the strategy's own (shadow-priced) cost, "
        "never the judge's; latency is the summed model and tool time the service reports."
    )
    out += ["", "## Per field", "", "accuracy over attempted cases [Wilson 95%] · attempted/expected", ""]
    if any(s["fields"][f].get("balanced") is not None for s in stats for f in BALANCED_FIELDS):
        out += [
            f"{', '.join(BALANCED_FIELDS)}: balanced accuracy (mean of TPR and TNR, each with its Wilson 95% "
            "interval), which is also their score in the composite (there with unattempted cases as wrong); "
            "⚑ marks a run whose cases have only one class, scored by that class's rate alone.",
            "",
        ]
    header = ["field", "weight"] + [f"R{i}" for i in range(1, len(stats) + 1)]
    rows = []
    for spec in FIELDS:
        if not any(s["fields"][spec.name]["expected"] for s in stats):
            continue
        weight = next((w[spec.name] for st in stats if spec.name in (w := st.get("weights") or {})), "")
        weight = format(weight, "g") if isinstance(weight, float) else weight
        rows.append([spec.name, weight, *(field_cell(s["fields"][spec.name]) for s in stats)])
    out += [md_table(header, rows), ""]
    for i, s in enumerate(stats, 1):
        ex = extras(s)
        if ex:
            out.append(f"- R{i}: " + "; ".join(ex))
    out.append("")
    if h2h:
        out += ["## Head-to-head", ""]
        for i, h in h2h:
            out.append(
                f"R{i} vs baseline R{base_i} on {h['pairs']} shared (case, repeat) pairs. "
                "only A = only the baseline right; unattempted counts as wrong; p = exact McNemar."
            )
            out.append("")
            rows = [
                [f, v["n"], v["only_a"], v["only_b"], v["both"], v["neither"], pval(v["p"])]
                for f, v in h["fields"].items()
            ]
            c = h["composite"]
            rows.append(
                [
                    "composite (per case)",
                    c["a_better"] + c["b_better"] + c["ties"],
                    c["a_better"],
                    c["b_better"],
                    "–",
                    f"{c['ties']} ties",
                    pval(c["p"]),
                ]
            )
            out += [md_table(["field", "n", f"only A (R{base_i})", f"only B (R{i})", "both", "neither", "p"], rows), ""]
    out += ["## Pareto frontier (cost per issue vs composite)", ""]
    rows = []
    for i, s in enumerate(stats, 1):
        tag = f"R{i}"
        status = "frontier" if tag in front["frontier"] else f"dominated by {front['dominated'].get(tag, '?')}"
        rows.append([tag, s["label"], usd(s["cost_mean"]), num(s["composite"] and 100 * s["composite"]), status])
    out += [md_table(["run", "label", "cost/issue", "composite", "status"], rows), ""]
    for st in front["steps"]:
        cpp = st["cost_per_point"]
        out.append(
            f"- {st['from']} → {st['to']}: +{usd(st['extra_cost'])} per issue for "
            f"+{st['extra_points']:.1f} points" + (f" ({usd(cpp)} per point)" if cpp is not None else "")
        )
    if not front["steps"]:
        out.append("- one point on the frontier, no trade-off to price")
    out += ["", "## Calibration", ""]
    any_cal = False
    for i, s in enumerate(stats, 1):
        for field, cal in s["calibration"].items():
            any_cal = True
            bins = ", ".join(
                f"{b['bin']}: {pct(b['accuracy'], 0)} of {b['n']} (conf {b['confidence']:.2f})" for b in cal["bins"]
            )
            out.append(f"- R{i} {field}: ECE {cal['ece']:.3f} over {cal['n']} — {bins}")
    if not any_cal:
        out.append("- no field reported a confidence")
    return "\n".join(out) + "\n"


def csv_rows(stats: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    header = [
        "run", "label", "path", "suite_id", "cases", "errors", "composite", "kind_macro_f1", "cost_mean",
        "input_tokens_mean", "output_tokens_mean", "llm_calls_mean", "decision_calls_mean", "tool_calls_mean",
        "latency_mean", "latency_p50", "latency_p95", "sanity_failures", "judge_cost",
    ]  # fmt: skip
    for spec in FIELDS:
        header += [
            f"{spec.name}_accuracy",
            f"{spec.name}_ci_lo",
            f"{spec.name}_ci_hi",
            f"{spec.name}_attempted",
            f"{spec.name}_expected",
        ]
    for name in BALANCED_FIELDS:
        header += [f"{name}_balanced", f"{name}_tpr", f"{name}_tnr", f"{name}_precision", f"{name}_recall"]
    rows = []
    for i, s in enumerate(stats, 1):
        row = [f"R{i}", *(s[k] for k in header[1:19])]
        for spec in FIELDS:
            st = s["fields"][spec.name]
            ci = st["ci"] or (None, None)
            row += [st["accuracy"], ci[0], ci[1], st["attempted"], st["expected"]]
        for name in BALANCED_FIELDS:
            st = s["fields"][name]
            ba = st.get("balanced") or {}
            row += [ba.get("value"), ba.get("tpr"), ba.get("tnr"), st.get("precision"), st.get("recall")]
        rows.append(row)
    return header, rows


def pareto_svg(stats: Sequence[Mapping[str, Any]], front: Mapping[str, Any]) -> str:
    """A small dependency-free scatter: cost per issue (x) against composite (y), frontier joined."""
    pts = [(f"R{i}", s["cost_mean"] or 0.0, (s["composite"] or 0.0) * 100) for i, s in enumerate(stats, 1)]
    w, h, m = 640, 400, 56
    xmax = max(p[1] for p in pts) or 1e-6
    ymin = min(p[2] for p in pts)
    ymax = max(p[2] for p in pts)
    ylo, yhi = max(0.0, ymin - 5), min(100.0, ymax + 5) if ymax + 5 > ymin else ymin + 10

    def sx(x: float) -> float:
        return m + (w - 2 * m) * x / (xmax * 1.1)

    def sy(y: float) -> float:
        return h - m - (h - 2 * m) * (y - ylo) / ((yhi - ylo) or 1)

    by_tag = {p[0]: p for p in pts}
    line = " ".join(f"{sx(by_tag[t][1]):.1f},{sy(by_tag[t][2]):.1f}" for t in front["frontier"])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="12">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<line x1="{m}" y1="{h - m}" x2="{w - m}" y2="{h - m}" stroke="#555"/>',
        f'<line x1="{m}" y1="{m}" x2="{m}" y2="{h - m}" stroke="#555"/>',
        f'<text x="{w / 2}" y="{h - 16}" text-anchor="middle">cost per issue (USD), max {usd(xmax)}</text>',
        f'<text x="16" y="{h / 2}" transform="rotate(-90 16 {h / 2})" text-anchor="middle">'
        f"composite ({ylo:.0f}–{yhi:.0f})</text>",
        f'<polyline points="{line}" fill="none" stroke="#2a6fdb" stroke-width="2"/>',
    ]
    for tag, x, y in pts:
        on = tag in front["frontier"]
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{"#2a6fdb" if on else "#999"}"/>')
        parts.append(f'<text x="{sx(x) + 8:.1f}" y="{sy(y) - 8:.1f}">{tag} {y:.1f}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def short_summary(run: Mapping[str, Any]) -> str:
    """A few lines for the end of a triage-eval run."""
    s = run_stats(run)
    lines = [
        f"{s['label']}: {s['cases'] - s['errors']}/{s['cases']} ok, "
        f"composite {num(s['composite'] and 100 * s['composite'])}, "
        f"kind macro-F1 {num(s['kind_macro_f1'] and 100 * s['kind_macro_f1'])}, cost/issue {usd(s['cost_mean'])}, "
        f"latency p50 {num(s['latency_p50'], '.0f')} ms, sanity failures {s['sanity_failures']}"
        + (f", judge cost {usd(s['judge_cost'])}" if s["judge_cost"] is not None else ""),
    ]
    for spec in FIELDS:
        st = s["fields"][spec.name]
        if st["expected"]:
            pr = ""
            if spec.name in BALANCED_FIELDS and st["attempted"]:
                pr = f" · P {pct(st['precision'], 0)} / R {pct(st['recall'], 0)}"
            lines.append(f"  {spec.name:<19} {field_cell(st)}{pr}")
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------------


def compare(
    paths: list[Path], baseline: Path | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[str]]:
    runs = [results.load(p) for p in paths]
    if not runs:
        raise SystemExit("no results files given")
    base_i = 0
    if baseline is not None:
        bl = results.load(baseline)
        matches = [i for i, r in enumerate(runs) if Path(r["path"]).resolve() == Path(bl["path"]).resolve()]
        if matches:
            base_i = matches[0]
        else:
            runs.insert(0, bl)
    notes = []
    suites = {r["metadata"].get("suite_id") for r in runs}
    if len(suites) > 1:
        notes.append(
            f"**different suites** ({', '.join(sorted(map(str, suites)))}): dataset, evaluators, rubric, judge "
            "or weights differ, so composites are not comparable; paired counts on shared cases still are."
        )
    for i, r in enumerate(runs, 1):
        m = r["metadata"]
        notes.append(
            f"R{i}: `{r['path']}` — run {m.get('run_id')}, dataset {Path(m.get('dataset', '')).name}, "
            f"judge {m.get('judge_model') or 'off'}, repeats {m.get('repeats')}, git {str(m.get('git_sha'))[:10]}"
            + (" (dirty)" if m.get("git_dirty") else "")
        )
    return runs, [run_stats(r) for r in runs], base_i, notes


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "judge-agreement":
        return judge_agreement_main(argv[1:])
    p = argparse.ArgumentParser(
        prog="compare-runs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("paths", nargs="+", type=Path, help="results files or directories of them")
    p.add_argument("--baseline", type=Path, help="the run the others are compared with (default: the first)")
    p.add_argument("--format", choices=["md", "json"], default="md")
    p.add_argument("--out", type=Path, help="write the report here instead of stdout")
    p.add_argument(
        "--csv", type=Path, help="per-run table as CSV (default: next to --out, else evals/results/compare.csv)"
    )
    p.add_argument(
        "--svg", type=Path, help="Pareto chart (default: next to --out, else evals/results/compare-pareto.svg)"
    )
    args = p.parse_args(argv)

    report = build_report(results.expand_paths(args.paths), args.baseline)
    stats, h2h, front, base_i = report["stats"], report["h2h"], report["front"], report["base_i"]
    if args.format == "json":
        text = json.dumps(
            {
                "runs": stats,
                "baseline": f"R{base_i + 1}",
                "head_to_head": {f"R{i}": h for i, h in h2h},
                "pareto": front,
            },
            indent=1,
            default=str,
        )
    else:
        text = report["markdown"]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    stem = args.out.with_suffix("") if args.out else RESULTS_DIR / "compare"
    csv_path = args.csv or Path(f"{stem}.csv")
    svg_path = args.svg or Path(f"{stem}-pareto.svg")
    write_csv_svg(stats, front, csv_path, svg_path)
    print(f"\nwrote {csv_path} and {svg_path}", file=sys.stderr)


def build_report(paths: list[Path], baseline: Path | None) -> dict[str, Any]:
    """Everything compare-runs reports, as data plus the Markdown text (used by run-experiments too)."""
    runs, stats, base_i, notes = compare(paths, baseline)
    for s, r in zip(stats, runs, strict=True):
        s["weights"] = r["metadata"].get("weights")
    h2h = [(i + 1, head_to_head(runs[base_i], r)) for i, r in enumerate(runs) if i != base_i]
    front = pareto([(f"R{i}", s["cost_mean"] or 0.0, s["composite"] or 0.0) for i, s in enumerate(stats, 1)])
    return {
        "runs": runs,
        "stats": stats,
        "h2h": h2h,
        "front": front,
        "base_i": base_i,
        "notes": notes,
        "markdown": render_markdown(stats, h2h, front, base_i + 1, notes),
    }


def write_csv_svg(stats: Sequence[Mapping[str, Any]], front: Mapping[str, Any], csv_path: Path, svg_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header, rows = csv_rows(stats)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(pareto_svg(stats, front), encoding="utf-8")


# --- judge vs hand labels ------------------------------------------------------


def load_gold(path: Path = GOLD_FILE) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    labels = data.get("labels") or {}
    out = {}
    for name, entry in labels.items():
        if isinstance(entry, int):
            entry = {"score": entry}
        out[str(name)] = dict(entry)
    return out


def agreement(pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Exact and within-one agreement, mean absolute error and quadratic-weighted kappa."""
    n = len(pairs)
    if not n:
        return {"n": 0}
    exact = sum(a == b for a, b in pairs)
    within = sum(abs(a - b) <= 1 for a, b in pairs)
    mae = sum(abs(a - b) for a, b in pairs) / n
    levels = range(1, 6)
    obs = Counter(pairs)
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    num_ = sum((i - j) ** 2 * obs[(i, j)] for i in levels for j in levels)
    den = sum((i - j) ** 2 * ca[i] * cb[j] / n for i in levels for j in levels)
    kappa = 1 - num_ / den if den else None
    return {
        "n": n,
        "exact": exact / n,
        "exact_ci": wilson(exact, n),
        "within_one": within / n,
        "mae": mae,
        "kappa": kappa,
    }


def judge_agreement(runs: Sequence[Mapping[str, Any]], gold: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Judge vs gold and strategy vs gold on the quality scores, over every case with a gold label."""
    judge_pairs, strategy_pairs = [], []
    seen = set()
    for run in runs:
        for c in run["cases"]:
            g = gold.get(c["name"])
            if g is None or c["repeat"] not in (0, None):
                continue
            j = c["scores"].get("quality_judge")
            if j is not None and (c["name"], run["metadata"].get("judge_model")) not in seen:
                seen.add((c["name"], run["metadata"].get("judge_model")))
                judge_pairs.append((int(g["score"]), int(j)))
            v = c["fields"]["quality"]["value"]
            if v is not None:
                strategy_pairs.append((int(g["score"]), int(v)))
    return {"gold_labels": len(gold), "judge": agreement(judge_pairs), "strategy": agreement(strategy_pairs)}


def judge_agreement_main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(
        prog="compare-runs judge-agreement", description="Judge vs hand-labeled quality scores."
    )
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--gold", type=Path, default=GOLD_FILE)
    args = p.parse_args(argv)
    gold = load_gold(args.gold)
    if not gold:
        print(f"{args.gold} has no labels yet; add `case-name: {{score: N, labeled_by: you}}` rows under `labels:`.")
        return
    runs = [results.load(x) for x in results.expand_paths(args.paths)]
    res = judge_agreement(runs, gold)
    for who in ("judge", "strategy"):
        a = res[who]
        if not a["n"]:
            print(f"{who}: no cases overlap with the gold labels")
            continue
        lo, hi = a["exact_ci"]
        print(
            f"{who} vs gold on {a['n']} cases: exact {pct(a['exact'])} [{100 * lo:.0f}–{100 * hi:.0f}], "
            f"within one {pct(a['within_one'])}, MAE {a['mae']:.2f}, weighted kappa {num(a['kappa'], '.2f')}"
        )

"""The local results files: one per (run, strategy, config).

`evals/results/<run_id>/<Strategy>-<config_hash>.json`:

```
{
  "format": 1,
  "metadata": {experiment, run_id, strategy, version, config, config_hash, git_sha, git_dirty,
               dataset, dataset_hash, suite_id, evaluators_version, rubric_version, judge_model,
               weights, repeats, limit, label, started_at, finished_at, judge_usage, ...},
  "cases": [
    {
      "name": "kubernetes-107415",          # the dataset case (inputs are not copied)
      "repeat": 0,                          # numbered per case, in start order
      "workflow_key": "...",
      "expected": {...},                    # the case's expected answers
      "output": {...} | null,               # the full TriageResponse
      "error": null | "...",                # task failure (HTTP error, schema mismatch)
      "scores": {"kind": true, "severity_distance": 1, ...},   # every evaluation, by name
      "reasons": {"sanity": "...", ...},
      "fields": {"kind": {"expected": true, "attempted": true, "score": 1.0,
                          "value": "bug", "confidence": 0.7}, ...},
      "composite": 0.83 | null,
      "metrics": {"input_tokens": ..., "cost": ..., ...},
      "judge_usage": {...} | null,          # never part of the strategy's cost
      "duration": 0.06,                     # task wall time, seconds
      "evaluator_failures": [...],
      "trace_id": "..."
    }
  ]
}
```

`fields` is the per-field view compare-runs works from: `expected` says the
case is scored on the field (for the judged fields: the judge ran), and
`attempted` that the strategy produced it. A field that is expected but not
attempted counts as 0 in the composite.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .evaluators import FIELDS, as_dict, triage_of
from .suite import composite

FORMAT = 1
_REPEAT_SUFFIX = re.compile(r" \[\d+/\d+\]$")


def base_case_name(name: str) -> str:
    """pydantic-evals names repeats `case [i/n]`; this strips that."""
    return _REPEAT_SUFFIX.sub("", name)


def field_view(
    output: Any, expected: Mapping[str, Any], scores: Mapping[str, Any], judged: bool, judge_failed: set[str]
) -> dict[str, dict[str, Any]]:
    triage = triage_of(output) if output else {}
    view = {}
    for spec in FIELDS:
        attempted = bool(output) and spec.produced(triage)
        score = scores.get(spec.score)
        if spec.expected is None:
            is_expected = judged and spec.name not in judge_failed and (not attempted or score is not None)
        else:
            is_expected = spec.expected(dict(expected))
        view[spec.name] = {
            "expected": is_expected,
            "attempted": attempted,
            "score": None if score is None else float(score),
            "value": spec.value(triage) if attempted else None,
            "confidence": spec.confidence(triage) if attempted else None,
        }
    return view


def composite_of(fields: Mapping[str, Mapping[str, Any]], weights: Mapping[str, float]) -> float | None:
    scores = {
        name: (f["score"] if f["attempted"] and f["score"] is not None else 0.0)
        for name, f in fields.items()
        if f["expected"]
    }
    return composite(scores, weights)


def case_record(
    *,
    name: str,
    repeat: int | None,
    workflow_key: str | None,
    expected: Any,
    output: Any,
    error: str | None,
    scores: Mapping[str, Any],
    reasons: Mapping[str, str],
    metrics: Mapping[str, Any],
    duration: float | None,
    judged: bool,
    weights: Mapping[str, float],
    judge_usage: Mapping[str, Any] | None = None,
    evaluator_failures: Iterable[Mapping[str, Any]] = (),
    trace_id: str | None = None,
) -> dict[str, Any]:
    failures = list(evaluator_failures)
    judge_failed = {f["field"] for f in failures if f.get("field")}
    expected_d = as_dict(expected)
    fields = field_view(output, expected_d, scores, judged, judge_failed)
    return {
        "name": name,
        "repeat": repeat,
        "workflow_key": workflow_key,
        "expected": expected_d,
        "output": as_dict(output) if output is not None else None,
        "error": error,
        "scores": dict(scores),
        "reasons": dict(reasons),
        "fields": fields,
        "composite": composite_of(fields, weights),
        "metrics": dict(metrics),
        "judge_usage": dict(judge_usage) if judge_usage else None,
        "duration": duration,
        "evaluator_failures": failures,
        "trace_id": trace_id,
    }


def results_path(results_dir: Path, run_id: str, strategy: str, cfg_hash: str) -> Path:
    return results_dir / run_id / f"{strategy}-{cfg_hash}.json"


def save(path: Path, metadata: Mapping[str, Any], cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = sorted(cases, key=lambda c: (c["name"], c["repeat"] if c["repeat"] is not None else -1))
    data = {"format": FORMAT, "metadata": dict(metadata), "cases": cases}
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported results format {data.get('format')!r}")
    data["path"] = str(path)
    return data


def expand_paths(paths: Iterable[Path]) -> list[Path]:
    """Files as given; directories expand to the results files inside them."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*.json") if q.is_file()))
        else:
            out.append(p)
    return out

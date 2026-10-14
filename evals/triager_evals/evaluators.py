"""Per-field evaluators for triage outputs.

The task output is a `TriageResponse` as JSON (a dict). Every evaluator
returns an empty dict when it has nothing to score, either because the case
has no expected answer for its field or because the strategy didn't produce
the field. pydantic-evals then records nothing, so a skip shows up as a
missing score, never as a pass or a fail. `Coverage` records how many of the
fields with an expected answer the strategy attempted, and `results.py`
keeps per-field attempted/expected flags so compare-runs can count
unattempted fields as 0 in the composite.

Evaluation names (the keys in the report and in the results files):

| Evaluator | Names |
|---|---|
| KindMatches | `kind` |
| AreaMatches | `area` |
| SeverityMatches | `severity` (exact), `severity_distance` (levels apart, 0..3) |
| RegressionMatches | `is_regression` |
| SecurityMatches | `security_sensitive` |
| EnoughInfoMatches | `enough_info` |
| DuplicateRecall | `duplicates` (recall@k of the labeled targets) |
| SanityChecks | `sanity` (False with the reason on any hard error) |
| Coverage | `coverage` |
| QualityJudge (judge.py) | `quality` (1 - abs error / 4), `quality_abs_error`, `quality_agree`, `quality_judge` |
| ReplyJudge (judge.py) | `reply` (judge score 0..1), `reply_pass` |

The per-case evaluations above are pass/fail. The run-level score of a field
(what the composite and the summaries use) is the mean of its per-case scores,
except for the fields in `BALANCED_FIELDS`: those are rare-positive booleans,
where plain accuracy rewards always answering false, so their run-level score
is balanced accuracy (the mean of the true-positive and true-negative rates),
computed in `compare.run_stats`. When the cases have only one class, it falls
back to the accuracy on that class and the summary flags it.

Bump `EVALUATORS_VERSION` whenever a score's meaning changes; it is part of
the suite ID.

Versions: 1 = plain accuracy everywhere; 2 = balanced accuracy for
`BALANCED_FIELDS`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

EVALUATORS_VERSION = "2"

# Run-level score is balanced accuracy (see the module docstring).
BALANCED_FIELDS = ("is_regression", "security_sensitive")

SEVERITY_ORDER = ("critical", "important_soon", "important_longterm", "backlog")


def as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value)


@dataclass(frozen=True)
class FieldSpec:
    """How one scored field is found in the expected answers and the output.

    `score` names the 0..1 evaluation that counts for accuracy and the
    composite. `expected` is None for the judged fields: they have no label,
    and count as expected whenever the judge ran.
    """

    name: str
    output_field: str
    score: str
    expected: Callable[[dict[str, Any]], bool] | None
    value: Callable[[dict[str, Any]], Any]

    def produced(self, triage: Mapping[str, Any]) -> bool:
        return self.value(triage) is not None

    def confidence(self, triage: Mapping[str, Any]) -> float | None:
        field = triage.get(self.output_field)
        return field.get("confidence") if isinstance(field, dict) else None


def _value(name: str) -> Callable[[dict[str, Any]], Any]:
    def get(triage: dict[str, Any]) -> Any:
        field = triage.get(name)
        return field.get("value") if isinstance(field, dict) else None

    return get


def _enough_info(triage: dict[str, Any]) -> bool | None:
    repro = _value("reproducibility")(triage)
    return repro.get("enough_info") if isinstance(repro, dict) else None


def _has(name: str) -> Callable[[dict[str, Any]], bool]:
    def check(expected: dict[str, Any]) -> bool:
        v = expected.get(name)
        return v is not None and v != []

    return check


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("kind", "kind", "kind", _has("kind"), _value("kind")),
    FieldSpec("area", "area", "area", _has("area"), _value("area")),
    FieldSpec("severity", "severity", "severity", _has("severity"), _value("severity")),
    FieldSpec("is_regression", "is_regression", "is_regression", _has("is_regression"), _value("is_regression")),
    FieldSpec(
        "security_sensitive",
        "security_sensitive",
        "security_sensitive",
        _has("security_sensitive"),
        _value("security_sensitive"),
    ),
    FieldSpec("enough_info", "reproducibility", "enough_info", _has("enough_info"), _enough_info),
    FieldSpec("duplicates", "duplicates", "duplicates", _has("duplicate_of"), _value("duplicates")),
    FieldSpec("quality", "quality", "quality", None, _value("quality")),
    FieldSpec("reply", "suggested_reply", "reply", None, _value("suggested_reply")),
)
FIELDS_BY_NAME = {f.name: f for f in FIELDS}
JUDGED_FIELDS = frozenset(f.name for f in FIELDS if f.expected is None)
LABELED_FIELDS = tuple(f for f in FIELDS if f.expected is not None)


def triage_of(output: Any) -> dict[str, Any]:
    out = as_dict(output)
    return out.get("triage") or {}


def severity_distance(a: str, b: str) -> int:
    return abs(SEVERITY_ORDER.index(a) - SEVERITY_ORDER.index(b))


def recall_at_k(predicted: list[int], targets: list[int], k: int) -> float:
    wanted = set(targets)
    return len(wanted & set(predicted[:k])) / len(wanted)


def sanity_problems(
    triage: Mapping[str, Any], expected: Mapping[str, Any], inputs: Mapping[str, Any], metadata: Mapping[str, Any]
) -> list[str]:
    """Hard errors: answers that contradict what happened to the issue, or the contract."""
    problems = []
    kind = _value("kind")(dict(triage))
    outcome = expected.get("close_outcome")
    fixing_prs = (metadata or {}).get("fixing_prs") or []
    if kind == "spam" and outcome == "fixed":
        problems.append("predicted spam, but the issue was fixed")
    if kind in ("question", "spam") and fixing_prs:
        problems.append(f"predicted {kind}, but PRs {fixing_prs} fixed it")
    issue = inputs.get("issue", {})
    number = issue.get("number")
    dupes = _value("duplicates")(dict(triage)) or []
    if number is not None and number in dupes:
        problems.append("lists the issue itself as its duplicate")
    if number is not None and any(d > number for d in dupes):
        problems.append("lists a duplicate filed after the issue")
    area = _value("area")(dict(triage))
    labels = {label["name"] for label in inputs.get("repo", {}).get("labels", [])}
    if area is not None and labels and area not in labels:
        problems.append(f"area {area!r} is not one of the repo's labels")
    quality = _value("quality")(dict(triage))
    if quality is not None and not 1 <= quality <= 5:
        problems.append(f"quality {quality} is outside 1..5")
    for name, field in triage.items():
        conf = field.get("confidence") if isinstance(field, dict) else None
        if conf is not None and not 0.0 <= conf <= 1.0:
            problems.append(f"{name}.confidence {conf} is outside 0..1")
    return problems


def scored_pair(ctx: EvaluatorContext[Any, Any, Any], field: str) -> tuple[Any, Any] | None:
    """(predicted, expected) for a labeled field, or None when either is missing."""
    spec = FIELDS_BY_NAME[field]
    expected = as_dict(ctx.expected_output)
    triage = triage_of(ctx.output)
    if spec.expected is None or not spec.expected(expected) or not spec.produced(triage):
        return None
    key = {"duplicates": "duplicate_of"}.get(field, field)
    return spec.value(triage), expected[key]


class _Versioned:
    """pydantic-evals' Evaluator base can't be subclassed abstractly, so this is a mixin."""

    def get_evaluator_version(self) -> str | None:
        return EVALUATORS_VERSION


@dataclass
class KindMatches(_Versioned, Evaluator[Any, Any, Any]):
    """The predicted kind is one of the issue's kind labels (some have two)."""

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> dict[str, bool]:
        pair = scored_pair(ctx, "kind")
        return {} if pair is None else {"kind": pair[0] in pair[1]}


@dataclass
class AreaMatches(_Versioned, Evaluator[Any, Any, Any]):
    """The predicted `sig/*` is one of the issue's sig labels."""

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> dict[str, bool]:
        pair = scored_pair(ctx, "area")
        return {} if pair is None else {"area": pair[0] in pair[1]}


@dataclass
class SeverityMatches(_Versioned, Evaluator[Any, Any, Any]):
    """Exact priority match, plus how many levels off it is."""

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> dict[str, bool | int]:
        pair = scored_pair(ctx, "severity")
        if pair is None:
            return {}
        return {"severity": pair[0] == pair[1], "severity_distance": severity_distance(pair[0], pair[1])}


@dataclass
class RegressionMatches(_Versioned, Evaluator[Any, Any, Any]):
    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> dict[str, bool]:
        pair = scored_pair(ctx, "is_regression")
        return {} if pair is None else {"is_regression": pair[0] == pair[1]}


@dataclass
class SecurityMatches(_Versioned, Evaluator[Any, Any, Any]):
    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> dict[str, bool]:
        pair = scored_pair(ctx, "security_sensitive")
        return {} if pair is None else {"security_sensitive": pair[0] == pair[1]}


@dataclass
class EnoughInfoMatches(_Versioned, Evaluator[Any, Any, Any]):
    """`reproducibility.enough_info` against needs-information (false) / fixed by a PR (true)."""

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> dict[str, bool]:
        pair = scored_pair(ctx, "enough_info")
        return {} if pair is None else {"enough_info": pair[0] == pair[1]}


@dataclass
class DuplicateRecall(_Versioned, Evaluator[Any, Any, Any]):
    """Recall@k of the labeled duplicate target(s) among the ranked candidates."""

    k: int = 5

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> dict[str, float]:
        pair = scored_pair(ctx, "duplicates")
        return {} if pair is None else {"duplicates": recall_at_k(pair[0], pair[1], self.k)}


@dataclass
class SanityChecks(_Versioned, Evaluator[Any, Any, Any]):
    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> dict[str, EvaluationReason]:
        problems = sanity_problems(
            triage_of(ctx.output), as_dict(ctx.expected_output), as_dict(ctx.inputs), ctx.metadata or {}
        )
        return {"sanity": EvaluationReason(value=not problems, reason="; ".join(problems) or None)}


@dataclass
class Coverage(_Versioned, Evaluator[Any, Any, Any]):
    """The share of labeled fields with an expected answer that the strategy produced."""

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> dict[str, float]:
        expected = as_dict(ctx.expected_output)
        triage = triage_of(ctx.output)
        wanted = [f for f in LABELED_FIELDS if f.expected(expected)]
        if not wanted:
            return {}
        return {"coverage": sum(f.produced(triage) for f in wanted) / len(wanted)}


def code_evaluators(duplicates_k: int = 5) -> list[Evaluator[Any, Any, Any]]:
    return [
        KindMatches(),
        AreaMatches(),
        SeverityMatches(),
        RegressionMatches(),
        SecurityMatches(),
        EnoughInfoMatches(),
        DuplicateRecall(k=duplicates_k),
        SanityChecks(),
        Coverage(),
    ]

"""What makes eval runs comparable: hashes, the quality rubric, the suite file.

- `config_hash` and `case_digest` build the Restate workflow key. The config
  hash must be byte-for-byte triage-core's `config_hash`, because the service
  puts its own value in every response and the harness checks the two agree.
- `QUALITY_RUBRIC` is a copy of triage-core's, so the judge rates issues on
  exactly the scale the strategies were prompted with. A test keeps the copy
  in sync with the Rust file; bump `RUBRIC_VERSION` when it changes.
- `Suite` is `evals/suite.yaml`: the composite weights. `suite_id` hashes
  everything a score depends on besides the strategy itself, and runs are
  only compared within one suite.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[2]
EVALS_DIR = ROOT / "evals"
SCHEMA_DIR = ROOT / "schema"
SUITE_FILE = EVALS_DIR / "suite.yaml"
RESULTS_DIR = EVALS_DIR / "results"
RUST_TRIAGE_FILE = ROOT / "crates" / "triage-core" / "src" / "triage.rs"
RUST_PRICE_FILE = ROOT / "crates" / "triage-core" / "src" / "price.rs"

RUBRIC_VERSION = "1"
QUALITY_RUBRIC: tuple[tuple[int, str], ...] = (
    (1, "Unusable: no clear problem or request; spam, gibberish, or a one-liner with no context."),
    (
        2,
        "Poor: the topic is clear but key information is missing; a maintainer would have to ask several questions.",
    ),
    (3, "Adequate: understandable and mostly complete, but needs at least one follow-up question."),
    (4, "Good: clear, complete and focused; a maintainer can act on it without follow-up."),
    (5, "Exemplary: clear, complete, focused, with a minimal reproduction or a well-argued proposal."),
)


def canonical_json(value: Any) -> str:
    """The bytes triage-core hashes: sorted keys, no whitespace, UTF-8 unescaped."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def short_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()[:12]


def config_hash(config: Mapping[str, Any]) -> str:
    """triage-core's `config_hash`."""
    return short_hash(config)


def case_digest(inputs: Mapping[str, Any]) -> str:
    """Identifies a case by its inputs, so an edited case gets a new workflow key."""
    return short_hash(inputs)


def workflow_key(run_id: str, cfg_hash: str, digest: str, repeat: int) -> str:
    return f"{run_id}:{cfg_hash}:{digest}:{repeat}"


# The JSON schema (in schema/) of each strategy's config.
CONFIG_SCHEMAS = {
    "Heuristic": "heuristic-config",
    "SingleShot": "single-shot-config",
    "Workflow": "workflow-config",
    "Agent": "agent-config",
    "Ensemble": "ensemble-config",
    "DraftVerify": "draft-verify-config",
}


def strict_schema(schema: Any) -> Any:
    """The schema with unknown keys rejected, so a typo in a config fails.

    The generated schemas allow extra keys, and the Rust config structs
    ignore them, so `escalaton:` would otherwise silently fall back to the
    default under a new config hash.
    """
    schema = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "properties" in node and "additionalProperties" not in node:
                node["additionalProperties"] = False
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


def config_errors(strategy: str, config: Mapping[str, Any]) -> list[str]:
    """What is wrong with a strategy config, checked against its strict schema."""
    name = CONFIG_SCHEMAS.get(strategy)
    path = SCHEMA_DIR / f"{name}.schema.json"
    if name is None or not path.exists():
        return []
    schema = strict_schema(json.loads(path.read_text(encoding="utf-8")))
    validator = jsonschema.validators.validator_for(schema)(schema)
    return [
        f"{'/'.join(map(str, e.absolute_path)) or '(top level)'}: {e.message}"
        for e in sorted(validator.iter_errors(config), key=lambda e: list(map(str, e.absolute_path)))
    ]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rubric_text() -> str:
    return "\n".join(f"{score}. {text}" for score, text in QUALITY_RUBRIC)


def parse_rust_rubric(path: Path = RUST_TRIAGE_FILE) -> tuple[tuple[int, str], ...]:
    """Reads `QUALITY_RUBRIC` out of triage.rs (for the sync test)."""
    import re

    src = path.read_text(encoding="utf-8")
    body = src[src.index("pub const QUALITY_RUBRIC") :]
    body = body[: body.index("];")]
    return tuple((int(n), text) for n, text in re.findall(r'\(\s*(\d+),\s*"((?:[^"\\]|\\.)*)",?\s*\)', body))


def rust_prices(path: Path = RUST_PRICE_FILE) -> list[tuple[str, float, float]]:
    """triage-core's price table as (prefix, input $/Mtok, output $/Mtok), in match order.

    The judge's cost uses the same (shadow) prices as the strategies, read from
    the Rust source so there is one table.
    """
    import re

    src = path.read_text(encoding="utf-8")
    return [
        (prefix, float(i), float(o))
        for prefix, i, o in re.findall(r'\("([^"]+)",\s*Price::new\(([\d.]+),\s*([\d.]+)\)\)', src)
    ]


def price_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    for prefix, inp, out in rust_prices():
        if model.startswith(prefix):
            return (input_tokens * inp + output_tokens * out) / 1_000_000
    return None


@dataclass(frozen=True)
class Suite:
    weights: dict[str, float]
    duplicates_k: int = 5

    def weights_for(self, judged: bool) -> dict[str, float]:
        """The judged fields only count when the judge ran."""
        from .evaluators import JUDGED_FIELDS

        return {f: w for f, w in self.weights.items() if judged or f not in JUDGED_FIELDS}


def load_suite(path: Path = SUITE_FILE) -> Suite:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    weights = {str(k): float(v) for k, v in data["composite_weights"].items()}
    return Suite(weights=weights, duplicates_k=int(data.get("duplicates_k", 5)))


def suite_id(dataset_hash: str, judge_model: str | None, suite: Suite) -> str:
    from .evaluators import EVALUATORS_VERSION

    return short_hash(
        {
            "dataset_hash": dataset_hash,
            "evaluators_version": EVALUATORS_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "judge_model": judge_model,
            "weights": suite.weights_for(judge_model is not None),
            "duplicates_k": suite.duplicates_k,
        }
    )


def composite(field_scores: Mapping[str, float], weights: Mapping[str, float]) -> float | None:
    """Weighted mean over the fields present in `field_scores`.

    The caller puts in every field that has an expected answer, with 0 for the
    ones the strategy didn't produce; fields with no expected answer are left
    out, so they don't count either way.
    """
    total = sum(weights[f] for f in field_scores if f in weights)
    if total == 0:
        return None
    return sum(weights[f] * s for f, s in field_scores.items() if f in weights) / total

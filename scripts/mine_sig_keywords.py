"""Mine per-SIG keyword tokens for the heuristic strategy from dev-split issues.

Writes crates/triage-service/src/strategies/sig_tokens.rs, the static table
heuristic v2 scores `area` with (next to the hand-written SIG_KEYWORDS).

Run from the repo root (uv lives in devenv, and the split rule comes from the
harness package):

    devenv shell -- bash -c 'cd evals && uv run python ../scripts/mine_sig_keywords.py'

How the table is derived:

1. Records come from data/normalized/kubernetes.jsonl, keeping only the dev
   split (triager_evals.build_dataset.split_of). Test-split records are never
   read past their issue number. The cases of datasets/kubernetes-dev-200.yaml
   are left out too, so dev-200 stays a held-out check of the mined table.
2. Each issue becomes a set of tokens: title and body, lowercased, HTML
   comments and @mentions removed, words matching
   `[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*` of 3 to 40 characters with fewer
   than three digits (hashes, issue and KEP numbers name a cluster of issues,
   not a topic). The Rust side tokenizes the same way (heuristic.rs,
   `tokens`).
3. For every token seen in at least --min-count issues with a sig label, the
   SIG it co-occurs with most is its candidate, and its precision is the
   share of the token's issues carrying that SIG (an issue with several sig
   labels counts for each). Tokens below --min-precision are dropped.
4. Each SIG keeps its --per-sig best tokens, ranked by count x precision. A
   token's weight is the log-odds of its precision, log(p / (1 - p)), capped.

The defaults (40 tokens, count >= 5, precision >= 0.55, i.e. weight >= 0.2)
come from a 2-fold split of the same tuning records, comparing area accuracy
for 20 to 200 tokens per SIG, counts of 3 to 8 and precisions of 0.4 to 0.7:
more tokens and lower counts helped a little (about 1 point) but made a far
bigger table. dev-200 and the test split played no part.

`--write-splits DIR` also writes the tuning records (dev minus dev-200) and
the dev-200 records as JSONL, for
`cargo run -p triage-service --example heuristic_dev -- DIR/dev-tune.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from triager_evals.build_dataset import split_of

ROOT = Path(__file__).resolve().parent.parent
TOKEN = re.compile(r"[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*")
COMMENT = re.compile(r"(?s)<!--.*?-->")
MENTION = re.compile(r"@[a-z0-9][a-z0-9-]*")
MAX_WEIGHT = 4.0
# Hashes, issue and KEP numbers and the like: they name one cluster of issues,
# not a topic.
NUMBERISH = re.compile(r"\d.*\d.*\d")


def tokens(title: str, body: str) -> set[str]:
    text = MENTION.sub(" ", COMMENT.sub("", f"{title}\n{body}").lower())
    return {t for t in TOKEN.findall(text) if 3 <= len(t) <= 40 and not NUMBERISH.search(t)}


def dataset_numbers(path: Path) -> set[int]:
    return {int(n) for n in re.findall(r"^- name: kubernetes-(\d+)$", path.read_text(), re.M)}


def dev_records(source: Path, held_out: set[int]) -> tuple[list[dict], list[dict]]:
    """(tuning records, held-out records), both from the dev split."""
    tune, held = [], []
    with source.open() as f:
        for line in f:
            record = json.loads(line)
            number = record["issue"]["number"]
            if split_of(record["repo"]["name"], number) != "dev":
                continue
            (held if number in held_out else tune).append(record)
    return tune, held


def mine(records: list[dict], per_sig: int, min_count: int, min_precision: float) -> dict[str, list[tuple[str, float]]]:
    seen = Counter()
    by_sig: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        sigs = [a for a in record["expected"].get("area") or [] if a.startswith("sig/")]
        if not sigs:
            continue
        for token in tokens(record["issue"]["title"], record["issue"]["body"]):
            seen[token] += 1
            for sig in sigs:
                by_sig[token][sig] += 1
    candidates: dict[str, list[tuple[float, str, float]]] = defaultdict(list)
    for token, count in seen.items():
        if count < min_count:
            continue
        # Most co-occurring SIG; ties go to the alphabetically first one.
        sig, hits = min(by_sig[token].items(), key=lambda kv: (-kv[1], kv[0]))
        precision = hits / count
        if precision < min_precision:
            continue
        candidates[sig].append((hits * precision, token, precision))
    table = {}
    for sig in sorted(candidates):
        ranked = sorted(candidates[sig], key=lambda c: (-c[0], c[1]))[:per_sig]
        table[sig] = [(token, weight(p)) for _, token, p in ranked]
    return table


def weight(precision: float) -> float:
    """Log-odds of the precision, capped (a token always seen with its SIG)."""
    if precision >= 1:
        return MAX_WEIGHT
    return min(MAX_WEIGHT, math.log(precision / (1 - precision)))


def rust_table(table: dict[str, list[tuple[str, float]]], meta: str) -> str:
    lines = [
        "//! Generated by `scripts/mine_sig_keywords.py`; do not edit by hand.",
        "//!",
        f"//! {meta}",
        "//!",
        "//! Per SIG: tokens that co-occur with its label in dev-split issues, with",
        "//! the log-odds of that co-occurrence as the weight. See the script for how",
        "//! they are derived.",
        "",
        "// Weights are data, so one may happen to look like a math constant.",
        "#[allow(clippy::approx_constant)]",
        "#[rustfmt::skip]",
        "pub(crate) const SIG_TOKENS: &[(&str, &[(&str, f64)])] = &[",
    ]
    for sig, entries in table.items():
        lines.append(f'    ("{sig}", &[')
        for token, weight in entries:
            lines.append(f'        ("{token}", {weight:.2f}),')
        lines.append("    ]),")
    lines.append("];")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=ROOT / "data/normalized/kubernetes.jsonl")
    parser.add_argument("--held-out", type=Path, default=ROOT / "datasets/kubernetes-dev-200.yaml")
    parser.add_argument("--per-sig", type=int, default=40)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--min-precision", type=float, default=0.55)
    parser.add_argument("--out", type=Path, default=ROOT / "crates/triage-service/src/strategies/sig_tokens.rs")
    parser.add_argument("--write-splits", type=Path, metavar="DIR")
    args = parser.parse_args()

    held_out = dataset_numbers(args.held_out)
    tune, held = dev_records(args.source, held_out)
    table = mine(tune, args.per_sig, args.min_count, args.min_precision)
    meta = (
        f"{len(tune)} dev-split issues from {args.source.relative_to(ROOT)} "
        f"({args.held_out.name} held out); per_sig={args.per_sig}, "
        f"min_count={args.min_count}, min_precision={args.min_precision}."
    )
    args.out.write_text(rust_table(table, meta))
    print(f"{args.out.relative_to(ROOT)}: {sum(len(v) for v in table.values())} tokens for {len(table)} SIGs; {meta}")

    if args.write_splits:
        args.write_splits.mkdir(parents=True, exist_ok=True)
        for name, records in [("dev-tune.jsonl", tune), ("dev-200.jsonl", held)]:
            with (args.write_splits / name).open("w") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote {len(tune)} tuning and {len(held)} held-out records to {args.write_splits}")


if __name__ == "__main__":
    main()

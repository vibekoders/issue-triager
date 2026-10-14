"""Builds a pydantic-evals dataset from normalized records.

    build-dataset --repo kubernetes --split dev --limit 200 --stratify kind

writes `datasets/kubernetes-dev-200.yaml` and its JSON schema.

Split: each issue goes to `dev` or `test` by a hash of `<owner>/<name>#<number>`
(80/20), so the splits never overlap and an issue stays in its split when the
dump is re-fetched.

Order: the cases of a split are put in one fixed order that does not depend
on `--limit`, and a dataset is the first `limit` of them. That is what makes
a larger limit a superset of a smaller one (and keeps every prefix, such as
`run.py --limit N`, stratified). With `--stratify kind` the order is:

1. a floor: the first `--min-per-stratum` cases of every kind, round-robin;
2. the rest, interleaved so that every prefix holds each kind roughly in
   proportion to its share of the split (systematic sampling: the r-th case
   of a kind with n cases sits at position (r + 0.5) / n).

Inside a kind, cases go in hash order. The order depends only on the hashes
and on the per-kind counts of the source file; rebuilding from a different
dump can reorder cases, which is why the dataset records the source's hash.

Dataset-level metadata (filters, source hash, build date) can't go into the
pydantic-evals file, whose top level is closed, so it is written as a JSON
comment line that `read_dataset_metadata` returns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel
from pydantic_evals import Case, Dataset

from . import fetch
from .normalize import load_mapping

DATASETS_DIR = fetch.ROOT / "datasets"
METADATA_PREFIX = "# dataset-metadata: "
DEV_SHARE = 0.8

Split = Literal["dev", "test"]


class Label(BaseModel):
    name: str
    description: str | None = None


class Issue(BaseModel):
    """triage-core's `Issue`."""

    number: int
    title: str
    body: str
    author_association: str
    created_at: str


class Repo(BaseModel):
    """triage-core's `Repo`."""

    name: str
    labels: list[Label]
    latest_release: str | None = None
    as_of: str


class TriageInput(BaseModel):
    """What every strategy gets: a `TriageRequest` without its config."""

    issue: Issue
    repo: Repo


class Expected(BaseModel):
    """The answers triagers gave. `None` means the field isn't scored for the case."""

    kind: (
        list[Literal["bug", "feature", "question", "docs", "cleanup", "failing_test", "flake", "other", "spam"]] | None
    ) = None
    area: list[str] | None = None
    severity: Literal["critical", "important_soon", "important_longterm", "backlog"] | None = None
    is_regression: bool | None = None
    security_sensitive: bool | None = None
    enough_info: bool | None = None
    duplicate_of: list[int] | None = None
    close_outcome: Literal["fixed", "duplicate", "invalid", "needs_info_timeout", "stale", "completed_other", "open"]


TriageDataset = Dataset[TriageInput, Expected, dict[str, Any]]


def case_hash(repo: str, number: int) -> int:
    return int(hashlib.sha256(f"{repo}#{number}".encode()).hexdigest()[:16], 16)


def split_of(repo: str, number: int) -> Split:
    return "dev" if case_hash(repo, number) < DEV_SHARE * 2**64 else "test"


def stratum(record: dict[str, Any], by: str | None) -> str:
    if by is None:
        return ""
    values = record["expected"].get(by)
    if isinstance(values, list):
        return values[0] if values else "none"
    return str(values) if values is not None else "none"


def order_cases(
    records: list[dict[str, Any]], repo: str, stratify: str | None, min_per_stratum: int
) -> list[dict[str, Any]]:
    """Puts the records in the one order every limit takes a prefix of."""
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sorted(records, key=lambda r: case_hash(repo, r["issue"]["number"])):
        by_stratum[stratum(r, stratify)].append(r)

    keyed = []
    for members in by_stratum.values():
        n = len(members)
        for rank, r in enumerate(members):
            h = case_hash(repo, r["issue"]["number"])
            if rank < min_per_stratum:
                keyed.append(((0, rank, h), r))
            else:
                keyed.append(((1, (rank + 0.5) / n, h), r))
    keyed.sort(key=lambda kr: kr[0])
    return [r for _, r in keyed]


def select(
    records: list[dict[str, Any]],
    repo: str,
    split: Split,
    limit: int | None,
    stratify: str | None,
    min_per_stratum: int,
    since: str | None = None,
    triager_verified: bool = False,
    keep_unknown_body: bool = False,
) -> list[dict[str, Any]]:
    """The cases of a split, in order, first `limit`.

    Issues whose filed body is lost (`original_body_unknown`: edited, with
    the creation snapshot deleted) are left out unless `keep_unknown_body`:
    their current body is mostly a security advisory written after triage,
    which gives the answers away.
    """
    pool = [
        r
        for r in records
        if split_of(repo, r["issue"]["number"]) == split
        and (since is None or r["issue"]["created_at"] >= since)
        and (not triager_verified or r["metadata"]["triager_verified"])
        and (keep_unknown_body or not r["metadata"].get("original_body_unknown"))
    ]
    ordered = order_cases(pool, repo, stratify, min_per_stratum)
    return ordered if limit is None else ordered[:limit]


def to_case(record: dict[str, Any], name: str) -> Case[TriageInput, Expected, dict[str, Any]]:
    metadata = {**record["metadata"], "close_outcome": record["expected"]["close_outcome"]}
    return Case(
        name=f"{name}-{record['issue']['number']}",
        inputs=TriageInput(issue=record["issue"], repo=record["repo"]),
        expected_output=Expected(**record["expected"]),
        metadata=metadata,
    )


class _DatasetDumper(yaml.SafeDumper):
    """Names the anchor of the label list every case shares, and writes
    multi-line text (issue bodies) as literal blocks so diffs stay readable."""

    def generate_anchor(self, node: yaml.Node) -> str:
        n = getattr(self, "_anchors_made", 0) + 1
        self._anchors_made = n
        return "repo_labels" if n == 1 else f"repo_labels_{n}"


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    # PyYAML falls back to a quoted style by itself when a block can't hold
    # the text (trailing spaces, tabs, control characters).
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)


_DatasetDumper.add_representer(str, _represent_str)


def write_dataset(dataset: TriageDataset, path: Path, metadata: dict[str, Any]) -> None:
    """Saves with `Dataset.to_file` (which also writes the JSON schema), then
    stores each distinct label list once.

    Every case carries the repo's ~150 labels; written out per case, a
    600-case file would be over 10 MB. YAML anchors keep one copy, and
    `Dataset.from_file` reads them back as plain lists.
    """
    dataset.to_file(path)
    text = path.read_text(encoding="utf-8")
    schema_line = text.splitlines()[0] if text.startswith("#") else None

    data = yaml.safe_load(text)
    shared: dict[str, list[Any]] = {}
    for case in data["cases"]:
        if case.get("evaluators") == []:
            del case["evaluators"]
        labels = case["inputs"]["repo"]["labels"]
        key = json.dumps(labels, sort_keys=True)
        case["inputs"]["repo"]["labels"] = shared.setdefault(key, labels)
    body = yaml.dump(data, Dumper=_DatasetDumper, sort_keys=False, allow_unicode=True, width=120)

    header = [schema_line] if schema_line else []
    header.append(METADATA_PREFIX + json.dumps(metadata, sort_keys=True))
    path.write_text("\n".join(header) + "\n" + body, encoding="utf-8")


def read_dataset_metadata(path: Path | str) -> dict[str, Any]:
    """The dataset-level metadata `write_dataset` stored in the file's header."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.startswith(METADATA_PREFIX):
                return json.loads(line[len(METADATA_PREFIX) :])
            if not line.startswith("#"):
                break
    return {}


def load_dataset(path: Path | str) -> TriageDataset:
    return TriageDataset.from_file(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="kubernetes", help="repo name or owner/name, as in repos/*.yaml")
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--limit", type=int, help="number of cases (default: the whole split)")
    p.add_argument("--stratify", choices=["kind", "none"], default="kind")
    p.add_argument("--min-per-stratum", type=int, default=2, help="cases of every kind placed first")
    p.add_argument("--since", help="only issues created on or after this date (YYYY-MM-DD)")
    p.add_argument("--triager-verified", action="store_true", help="only issues whose kind a triager confirmed")
    p.add_argument(
        "--keep-unknown-body",
        action="store_true",
        help="keep issues whose body as filed is lost (shown with their current, post-triage body)",
    )
    p.add_argument("--build-date", default=date.today().isoformat())
    p.add_argument("--source", type=Path, help="normalized records (default: data/normalized/<name>.jsonl)")
    p.add_argument("--out", type=Path, help="output file (default: datasets/<name>-<split>-<limit>.yaml)")
    args = p.parse_args()

    mapping = load_mapping(args.repo)
    repo, name = mapping.repo, mapping.name
    source = args.source or fetch.DATA_DIR / "normalized" / f"{name}.jsonl"
    with source.open() as f:
        records = [json.loads(line) for line in f]

    stratify = None if args.stratify == "none" else args.stratify
    chosen = select(
        records,
        repo,
        args.split,
        args.limit,
        stratify,
        args.min_per_stratum,
        args.since,
        args.triager_verified,
        args.keep_unknown_body,
    )
    suffix = (
        ""
        if not (args.since or args.triager_verified)
        else "-"
        + "-".join(
            x for x in (args.since and f"since{args.since.replace('-', '')}", args.triager_verified and "verified") if x
        )
    )
    ds_name = f"{name}-{args.split}-{args.limit or 'all'}{suffix}"
    out = args.out or DATASETS_DIR / f"{ds_name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "repo": repo,
        "split": args.split,
        "limit": args.limit,
        "stratify": stratify,
        "min_per_stratum": args.min_per_stratum,
        "since": args.since,
        "triager_verified": args.triager_verified,
        "keep_unknown_body": args.keep_unknown_body,
        "split_rule": f"sha256('{repo}#<number>')[:16] < {DEV_SHARE} * 2**64 -> dev",
        "source": str(source.relative_to(fetch.ROOT) if source.is_relative_to(fetch.ROOT) else source),
        "source_sha256": file_sha256(source),
        "source_records": len(records),
        "build_date": args.build_date,
        "cases": len(chosen),
    }
    dataset = TriageDataset(name=ds_name, cases=[to_case(r, name) for r in chosen])
    write_dataset(dataset, out, metadata)

    counts: dict[str, int] = defaultdict(int)
    for r in chosen:
        counts[stratum(r, "kind")] += 1
    print(f"{out}: {len(chosen)} cases; first kind: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()

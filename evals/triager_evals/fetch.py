"""Fetches a repo's issues from GitHub into `data/raw/<owner>__<name>/`.

Modes:

- `eligible`: newest issues first, with everything `normalize` needs for the
  expected answers (labels, label history, close events, first comments).
  It stops once `--target` issues pass the eligibility filter.
- `corpus`: every issue, with only number, title, body, created_at and state,
  for the agents' search tools.
- `backfill`: for an `eligible` dump made before title renames and both
  sides of duplicate markings were fetched, fetches just those events into
  `eligible/backfill.jsonl`, which `normalize` merges in.

`eligible` and `corpus` write one JSON line per page to `<mode>/pages.jsonl` and save the cursor
after every page in `<mode>/checkpoint.json`, so an interrupted run continues
where it stopped. Releases, labels and issue templates go to the repo
directory itself.
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .env import load_repo_env
from .github import GitHub, QueryTooHeavy, log

# The repo root, so the scripts write to the same `data/` whether they run
# from the root or from `evals/`.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

RATE = "rateLimit { cost remaining resetAt }"

ACTOR = "actor { login __typename }"

# Event types `backfill` adds to issues fetched before they were part of
# TIMELINE_TYPES: title renames (the case shows the title as filed) and both
# sides of duplicate markings (the event shows up on the canonical issue too).
BACKFILL_TYPES = ("RENAMED_TITLE_EVENT", "MARKED_AS_DUPLICATE_EVENT", "UNMARKED_AS_DUPLICATE_EVENT")
BACKFILL_TYPENAMES = ("RenamedTitleEvent", "MarkedAsDuplicateEvent", "UnmarkedAsDuplicateEvent")

TIMELINE_TYPES = (
    "[LABELED_EVENT, UNLABELED_EVENT, CLOSED_EVENT, REOPENED_EVENT, "
    "MARKED_AS_DUPLICATE_EVENT, UNMARKED_AS_DUPLICATE_EVENT, RENAMED_TITLE_EVENT, "
    "CROSS_REFERENCED_EVENT, TRANSFERRED_EVENT]"
)

_ISSUE_OR_PR = """{
      __typename
      ... on Issue { number repository { nameWithOwner } }
      ... on PullRequest { number repository { nameWithOwner } }
    }"""

DUPLICATE_NODES = f"""
  ... on MarkedAsDuplicateEvent {{
    createdAt {ACTOR} isCrossRepository
    canonical {_ISSUE_OR_PR}
    duplicate {_ISSUE_OR_PR}
  }}
  ... on UnmarkedAsDuplicateEvent {{
    createdAt {ACTOR}
    canonical {_ISSUE_OR_PR}
    duplicate {_ISSUE_OR_PR}
  }}
"""

RENAME_NODES = f"""
  ... on RenamedTitleEvent {{ createdAt {ACTOR} previousTitle currentTitle }}
"""

TIMELINE_NODES = f"""
  __typename
  ... on LabeledEvent {{ createdAt {ACTOR} label {{ name }} }}
  ... on UnlabeledEvent {{ createdAt {ACTOR} label {{ name }} }}
  ... on ClosedEvent {{
    createdAt {ACTOR} stateReason
    closer {{
      __typename
      ... on PullRequest {{ number merged repository {{ nameWithOwner }} }}
      ... on Commit {{ oid }}
    }}
  }}
  ... on ReopenedEvent {{ createdAt {ACTOR} }}
  {DUPLICATE_NODES}
  {RENAME_NODES}
  ... on CrossReferencedEvent {{
    createdAt willCloseTarget {ACTOR}
    source {{
      __typename
      ... on Issue {{ number repository {{ nameWithOwner }} }}
      ... on PullRequest {{ number merged repository {{ nameWithOwner }} }}
    }}
  }}
  ... on TransferredEvent {{ createdAt {ACTOR} fromRepository {{ nameWithOwner }} }}
"""

ISSUE_FIELDS = f"""
  number title body createdAt authorAssociation
  author {{ login __typename }}
  state stateReason closedAt locked activeLockReason
  milestone {{ title }}
  comments(first: 10) {{
    totalCount
    nodes {{ author {{ login __typename }} authorAssociation body createdAt }}
  }}
  participants {{ totalCount }}
  reactions(content: THUMBS_UP) {{ totalCount }}
  labels(first: 50) {{ nodes {{ name }} }}
  userContentEdits(last: 1) {{ totalCount nodes {{ diff editedAt }} }}
  closedByPullRequestsReferences(first: 10, includeClosedPrs: true) {{
    nodes {{ number merged repository {{ nameWithOwner }} }}
  }}
  timelineItems(first: 100, itemTypes: {TIMELINE_TYPES}) {{
    pageInfo {{ hasNextPage endCursor }}
    nodes {{ {TIMELINE_NODES} }}
  }}
"""

ELIGIBLE_QUERY = f"""
query($owner: String!, $name: String!, $first: Int!, $after: String) {{
  {RATE}
  repository(owner: $owner, name: $name) {{
    issues(first: $first, after: $after, orderBy: {{field: CREATED_AT, direction: DESC}}) {{
      pageInfo {{ hasNextPage endCursor }}
      nodes {{ {ISSUE_FIELDS} }}
    }}
  }}
}}
"""

# Used to page past the issues that are too recent to be settled without
# paying for their timelines. Cursors are interchangeable between queries
# with the same connection arguments.
SKIP_QUERY = f"""
query($owner: String!, $name: String!, $first: Int!, $after: String) {{
  {RATE}
  repository(owner: $owner, name: $name) {{
    issues(first: $first, after: $after, orderBy: {{field: CREATED_AT, direction: DESC}}) {{
      pageInfo {{ hasNextPage endCursor }}
      nodes {{ number createdAt }}
    }}
  }}
}}
"""

TIMELINE_QUERY = f"""
query($owner: String!, $name: String!, $number: Int!, $after: String) {{
  {RATE}
  repository(owner: $owner, name: $name) {{
    issue(number: $number) {{
      timelineItems(first: 100, after: $after, itemTypes: {TIMELINE_TYPES}) {{
        pageInfo {{ hasNextPage endCursor }}
        nodes {{ {TIMELINE_NODES} }}
      }}
    }}
  }}
}}
"""

CORPUS_QUERY = f"""
query($owner: String!, $name: String!, $first: Int!, $after: String) {{
  {RATE}
  repository(owner: $owner, name: $name) {{
    issues(first: $first, after: $after, orderBy: {{field: CREATED_AT, direction: DESC}}) {{
      totalCount
      pageInfo {{ hasNextPage endCursor }}
      nodes {{ number title body createdAt state }}
    }}
  }}
}}
"""

RELEASES_QUERY = f"""
query($owner: String!, $name: String!, $after: String) {{
  {RATE}
  repository(owner: $owner, name: $name) {{
    releases(first: 100, after: $after, orderBy: {{field: CREATED_AT, direction: DESC}}) {{
      pageInfo {{ hasNextPage endCursor }}
      nodes {{ tagName publishedAt isPrerelease isDraft }}
    }}
  }}
}}
"""

# More pages than this for one issue's timeline is a bot-heavy outlier; the
# rest is dropped and the issue is marked truncated.
MAX_TIMELINE_PAGES = 5


def repo_dir(repo: str, data_dir: Path | None = None) -> Path:
    owner, name = repo.split("/")
    return (data_dir or DATA_DIR) / "raw" / f"{owner}__{name}"


class Checkpoint:
    """The cursor and counters of one fetch, saved after every page."""

    def __init__(self, path: Path):
        self.path = path
        self.state: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    def __getitem__(self, key: str) -> Any:
        return self.state.get(key)

    def update(self, **kw: Any) -> None:
        self.state.update(kw)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.path)


def append_page(path: Path, after: str | None, issues: list[dict[str, Any]]) -> None:
    """Appends one page, first cutting off a line a crash left half-written."""
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.seek(0)
                data = f.read()
                f.truncate(data.rfind(b"\n") + 1)
    with path.open("a") as f:
        f.write(json.dumps({"after": after, "issues": issues}) + "\n")


def paged(
    gh: GitHub,
    query: str,
    variables: dict[str, Any],
    after: str | None,
    first: int,
    *,
    min_first: int = 10,
) -> Iterator[tuple[dict[str, Any], str | None, bool]]:
    """Yields (issues connection, end cursor, has next page) per page.

    The page size halves when GitHub times out on a heavy page and creeps
    back up afterwards, so one huge issue doesn't slow down the whole run.
    """
    size = first
    while True:
        try:
            data = gh.graphql(query, {**variables, "first": size, "after": after})
        except QueryTooHeavy as e:
            if size <= min_first:
                raise
            size = max(min_first, size // 2)
            log(f"page too heavy ({e}); retrying with {size} per page")
            continue
        if errs := data.get("__errors__"):
            log(f"partial GraphQL errors: {json.dumps(errs)[:500]}")
        conn = data["repository"]["issues"]
        info = conn["pageInfo"]
        yield conn, info["endCursor"], info["hasNextPage"]
        if not info["hasNextPage"]:
            return
        after = info["endCursor"]
        size = min(first, size + max(1, size // 2))


def complete_timeline(gh: GitHub, owner: str, name: str, issue: dict[str, Any]) -> None:
    """Fetches the rest of a timeline that overflowed the first 100 events."""
    tl = issue["timelineItems"]
    pages = 1
    while tl["pageInfo"]["hasNextPage"] and pages < MAX_TIMELINE_PAGES:
        data = gh.graphql(
            TIMELINE_QUERY,
            {"owner": owner, "name": name, "number": issue["number"], "after": tl["pageInfo"]["endCursor"]},
        )
        more = data["repository"]["issue"]["timelineItems"]
        tl["nodes"].extend(more["nodes"])
        tl["pageInfo"] = more["pageInfo"]
        pages += 1
    issue["timeline_truncated"] = tl["pageInfo"]["hasNextPage"]


def fetch_eligible(gh: GitHub, repo: str, target: int, skip_recent_days: int, page_size: int) -> None:
    from .normalize import is_eligible

    owner, name = repo.split("/")
    out = repo_dir(repo) / "eligible"
    out.mkdir(parents=True, exist_ok=True)
    cp = Checkpoint(out / "checkpoint.json")
    pages_path = out / "pages.jsonl"

    if cp["done"]:
        log(f"eligible fetch already done ({cp['eligible']} eligible); delete {cp.path} to refetch")
        return
    if cp["cutoff"] is None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=skip_recent_days)
        cp.update(cutoff=cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"), eligible=0, fetched=0, after=None, skipped=False)
    cutoff = cp["cutoff"]
    variables = {"owner": owner, "name": name}

    if not cp["skipped"]:
        # The cursor stops at the last page made only of too-recent issues;
        # the heavy query re-reads the boundary page and drops the recent ones.
        for conn, end, has_next in paged(gh, SKIP_QUERY, variables, cp["after"], 100):
            if not conn["nodes"] or conn["nodes"][-1]["createdAt"] >= cutoff and has_next:
                cp.update(after=end)
                continue
            break
        cp.update(skipped=True)
        log(f"skipped issues created after {cutoff}")

    for conn, end, has_next in paged(gh, ELIGIBLE_QUERY, variables, cp["after"], page_size):
        issues = [i for i in conn["nodes"] if i and i["createdAt"] < cutoff]
        for issue in issues:
            if issue["timelineItems"]["pageInfo"]["hasNextPage"]:
                complete_timeline(gh, owner, name, issue)
        eligible = sum(1 for i in issues if is_eligible(i, repo))
        append_page(pages_path, cp["after"], issues)
        cp.update(
            after=end,
            fetched=cp["fetched"] + len(issues),
            eligible=cp["eligible"] + eligible,
            oldest=issues[-1]["createdAt"] if issues else cp["oldest"],
        )
        log(f"eligible: {cp['eligible']}/{target} (fetched {cp['fetched']}, oldest {cp['oldest']})")
        if cp["eligible"] >= target or not has_next:
            break
    cp.update(done=True)


def fetch_corpus(gh: GitHub, repo: str, since: str | None, page_size: int) -> None:
    owner, name = repo.split("/")
    out = repo_dir(repo) / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    cp = Checkpoint(out / "checkpoint.json")
    pages_path = out / "pages.jsonl"
    if cp["done"]:
        log(f"corpus fetch already done ({cp['fetched']} issues); delete {cp.path} to refetch")
        return
    if cp["fetched"] is None:
        cp.update(fetched=0, after=None, since=since)

    for conn, end, has_next in paged(gh, CORPUS_QUERY, {"owner": owner, "name": name}, cp["after"], page_size):
        issues = [i for i in conn["nodes"] if i]
        reached_since = bool(since and issues and issues[-1]["createdAt"] < since)
        if since:
            issues = [i for i in issues if i["createdAt"] >= since]
        append_page(pages_path, cp["after"], issues)
        cp.update(
            after=end,
            fetched=cp["fetched"] + len(issues),
            total=conn["totalCount"],
            oldest=issues[-1]["createdAt"] if issues else cp["oldest"],
        )
        log(f"corpus: {cp['fetched']}/{conn['totalCount']} (oldest {cp['oldest']})")
        if reached_since or not has_next:
            break
    cp.update(done=True)


BACKFILL_NODES = f"__typename {DUPLICATE_NODES} {RENAME_NODES}"
BACKFILL_TIMELINE = f"""
  timelineItems(first: 100, itemTypes: [{", ".join(BACKFILL_TYPES)}]) {{
    pageInfo {{ hasNextPage endCursor }}
    nodes {{ {BACKFILL_NODES} }}
  }}
"""

BACKFILL_MORE_QUERY = f"""
query($owner: String!, $name: String!, $number: Int!, $after: String) {{
  {RATE}
  repository(owner: $owner, name: $name) {{
    issue(number: $number) {{
      timelineItems(first: 100, after: $after, itemTypes: [{", ".join(BACKFILL_TYPES)}]) {{
        pageInfo {{ hasNextPage endCursor }}
        nodes {{ {BACKFILL_NODES} }}
      }}
    }}
  }}
}}
"""


def backfill_query(numbers: list[int]) -> str:
    """One query for the BACKFILL_TYPES events of several issues, by alias."""
    issues = "\n".join(f"i{n}: issue(number: {n}) {{ number {BACKFILL_TIMELINE} }}" for n in numbers)
    return f"""
query($owner: String!, $name: String!) {{
  {RATE}
  repository(owner: $owner, name: $name) {{
    {issues}
  }}
}}
"""


def backfill_path(pages: Path) -> Path:
    """Where `backfill` writes the extra events for a pages file."""
    return pages.parent / "backfill.jsonl"


def fetch_backfill(gh: GitHub, repo: str, batch: int) -> None:
    """Fetches the BACKFILL_TYPES events of every issue in `eligible/pages.jsonl`.

    For dumps made before those types were in TIMELINE_TYPES. Writes one line
    per issue to `eligible/backfill.jsonl`, `{"number", "nodes", "truncated"}`;
    `normalize.raw_issues` puts these nodes in place of the issue's own nodes
    of the same types. Resumes by skipping the issues already written.
    """
    from .normalize import raw_issues

    owner, name = repo.split("/")
    pages = repo_dir(repo) / "eligible" / "pages.jsonl"
    out = backfill_path(pages)
    done: set[int] = set()
    if out.exists():
        # a crash can leave a half-written last line; append_page-style repair
        lines = out.read_bytes().split(b"\n")
        good = [ln for ln in lines if ln.strip()]
        try:
            json.loads(good[-1]) if good else None
        except ValueError:
            good = good[:-1]
        out.write_bytes(b"".join(ln + b"\n" for ln in good))
        done = {json.loads(ln)["number"] for ln in good}
    todo = [i["number"] for i in raw_issues(pages, backfill=False) if i["number"] not in done]
    log(f"backfill: {len(done)} done, {len(todo)} to fetch")
    for start in range(0, len(todo), batch):
        numbers = todo[start : start + batch]
        data = gh.graphql(backfill_query(numbers), {"owner": owner, "name": name})
        if errs := data.get("__errors__"):
            log(f"partial GraphQL errors: {json.dumps(errs)[:500]}")
        lines = []
        for n in numbers:
            issue = data["repository"].get(f"i{n}")
            if issue is None:
                # deleted or moved since the dump: nothing to add
                lines.append({"number": n, "nodes": [], "truncated": False, "missing": True})
                continue
            tl = issue["timelineItems"]
            pages_read = 1
            while tl["pageInfo"]["hasNextPage"] and pages_read < MAX_TIMELINE_PAGES:
                more = gh.graphql(
                    BACKFILL_MORE_QUERY,
                    {"owner": owner, "name": name, "number": n, "after": tl["pageInfo"]["endCursor"]},
                )["repository"]["issue"]["timelineItems"]
                tl["nodes"].extend(more["nodes"])
                tl["pageInfo"] = more["pageInfo"]
                pages_read += 1
            lines.append({"number": n, "nodes": tl["nodes"], "truncated": tl["pageInfo"]["hasNextPage"]})
        with out.open("a") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        log(f"backfill: {len(done) + start + len(numbers)}/{len(done) + len(todo)}")


def fetch_repo_context(gh: GitHub, repo: str) -> None:
    """Releases, labels and issue templates: small, so refetched every time."""
    out = repo_dir(repo)
    out.mkdir(parents=True, exist_ok=True)

    # GraphQL rather than REST: the REST pages carry the full release notes
    # and GitHub fails on them with 500s.
    owner, name = repo.split("/")
    releases: list[dict[str, Any]] = []
    after = None
    while True:
        data = gh.graphql(RELEASES_QUERY, {"owner": owner, "name": name, "after": after})
        conn = data["repository"]["releases"]
        releases += [
            {
                "tag": r["tagName"],
                "published_at": r["publishedAt"],
                "prerelease": r["isPrerelease"],
                "draft": r["isDraft"],
            }
            for r in conn["nodes"]
        ]
        if not conn["pageInfo"]["hasNextPage"]:
            break
        after = conn["pageInfo"]["endCursor"]
    (out / "releases.json").write_text(json.dumps(releases, indent=1))

    labels = [
        {"name": lb["name"], "description": lb.get("description") or None}
        for lb in gh.rest_paginate(f"repos/{repo}/labels")
    ]
    (out / "labels.json").write_text(json.dumps(labels, indent=1))

    templates = {}
    try:
        listing = gh.rest_get(f"repos/{repo}/contents/.github/ISSUE_TEMPLATE")
    except Exception as e:  # noqa: BLE001 - a repo without templates is fine
        log(f"no issue templates: {e}")
        listing = []
    for entry in listing:
        if entry["type"] == "file":
            content = gh.rest_get(entry["url"].removeprefix("https://api.github.com/"))
            templates[entry["name"]] = base64.b64decode(content["content"]).decode()
    (out / "templates.json").write_text(json.dumps(templates, indent=1))
    log(f"repo context: {len(releases)} releases, {len(labels)} labels, {len(templates)} templates")


def main() -> None:
    global DATA_DIR
    load_repo_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="kubernetes/kubernetes")
    p.add_argument("--mode", choices=["eligible", "corpus", "context", "backfill"], required=True)
    p.add_argument("--target", type=int, default=3000, help="eligible issues to collect (eligible mode)")
    p.add_argument("--skip-recent-days", type=int, default=60)
    p.add_argument("--since", help="corpus mode: stop at issues created before this date (YYYY-MM-DD)")
    p.add_argument("--page-size", type=int, help="issues per page (default: 50 eligible, 100 corpus)")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = p.parse_args()

    DATA_DIR = args.data_dir
    gh = GitHub()
    try:
        if args.mode != "backfill":
            # backfill only adds events to an existing dump; keep its context as fetched
            fetch_repo_context(gh, args.repo)
        if args.mode == "eligible":
            fetch_eligible(gh, args.repo, args.target, args.skip_recent_days, args.page_size or 50)
        elif args.mode == "corpus":
            fetch_corpus(gh, args.repo, args.since, args.page_size or 100)
        elif args.mode == "backfill":
            fetch_backfill(gh, args.repo, args.page_size or 50)
    finally:
        gh.close()


if __name__ == "__main__":
    main()

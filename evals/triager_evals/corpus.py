"""Builds the search corpus the agents' tools read: `data/corpus/<owner>__<name>.sqlite`.

It holds every issue of the repo (not only the eligible ones) with nothing
that reveals triage: number, title, body without slash commands, creation
time and state. The Rust service filters by `created_at < repo.as_of`, which
works as a string comparison because every timestamp is RFC 3339 UTC with a
`Z` suffix, as GitHub returns them.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import fetch
from .normalize import load_mapping, original_body, original_title, raw_issues, strip_slash_commands

SCHEMA = """
CREATE TABLE issues (
  number INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  created_at TEXT NOT NULL,
  state TEXT NOT NULL
);
CREATE VIRTUAL TABLE issues_fts USING fts5(title, body, content='issues', content_rowid='number');
CREATE TABLE releases (tag TEXT PRIMARY KEY, published_at TEXT NOT NULL, prerelease INTEGER NOT NULL);
CREATE TABLE templates (name TEXT PRIMARY KEY, body TEXT NOT NULL);
"""


def corpus_path(repo: str, data_dir: Path | None = None) -> Path:
    owner, name = repo.split("/")
    return (data_dir or fetch.DATA_DIR) / "corpus" / f"{owner}__{name}.sqlite"


def build(repo: str, data_dir: Path | None = None) -> tuple[Path, dict[str, int]]:
    """Rebuilds the corpus from the raw corpus pages, plus the eligible pages.

    The eligible pages add the titles and bodies as filed (from rename
    events and `userContentEdits`), so an issue reads the same in the corpus
    as in its eval case, and fill in issues the corpus fetch missed (for
    example with `--since`). The other issues have their current title and
    body: the corpus fetch has no history, and they are older references
    rather than cases, so a later retitle there gives less away.
    """
    mapping = load_mapping(repo)
    raw_dir = fetch.repo_dir(repo, data_dir)
    out = corpus_path(repo, data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)

    issues: dict[int, tuple[str, str, str, str]] = {}
    corpus_pages = raw_dir / "corpus" / "pages.jsonl"
    if corpus_pages.exists():
        for i in raw_issues(corpus_pages):
            issues[i["number"]] = (i["title"], i.get("body") or "", i["createdAt"], i["state"])
    eligible_pages = raw_dir / "eligible" / "pages.jsonl"
    if eligible_pages.exists():
        for i in raw_issues(eligible_pages):
            issues[i["number"]] = (original_title(i)[0], original_body(i)[0], i["createdAt"], i["state"])

    releases = json.loads((raw_dir / "releases.json").read_text())
    templates = json.loads((raw_dir / "templates.json").read_text())

    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    try:
        db.executescript(SCHEMA)
        db.executemany(
            "INSERT INTO issues VALUES (?, ?, ?, ?, ?)",
            (
                (n, title, strip_slash_commands(body, mapping.slash_commands), created, state)
                for n, (title, body, created, state) in sorted(issues.items())
            ),
        )
        db.execute("INSERT INTO issues_fts(issues_fts) VALUES('rebuild')")
        db.executemany(
            "INSERT OR REPLACE INTO releases VALUES (?, ?, ?)",
            (
                (r["tag"], r["published_at"], int(r["prerelease"]))
                for r in releases
                if r.get("published_at") and not r.get("draft")
            ),
        )
        db.executemany("INSERT INTO templates VALUES (?, ?)", sorted(templates.items()))
        db.commit()
        db.execute("VACUUM")
    finally:
        db.close()
    tmp.replace(out)
    return out, {"issues": len(issues), "releases": len(releases), "templates": len(templates)}

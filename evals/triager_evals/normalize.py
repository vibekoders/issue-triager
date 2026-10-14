"""Turns raw GitHub pages into eval records: inputs, expected answers, metadata.

Reads `data/raw/<owner>__<name>/eligible/pages.jsonl` and writes
`data/normalized/<name>.jsonl`, one record per eligible issue, newest first:

    {"issue": ..., "repo": ..., "expected": ..., "metadata": ...}

`issue` and `repo` are exactly triage-core's `Issue` and `Repo`: what a
strategy gets. Everything that shows how the issue was triaged stays in
`expected` and `metadata`. It also builds the search corpus (see `corpus`).
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from . import fetch

REPOS_DIR = Path(__file__).resolve().parent.parent / "repos"

SEVERITIES = ["critical", "important_soon", "important_longterm", "backlog"]

# The creation snapshot in `userContentEdits` is stamped with the issue's
# creation time; allow a little clock skew.
ORIGINAL_EDIT_SKEW_S = 60

# A bot applying a label this soon after creation is acting on the body's
# commands or the template, not on a triager's comment.
AUTHOR_LABEL_WINDOW_S = 15 * 60

EARLY_COMMENTS = 5
EARLY_COMMENT_CHARS = 2000


@dataclass(frozen=True)
class Mapping:
    """A repo's label mapping, from `repos/<name>.yaml`."""

    raw: dict[str, Any]

    def __getattr__(self, key: str) -> Any:
        try:
            return self.raw[key]
        except KeyError:
            raise AttributeError(key) from None

    @property
    def bot_logins(self) -> set[str]:
        return set(self.raw.get("bots", []))

    def is_context_label(self, name: str) -> bool:
        return name.startswith(tuple(self.context_label_prefixes))


@cache
def load_mapping(repo: str) -> Mapping:
    """The mapping whose `repo` (owner/name) or `name` matches."""
    for path in sorted(REPOS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        if repo in (raw.get("repo"), raw.get("name")):
            return Mapping(raw)
    raise SystemExit(f"no label mapping for {repo} in {REPOS_DIR}")


# --- Slash commands ---------------------------------------------------------

# Prow's label plugin: `/<cmd> <arg>` adds the label `<cmd>/<arg>`.
LABEL_COMMANDS = ("kind", "sig", "area", "priority", "triage", "lifecycle", "committee", "wg", "language")


@cache
def _command_re(commands: tuple[str, ...]) -> re.Pattern[str]:
    names = "|".join(re.escape(c).replace(r"\*", r"[\w-]+") for c in commands)
    # A command at the start of a line, optionally after whitespace, a quote
    # marker or an HTML comment opener (the bug template's comment block),
    # followed by whitespace or the end of the line: `/kind bug`, `/sig`,
    # but not `/home/user` or `/closed-loop`. Misspelled label commands
    # (`/sig-network`, `/triage-accepted`) do nothing in prow but give the
    # answer away all the same, so they match too, as `bad`.
    return re.compile(
        rf"^(?P<lead>[ \t]*(?:>[ \t]*)*(?:<!--[ \t]*)?)/"
        rf"(?:(?P<cmd>{names})(?=[ \t]|-->|$)|(?P<bad>{'|'.join(LABEL_COMMANDS)})[-/][\w-]+)"
        rf"(?P<rest>.*)$",
        re.IGNORECASE | re.MULTILINE,
    )


def strip_slash_commands(body: str, commands: Iterable[str]) -> str:
    """Removes prow command lines from a body.

    A command inside a one-line HTML comment (`<!-- /kind bug -->`) is removed
    with its comment. Runs of blank lines left behind are collapsed.
    """
    pattern = _command_re(tuple(commands))

    def repl(m: re.Match[str]) -> str:
        # Keep a comment opener when the comment continues on later lines.
        if "<!--" in m["lead"] and "-->" not in m["rest"]:
            return m["lead"].rstrip()
        return "\x00"

    out = pattern.sub(repl, body.replace("\r\n", "\n"))
    out = "\n".join(line for line in out.split("\n") if line != "\x00")
    out = re.sub(r"<!--\s*-->", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def body_commands(body: str, commands: Iterable[str]) -> list[tuple[str, list[str]]]:
    """The (command, args) pairs found in a body, e.g. ("kind", ["bug"])."""
    pattern = _command_re(tuple(commands))
    found = []
    for m in pattern.finditer(body.replace("\r\n", "\n")):
        if m["bad"]:
            continue
        rest = m["rest"].split("-->")[0]
        found.append((m["cmd"].lower(), rest.split()))
    return found


def labels_from_commands(cmds: list[tuple[str, list[str]]]) -> set[str]:
    """The labels prow would add for these commands."""
    labels = set()
    for cmd, args in cmds:
        if cmd in LABEL_COMMANDS:
            labels |= {f"{cmd}/{a.lower()}" for a in args}
        elif cmd == "label":
            labels |= {a for a in args}
        elif cmd == "good-first-issue":
            labels.add("good first issue")
        elif cmd == "help":
            labels.add("help wanted")
    return labels


# --- Raw issue helpers ------------------------------------------------------


def _ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _login(actor: dict[str, Any] | None) -> str | None:
    return (actor or {}).get("login")


def is_bot(actor: dict[str, Any] | None, mapping: Mapping) -> bool:
    if not actor:
        return False
    return actor.get("__typename") == "Bot" or actor.get("login") in mapping.bot_logins


def events(raw: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [n for n in raw["timelineItems"]["nodes"] if n and n["__typename"] == kind]


def original_body(raw: dict[str, Any]) -> tuple[str, bool, bool]:
    """The body as it was filed, whether it differs from the current one, and
    whether the filed body is known.

    `userContentEdits(last: 1)` is the oldest edit, which for an edited issue
    is the creation snapshot. When the issue was edited but that snapshot is
    gone (its content was deleted, as for security placeholder issues later
    filled in with the advisory), only the current body is left: it is
    returned as edited, with the filed body unknown.
    """
    body = raw.get("body") or ""
    edits = raw.get("userContentEdits") or {}
    for e in edits.get("nodes") or []:
        if e and e.get("diff") is not None and abs(_ts(e["editedAt"]) - _ts(raw["createdAt"])) <= ORIGINAL_EDIT_SKEW_S:
            return e["diff"], e["diff"] != body, True
    if edits.get("totalCount"):
        return body, True, False
    return body, False, True


def original_title(raw: dict[str, Any]) -> tuple[str, bool]:
    """The title as it was filed, and whether it differs from the current one.

    Triagers often retitle issues (prow's `/retitle`), adding the sig, the
    kind or the diagnosis; the oldest rename's `previousTitle` is the title
    before any of that.
    """
    renames = sorted(events(raw, "RenamedTitleEvent"), key=lambda e: e["createdAt"])
    title = renames[0].get("previousTitle") if renames else None
    if title is None:
        return raw["title"], False
    return title, title != raw["title"]


@dataclass
class LabelHistory:
    """Label state over time, replayed from labeled/unlabeled events."""

    final: set[str]
    ever: set[str] = field(default_factory=set)
    # label -> the event that last applied it
    applied_by: dict[str, dict[str, Any]] = field(default_factory=dict)
    removed: set[str] = field(default_factory=set)
    timeline: list[tuple[float, str, str]] = field(default_factory=list)

    @classmethod
    def of(cls, raw: dict[str, Any]) -> LabelHistory:
        final = {lb["name"] for lb in raw["labels"]["nodes"]}
        h = cls(final=final, ever=set(final))
        evs = [n for n in raw["timelineItems"]["nodes"] if n and n["__typename"] in ("LabeledEvent", "UnlabeledEvent")]
        for ev in sorted(evs, key=lambda e: e["createdAt"]):
            name = (ev.get("label") or {}).get("name")
            if not name:
                continue
            if ev["__typename"] == "LabeledEvent":
                h.ever.add(name)
                h.applied_by[name] = ev
                h.timeline.append((_ts(ev["createdAt"]), "+", name))
            else:
                h.removed.add(name)
                h.timeline.append((_ts(ev["createdAt"]), "-", name))
        return h

    def at(self, when: float) -> set[str]:
        """The labels set at a moment, from the events up to it.

        Labels with no events at all (applied before GitHub recorded them)
        count as set the whole time if they are still set.
        """
        with_events = {name for _, _, name in self.timeline}
        state = {lb for lb in self.final if lb not in with_events}
        for t, op, name in self.timeline:
            if t > when:
                break
            if op == "+":
                state.add(name)
            else:
                state.discard(name)
        return state


def label_provenance(raw: dict[str, Any], body: str, history: LabelHistory, mapping: Mapping) -> dict[str, str]:
    """Who chose each final triage label: the author or a triager.

    A label is the author's when the author applied it (issue templates add
    their label as the author), or when prow applied it shortly after the
    issue was filed and the body carries the matching command. A label
    without events is the author's only if the body has its command.
    """
    author = _login(raw.get("author"))
    created = _ts(raw["createdAt"])
    from_body = labels_from_commands(body_commands(body, mapping.slash_commands))
    out = {}
    for name in sorted(history.final):
        if not mapping.is_context_label(name) and name not in mapping.security_sensitive:
            continue
        ev = history.applied_by.get(name)
        if ev is None:
            out[name] = "author" if name in from_body else "triager"
            continue
        actor = ev.get("actor")
        if author and _login(actor) == author:
            out[name] = "author"
        elif is_bot(actor, mapping) and name in from_body and _ts(ev["createdAt"]) - created <= AUTHOR_LABEL_WINDOW_S:
            out[name] = "author"
        else:
            out[name] = "triager"
    return out


def last_close(raw: dict[str, Any]) -> dict[str, Any] | None:
    closes = sorted(events(raw, "ClosedEvent"), key=lambda e: e["createdAt"])
    return closes[-1] if closes else None


def fixing_prs(raw: dict[str, Any], repo: str) -> list[int]:
    """Merged PRs in the repo that closed or are linked as closing the issue."""
    prs = set()
    for pr in (raw.get("closedByPullRequestsReferences") or {}).get("nodes") or []:
        if pr and pr.get("merged") and pr["repository"]["nameWithOwner"] == repo:
            prs.add(pr["number"])
    close = last_close(raw)
    if raw["state"] == "CLOSED" and close and (closer := close.get("closer")):
        if closer.get("__typename") == "PullRequest" and closer.get("merged"):
            prs.add(closer["number"])
    return sorted(prs)


def closed_by_fix(raw: dict[str, Any], repo: str) -> bool:
    close = last_close(raw)
    if raw["state"] != "CLOSED":
        return False
    if close and (closer := close.get("closer")):
        if closer.get("__typename") == "Commit":
            return True
        if closer.get("__typename") == "PullRequest" and closer.get("merged"):
            return True
    return bool(fixing_prs(raw, repo))


_DUP_RE = re.compile(
    r"\b(?:duplicate|dup)\s+of\s*:?\s*"
    r"(?:(?:https://github\.com/)?([\w.-]+/[\w.-]+)(?:#|/issues/)|#)(\d+)",
    re.I,
)


def _number_in(ref: dict[str, Any] | None, repo: str) -> int | None:
    ref = ref or {}
    if (ref.get("repository") or {}).get("nameWithOwner") == repo:
        return ref.get("number")
    return None


def own_duplicate_events(raw: dict[str, Any], repo: str) -> list[dict[str, Any]]:
    """The duplicate markings of this issue that were not undone.

    A `MarkedAsDuplicateEvent` shows up on both issues' timelines, so on the
    canonical one it is about some other issue. Events fetched without their
    `duplicate` side (older dumps) count unless their canonical is this issue.
    """
    me = raw["number"]
    own = []
    for ev in sorted(
        events(raw, "MarkedAsDuplicateEvent") + events(raw, "UnmarkedAsDuplicateEvent"), key=lambda e: e["createdAt"]
    ):
        if "duplicate" in ev and _number_in(ev.get("duplicate"), repo) != me:
            continue
        canonical = ev.get("canonical") or {}
        if _number_in(canonical, repo) == me:
            continue
        if ev["__typename"] == "MarkedAsDuplicateEvent":
            own.append(ev)
        else:
            # an unmarking undoes the earlier marking with the same canonical
            key = (canonical.get("number"), (canonical.get("repository") or {}).get("nameWithOwner"))
            own = [
                e
                for e in own
                if (
                    (e.get("canonical") or {}).get("number"),
                    ((e.get("canonical") or {}).get("repository") or {}).get("nameWithOwner"),
                )
                != key
            ]
    return own


def event_duplicate_targets(raw: dict[str, Any], repo: str) -> list[int]:
    found = []
    for ev in own_duplicate_events(raw, repo):
        if n := _number_in(ev.get("canonical"), repo):
            found.append(n)
    return list(dict.fromkeys(found))


# Words that make a "duplicate of #N" a guess or a question, not a verdict.
_DUP_HEDGE_RE = re.compile(
    r"\b(?:might|maybe|may\s+be|possibly|perhaps|probably|sounds\s+like|is\s+this|is\s+it|could\s+(?:this|it|be)"
    r"|not\s+sure|i\s+think|i\s+guess|wonder(?:ing)?|further\s+investigation)\b",
    re.I,
)


def _sentence_around(text: str, start: int, end: int) -> str:
    """The sentence holding text[start:end], with its closing punctuation."""
    left = max(text.rfind("\n", 0, start), *(text.rfind(p, 0, start) for p in (". ", "! ", "? ")))
    m = re.compile(r"[.!?](?=\s|$)|\n").search(text, end)
    right = m.end() if m else len(text)
    return text[left + 1 : right].strip()


def comment_duplicate_targets(
    raw: dict[str, Any], repo: str, mapping: Mapping, *, confirmed: bool = False
) -> list[int]:
    """Issues people called the original in "duplicate of #N" comments.

    Most k8s duplicates are closed with `/close` and such a comment rather
    than GitHub's duplicate marker, so without this they look like ordinary
    closes. Only the fetched first comments are searched. A question or a
    guess ("is this a duplicate of #N?", "might be a duplicate of #N") only
    counts when the issue is `confirmed` a duplicate some other way (state
    reason, label, marker), which settles the question.
    """
    found = []
    for c in (raw.get("comments") or {}).get("nodes") or []:
        if not c or is_bot(c.get("author"), mapping):
            continue
        text = (c.get("body") or "").replace("\r\n", "\n")
        for m in _DUP_RE.finditer(text):
            if m[1] not in (None, repo) or int(m[2]) == raw["number"]:
                continue
            sentence = _sentence_around(text, m.start(), m.end())
            if not confirmed and (sentence.endswith("?") or _DUP_HEDGE_RE.search(sentence)):
                continue
            found.append(int(m[2]))
    return list(dict.fromkeys(found))


def duplicate_targets(raw: dict[str, Any], repo: str, mapping: Mapping, *, confirmed: bool = False) -> list[int]:
    """Canonical issues from duplicate events, else from comments.

    Only issues filed before this one: the tools can't find a later one (they
    search as of the issue's creation), so it can't be scored. Issue numbers
    in a repo grow with creation time, which makes that a number comparison.
    """
    from_comments = comment_duplicate_targets(raw, repo, mapping, confirmed=confirmed)
    for targets in (event_duplicate_targets(raw, repo), from_comments):
        if targets:
            return [n for n in targets if n < raw["number"]]
    return []


def duplicate_confirmed(raw: dict[str, Any], history: LabelHistory, mapping: Mapping, repo: str) -> bool:
    """Closed as a duplicate by state reason, GitHub's marker or the duplicate
    label as of the close or now (one added and soon removed was a mistake)."""
    close = last_close(raw)
    at_close = history.at(_ts(close["createdAt"])) if close else history.final
    return (
        raw.get("stateReason") == "DUPLICATE"
        or bool((at_close | history.final) & set(mapping.duplicate))
        or bool(own_duplicate_events(raw, repo))
    )


def close_outcome(raw: dict[str, Any], history: LabelHistory, mapping: Mapping, repo: str) -> str:
    """One label for how the issue ended, in PLAN.md's order of precedence.

    Prow closes issues for everyone who comments `/close`, so the closing
    actor is a bot for humans and the lifecycle bot alike; a close is the
    lifecycle bot's only when `lifecycle/rotten` was set at that moment.
    """
    if raw["state"] == "OPEN":
        return "open"
    close = last_close(raw)
    at_close = history.at(_ts(close["createdAt"])) if close else history.final
    reason = raw.get("stateReason")

    if closed_by_fix(raw, repo):
        return "fixed"
    if duplicate_confirmed(raw, history, mapping, repo):
        return "duplicate"
    if mapping.needs_information in at_close:
        return "needs_info_timeout"
    if mapping.lifecycle["rotten"] in at_close and (close is None or is_bot(close.get("actor"), mapping)):
        return "stale"
    if comment_duplicate_targets(raw, repo, mapping):
        return "duplicate"
    if (
        at_close & set(mapping.invalid)
        or (raw.get("activeLockReason") or "").upper() in ("SPAM", "OFF_TOPIC")
        or reason == "NOT_PLANNED"
    ):
        return "invalid"
    return "completed_other"


def detect_template(body: str, mapping: Mapping) -> str | None:
    """The issue template the body follows, from its first heading."""
    for line in body.splitlines():
        text = line.strip().lstrip("#*").strip().rstrip(":*").strip().lower()
        if not text:
            continue
        for name, starts in mapping.templates.items():
            if any(text.startswith(s) for s in starts):
                return name
        if line.lstrip().startswith(("#", "**")):
            return None
    return None


# --- Records -----------------------------------------------------------------


def _version_key(tag: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", tag)[:3])


@dataclass(frozen=True)
class RepoContext:
    name: str
    labels: list[dict[str, Any]]
    # (published_at, tag), oldest first, stable releases only
    releases: list[tuple[str, str]]

    @classmethod
    def load(cls, repo: str, mapping: Mapping, data_dir: Path | None = None) -> RepoContext:
        d = fetch.repo_dir(repo, data_dir)
        labels = json.loads((d / "labels.json").read_text())
        releases = json.loads((d / "releases.json").read_text())
        return cls(
            name=repo,
            labels=sorted(
                (
                    {"name": lb["name"], "description": lb.get("description")}
                    for lb in labels
                    if mapping.is_context_label(lb["name"])
                ),
                key=lambda lb: lb["name"],
            ),
            releases=sorted(
                (r["published_at"], r["tag"])
                for r in releases
                if r.get("published_at") and not r["prerelease"] and not r.get("draft")
            ),
        )

    def latest_release(self, as_of: str) -> str | None:
        """The highest version released before `as_of`.

        Not the most recently published one: patch releases of older
        branches (v1.30.13 a day after v1.33.1) would make a current report
        look outdated.
        """
        best: tuple[tuple[int, ...], str] | None = None
        for published, tag in self.releases:
            if published >= as_of:
                break
            key = _version_key(tag)
            if best is None or key >= best[0]:
                best = (key, tag)
        return best[1] if best else None

    def snapshot(self, as_of: str) -> dict[str, Any]:
        return {"name": self.name, "labels": self.labels, "latest_release": self.latest_release(as_of), "as_of": as_of}


def is_eligible(raw: dict[str, Any], repo: str) -> bool:
    return eligibility(raw, load_mapping(repo)) is None


def eligibility(raw: dict[str, Any], mapping: Mapping, *, closed_untriaged: bool = False) -> str | None:
    """Why an issue is not eligible, or None when it is.

    Many k8s issues are closed (fixed, duplicate, stale) while `needs-triage`
    is still set, because nobody ran `/triage accepted` first. PLAN.md
    excludes them; `closed_untriaged` keeps them, since closing settles the
    issue, at the price of labels no triager confirmed.
    """
    labels = {lb["name"] for lb in raw["labels"]["nodes"]}
    if labels & set(mapping.needs_triage) and not (closed_untriaged and raw["state"] == "CLOSED"):
        return "needs-triage"
    if not labels & set(mapping.kind):
        return "no kind"
    return None


def expected_answers(
    raw: dict[str, Any], history: LabelHistory, mapping: Mapping, repo: str, outcome: str
) -> dict[str, Any]:
    labels = history.final
    kinds = list(dict.fromkeys(mapping.kind[lb] for lb in sorted(labels) if lb in mapping.kind))
    areas = sorted(lb for lb in labels if lb.startswith(mapping.area_prefix))

    severity = None
    if not labels & set(mapping.severity_undecided):
        levels = [mapping.severity[lb] for lb in labels if lb in mapping.severity]
        # Two priorities happen when one wasn't removed; the higher one is
        # what triagers acted on.
        severity = min(levels, key=SEVERITIES.index) if levels else None

    if labels & set(mapping.is_regression):
        is_regression: bool | None = True
    elif set(kinds) & set(mapping.regression_kinds):
        is_regression = False
    else:
        is_regression = None

    enough_info = None
    if "bug" in kinds:
        if history.ever & set(mapping.not_enough_info):
            enough_info = False
        elif closed_by_fix(raw, repo):
            enough_info = True

    duplicate_of = None
    if outcome == "duplicate":
        confirmed = duplicate_confirmed(raw, history, mapping, repo)
        duplicate_of = duplicate_targets(raw, repo, mapping, confirmed=confirmed) or None

    return {
        "kind": kinds or None,
        "area": areas or None,
        "severity": severity,
        "is_regression": is_regression,
        "security_sensitive": bool(labels & set(mapping.security_sensitive)),
        "enough_info": enough_info,
        "duplicate_of": duplicate_of,
        "close_outcome": outcome,
    }


def accepted_by_other(raw: dict[str, Any], history: LabelHistory, mapping: Mapping) -> bool:
    """Whether someone other than the author accepted the triage.

    Prow applies `triage/accepted` for whoever commented `/triage accepted`,
    so the label event's actor doesn't say who that was; the comment does.
    Only the first comments are fetched, so a late acceptance is missed.
    """
    if mapping.accepted not in history.final:
        return False
    author = _login(raw.get("author"))
    ev = history.applied_by.get(mapping.accepted)
    if ev and not is_bot(ev.get("actor"), mapping) and _login(ev.get("actor")) != author:
        return True
    for c in (raw.get("comments") or {}).get("nodes") or []:
        if not c or is_bot(c.get("author"), mapping) or _login(c.get("author")) == author:
            continue
        if mapping.accepted in labels_from_commands(body_commands(c.get("body") or "", mapping.slash_commands)):
            return True
    return False


def early_maintainer_comments(raw: dict[str, Any], mapping: Mapping) -> list[str]:
    """The first comments by people other than the author, bots left out."""
    author = _login(raw.get("author"))
    out = []
    for c in (raw.get("comments") or {}).get("nodes") or []:
        if not c or is_bot(c.get("author"), mapping) or _login(c.get("author")) == author:
            continue
        out.append((c.get("body") or "").replace("\r\n", "\n")[:EARLY_COMMENT_CHARS])
        if len(out) == EARLY_COMMENTS:
            break
    return out


def normalize_issue(raw: dict[str, Any], ctx: RepoContext, mapping: Mapping) -> dict[str, Any]:
    repo = ctx.name
    body, edited, body_known = original_body(raw)
    title, title_edited = original_title(raw)
    history = LabelHistory.of(raw)
    outcome = close_outcome(raw, history, mapping, repo)
    provenance = label_provenance(raw, body, history, mapping)
    close = last_close(raw) if raw["state"] == "CLOSED" else None
    kind_labels = set(mapping.kind)
    kind_provenance = {p for lb, p in provenance.items() if lb in kind_labels}

    cross_refs = sorted(
        {
            ev["source"]["number"]
            for ev in events(raw, "CrossReferencedEvent")
            if (ev.get("source") or {}).get("number")
            and (ev["source"].get("repository") or {}).get("nameWithOwner") == repo
        }
    )
    transferred_from = [
        ev["fromRepository"]["nameWithOwner"] for ev in events(raw, "TransferredEvent") if ev.get("fromRepository")
    ]
    lifecycle = sorted(lb for lb in history.final if lb.startswith("lifecycle/"))

    return {
        "issue": {
            "number": raw["number"],
            "title": title,
            "body": strip_slash_commands(body, mapping.slash_commands),
            "author_association": raw["authorAssociation"],
            "created_at": raw["createdAt"],
        },
        "repo": ctx.snapshot(raw["createdAt"]),
        "expected": expected_answers(raw, history, mapping, repo, outcome),
        "metadata": {
            "created_at": raw["createdAt"],
            "state": raw["state"],
            "state_reason": raw.get("stateReason"),
            "closed_at": raw.get("closedAt"),
            "closed_by": _login(close.get("actor")) if close else None,
            "closed_by_bot": bool(close and is_bot(close.get("actor"), mapping)),
            "stale": outcome == "stale"
            or bool(set(lifecycle) & {mapping.lifecycle["stale"], mapping.lifecycle["rotten"]}),
            "lifecycle": lifecycle,
            "labels": sorted(history.final),
            "label_provenance": provenance,
            "triager_verified": "triager" in kind_provenance or accepted_by_other(raw, history, mapping),
            "kind_changed": bool(history.removed & kind_labels)
            or len(history.ever & kind_labels) > len(history.final & kind_labels),
            "needed_information": bool(history.ever & set(mapping.not_enough_info)),
            "author_is_bot": is_bot(raw.get("author"), mapping),
            "body_edited": edited,
            # edited, but the body as filed is lost: the case shows the current body
            "original_body_unknown": not body_known,
            "title_edited": title_edited,
            "comments": raw["comments"]["totalCount"],
            "participants": (raw.get("participants") or {}).get("totalCount"),
            "reactions_plus_one": raw["reactions"]["totalCount"],
            "milestone": (raw.get("milestone") or {}).get("title"),
            "template": detect_template(body, mapping),
            "locked_reason": raw.get("activeLockReason"),
            "fixing_prs": fixing_prs(raw, repo),
            "cross_refs": cross_refs,
            "transferred_from": transferred_from[0] if transferred_from else None,
            "timeline_truncated": bool(raw.get("timeline_truncated")),
            "early_maintainer_comments": early_maintainer_comments(raw, mapping),
        },
    }


def raw_issues(path: Path, *, backfill: bool = True) -> Iterator[dict[str, Any]]:
    """Issues from a pages file, each number once (a resumed fetch can repeat a page).

    With `backfill`, the events in the `backfill.jsonl` next to the pages
    file (see `fetch.fetch_backfill`) replace the issue's own events of the
    backfilled types.
    """
    extra: dict[int, dict[str, Any]] = {}
    extra_path = fetch.backfill_path(path)
    if backfill and extra_path.exists():
        with extra_path.open() as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    extra[rec["number"]] = rec
    seen: set[int] = set()
    with path.open() as f:
        for line in f:
            for issue in json.loads(line)["issues"]:
                if issue and issue["number"] not in seen:
                    seen.add(issue["number"])
                    if (rec := extra.get(issue["number"])) is not None:
                        apply_backfill(issue, rec)
                    yield issue


def apply_backfill(issue: dict[str, Any], rec: dict[str, Any]) -> None:
    tl = issue["timelineItems"]
    kept = [n for n in tl["nodes"] if not (n and n["__typename"] in fetch.BACKFILL_TYPENAMES)]
    tl["nodes"] = kept + [n for n in rec["nodes"] if n]
    issue["timeline_truncated"] = bool(issue.get("timeline_truncated") or rec.get("truncated"))


def normalize(
    repo: str, limit: int | None, data_dir: Path | None = None, *, closed_untriaged: bool = False
) -> tuple[Path, dict[str, int]]:
    mapping = load_mapping(repo)
    ctx = RepoContext.load(repo, mapping, data_dir)
    src = fetch.repo_dir(repo, data_dir) / "eligible" / "pages.jsonl"
    out_dir = (data_dir or fetch.DATA_DIR) / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{mapping.name}.jsonl"

    records = []
    skipped: dict[str, int] = {}
    for raw in raw_issues(src):
        if (why := eligibility(raw, mapping, closed_untriaged=closed_untriaged)) is not None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        records.append(normalize_issue(raw, ctx, mapping))
    records.sort(key=lambda r: r["issue"]["created_at"], reverse=True)
    if limit is not None:
        skipped["over limit"] = max(0, len(records) - limit)
        records = records[:limit]

    tmp = out.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    return out, {"written": len(records), **{f"skipped: {k}": v for k, v in skipped.items()}}


def stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts worth eyeballing after a normalize run."""
    from collections import Counter

    exp = [r["expected"] for r in records]
    meta = [r["metadata"] for r in records]
    dates = sorted(r["issue"]["created_at"] for r in records)
    return {
        "records": len(records),
        "date_range": [dates[0], dates[-1]] if dates else None,
        "first_kind": dict(Counter(e["kind"][0] for e in exp if e["kind"]).most_common()),
        "any_kind": dict(Counter(k for e in exp for k in e["kind"] or []).most_common()),
        "multi_kind": sum(len(e["kind"] or []) > 1 for e in exp),
        "close_outcome": dict(Counter(e["close_outcome"] for e in exp).most_common()),
        "severity": dict(Counter(e["severity"] for e in exp).most_common()),
        "has_area": sum(e["area"] is not None for e in exp),
        "multi_area": sum(len(e["area"] or []) > 1 for e in exp),
        "is_regression": dict(Counter(e["is_regression"] for e in exp).most_common()),
        "security_sensitive": sum(bool(e["security_sensitive"]) for e in exp),
        "enough_info": dict(Counter(e["enough_info"] for e in exp).most_common()),
        "has_duplicate_of": sum(e["duplicate_of"] is not None for e in exp),
        "triager_verified": sum(m["triager_verified"] for m in meta),
        "kind_changed": sum(m["kind_changed"] for m in meta),
        "stale": sum(m["stale"] for m in meta),
        "body_edited": sum(m["body_edited"] for m in meta),
        "original_body_unknown": sum(m["original_body_unknown"] for m in meta),
        "title_edited": sum(m["title_edited"] for m in meta),
        "timeline_truncated": sum(m["timeline_truncated"] for m in meta),
        "template": dict(Counter(m["template"] for m in meta).most_common()),
        "with_latest_release": sum(r["repo"]["latest_release"] is not None for r in records),
    }


def main() -> None:
    from . import corpus

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="kubernetes/kubernetes")
    p.add_argument("--limit", type=int, default=3000, help="keep the newest N eligible issues (0: all)")
    p.add_argument("--closed-untriaged", action="store_true", help="keep closed issues that still carry needs-triage")
    p.add_argument("--skip-corpus", action="store_true", help="don't (re)build the search corpus")
    p.add_argument("--data-dir", type=Path, default=None)
    args = p.parse_args()

    out, counts = normalize(args.repo, args.limit or None, args.data_dir, closed_untriaged=args.closed_untriaged)
    print(f"{out}: {json.dumps(counts)}")
    with out.open() as f:
        print(json.dumps(stats([json.loads(line) for line in f]), indent=1))
    if not args.skip_corpus:
        db, n = corpus.build(args.repo, args.data_dir)
        print(f"{db}: {json.dumps(n)}")


if __name__ == "__main__":
    main()

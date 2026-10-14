"""Builders for raw issues shaped like the GraphQL responses `fetch` saves."""

from __future__ import annotations

from typing import Any

import pytest

from triager_evals.normalize import load_mapping

REPO = "kubernetes/kubernetes"
BOT = {"login": "kubernetes-prow", "__typename": "Bot"}
AUTHOR = {"login": "alice", "__typename": "User"}
TRIAGER = {"login": "bob", "__typename": "User"}


@pytest.fixture
def mapping():
    return load_mapping(REPO)


def labeled(name: str, at: str = "2025-03-01T12:00:30Z", actor: dict | None = None, op: str = "LabeledEvent") -> dict:
    return {"__typename": op, "createdAt": at, "actor": actor or BOT, "label": {"name": name}}


def unlabeled(name: str, at: str, actor: dict | None = None) -> dict:
    return labeled(name, at, actor, "UnlabeledEvent")


def closed(
    at: str = "2025-04-01T00:00:00Z", actor: dict | None = None, reason: str = "COMPLETED", closer: dict | None = None
) -> dict:
    return {
        "__typename": "ClosedEvent",
        "createdAt": at,
        "actor": actor or BOT,
        "stateReason": reason,
        "closer": closer,
    }


def raw_issue(
    labels: list[str],
    *,
    number: int = 1000,
    body: str = "### What happened?\n\nIt broke.",
    timeline: list[dict] | None = None,
    state: str = "OPEN",
    state_reason: str | None = None,
    closed_by_prs: list[dict] | None = None,
    comments: list[dict] | None = None,
    lock: str | None = None,
    edits: list[dict] | None = None,
) -> dict[str, Any]:
    return {
        "number": number,
        "title": "Something broke",
        "body": body,
        "createdAt": "2025-03-01T12:00:00Z",
        "authorAssociation": "NONE",
        "author": AUTHOR,
        "state": state,
        "stateReason": state_reason,
        "closedAt": "2025-04-01T00:00:00Z" if state == "CLOSED" else None,
        "locked": lock is not None,
        "activeLockReason": lock,
        "milestone": None,
        "comments": {"totalCount": len(comments or []), "nodes": comments or []},
        "participants": {"totalCount": 1},
        "reactions": {"totalCount": 0},
        "labels": {"nodes": [{"name": n} for n in labels]},
        "userContentEdits": {"totalCount": len(edits or []), "nodes": edits or []},
        "closedByPullRequestsReferences": {"nodes": closed_by_prs or []},
        "timelineItems": {"pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": timeline or []},
    }

import pytest
from conftest import AUTHOR, REPO, TRIAGER, closed, labeled, raw_issue, unlabeled

from triager_evals.normalize import (
    LabelHistory,
    RepoContext,
    close_outcome,
    detect_template,
    duplicate_targets,
    eligibility,
    expected_answers,
    label_provenance,
    normalize_issue,
    original_body,
    original_title,
)


def dup_event(canonical: int, duplicate: int, at: str = "2025-03-02T00:00:00Z", op: str = "MarkedAsDuplicateEvent"):
    return {
        "__typename": op,
        "createdAt": at,
        "canonical": {"number": canonical, "repository": {"nameWithOwner": REPO}},
        "duplicate": {"number": duplicate, "repository": {"nameWithOwner": REPO}},
    }


def comment(body: str, author: dict = TRIAGER) -> dict:
    return {"author": author, "authorAssociation": "MEMBER", "body": body, "createdAt": "2025-03-02T00:00:00Z"}


def outcome(raw, mapping):
    return close_outcome(raw, LabelHistory.of(raw), mapping, REPO)


class TestCloseOutcome:
    def test_open(self, mapping):
        assert outcome(raw_issue(["kind/bug"]), mapping) == "open"

    def test_fixed_by_pr_closer(self, mapping):
        pr = {"__typename": "PullRequest", "number": 5, "merged": True, "repository": {"nameWithOwner": REPO}}
        raw = raw_issue(["kind/bug"], state="CLOSED", state_reason="COMPLETED", timeline=[closed(closer=pr)])
        assert outcome(raw, mapping) == "fixed"

    def test_fixed_by_commit(self, mapping):
        raw = raw_issue(["kind/bug"], state="CLOSED", timeline=[closed(closer={"__typename": "Commit", "oid": "abc"})])
        assert outcome(raw, mapping) == "fixed"

    def test_fixed_by_linked_pr_even_if_not_planned(self, mapping):
        pr = {"number": 7, "merged": True, "repository": {"nameWithOwner": REPO}}
        raw = raw_issue(
            ["kind/bug"],
            state="CLOSED",
            state_reason="NOT_PLANNED",
            timeline=[closed(reason="NOT_PLANNED")],
            closed_by_prs=[pr],
        )
        assert outcome(raw, mapping) == "fixed"

    def test_unmerged_or_foreign_pr_is_not_a_fix(self, mapping):
        prs = [
            {"number": 7, "merged": False, "repository": {"nameWithOwner": REPO}},
            {"number": 8, "merged": True, "repository": {"nameWithOwner": "kubernetes/website"}},
        ]
        raw = raw_issue(["kind/bug"], state="CLOSED", state_reason="COMPLETED", timeline=[closed()], closed_by_prs=prs)
        assert outcome(raw, mapping) == "completed_other"

    @pytest.mark.parametrize(
        "labels,reason,extra",
        [
            (["kind/bug"], "DUPLICATE", []),
            (["kind/bug", "triage/duplicate"], "COMPLETED", []),
            (
                ["kind/bug"],
                "COMPLETED",
                [
                    {
                        "__typename": "MarkedAsDuplicateEvent",
                        "createdAt": "2025-03-02T00:00:00Z",
                        "canonical": {"number": 9, "repository": {"nameWithOwner": REPO}},
                    }
                ],
            ),
        ],
    )
    def test_duplicate(self, mapping, labels, reason, extra):
        raw = raw_issue(labels, state="CLOSED", state_reason=reason, timeline=[*extra, closed(reason=reason)])
        assert outcome(raw, mapping) == "duplicate"

    def test_canonical_side_of_a_duplicate_marking_is_not_a_duplicate(self, mapping):
        # Another issue was marked a duplicate of this one; the event shows up here too.
        ev = dup_event(canonical=1000, duplicate=2000)
        raw = raw_issue(["kind/bug"], state="CLOSED", state_reason="COMPLETED", timeline=[ev, closed()])
        assert outcome(raw, mapping) == "completed_other"
        assert duplicate_targets(raw, REPO, mapping) == []
        # Older dumps have no `duplicate` side: a self-canonical event still doesn't count.
        del ev["duplicate"]
        assert outcome(raw, mapping) == "completed_other"

    def test_unmarked_duplicate_is_not_a_duplicate(self, mapping):
        tl = [
            dup_event(canonical=9, duplicate=1000),
            dup_event(canonical=9, duplicate=1000, at="2025-03-03T00:00:00Z", op="UnmarkedAsDuplicateEvent"),
            closed(),
        ]
        raw = raw_issue(["kind/bug"], state="CLOSED", state_reason="COMPLETED", timeline=tl)
        assert outcome(raw, mapping) == "completed_other"

    def test_duplicate_label_removed_before_close_is_not_a_duplicate(self, mapping):
        tl = [
            labeled("triage/duplicate", "2025-03-02T00:00:00Z", actor=TRIAGER),
            unlabeled("triage/duplicate", "2025-03-02T00:10:00Z", actor=TRIAGER),
            closed(),
        ]
        raw = raw_issue(["kind/bug"], state="CLOSED", state_reason="COMPLETED", timeline=tl)
        assert outcome(raw, mapping) == "completed_other"

    @pytest.mark.parametrize(
        "text",
        [
            "@pacoxu is this issue duplicate of #129?",
            "This sounds like a duplicate of #60",
            "I think this might be a duplicate of #116 but needs further investigation.",
            "Maybe a duplicate of #42.",
        ],
    )
    def test_questions_and_guesses_are_not_duplicate_verdicts(self, mapping, text):
        raw = raw_issue(
            ["kind/bug"], state="CLOSED", state_reason="COMPLETED", timeline=[closed()], comments=[comment(text)]
        )
        assert outcome(raw, mapping) == "completed_other"

    def test_a_guess_names_the_target_once_the_label_confirms_the_duplicate(self, mapping):
        raw = raw_issue(
            ["kind/bug", "triage/duplicate"],
            state="CLOSED",
            state_reason="COMPLETED",
            timeline=[closed()],
            comments=[comment("I think this is a duplicate of #77")],
        )
        h = LabelHistory.of(raw)
        assert expected_answers(raw, h, mapping, REPO, close_outcome(raw, h, mapping, REPO))["duplicate_of"] == [77]

    def test_duplicate_verdict_next_to_a_question(self, mapping):
        text = "Thanks for the report. Duplicate of #77. Can you follow up there?"
        raw = raw_issue(
            ["kind/bug"], state="CLOSED", state_reason="COMPLETED", timeline=[closed()], comments=[comment(text)]
        )
        assert outcome(raw, mapping) == "duplicate"
        assert duplicate_targets(raw, REPO, mapping) == [77]

    def test_duplicate_of_a_later_issue_has_no_target(self, mapping):
        raw = raw_issue(
            ["kind/bug"],
            state="CLOSED",
            state_reason="COMPLETED",
            timeline=[closed()],
            comments=[comment("/close\nDuplicate of #1500")],
        )
        assert outcome(raw, mapping) == "duplicate"
        assert duplicate_targets(raw, REPO, mapping) == []
        ev = dup_event(canonical=1500, duplicate=1000)
        raw = raw_issue(["kind/bug"], state="CLOSED", state_reason="DUPLICATE", timeline=[ev, closed()])
        assert duplicate_targets(raw, REPO, mapping) == []

    def test_needs_info_timeout_uses_labels_at_close_time(self, mapping):
        tl = [
            labeled("triage/needs-information", "2025-03-02T00:00:00Z"),
            closed("2025-04-01T00:00:00Z", reason="NOT_PLANNED"),
        ]
        raw = raw_issue(
            ["kind/bug", "triage/needs-information"], state="CLOSED", state_reason="NOT_PLANNED", timeline=tl
        )
        assert outcome(raw, mapping) == "needs_info_timeout"
        # Removed before the close: not a timeout.
        tl = [
            labeled("triage/needs-information", "2025-03-02T00:00:00Z"),
            unlabeled("triage/needs-information", "2025-03-03T00:00:00Z"),
            closed(reason="COMPLETED"),
        ]
        raw = raw_issue(["kind/bug"], state="CLOSED", state_reason="COMPLETED", timeline=tl)
        assert outcome(raw, mapping) == "completed_other"

    def test_stale(self, mapping):
        tl = [
            labeled("lifecycle/stale", "2025-06-01T00:00:00Z"),
            labeled("lifecycle/rotten", "2025-07-01T00:00:00Z"),
            closed("2025-08-01T00:00:00Z", reason="NOT_PLANNED"),
        ]
        raw = raw_issue(["kind/bug", "lifecycle/rotten"], state="CLOSED", state_reason="NOT_PLANNED", timeline=tl)
        assert outcome(raw, mapping) == "stale"

    def test_rotten_but_closed_by_a_person_directly_is_not_stale(self, mapping):
        tl = [
            labeled("lifecycle/rotten", "2025-07-01T00:00:00Z"),
            closed("2025-08-01T00:00:00Z", actor=TRIAGER, reason="COMPLETED"),
        ]
        raw = raw_issue(["kind/bug", "lifecycle/rotten"], state="CLOSED", state_reason="COMPLETED", timeline=tl)
        assert outcome(raw, mapping) == "completed_other"

    @pytest.mark.parametrize(
        "labels,reason,lock",
        [
            (["kind/support"], "COMPLETED", None),
            (["kind/bug", "triage/unresolved"], "COMPLETED", None),
            (["kind/bug"], "COMPLETED", "SPAM"),
            (["kind/bug"], "NOT_PLANNED", None),
        ],
    )
    def test_invalid(self, mapping, labels, reason, lock):
        raw = raw_issue(labels, state="CLOSED", state_reason=reason, timeline=[closed(reason=reason)], lock=lock)
        assert outcome(raw, mapping) == "invalid"

    def test_completed_other(self, mapping):
        raw = raw_issue(["kind/feature"], state="CLOSED", state_reason="COMPLETED", timeline=[closed()])
        assert outcome(raw, mapping) == "completed_other"


class TestProvenance:
    def prov(self, raw, mapping):
        body, _, _ = original_body(raw)
        return label_provenance(raw, body, LabelHistory.of(raw), mapping)

    def test_template_label_applied_by_author(self, mapping):
        raw = raw_issue(["kind/bug"], timeline=[labeled("kind/bug", actor=AUTHOR)])
        assert self.prov(raw, mapping) == {"kind/bug": "author"}

    def test_body_command_applied_by_bot_right_away(self, mapping):
        raw = raw_issue(
            ["kind/bug", "sig/node"],
            body="It broke\n/kind bug\n/sig node",
            timeline=[labeled("kind/bug"), labeled("sig/node")],
        )
        assert self.prov(raw, mapping) == {"kind/bug": "author", "sig/node": "author"}

    def test_bot_label_without_body_command_is_triager(self, mapping):
        raw = raw_issue(
            ["kind/bug", "sig/node"],
            body="It broke\n/kind bug",
            timeline=[labeled("kind/bug"), labeled("sig/node", "2025-03-01T12:05:00Z")],
        )
        assert self.prov(raw, mapping) == {"kind/bug": "author", "sig/node": "triager"}

    def test_late_bot_label_is_triager_even_with_body_command(self, mapping):
        # Author wrote /kind bug, a triager removed it, then re-added it days later.
        raw = raw_issue(
            ["kind/bug"],
            body="/kind bug",
            timeline=[
                labeled("kind/bug"),
                unlabeled("kind/bug", "2025-03-02T00:00:00Z"),
                labeled("kind/bug", "2025-03-05T00:00:00Z"),
            ],
        )
        assert self.prov(raw, mapping) == {"kind/bug": "triager"}

    def test_label_applied_by_other_human(self, mapping):
        raw = raw_issue(["priority/backlog"], timeline=[labeled("priority/backlog", actor=TRIAGER)])
        assert self.prov(raw, mapping) == {"priority/backlog": "triager"}

    def test_original_body_from_edits(self, mapping):
        edits = [{"diff": "Original\n/kind bug", "editedAt": "2025-03-01T12:00:00Z"}]
        raw = raw_issue(["kind/bug"], body="Edited, command gone", edits=edits, timeline=[labeled("kind/bug")])
        assert original_body(raw) == ("Original\n/kind bug", True, True)
        assert self.prov(raw, mapping) == {"kind/bug": "author"}

    def test_later_edit_is_not_the_original(self):
        # edited, but the creation snapshot isn't the one fetched: the filed body is unknown
        raw = raw_issue([], body="now", edits=[{"diff": "later", "editedAt": "2025-03-05T00:00:00Z"}])
        assert original_body(raw) == ("now", True, False)

    def test_deleted_creation_snapshot_is_unknown(self):
        raw = raw_issue([], body="CVE text", edits=[{"diff": None, "editedAt": "2025-03-01T12:00:00Z"}])
        assert original_body(raw) == ("CVE text", True, False)

    def test_unedited_body(self):
        assert original_body(raw_issue([], body="as filed")) == ("as filed", False, True)

    def test_author_typed_committee_command(self, mapping):
        raw = raw_issue(
            ["kind/bug", "committee/security-response"],
            body="/committee security-response",
            timeline=[labeled("kind/bug", actor=AUTHOR), labeled("committee/security-response")],
        )
        assert self.prov(raw, mapping) == {"kind/bug": "author", "committee/security-response": "author"}


class TestEligibility:
    def test_needs_triage_excluded(self, mapping):
        assert eligibility(raw_issue(["kind/bug", "needs-triage"]), mapping) == "needs-triage"

    def test_no_kind_excluded(self, mapping):
        assert eligibility(raw_issue(["sig/node", "triage/accepted"]), mapping) == "no kind"

    def test_kind_tag_alone_is_not_a_kind(self, mapping):
        assert eligibility(raw_issue(["kind/api-change", "kind/regression"]), mapping) == "no kind"

    def test_eligible(self, mapping):
        assert eligibility(raw_issue(["kind/bug", "triage/accepted"]), mapping) is None

    def test_stale_closed_kept(self, mapping):
        raw = raw_issue(
            ["kind/bug", "lifecycle/rotten"],
            state="CLOSED",
            state_reason="NOT_PLANNED",
            timeline=[labeled("lifecycle/rotten", "2025-03-05T00:00:00Z"), closed(reason="NOT_PLANNED")],
        )
        assert eligibility(raw, mapping) is None


class TestExpected:
    def expected(self, raw, mapping):
        h = LabelHistory.of(raw)
        return expected_answers(raw, h, mapping, REPO, close_outcome(raw, h, mapping, REPO))

    def test_fields_from_labels(self, mapping):
        raw = raw_issue(
            [
                "kind/bug",
                "kind/regression",
                "sig/node",
                "sig/storage",
                "priority/backlog",
                "priority/important-soon",
                "area/security",
                "kind/api-change",
            ]
        )
        e = self.expected(raw, mapping)
        assert e["kind"] == ["bug"]
        assert e["area"] == ["sig/node", "sig/storage"]
        assert e["severity"] == "important_soon"
        assert e["is_regression"] is True
        assert e["security_sensitive"] is True
        assert e["enough_info"] is None
        assert e["close_outcome"] == "open"

    def test_missing_labels_are_not_answers(self, mapping):
        e = self.expected(raw_issue(["kind/feature", "priority/awaiting-more-evidence"]), mapping)
        assert e["area"] is None and e["severity"] is None and e["is_regression"] is None
        assert e["security_sensitive"] is False

    def test_enough_info(self, mapping):
        tl = [
            labeled("triage/needs-information", "2025-03-02T00:00:00Z"),
            unlabeled("triage/needs-information", "2025-03-03T00:00:00Z"),
            closed(closer={"__typename": "Commit", "oid": "x"}),
        ]
        raw = raw_issue(["kind/bug"], state="CLOSED", state_reason="COMPLETED", timeline=tl)
        assert self.expected(raw, mapping)["enough_info"] is False
        raw = raw_issue(
            ["kind/bug"],
            state="CLOSED",
            state_reason="COMPLETED",
            timeline=[closed(closer={"__typename": "Commit", "oid": "x"})],
        )
        assert self.expected(raw, mapping)["enough_info"] is True
        raw = raw_issue(
            ["kind/feature"],
            state="CLOSED",
            state_reason="COMPLETED",
            timeline=[closed(closer={"__typename": "Commit", "oid": "x"})],
        )
        assert self.expected(raw, mapping)["enough_info"] is None

    def test_duplicate_of_from_event_then_comment(self, mapping):
        ev = {
            "__typename": "MarkedAsDuplicateEvent",
            "createdAt": "2025-03-02T00:00:00Z",
            "canonical": {"number": 42, "repository": {"nameWithOwner": REPO}},
        }
        raw = raw_issue(
            ["kind/bug"], state="CLOSED", state_reason="DUPLICATE", timeline=[ev, closed(reason="DUPLICATE")]
        )
        assert self.expected(raw, mapping)["duplicate_of"] == [42]
        comments = [
            {
                "author": {"login": "bob", "__typename": "User"},
                "authorAssociation": "MEMBER",
                "body": "/close\nDuplicate of #77",
                "createdAt": "2025-03-02T00:00:00Z",
            }
        ]
        raw = raw_issue(
            ["kind/bug", "triage/duplicate"],
            state="CLOSED",
            state_reason="COMPLETED",
            timeline=[closed()],
            comments=comments,
        )
        assert self.expected(raw, mapping)["duplicate_of"] == [77]


def test_normalize_issue_shape(mapping):
    ctx = RepoContext(
        name=REPO,
        labels=[{"name": "kind/bug", "description": "Categorizes issue or PR as related to a bug."}],
        releases=[("2025-02-01T00:00:00Z", "v1.32.2"), ("2025-03-02T00:00:00Z", "v1.32.3")],
    )
    raw = raw_issue(
        ["kind/bug", "sig/node", "triage/accepted"],
        body="### What happened?\n\nboom\n\n/kind bug\n/sig node",
        timeline=[labeled("kind/bug"), labeled("sig/node"), labeled("triage/accepted", "2025-03-03T00:00:00Z")],
        comments=[
            {
                "author": TRIAGER,
                "authorAssociation": "MEMBER",
                "body": "/triage accepted",
                "createdAt": "2025-03-03T00:00:00Z",
            }
        ],
    )
    rec = normalize_issue(raw, ctx, mapping)
    assert set(rec) == {"issue", "repo", "expected", "metadata"}
    assert rec["issue"] == {
        "number": 1000,
        "title": "Something broke",
        "body": "### What happened?\n\nboom",
        "author_association": "NONE",
        "created_at": "2025-03-01T12:00:00Z",
    }
    assert rec["repo"]["latest_release"] == "v1.32.2"
    assert rec["repo"]["as_of"] == "2025-03-01T12:00:00Z"
    assert rec["metadata"]["template"] == "bug-report"
    assert rec["metadata"]["label_provenance"] == {
        "kind/bug": "author",
        "sig/node": "author",
        "triage/accepted": "triager",
    }
    assert rec["metadata"]["triager_verified"] is True


def test_original_title_is_the_title_before_the_first_rename():
    def rename(at: str, before: str, after: str) -> dict:
        return {"__typename": "RenamedTitleEvent", "createdAt": at, "previousTitle": before, "currentTitle": after}

    renames = [rename("2025-03-05T00:00:00Z", "B", "C"), rename("2025-03-02T00:00:00Z", "A", "B")]
    raw = raw_issue([], timeline=renames)
    raw["title"] = "C"
    assert original_title(raw) == ("A", True)
    assert original_title(raw_issue([])) == ("Something broke", False)


def test_normalize_issue_uses_title_as_filed_and_flags_lost_bodies(mapping):
    ctx = RepoContext(name=REPO, labels=[], releases=[])
    rename = {
        "__typename": "RenamedTitleEvent",
        "createdAt": "2025-03-02T00:00:00Z",
        "previousTitle": "Placeholder Issue",
        "currentTitle": "CVE-2025-0001: something",
    }
    raw = raw_issue(
        ["kind/bug", "committee/security-response"],
        body="CVSS Rating: High\n/committee security-response",
        timeline=[rename],
        edits=[{"diff": None, "editedAt": "2025-03-01T12:00:00Z"}],
    )
    raw["title"] = "CVE-2025-0001: something"
    rec = normalize_issue(raw, ctx, mapping)
    assert rec["issue"]["title"] == "Placeholder Issue"
    assert rec["issue"]["body"] == "CVSS Rating: High"
    assert rec["metadata"]["title_edited"] is True
    assert rec["metadata"]["body_edited"] is True
    assert rec["metadata"]["original_body_unknown"] is True


def test_detect_template(mapping):
    assert detect_template("### Which jobs are flaking?\n\nfoo", mapping) == "flaking-test"
    assert detect_template("**What happened**:\nfoo", mapping) == "bug-report"
    assert detect_template("Just some text", mapping) is None


class TestTriagerVerified:
    def verified(self, raw, mapping):
        ctx = RepoContext(name=REPO, labels=[], releases=[])
        return normalize_issue(raw, ctx, mapping)["metadata"]["triager_verified"]

    def comment(self, who, body):
        return {"author": who, "authorAssociation": "MEMBER", "body": body, "createdAt": "2025-03-02T00:00:00Z"}

    def test_author_self_accepting_is_not_verification(self, mapping):
        raw = raw_issue(
            ["kind/bug", "triage/accepted"],
            timeline=[labeled("kind/bug", actor=AUTHOR), labeled("triage/accepted", "2025-03-02T00:00:00Z")],
            comments=[self.comment(AUTHOR, "/triage accepted")],
        )
        assert self.verified(raw, mapping) is False

    def test_other_person_accepting_is(self, mapping):
        raw = raw_issue(
            ["kind/bug", "triage/accepted"],
            timeline=[labeled("kind/bug", actor=AUTHOR), labeled("triage/accepted", "2025-03-02T00:00:00Z")],
            comments=[self.comment(TRIAGER, "Thanks!\n/triage accepted\n/priority backlog")],
        )
        assert self.verified(raw, mapping) is True

    def test_kind_set_by_triager_is(self, mapping):
        raw = raw_issue(["kind/bug"], timeline=[labeled("kind/bug", "2025-03-03T00:00:00Z")])
        assert self.verified(raw, mapping) is True


def test_latest_release_is_highest_version_before_as_of():
    ctx = RepoContext(
        name=REPO,
        labels=[],
        releases=[
            ("2025-05-15T00:00:00Z", "v1.33.1"),
            ("2025-06-17T00:00:00Z", "v1.30.13"),
            ("2025-06-17T01:00:00Z", "v1.33.2"),
            ("2025-06-18T00:00:00Z", "v1.31.10"),
        ],
    )
    assert ctx.latest_release("2025-06-17T00:30:00Z") == "v1.33.1"
    assert ctx.latest_release("2025-06-30T00:00:00Z") == "v1.33.2"
    assert ctx.latest_release("2025-01-01T00:00:00Z") is None


class TestCommentDuplicates:
    def comment(self, body, who=TRIAGER):
        return {"author": who, "authorAssociation": "MEMBER", "body": body, "createdAt": "2025-03-02T00:00:00Z"}

    @pytest.mark.parametrize(
        "body,target",
        [
            ("dup of https://github.com/kubernetes/kubernetes/issues/139470\n/close", 139470),
            ("I'll close this issue because this is a duplicate of #139494.", 139494),
            ("Duplicate of kubernetes/kubernetes#77", 77),
        ],
    )
    def test_closed_with_duplicate_comment(self, mapping, body, target):
        raw = raw_issue(
            ["kind/bug"],
            number=140000,
            state="CLOSED",
            state_reason="NOT_PLANNED",
            timeline=[closed(reason="NOT_PLANNED")],
            comments=[self.comment(body)],
        )
        h = LabelHistory.of(raw)
        assert close_outcome(raw, h, mapping, REPO) == "duplicate"
        assert expected_answers(raw, h, mapping, REPO, "duplicate")["duplicate_of"] == [target]

    def test_other_repo_or_question_is_not_a_duplicate(self, mapping):
        for body in ["duplicate of kubernetes/website#12", "Is it the same as #5?"]:
            raw = raw_issue(
                ["kind/bug"],
                state="CLOSED",
                state_reason="COMPLETED",
                timeline=[closed()],
                comments=[self.comment(body)],
            )
            assert outcome(raw, mapping) == "completed_other"

    def test_open_issue_with_duplicate_comment_stays_open(self, mapping):
        raw = raw_issue(["kind/bug"], comments=[self.comment("duplicate of #5")])
        assert outcome(raw, mapping) == "open"

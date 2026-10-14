import json

from triager_evals.fetch import append_page, backfill_path, backfill_query
from triager_evals.normalize import raw_issues


def test_append_page_repairs_a_half_written_line(tmp_path):
    path = tmp_path / "pages.jsonl"
    append_page(path, None, [{"number": 1}])
    with path.open("a") as f:
        f.write('{"after": "x", "issues": [{"numb')
    append_page(path, "x", [{"number": 2}, {"number": 1}])
    lines = path.read_text().splitlines()
    assert [json.loads(line)["after"] for line in lines] == [None, "x"]
    # A page fetched twice after a resume yields its issues once.
    assert [i["number"] for i in raw_issues(path)] == [1, 2]


def test_backfill_replaces_the_backfilled_event_types(tmp_path):
    old_dup = {"__typename": "MarkedAsDuplicateEvent", "createdAt": "t", "canonical": {"number": 1}}
    label = {"__typename": "LabeledEvent", "createdAt": "t", "label": {"name": "kind/bug"}}
    issue = {"number": 1, "timelineItems": {"pageInfo": {}, "nodes": [label, old_dup]}}
    path = tmp_path / "pages.jsonl"
    append_page(path, None, [issue, {"number": 2, "timelineItems": {"pageInfo": {}, "nodes": [label]}}])
    new_dup = {**old_dup, "duplicate": {"number": 5}}
    rename = {"__typename": "RenamedTitleEvent", "createdAt": "t", "previousTitle": "a", "currentTitle": "b"}
    backfill_path(path).write_text(json.dumps({"number": 1, "nodes": [new_dup, rename], "truncated": False}) + "\n")

    got = {i["number"]: i["timelineItems"]["nodes"] for i in raw_issues(path)}
    assert got[1] == [label, new_dup, rename]
    assert got[2] == [label]
    assert [i["timelineItems"]["nodes"] for i in raw_issues(path, backfill=False)][0] == [label, old_dup]


def test_backfill_query_aliases_each_issue():
    q = backfill_query([12, 34])
    assert "i12: issue(number: 12)" in q and "i34: issue(number: 34)" in q
    assert "RENAMED_TITLE_EVENT" in q and "previousTitle" in q and "duplicate {" in q

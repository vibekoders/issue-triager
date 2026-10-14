from collections import Counter

from triager_evals.build_dataset import (
    TriageDataset,
    load_dataset,
    read_dataset_metadata,
    select,
    split_of,
    to_case,
    write_dataset,
)

REPO = "kubernetes/kubernetes"
KINDS = ["bug"] * 60 + ["feature"] * 20 + ["cleanup"] * 10 + ["failing_test"] * 5 + ["flake"] * 3 + ["docs"] * 2


def record(number: int, kind: str) -> dict:
    return {
        "issue": {
            "number": number,
            "title": f"t{number}",
            "body": "b",
            "author_association": "NONE",
            "created_at": f"2025-{1 + number % 12:02d}-01T00:00:00Z",
        },
        "repo": {
            "name": REPO,
            "labels": [{"name": "kind/bug", "description": "d"}],
            "latest_release": "v1.32.0",
            "as_of": "2025-01-01T00:00:00Z",
        },
        "expected": {
            "kind": [kind],
            "area": None,
            "severity": None,
            "is_regression": None,
            "security_sensitive": False,
            "enough_info": None,
            "duplicate_of": None,
            "close_outcome": "open",
        },
        "metadata": {"triager_verified": number % 2 == 0, "created_at": "2025-01-01T00:00:00Z"},
    }


def records(n: int = 1000) -> list[dict]:
    return [record(100000 + i, KINDS[i % len(KINDS)]) for i in range(n)]


def numbers(rs):
    return [r["issue"]["number"] for r in rs]


def test_split_is_stable_and_roughly_80_20():
    assert split_of(REPO, 123456) == split_of(REPO, 123456)
    counts = Counter(split_of(REPO, n) for n in range(10000))
    assert 0.78 < counts["dev"] / 10000 < 0.82


def test_splits_never_overlap():
    rs = records()
    dev = set(numbers(select(rs, REPO, "dev", None, "kind", 2)))
    test = set(numbers(select(rs, REPO, "test", None, "kind", 2)))
    assert not dev & test and len(dev | test) == len(rs)


def test_larger_limit_is_superset_and_prefix():
    rs = records()
    full = numbers(select(rs, REPO, "dev", None, "kind", 2))
    for small, large in [(5, 20), (20, 200), (200, 400), (1, 799)]:
        a = numbers(select(rs, REPO, "dev", small, "kind", 2))
        b = numbers(select(rs, REPO, "dev", large, "kind", 2))
        assert b[: len(a)] == a
        assert full[: len(b)] == b


def test_order_does_not_depend_on_input_order():
    rs = records()
    assert numbers(select(rs, REPO, "dev", 50, "kind", 2)) == numbers(select(rs[::-1], REPO, "dev", 50, "kind", 2))


def test_small_datasets_contain_rare_kinds_and_stay_proportional():
    rs = records()
    small = Counter(r["expected"]["kind"][0] for r in select(rs, REPO, "dev", 20, "kind", 2))
    assert set(small) == set(KINDS)
    big = Counter(r["expected"]["kind"][0] for r in select(rs, REPO, "dev", 400, "kind", 2))
    pool = Counter(r["expected"]["kind"][0] for r in select(rs, REPO, "dev", None, "kind", 2))
    for kind, n in pool.items():
        assert abs(big[kind] / 400 - n / sum(pool.values())) < 0.03


def test_filters():
    rs = records()
    assert all(
        r["metadata"]["triager_verified"] for r in select(rs, REPO, "dev", None, "kind", 2, triager_verified=True)
    )
    assert all(
        r["issue"]["created_at"] >= "2025-06-01" for r in select(rs, REPO, "dev", None, "kind", 2, since="2025-06-01")
    )


def test_issues_with_a_lost_body_are_left_out_by_default():
    rs = records()
    for r in rs[::3]:
        r["metadata"]["original_body_unknown"] = True
    lost = {r["issue"]["number"] for r in rs[::3]}
    default = numbers(select(rs, REPO, "dev", None, "kind", 2))
    assert default and not set(default) & lost
    kept = numbers(select(rs, REPO, "dev", None, "kind", 2, keep_unknown_body=True))
    assert set(kept) & lost


def test_round_trip_with_shared_labels(tmp_path):
    rs = select(records(), REPO, "dev", 10, "kind", 2)
    path = tmp_path / "ds.yaml"
    write_dataset(TriageDataset(name="ds", cases=[to_case(r, "kubernetes") for r in rs]), path, {"limit": 10})
    text = path.read_text()
    assert text.startswith("# yaml-language-server: $schema=ds_schema.json\n# dataset-metadata: ")
    assert text.count("&repo_labels") == 1 and text.count("*repo_labels") == 9
    assert (tmp_path / "ds_schema.json").exists()
    assert read_dataset_metadata(path) == {"limit": 10}
    ds = load_dataset(path)
    assert [c.name for c in ds.cases] == [f"kubernetes-{n}" for n in numbers(rs)]
    assert ds.cases[0].inputs.repo.labels[0].name == "kind/bug"
    assert ds.cases[0].expected_output.close_outcome == "open"
    assert ds.cases[0].metadata["close_outcome"] == "open"

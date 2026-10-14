import os

from triager_evals.env import load_repo_env, parse


def test_parse():
    text = '# c\nA=1\nexport B="two words"\nC=x # note\nD=\nA=3\nnot a line\n'
    assert parse(text) == {"A": "3", "B": "two words", "C": "x", "D": ""}


def test_existing_env_wins(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("TRIAGE_TEST_A=file\nTRIAGE_TEST_B=file\nTRIAGE_TEST_C=file\n")
    monkeypatch.setenv("TRIAGE_TEST_A", "shell")
    monkeypatch.delenv("TRIAGE_TEST_B", raising=False)
    monkeypatch.setenv("TRIAGE_TEST_C", "")
    load_repo_env(f)
    assert os.environ["TRIAGE_TEST_A"] == "shell"
    assert os.environ["TRIAGE_TEST_B"] == "file"
    assert os.environ["TRIAGE_TEST_C"] == "file"  # empty counts as unset
    monkeypatch.delenv("TRIAGE_TEST_B")

"""Loads the repo's `.env`, so the harness works without devenv.

devenv already exports `.env` into the shell; with plain `uv run` nothing
does. Variables that are already set win over the file, the same way devenv
and docker compose treat it, so a one-off `FOO=bar uv run ...` still works.
Only simple `KEY=value` lines are read (optionally quoted, `export `
allowed); anything fancier belongs in the shell.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def parse(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            # Later lines win within the file, as in docker compose.
            values[key] = value
    return values


def load_repo_env(path: Path = ENV_FILE) -> None:
    if not path.is_file():
        return
    for key, value in parse(path.read_text(encoding="utf-8")).items():
        # The shell wins, but an empty value counts as unset (a stale
        # `LOGFIRE_TOKEN=` would otherwise turn Logfire off without a word).
        if not os.environ.get(key):
            os.environ[key] = value

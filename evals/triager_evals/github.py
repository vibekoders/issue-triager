"""A small GitHub API client that survives a long, unattended fetch.

The fetch runs for up to an hour against an API with a points budget,
secondary rate limits and occasional 502s on heavy queries, so every request
goes through one place that waits, backs off and retries.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import httpx

GRAPHQL_URL = "https://api.github.com/graphql"
REST_URL = "https://api.github.com"


def token() -> str:
    """The token from `GITHUB_TOKEN`, or from the `gh` CLI.

    devenv loads `.env`, which may set `GITHUB_TOKEN` to an empty string, so an
    empty value counts as unset.
    """
    if tok := os.environ.get("GITHUB_TOKEN", "").strip():
        return tok
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit("no GitHub token: set GITHUB_TOKEN or log in with `gh auth login`") from e
    return out.stdout.strip()


class GraphQLError(Exception):
    def __init__(self, errors: list[dict[str, Any]]):
        super().__init__("; ".join(e.get("message", str(e)) for e in errors))
        self.errors = errors


class QueryTooHeavy(Exception):
    """GitHub timed out or refused the query; a smaller page may get through."""


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class GitHub:
    def __init__(self, tok: str | None = None, *, max_retries: int = 8):
        self.http = httpx.Client(
            headers={
                "Authorization": f"bearer {tok or token()}",
                "User-Agent": "triager-evals",
                "Accept": "application/vnd.github+json",
            },
            timeout=httpx.Timeout(90.0, connect=15.0),
        )
        self.max_retries = max_retries

    def close(self) -> None:
        self.http.close()

    def graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Runs a query and returns `data`.

        Partial results (data plus errors) are returned with the errors under
        `data["__errors__"]` so the caller can decide whether the missing parts
        matter; a response with no data at all raises.
        """
        payload = {"query": query, "variables": variables or {}}
        for attempt in range(self.max_retries):
            try:
                resp = self.http.post(GRAPHQL_URL, json=payload)
            except httpx.TransportError as e:
                self._backoff(attempt, f"transport error: {e!r}")
                continue

            if resp.status_code in (500, 502, 503, 504):
                # Heavy queries time out on GitHub's side as 502s. Retrying
                # the same size a couple of times usually works; after that
                # the caller should shrink the page.
                if attempt >= 2:
                    raise QueryTooHeavy(f"HTTP {resp.status_code}")
                self._backoff(attempt, f"HTTP {resp.status_code}")
                continue
            if resp.status_code in (403, 429):
                self._wait_rate_limited(resp, attempt)
                continue
            resp.raise_for_status()

            try:
                body = resp.json()
            except ValueError:
                # A 200 with a cut-off body: GitHub gave up on a slow query.
                if attempt >= 2:
                    raise QueryTooHeavy("truncated response") from None
                self._backoff(attempt, "response is not JSON")
                continue
            errors = body.get("errors") or []
            data = body.get("data")
            if errors and any(e.get("type") == "RATE_LIMITED" for e in errors):
                self._wait_until(self._reset_from_headers(resp), "GraphQL rate limit")
                continue
            if errors and not data:
                msgs = " ".join(e.get("message", "") for e in errors).lower()
                if "timeout" in msgs or "timedout" in msgs or "something went wrong" in msgs:
                    if attempt >= 2:
                        raise QueryTooHeavy(msgs)
                    self._backoff(attempt, f"GraphQL: {msgs[:200]}")
                    continue
                raise GraphQLError(errors)
            if errors:
                data["__errors__"] = errors
            self._respect_budget(data)
            return data
        raise RuntimeError(f"GraphQL request failed after {self.max_retries} attempts")

    def rest_paginate(self, path: str, params: dict[str, Any] | None = None) -> list[Any]:
        """GETs every page of a REST list endpoint."""
        url: str | None = f"{REST_URL}/{path.lstrip('/')}"
        params = {"per_page": 100, **(params or {})}
        items: list[Any] = []
        while url:
            resp = self._rest_get(url, params)
            items.extend(resp.json())
            url = resp.links.get("next", {}).get("url")
            params = None
        return items

    def rest_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._rest_get(f"{REST_URL}/{path.lstrip('/')}", params).json()

    def _rest_get(self, url: str, params: dict[str, Any] | None) -> httpx.Response:
        for attempt in range(self.max_retries):
            try:
                resp = self.http.get(url, params=params)
            except httpx.TransportError as e:
                self._backoff(attempt, f"transport error: {e!r}")
                continue
            if resp.status_code in (500, 502, 503, 504):
                self._backoff(attempt, f"HTTP {resp.status_code}")
                continue
            if resp.status_code in (403, 429) and self._is_rate_limited(resp):
                self._wait_rate_limited(resp, attempt)
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"GET {url} failed after {self.max_retries} attempts")

    @staticmethod
    def _is_rate_limited(resp: httpx.Response) -> bool:
        return (
            resp.status_code == 429
            or resp.headers.get("x-ratelimit-remaining") == "0"
            or "rate limit" in resp.text.lower()
        )

    def _wait_rate_limited(self, resp: httpx.Response, attempt: int) -> None:
        if retry_after := resp.headers.get("retry-after"):
            log(f"secondary rate limit, sleeping {retry_after}s")
            time.sleep(int(retry_after) + 1)
        elif resp.headers.get("x-ratelimit-remaining") == "0":
            self._wait_until(self._reset_from_headers(resp), "primary rate limit")
        else:
            # Secondary limits without Retry-After: GitHub asks for at least a
            # minute between retries.
            self._backoff(attempt, f"HTTP {resp.status_code}: {resp.text[:200]}", base=60)

    @staticmethod
    def _reset_from_headers(resp: httpx.Response) -> float:
        try:
            return float(resp.headers["x-ratelimit-reset"])
        except (KeyError, ValueError):
            return time.time() + 60

    def _respect_budget(self, data: dict[str, Any]) -> None:
        """Sleeps until the budget resets when the next query might not fit."""
        rl = data.get("rateLimit")
        if not rl:
            return
        if rl["remaining"] < max(50, 3 * rl.get("cost", 1)):
            reset = datetime.fromisoformat(rl["resetAt"].replace("Z", "+00:00")).timestamp()
            self._wait_until(reset, f"GraphQL budget low ({rl['remaining']} left)")

    @staticmethod
    def _wait_until(epoch: float, why: str) -> None:
        delay = max(5.0, epoch - time.time() + 5)
        until = datetime.fromtimestamp(epoch, timezone.utc).strftime("%H:%M:%S UTC")
        log(f"{why}: sleeping {delay:.0f}s until {until}")
        time.sleep(delay)

    @staticmethod
    def _backoff(attempt: int, why: str, base: float = 2.0) -> None:
        delay = min(base * 2**attempt, 300)
        log(f"{why}; retry {attempt + 1} in {delay:.0f}s")
        time.sleep(delay)

"""Basic API smoke checks for CI/deploy pipelines."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def _get_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"{url} returned status {resp.status}")
        return json.loads(resp.read().decode("utf-8"))


def _wait_ready(url: str, retries: int = 20, delay_s: float = 1.0) -> None:
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            payload = _get_json(url, timeout=3.0)
            if payload.get("status") == "ready":
                return
        except Exception as e:  # pragma: no cover - CI startup race
            last_err = e
        time.sleep(delay_s)
    if last_err:
        raise RuntimeError(f"Readiness failed: {last_err}") from last_err
    raise RuntimeError("Readiness failed: unknown error")


def main() -> None:
    base = "http://127.0.0.1:8000"
    _wait_ready(f"{base}/readyz")
    health = _get_json(f"{base}/health")
    if health.get("api") != "ok":
        raise RuntimeError(f"Unexpected /health payload: {health}")
    print("Smoke checks passed.")


if __name__ == "__main__":
    main()

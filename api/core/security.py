from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock

from fastapi import Header, HTTPException

from api.core.settings import get_settings


def require_admin_token(x_admin_token: str = Header(default="", alias="X-Admin-Token")) -> None:
    """Optionally enforce admin token for sensitive routes in production."""
    settings = get_settings()
    if not settings.enforce_admin_auth:
        return
    if not x_admin_token or x_admin_token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Missing or invalid admin token")


@dataclass
class _Window:
    timestamps: deque[float]


class InMemoryRateLimiter:
    """Simple process-local sliding-window rate limiter by key."""

    def __init__(self, default_limit_per_min: int, heavy_limit_per_min: int) -> None:
        self.default_limit_per_min = int(default_limit_per_min)
        self.heavy_limit_per_min = int(heavy_limit_per_min)
        self._store: dict[str, _Window] = {}
        self._lock = Lock()

    def _allow(self, key: str, limit: int, now: float) -> bool:
        cutoff = now - 60.0
        win = self._store.get(key)
        if win is None:
            win = _Window(timestamps=deque())
            self._store[key] = win
        dq = win.timestamps
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit:
            return False
        dq.append(now)
        return True

    def check(self, *, client_key: str, heavy: bool = False) -> bool:
        limit = self.heavy_limit_per_min if heavy else self.default_limit_per_min
        now = time.time()
        with self._lock:
            return self._allow(f"{'H' if heavy else 'D'}:{client_key}", limit, now)

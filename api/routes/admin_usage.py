"""Usage & cost analytics for the management dashboard.

Aggregates the cost events the cost meter appends to data/cost_events.jsonl
(one per generation call). Read-only; mounted behind the admin-token guard.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter

from api.core.settings import get_settings

router = APIRouter()


def _events_path() -> Path:
    return Path(get_settings().data_dir) / "cost_events.jsonl"


def _load_events() -> list[dict]:
    path = _events_path()
    if not path.is_file():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


@router.get("/admin/usage/summary")
def usage_summary(recent: int = 50) -> dict:
    """Totals + per-model and per-day breakdowns from the cost-event log."""
    events = _load_events()
    currency = "USD"

    total_cost = 0.0
    total_in = 0.0
    total_out = 0.0
    by_model: dict[str, dict] = defaultdict(
        lambda: {"provider": "", "cost": 0.0, "events": 0, "units_in": 0.0, "units_out": 0.0}
    )
    by_day: dict[str, dict] = defaultdict(lambda: {"cost": 0.0, "events": 0})

    for e in events:
        cost = float(e.get("estimated_cost") or 0.0)
        uin = float(e.get("units_in") or 0.0)
        uout = float(e.get("units_out") or 0.0)
        currency = str(e.get("currency") or currency)
        total_cost += cost
        total_in += uin
        total_out += uout

        model = str(e.get("model") or "unknown")
        m = by_model[model]
        m["provider"] = str(e.get("provider") or m["provider"])
        m["cost"] += cost
        m["events"] += 1
        m["units_in"] += uin
        m["units_out"] += uout

        day = str(e.get("ts") or "")[:10] or "unknown"
        by_day[day]["cost"] += cost
        by_day[day]["events"] += 1

    model_rows = sorted(
        ({"model": k, **v} for k, v in by_model.items()),
        key=lambda r: r["cost"],
        reverse=True,
    )
    day_rows = sorted(
        ({"date": k, **v} for k, v in by_day.items()),
        key=lambda r: r["date"],
    )
    recent_rows = list(reversed(events))[: max(0, int(recent))]

    return {
        "currency": currency,
        "total_cost": round(total_cost, 6),
        "total_events": len(events),
        "total_units_in": int(total_in),
        "total_units_out": int(total_out),
        "by_model": model_rows,
        "by_day": day_rows,
        "recent": recent_rows,
    }

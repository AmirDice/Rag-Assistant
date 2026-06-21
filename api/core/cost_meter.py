"""Cost metering for model provider calls.

Prices a model call from token-usage units using versioned pricing in
config/api_pricing/. Pure and dependency-light: no telemetry DB. Each metered
call is appended best-effort to data/cost_events.jsonl as an observability
artifact, and the per-call estimate is surfaced on the API response.

Token units come from ``GenerationResult`` (api/core/backends/base.py), so the
meter never needs to know which provider produced the call.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from api.core.settings import get_settings

logger = logging.getLogger(__name__)

_FALLBACK_PRICE = {
    "currency": "USD",
    "metric": "io_tokens",
    "price_per_unit": 0.0,
    "price_per_1k": 0.0,
    "input_per_1k": 0.0,
    "output_per_1k": 0.0,
    "thoughts_per_1k": 0.0,
    "pricing_source": "fallback_zero",
}


def resolve_price(provider: str, model: str, operation: str) -> dict[str, Any]:
    """Resolve pricing for provider/model/operation from the pricing config.

    Model-specific entries win over provider-wide operations; missing entries
    fall back to zero cost (with ``pricing_source == 'fallback_zero'``).
    """
    cfg = get_settings().api_pricing_config()
    default_currency = str(cfg.get("currency", "USD"))
    providers = cfg.get("providers") or {}
    p_cfg = providers.get(provider) if isinstance(providers, dict) else None
    if not isinstance(p_cfg, dict):
        return {**_FALLBACK_PRICE, "currency": default_currency}

    provider_ops = p_cfg.get("operations") if isinstance(p_cfg.get("operations"), dict) else {}
    model_ops = ((p_cfg.get("models") or {}).get(model) or {}).get("operations", {})
    op_cfg = None
    source = "fallback_zero"
    if isinstance(model_ops, dict) and isinstance(model_ops.get(operation), dict):
        op_cfg = model_ops[operation]
        source = "model"
    elif isinstance(provider_ops, dict) and isinstance(provider_ops.get(operation), dict):
        op_cfg = provider_ops[operation]
        source = "provider"

    if not isinstance(op_cfg, dict):
        return {**_FALLBACK_PRICE, "currency": default_currency}

    return {
        "currency": str(op_cfg.get("currency") or default_currency),
        "metric": str(op_cfg.get("metric") or "io_tokens"),
        "price_per_unit": float(op_cfg.get("price_per_unit") or 0.0),
        "price_per_1k": float(op_cfg.get("price_per_1k") or 0.0),
        "input_per_1k": float(op_cfg.get("input_per_1k") or 0.0),
        "output_per_1k": float(op_cfg.get("output_per_1k") or 0.0),
        "thoughts_per_1k": float(op_cfg.get("thoughts_per_1k") or 0.0),
        "pricing_source": source,
    }


def estimate_cost(
    price: dict[str, Any],
    *,
    units_in: float = 0.0,
    units_out: float = 0.0,
    units_thoughts: float = 0.0,
    billable_quantity: Optional[float] = None,
) -> float:
    """Compute estimated cost (in ``price['currency']``) for the given units."""
    metric = str(price.get("metric") or "io_tokens")
    if metric == "search_units":
        qty = float(billable_quantity if billable_quantity is not None else 0.0)
        return qty * float(price.get("price_per_unit") or 0.0)
    if metric == "total_tokens":
        qty = float(
            billable_quantity
            if billable_quantity is not None
            else (units_in + units_out + units_thoughts)
        )
        return (qty / 1000.0) * float(price.get("price_per_1k") or 0.0)
    # io_tokens (default)
    return (
        (units_in / 1000.0) * float(price.get("input_per_1k") or 0.0)
        + (units_out / 1000.0) * float(price.get("output_per_1k") or 0.0)
        + (units_thoughts / 1000.0) * float(price.get("thoughts_per_1k") or 0.0)
    )


def price_generation(
    *,
    provider: str,
    model: str,
    units_in: float,
    units_out: float,
    units_thoughts: float = 0.0,
    operation: str = "generate_text",
) -> dict[str, Any]:
    """Price one generation call. Returns cost + the inputs for transparency."""
    price = resolve_price(provider=provider, model=model, operation=operation)
    cost = estimate_cost(
        price,
        units_in=units_in,
        units_out=units_out,
        units_thoughts=units_thoughts,
    )
    return {
        "operation": operation,
        "provider": provider,
        "model": model,
        "units_in": float(units_in),
        "units_out": float(units_out),
        "units_thoughts": float(units_thoughts),
        "estimated_cost": float(cost),
        "currency": price["currency"],
        "pricing_source": price["pricing_source"],
    }


def record_cost_event(event: dict[str, Any]) -> None:
    """Append a cost event to data/cost_events.jsonl (best-effort)."""
    try:
        path = Path(get_settings().data_dir) / "cost_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - logging side effect only
        logger.warning("Could not persist cost event: %s", exc)

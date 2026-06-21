"""Cost metering — pricing resolution + cost computation from token units."""

from __future__ import annotations

import math

from api.core.cost_meter import estimate_cost, price_generation, resolve_price


def test_provider_default_pricing_resolves():
    price = resolve_price("google", "gemini-2.5-flash-lite", "generate_text")
    assert price["pricing_source"] == "provider"
    assert price["input_per_1k"] == 0.0001
    assert price["output_per_1k"] == 0.0004


def test_model_override_wins_over_provider():
    price = resolve_price("google", "gemini-2.5-flash", "generate_text")
    assert price["pricing_source"] == "model"
    assert price["input_per_1k"] == 0.0003
    assert price["output_per_1k"] == 0.0025


def test_unknown_provider_falls_back_to_zero():
    price = resolve_price("mystery", "x", "generate_text")
    assert price["pricing_source"] == "fallback_zero"
    assert estimate_cost(price, units_in=1000, units_out=1000) == 0.0


def test_io_tokens_cost_computation():
    price = resolve_price("google", "gemini-2.5-flash-lite", "generate_text")
    # 1000 in + 1000 out = 0.0001 + 0.0004 = 0.0005
    cost = estimate_cost(price, units_in=1000, units_out=1000)
    assert math.isclose(cost, 0.0005, rel_tol=1e-9)


def test_price_generation_end_to_end():
    priced = price_generation(
        provider="openai_compat",
        model="gpt-4o-mini",
        units_in=2000,
        units_out=500,
    )
    # gpt-4o-mini default: in 0.00015/1k, out 0.0006/1k
    expected = (2000 / 1000) * 0.00015 + (500 / 1000) * 0.0006
    assert math.isclose(priced["estimated_cost"], expected, rel_tol=1e-9)
    assert priced["currency"] == "USD"
    assert priced["provider"] == "openai_compat"


def test_ollama_is_free():
    priced = price_generation(
        provider="ollama", model="llama3.1", units_in=5000, units_out=5000
    )
    assert priced["estimated_cost"] == 0.0

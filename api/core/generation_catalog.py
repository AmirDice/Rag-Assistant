"""Allowed chat / query generation model IDs (UI, preferences, and API validation)."""

from __future__ import annotations

# Single source of truth: Gemini 2.5 family + OpenAI GPT-4 / GPT-4o mini.
GENERATION_MODEL_IDS: tuple[str, ...] = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gpt-4",
    "gpt-4o-mini",
)

GENERATION_MODEL_LABELS: dict[str, str] = {
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gpt-4": "GPT-4",
    "gpt-4o-mini": "GPT-4o mini",
}


def generation_options_payload() -> list[dict[str, str]]:
    return [
        {"id": mid, "label": GENERATION_MODEL_LABELS.get(mid, mid)}
        for mid in GENERATION_MODEL_IDS
    ]


def uses_openai_api(model_id: str) -> bool:
    return model_id.strip().lower().startswith("gpt-")

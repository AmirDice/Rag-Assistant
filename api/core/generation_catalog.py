"""Generation model catalog (UI options, API validation, provider resolution).

The catalog is config-driven: `generation.available_models` in
config/models.yaml maps a public model id to a provider + connection details.
The factory (`api/core/backends/factory.py`) turns a catalog entry into a
concrete backend. When `available_models` is absent, we synthesize a catalog
from the static `GENERATION_MODEL_IDS` below so older configs keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Static fallback catalog: Gemini 2.5 family + OpenAI GPT-4 / GPT-4o mini.
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


@dataclass(frozen=True)
class GenerationModelConfig:
    """A single catalog entry: how to reach one generation model."""

    id: str
    provider: str  # gemini | openai_compat | ollama
    model: str
    base_url: str = ""
    base_url_ref: str = ""   # name of a Settings attribute holding the base_url
    api_key: str = ""
    api_key_ref: str = ""    # name of a Settings attribute holding the api_key
    thinking: bool = False
    max_completion_tokens: Optional[int] = None


def _synthesized_entry(model_id: str) -> GenerationModelConfig:
    """Build a catalog entry for a static id when no YAML catalog is present."""
    if model_id.strip().lower().startswith("gpt-"):
        return GenerationModelConfig(
            id=model_id,
            provider="openai_compat",
            model=model_id,
            base_url="https://api.openai.com/v1",
            api_key_ref="openai_api_key",
        )
    return GenerationModelConfig(id=model_id, provider="gemini", model=model_id)


def _load_catalog() -> tuple[dict[str, GenerationModelConfig], str]:
    from api.core.settings import get_settings

    section = (get_settings().models_config().get("generation") or {})
    raw = section.get("available_models")
    out: dict[str, GenerationModelConfig] = {}
    if isinstance(raw, dict):
        for raw_id, raw_cfg in raw.items():
            model_id = str(raw_id).strip()
            if not model_id or not isinstance(raw_cfg, dict):
                continue
            provider = str(raw_cfg.get("provider") or "").strip().lower()
            model = str(raw_cfg.get("model") or "").strip()
            if not provider or not model:
                continue
            cap = raw_cfg.get("max_completion_tokens")
            out[model_id] = GenerationModelConfig(
                id=model_id,
                provider=provider,
                model=model,
                base_url=str(raw_cfg.get("base_url") or "").strip(),
                base_url_ref=str(raw_cfg.get("base_url_ref") or "").strip(),
                api_key=str(raw_cfg.get("api_key") or "").strip(),
                api_key_ref=str(raw_cfg.get("api_key_ref") or "").strip(),
                thinking=bool(raw_cfg.get("thinking", False)),
                max_completion_tokens=int(cap) if cap is not None else None,
            )

    default_model = str(section.get("default_model") or section.get("model") or "").strip()

    if not out:
        for model_id in GENERATION_MODEL_IDS:
            out[model_id] = _synthesized_entry(model_id)
        if not default_model or default_model not in out:
            default_model = GENERATION_MODEL_IDS[0]

    if default_model not in out:
        default_model = next(iter(out))
    return out, default_model


def generation_catalog() -> dict[str, GenerationModelConfig]:
    return _load_catalog()[0]


def generation_model_ids() -> tuple[str, ...]:
    return tuple(generation_catalog().keys())


def default_generation_model_id() -> str:
    return _load_catalog()[1]


def resolve_generation_model_id(model_id: Optional[str]) -> str:
    requested = str(model_id or "").strip()
    catalog, default_model = _load_catalog()
    if requested and requested in catalog:
        return requested
    return default_model


def generation_model_config(model_id: Optional[str]) -> GenerationModelConfig:
    """Resolve a (possibly None / unknown) id to a concrete catalog entry."""
    catalog, default_model = _load_catalog()
    resolved = model_id.strip() if (model_id and model_id.strip() in catalog) else default_model
    return catalog[resolved]


def generation_options_payload() -> list[dict[str, str]]:
    return [
        {"id": mid, "label": GENERATION_MODEL_LABELS.get(mid, mid.replace("-", " "))}
        for mid in generation_model_ids()
    ]


def uses_openai_api(model_id: str) -> bool:
    return model_id.strip().lower().startswith("gpt-")

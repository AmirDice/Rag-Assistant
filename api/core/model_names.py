"""Resolve LLM model IDs from config YAML with optional env overrides.

Switching Gemini versions (e.g. to Flash-Lite or a future release) should not
require code edits — set GEMINI_VISION_MODEL / GEMINI_GENERATION_MODEL or edit
config/models.yaml.
"""

from __future__ import annotations

from api.core.settings import get_settings

_DEFAULT_GEN = "gemini-2.5-flash"
_DEFAULT_VISION = "gemini-2.5-flash"


def gemini_generation_model() -> str:
    s = get_settings()
    if s.gemini_generation_model.strip():
        return s.gemini_generation_model.strip()
    cfg = s.models_config()
    return cfg.get("generation", {}).get("model", _DEFAULT_GEN)


def gemini_vision_model() -> str:
    s = get_settings()
    if s.gemini_vision_model.strip():
        return s.gemini_vision_model.strip()
    cfg = s.models_config()
    return cfg.get("vision", {}).get("gemini", {}).get("model", _DEFAULT_VISION)

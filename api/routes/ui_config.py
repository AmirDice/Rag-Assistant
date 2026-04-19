"""UI-facing model options and persisted chat preferences (data/ui_preferences.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from api.core.generation_catalog import GENERATION_MODEL_IDS, generation_options_payload
from api.core.settings import get_settings

router = APIRouter()

_PREFS_NAME = "ui_preferences.json"

RERANKER_IDS = ("voyage", "cohere", "bge", "none")


def _prefs_path() -> Path:
    return Path(get_settings().data_dir) / _PREFS_NAME


class UiPreferences(BaseModel):
    reranker: Optional[str] = None
    generation_model: Optional[str] = None

    @field_validator("reranker")
    @classmethod
    def _rerank(cls, v: Optional[str]) -> Optional[str]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        low = v.strip().lower()
        if low not in RERANKER_IDS:
            raise ValueError(f"reranker must be one of {RERANKER_IDS}")
        return low

    @field_validator("generation_model")
    @classmethod
    def _gen(cls, v: Optional[str]) -> Optional[str]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        s = v.strip()
        if len(s) > 128:
            raise ValueError("generation_model too long")
        if s not in GENERATION_MODEL_IDS:
            raise ValueError(f"generation_model must be one of {GENERATION_MODEL_IDS}")
        return s


@router.get("/config/model-options")
async def model_options() -> dict[str, Any]:
    """Reranker choices and generation model list + YAML defaults."""
    cfg = get_settings().models_config()
    rer = cfg.get("reranker", {})
    gen = cfg.get("generation", {})
    return {
        "reranker_yaml_active": str(rer.get("active", "voyage")),
        "reranker_options": [{"id": rid, "label": rid} for rid in RERANKER_IDS],
        "generation_yaml_default": str(gen.get("model", "gemini-2.5-flash")),
        "generation_options": generation_options_payload(),
    }


@router.get("/config/ui-preferences")
async def get_ui_preferences() -> dict[str, Any]:
    p = _prefs_path()
    if not p.is_file():
        return {"reranker": None, "generation_model": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"reranker": None, "generation_model": None}


@router.put("/config/ui-preferences")
async def put_ui_preferences(body: UiPreferences) -> dict[str, str]:
    p = _prefs_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"reranker": body.reranker, "generation_model": body.generation_model}
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"status": "saved"}

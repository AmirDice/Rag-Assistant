"""Pydantic models for the audio-pipeline call-analysis output."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


_SEC_SUFFIX = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*s?\s*$", re.IGNORECASE)

_TRUTHY_STR = frozenset({"true", "1", "yes", "y", "si", "sí", "verdadero", "ok"})
_FALSY_STR = frozenset({"false", "0", "no", "n", "falso"})
# Substrings suggesting unresolved vs resolved (Spanish call summaries).
_RESOL_NEG_HINTS = (
    "no se confirma",
    "no se confir",
    "sin resolver",
    "pendiente",
    "no resuelto",
    "no queda",
    "no se puede confirmar",
)
_RESOL_POS_HINTS = ("resuelto", "confirmado", "solucionado", "exitosamente", "éxito", "exito")

# LLMs often return words instead of a float for RAGPair.confidence.
_CONFIDENCE_WORDS: dict[str, float] = {
    "high": 0.85,
    "very high": 0.95,
    "veryhigh": 0.95,
    "medium": 0.6,
    "med": 0.6,
    "mid": 0.6,
    "low": 0.35,
    "very low": 0.15,
    "verylow": 0.15,
    "alto": 0.85,
    "medio": 0.6,
    "bajo": 0.35,
}


def _coerce_confidence(v: Any) -> float:
    if v is None:
        return 0.5
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return max(0.0, min(1.0, float(v)))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _CONFIDENCE_WORDS:
            return _CONFIDENCE_WORDS[s]
        try:
            return max(0.0, min(1.0, float(s)))
        except ValueError:
            return 0.5
    return 0.5


def _normalize_resolucion_exitosa(raw: Any, resolucion: str) -> tuple[bool, str]:
    """Coerce LLM output to bool; prose in this field is merged into ``resolucion``."""
    resolucion = resolucion if isinstance(resolucion, str) else ""
    if isinstance(raw, bool):
        return raw, resolucion
    if raw is None:
        return False, resolucion
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw != 0, resolucion
    if isinstance(raw, float):
        return raw != 0.0, resolucion
    if not isinstance(raw, str):
        return False, resolucion
    s = raw.strip()
    if not s:
        return False, resolucion
    low = s.lower()
    if low in _TRUTHY_STR:
        return True, resolucion
    if low in _FALSY_STR:
        return False, resolucion
    if any(h in low for h in _RESOL_NEG_HINTS):
        return False, resolucion
    if any(h in low for h in _RESOL_POS_HINTS):
        return True, resolucion
    # Sentence or garbage in the bool slot: keep text, default to False.
    base = resolucion.strip()
    merged = (base + ("\n\n" if base else "") + s).strip()
    return False, merged


class TranscriptLine(BaseModel):
    start: float
    end: float
    speaker: str
    text: str

    @field_validator("start", "end", mode="before")
    @classmethod
    def coerce_seconds(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("boolean is not a valid timestamp")
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            m = _SEC_SUFFIX.match(v.strip())
            if m:
                return float(m.group(1))
            return float(v.strip())
        raise ValueError(f"invalid timestamp: {v!r}")

    @field_validator("speaker", "text", mode="before")
    @classmethod
    def empty_str_if_none(cls, v: Any) -> str:
        if v is None:
            return ""
        return v if isinstance(v, str) else str(v)


class RAGPair(BaseModel):
    question: str = Field(..., description="Natural-language question a user might ask")
    answer: str = Field(..., description="Step-by-step resolution as if from documentation")
    category: str = Field(..., description='Domain category, e.g. "stock-management"')
    confidence: float = Field(..., ge=0, le=1, description="LLM self-assessment 0-1")

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> float:
        return _coerce_confidence(v)


_CALL_REQ_STR_KEYS = (
    "call_id",
    "source_file",
    "source_file_hash",
    "timestamp_start",
    "timestamp_end",
    "problema_corto",
    "descripcion_problema",
    "resolucion",
    "resumen",
)


class CallAnalysis(BaseModel):
    call_id: str = Field(..., description='E.g. "CALL-001"')
    source_file: str
    source_file_hash: str = Field(..., description="SHA-256 of the normalised source audio")
    timestamp_start: str = Field(..., description='Relative to recording, e.g. "01:41"')
    timestamp_end: str
    farmacia: str | None = None
    llamante: str | None = None
    agent: str | None = None
    problema_corto: str = Field(..., description="One-line problem statement")
    descripcion_problema: str = Field(..., description="Full description with context")
    causa_raiz: str | None = None
    resolucion: str = Field(..., description="Step-by-step resolution path")
    resolucion_exitosa: bool = Field(..., description="Was the issue resolved in the call?")
    resumen: str = Field(..., description="Narrative summary")
    rag_qa: list[RAGPair] = Field(default_factory=list, description="1-4 Q&A pairs for RAG")
    software_features: list[str] = Field(default_factory=list)
    error_codes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    transcript: list[TranscriptLine] = Field(default_factory=list)
    processing_metadata: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def coerce_llm_nulls(cls, data: Any) -> Any:
        """LLMs often emit null for optional-looking fields and times like '904.5s'."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for key in _CALL_REQ_STR_KEYS:
            if out.get(key) is None:
                out[key] = ""
        rex = out.get("resolucion_exitosa")
        if rex is not None and not isinstance(rex, bool):
            b, res = _normalize_resolucion_exitosa(rex, out.get("resolucion") or "")
            out["resolucion_exitosa"] = b
            out["resolucion"] = res
        elif rex is None:
            out["resolucion_exitosa"] = False
        for key in ("rag_qa", "software_features", "error_codes", "tags", "transcript"):
            if out.get(key) is None:
                out[key] = []
        if out.get("processing_metadata") is None:
            out["processing_metadata"] = {}
        return out

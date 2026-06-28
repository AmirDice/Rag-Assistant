"""WP12 — Document classifier + metadata enrichment.

Uses the self-contained LLM classifier (api/pipeline/doc_type_llm.py) for
accurate doc-type detection, then enriches with version/module metadata.
Falls back to heuristic rules if no LLM API key is available.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from langdetect import detect

from api.core.models import DocumentMeta
from api.pipeline.doc_type_llm import classify_document_text

logger = logging.getLogger(__name__)


# ── LLM-based classification ─────────────────────────────────

def _llm_classify(filepath: Path, text_sample: str) -> Optional[dict]:
    """Classify via the self-contained doc_type LLM. Returns result dict or None."""
    result = classify_document_text(filename=filepath.name, text=text_sample or "")
    if result:
        logger.info(
            "LLM classified %s → %s (confidence=%.0f%%)",
            filepath.name,
            result.get("type_id", "unknown"),
            result.get("confidence", 0) * 100,
        )
    return result


# ── Mapping from LLM result to DocumentMeta ───────────────────

_VALID_TYPES = {
    "structured_manual", "module_manual", "operational_guide",
    "changelog_as_manual", "changelog_pure", "faq_document",
    "technical_spec", "training_material", "technical_note",
    "presentation",
}

# Normalize LLM type_ids to our canonical types
_TYPE_ALIASES = {
    "technical_note": "technical_spec",
    "presentation": "training_material",
    "release_notes": "changelog_pure",
    "faq": "faq_document",
    "step_by_step_guide": "operational_guide",
    "procedure_guide": "operational_guide",
    "user_manual": "structured_manual",
    "installation_manual": "module_manual",
    "configuration_guide": "module_manual",
    "integration_manual": "module_manual",
}

# Map LLM chunking_strategy to our doc_types.yaml chunk_unit
_CHUNKING_MAP = {
    "by_section": "section",
    "by_page": "section",
    "single_chunk": "single",
    "by_procedure": "procedure",
}

_VERSION_PATTERN = re.compile(
    r"(?:versi[oó]n|v\.?)\s*(\d+(?:\.\d+){0,2})",
    re.IGNORECASE,
)
_MODULE_PATTERN = re.compile(
    r"(?:m[oó]dulo|module)\s*[:\-]?\s*(\w[\w\s]{1,30})",
    re.IGNORECASE,
)
# Generic integration/automation keywords (override per domain as needed).
_ROBOT_KEYWORDS = {"robot", "automation", "integration", "connector"}


def classify_document(
    filepath: Path,
    text_sample: str,
    *,
    max_sample_chars: int = 3000,
) -> DocumentMeta:
    """Return enriched metadata for a single document.

    Tries the LLM classifier first; falls back to heuristics if unavailable.
    """
    fname = filepath.stem.lower()
    suffix = filepath.suffix.lower().lstrip(".")
    sample = text_sample[:max_sample_chars].lower()

    # 1) Try LLM classification
    llm_result = _llm_classify(filepath, text_sample[:max_sample_chars])

    if llm_result and llm_result.get("type_id") != "unknown":
        doc_type = _normalize_type(llm_result["type_id"])
        hints = llm_result.get("pipeline_hints", {})
        discard = hints.get("discard", False)
        confidence = llm_result.get("confidence", 0)
    else:
        doc_type = _heuristic_type(fname, sample)
        discard = False
        confidence = 0.0

    # 2) Version range (always heuristic — LLM doesn't extract this)
    version_min, version_max = _extract_versions(sample)

    # 3) Module
    module_id = _extract_module(fname, sample)

    # 4) Robot doc?
    is_robot = any(kw in fname or kw in sample for kw in _ROBOT_KEYWORDS)
    if "robots" in str(filepath).lower():
        is_robot = True

    # 5) Language
    lang = _detect_lang(text_sample[:500])

    # 6) Changelog purity check
    if doc_type == "changelog_pure" and _has_functional_content(sample):
        doc_type = "changelog_as_manual"
        discard = False

    # 7) Discard rule — only truly empty changelogs
    if doc_type == "changelog_pure" and len(text_sample.strip()) < 200:
        discard = True

    return DocumentMeta(
        doc_id=filepath.stem,
        doc_type=doc_type,
        module_id=module_id,
        version_min=version_min,
        version_max=version_max,
        is_robot_doc=is_robot,
        ocr_needed=False,
        format=suffix if suffix in ("pdf", "docx", "pptx", "xlsx") else "pdf",
        lang=lang,
        discard=discard,
        source_file=filepath.name,
    )


def _normalize_type(type_id: str) -> str:
    """Map an LLM-proposed type_id to our canonical set."""
    type_id = type_id.lower().strip()
    if type_id in _VALID_TYPES:
        return type_id
    return _TYPE_ALIASES.get(type_id, "structured_manual")


# ── Heuristic fallback (when no LLM is available) ────────────

_RULES: list[tuple[str, list[str], list[str]]] = [
    ("faq_document",       ["faq", "preguntas"],            ["pregunta", "respuesta", "¿"]),
    ("operational_guide",  ["guia", "procedimiento", "paso"], ["paso 1", "instrucciones", "procedimiento"]),
    ("changelog_pure",     ["changelog", "novedades", "cambios"], []),
    ("changelog_as_manual", [],                              ["versión", "novedad", "configuración"]),
    ("module_manual",      ["modulo", "módulo", "robot"],   ["módulo", "instalación del módulo"]),
    ("training_material",  ["formacion", "formación", "training"], ["diapositiva", "slide"]),
    ("technical_spec",     ["spec", "api", "técnic"],       ["endpoint", "parámetro", "api"]),
    ("structured_manual",  ["manual"],                       ["capítulo", "sección", "índice"]),
]


def _heuristic_type(fname: str, sample: str) -> str:
    for doc_type, fname_kws, content_kws in _RULES:
        if any(kw in fname for kw in fname_kws):
            return doc_type
        if content_kws and sum(1 for kw in content_kws if kw in sample) >= 2:
            return doc_type
    return "structured_manual"


def _extract_versions(sample: str) -> tuple[float, Optional[float]]:
    matches = _VERSION_PATTERN.findall(sample)
    if not matches:
        return 0.0, None
    versions = sorted(set(_parse_version(m) for m in matches))
    return versions[0], versions[-1] if len(versions) > 1 else None


def _parse_version(raw: str) -> float:
    parts = raw.split(".")
    try:
        return float(f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else parts[0])
    except (ValueError, IndexError):
        return 0.0


def _extract_module(fname: str, sample: str) -> Optional[str]:
    m = _MODULE_PATTERN.search(sample)
    if m:
        return m.group(1).strip().lower()
    for kw in ("integration", "connector", "automation", "api", "module", "robot"):
        if kw in fname or kw in sample:
            return kw
    return None


def _detect_lang(text: str) -> str:
    """Detect the document language (ISO 639-1, e.g. 'en', 'es', 'ca')."""
    try:
        return detect(text)
    except Exception:
        return "en"


def _has_functional_content(sample: str) -> bool:
    functional_kws = ["configuración", "pantalla", "botón", "campo", "paso"]
    return sum(1 for kw in functional_kws if kw in sample) >= 2

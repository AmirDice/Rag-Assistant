"""Query preprocessing — spell/accent normalization + glossary-aware enrichment.

Runs before retrieval so the embedder (and, later, sparse BM25) sees the
canonical terminology instead of the user's shorthand/typos. The original
question is ALWAYS preserved and used for reranking + answer generation — we
only substitute for retrieval.

Pipeline
--------
When ``query_preprocessor.enabled`` is false, the raw question is used as-is.

When enabled:
1. Deterministic glossary match (no LLM cost, no hallucination risk): match
   enabled glossary aliases against the normalized question (accent/case
   insensitive, whole-word) and collect canonical term + optional expansion.
2. Optional AI pass (``ai_enabled`` + ``ai_model_id`` in config/models.yaml).
   The LLM returns ``{"corrected": str, "expansions": [...], "ambiguous": bool}``
   and is instructed to preserve intent and skip ambiguous rewrites. Failures
   are swallowed — the deterministic result still stands.
3. Final retrieval query = ``<corrected> [<canonical terms>] [<expansions>]``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from api.core.glossary_store import (
    GlossaryEntry,
    glossary_fingerprint,
    normalize_text,
    resolve_effective_entries,
)
from api.core.settings import get_settings

logger = logging.getLogger(__name__)

_MIN_QUERY_LEN = 2
_MAX_ENRICHMENT_CHARS = 400  # keep enriched query reasonable for embedding


@dataclass
class PreprocessedQuery:
    """Output of :func:`preprocess_query`.

    original : the untouched user question (used for rerank + generation).
    corrected : original with obvious spelling fixes (may equal original).
    retrieval_query : what gets embedded (corrected + glossary tokens).
    matched_terms : canonical glossary terms that matched.
    expansions : extra tokens appended to widen lexical recall.
    ai_used : True iff the LLM pass ran successfully.
    fingerprint : stable short hash for cache / telemetry keying.
    notes : debug messages (swallowed LLM errors, trimmed expansions, etc.).
    """

    original: str
    corrected: str
    retrieval_query: str
    matched_terms: list[str] = field(default_factory=list)
    expansions: list[str] = field(default_factory=list)
    ai_used: bool = False
    fingerprint: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "corrected": self.corrected,
            "retrieval_query": self.retrieval_query,
            "matched_terms": list(self.matched_terms),
            "expansions": list(self.expansions),
            "ai_used": self.ai_used,
            "fingerprint": self.fingerprint,
            "notes": list(self.notes),
        }


def _preprocessor_cfg() -> dict:
    try:
        models = get_settings().models_config() or {}
    except Exception:
        return {}
    cfg = models.get("query_preprocessor") or {}
    return cfg if isinstance(cfg, dict) else {}


def _preprocessor_enabled() -> bool:
    return bool(_preprocessor_cfg().get("enabled", True))


def _ai_enabled() -> bool:
    return bool(_preprocessor_cfg().get("ai_enabled", False))


def _ai_model_id() -> str:
    return str(_preprocessor_cfg().get("ai_model_id") or "").strip()


def _match_glossary(
    question: str,
    entries: list[GlossaryEntry],
) -> tuple[list[str], list[str]]:
    """Return (matched canonical terms, expansions) using normalized whole-word match."""
    if not question or not entries:
        return [], []
    norm_q = normalize_text(question)
    if not norm_q:
        return [], []
    padded = f" {norm_q} "

    matched: list[str] = []
    expansions: list[str] = []
    seen_terms: set[str] = set()

    for entry in entries:
        aliases = list(entry.aliases) + [entry.term]
        hit = False
        for alias in aliases:
            norm_a = normalize_text(alias)
            if not norm_a:
                continue
            if f" {norm_a} " in padded:
                hit = True
                break
        if not hit:
            continue
        term_key = entry.term.strip()
        if term_key and term_key not in seen_terms:
            matched.append(term_key)
            seen_terms.add(term_key)
        if entry.expansion.strip():
            expansions.append(entry.expansion.strip())
    return matched, expansions


def _build_retrieval_query(
    corrected: str,
    matched_terms: list[str],
    expansions: list[str],
) -> str:
    parts: list[str] = []
    if corrected.strip():
        parts.append(corrected.strip())
    tokens: list[str] = []
    for t in matched_terms + expansions:
        t_clean = t.strip()
        if not t_clean:
            continue
        if normalize_text(t_clean) in normalize_text(corrected):
            continue
        tokens.append(t_clean)
    seen: set[str] = set()
    deduped: list[str] = []
    for t in tokens:
        key = normalize_text(t)
        if key and key not in seen:
            seen.add(key)
            deduped.append(t)
    if deduped:
        enrichment = " ".join(deduped)
        if len(enrichment) > _MAX_ENRICHMENT_CHARS:
            enrichment = enrichment[:_MAX_ENRICHMENT_CHARS].rsplit(" ", 1)[0]
        parts.append(enrichment)
    return " ".join(parts).strip()


def _ai_system_prompt() -> str:
    return (
        "Eres un corrector de consultas de búsqueda para un asistente RAG. "
        "Tu tarea es:\n"
        "1. Detectar y corregir ÚNICAMENTE errores ortográficos claros y "
        "acentos mal puestos en la pregunta del usuario.\n"
        "2. Si hay un término del glosario aplicable, sugerirlo en la lista "
        "`expansions` — pero solo si no está ya en la pregunta.\n"
        "3. NO cambies el significado ni la intención de la pregunta.\n"
        "4. NO inventes términos; si la consulta es ambigua, devuélvela sin "
        "cambios y marca `ambiguous` como true.\n"
        "5. NO traduzcas ni alteres el idioma original.\n"
        "6. Responde EXCLUSIVAMENTE con JSON válido:\n"
        '{"corrected": "...", "expansions": ["..."], "ambiguous": false}'
    )


def _ai_user_prompt(question: str, matched_terms: list[str], entries: list[GlossaryEntry]) -> str:
    glossary_lines = []
    for e in entries[:60]:
        alias_str = ", ".join(e.aliases[:6]) if e.aliases else ""
        glossary_lines.append(f"- {e.term}: {alias_str}")
    ctx_block = "\n".join(glossary_lines) or "(glosario vacío)"
    matched_block = ", ".join(matched_terms) if matched_terms else "(ninguno)"
    return (
        f"Glosario disponible:\n{ctx_block}\n\n"
        f"Términos ya detectados por coincidencia literal: {matched_block}\n\n"
        f"Pregunta del usuario: {question}"
    )


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_ai_json(text: str) -> Optional[dict]:
    if not text:
        return None
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass
    m = _JSON_BLOCK_RE.search(stripped)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _is_over_correction(original: str, corrected: str) -> bool:
    """Guard against the LLM rewriting the user's intent.

    Heuristic: reject if >~40% of normalized tokens disappeared, or the
    corrected version is substantially longer than the original.
    """
    orig_tokens = set(normalize_text(original).split())
    new_tokens = set(normalize_text(corrected).split())
    if not orig_tokens:
        return False
    lost = orig_tokens - new_tokens
    if len(lost) / max(len(orig_tokens), 1) > 0.4:
        return True
    if len(corrected) > max(len(original) * 2, len(original) + 60):
        return True
    return False


async def _run_ai_correction(
    question: str,
    matched_terms: list[str],
    entries: list[GlossaryEntry],
) -> tuple[Optional[str], list[str], Optional[str]]:
    """Return (corrected or None, extra_expansions, error-or-None).

    Best-effort: failures are swallowed and the deterministic result is kept.
    """
    model_id = _ai_model_id()
    if not model_id:
        return None, [], "no_model_id"
    try:
        from api.core.backends import build_generation_backend
        from api.core.generation_catalog import generation_catalog, resolve_generation_model_id
    except Exception as exc:
        return None, [], f"import_error:{exc}"

    try:
        resolved = resolve_generation_model_id(model_id)
        model_cfg = generation_catalog().get(resolved)
        if model_cfg is None:
            return None, [], f"unknown_model_id:{model_id}"
        backend = build_generation_backend(get_settings(), model_cfg)
        sys_prompt = _ai_system_prompt()
        user_prompt = _ai_user_prompt(question, matched_terms, entries)
        timeout_sec = float(_preprocessor_cfg().get("ai_timeout_sec", 4.0))
        result = await asyncio.wait_for(
            backend.generate(prompt=sys_prompt, chunks=[user_prompt], images=None),
            timeout=timeout_sec,
        )
        raw_response = result.text or ""
    except asyncio.TimeoutError:
        return None, [], "ai_timeout"
    except Exception as exc:
        return None, [], f"ai_error:{exc}"

    parsed = _parse_ai_json(raw_response)
    if not parsed:
        return None, [], "ai_parse_error"
    if parsed.get("ambiguous") is True:
        return None, [], "ambiguous"
    corrected_raw = str(parsed.get("corrected") or "").strip()
    corrected = corrected_raw if corrected_raw else None
    if corrected and _is_over_correction(question, corrected):
        return None, [], "over_correction"
    extras = [str(x).strip() for x in (parsed.get("expansions") or []) if str(x).strip()]
    return corrected, extras, None


def _compute_fingerprint(
    question: str,
    retrieval_query: str,
    tenant_id: Optional[str],
    ai_used: bool,
) -> str:
    gloss_hash = glossary_fingerprint(tenant_id)
    raw = f"{question}||{retrieval_query}||{gloss_hash}||{int(ai_used)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


async def preprocess_query(
    question: str,
    *,
    tenant_id: Optional[str] = None,
    use_ai: Optional[bool] = None,
) -> PreprocessedQuery:
    """Return a :class:`PreprocessedQuery` for ``question``.

    Always returns something usable — even if the AI step fails, the
    deterministic glossary pass still produces an enriched query.
    """
    original = (question or "").strip()
    if len(original) < _MIN_QUERY_LEN or not _preprocessor_enabled():
        return PreprocessedQuery(
            original=original,
            corrected=original,
            retrieval_query=original,
            fingerprint=_compute_fingerprint(original, original, tenant_id, False),
        )

    try:
        entries = resolve_effective_entries(tenant_id)
    except Exception as exc:  # pragma: no cover - glossary store should not fail
        logger.warning("Glossary resolve failed: %s", exc)
        entries = []
    matched_terms, expansions = _match_glossary(original, entries)

    notes: list[str] = []
    corrected = original
    ai_used = False
    should_use_ai = use_ai if use_ai is not None else _ai_enabled()
    if should_use_ai:
        ai_corrected, ai_expansions, err = await _run_ai_correction(original, matched_terms, entries)
        if err:
            notes.append(f"ai:{err}")
        if ai_corrected and ai_corrected != original:
            corrected = ai_corrected
            ai_used = True
        if ai_expansions:
            ai_used = True
            expansions = list(expansions) + ai_expansions

    retrieval_query = _build_retrieval_query(corrected, matched_terms, expansions) or original
    fingerprint = _compute_fingerprint(original, retrieval_query, tenant_id, ai_used)
    out = PreprocessedQuery(
        original=original,
        corrected=corrected,
        retrieval_query=retrieval_query,
        matched_terms=matched_terms,
        expansions=expansions,
        ai_used=ai_used,
        fingerprint=fingerprint,
        notes=notes,
    )
    if retrieval_query != original or matched_terms:
        logger.info(
            "query_preprocessed tenant=%s original=%r retrieval=%r matched=%s ai_used=%s",
            tenant_id, original, retrieval_query, matched_terms, ai_used,
        )
    return out

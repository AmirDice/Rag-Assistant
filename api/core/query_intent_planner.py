"""Grounded query refinement planner for the query endpoint.

Runs after a preliminary retrieval. It is deliberately conservative: it may
refine a search query, but it should only ask the user for clarification when
the retrieved evidence supports multiple plausible interpretations. Disabled by
default (config ``query_refinement.enabled``); on any error it falls back to a
plain "search" decision so the normal answer path is unaffected.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from api.core.backends import build_generation_backend
from api.core.generation_catalog import generation_catalog, resolve_generation_model_id
from api.core.models import AnswerChunk, PriorTurn
from api.core.query_preprocessor import PreprocessedQuery
from api.core.settings import get_settings

logger = logging.getLogger(__name__)

PlannerAction = Literal["search", "ask_clarification", "no_answer"]

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_THOUGHT_RE = re.compile(r"<(?:think|thought)\b[^>]*>.*?</(?:think|thought)>", re.IGNORECASE | re.DOTALL)


@dataclass
class QueryRefinementDecision:
    action: PlannerAction = "search"
    original_query: str = ""
    corrected_query: str = ""
    final_query: str = ""
    retrieval_query: str = ""
    clarification_question: str = ""
    confidence: float = 0.0
    reasoning_summary: str = ""
    is_followup_to_clarification: bool = False
    query_session_id: str | None = field(default_factory=lambda: str(uuid4()))
    parent_query_id: str | None = None
    model_id: str = ""
    used: bool = False
    fallback_reason: str = ""

    @property
    def needs_clarification(self) -> bool:
        return self.action == "ask_clarification" and bool(self.clarification_question.strip())

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "original_query": self.original_query,
            "corrected_query": self.corrected_query,
            "final_query": self.final_query,
            "retrieval_query": self.retrieval_query,
            "clarification_question": self.clarification_question,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "is_followup_to_clarification": self.is_followup_to_clarification,
            "model_id": self.model_id,
            "used": self.used,
            "fallback_reason": self.fallback_reason,
        }


def query_refinement_config() -> dict[str, Any]:
    try:
        cfg = get_settings().models_config().get("query_refinement") or {}
    except Exception:
        cfg = {}
    return cfg if isinstance(cfg, dict) else {}


def query_refinement_enabled() -> bool:
    return bool(query_refinement_config().get("enabled", False))


def preliminary_top_k() -> int:
    try:
        value = int(query_refinement_config().get("preliminary_top_k", 5))
    except (TypeError, ValueError):
        value = 5
    return max(1, min(value, 20))


def _model_id() -> str:
    return str(query_refinement_config().get("model_id") or "").strip()


def _timeout_sec() -> float:
    try:
        return max(0.5, float(query_refinement_config().get("timeout_sec", 4.0)))
    except (TypeError, ValueError):
        return 4.0


def _clarification_min_confidence() -> float:
    try:
        return max(0.0, min(float(query_refinement_config().get("clarification_min_confidence", 0.85)), 1.0))
    except (TypeError, ValueError):
        return 0.85


def _system_prompt() -> str:
    return (
        "Eres un planificador de intención para un asistente RAG técnico.\n"
        "Tu trabajo es refinar la consulta del usuario para mejorar la recuperación.\n"
        "SOLO debes pedir aclaración cuando sea estrictamente necesario.\n\n"
        "Acciones posibles:\n"
        "- search: la intención es clara o razonablemente deducible. Devuelve una consulta final.\n"
        "- ask_clarification: SOLO si hay 2+ interpretaciones MUTUAMENTE EXCLUYENTES que llevan a respuestas completamente distintas.\n"
        "- no_answer: no hay evidencia útil en los fragmentos.\n\n"
        "REGLAS CRÍTICAS (prioridad máxima):\n"
        "1. PREFIERE responder (search). Solo pide aclaración si es imposible dar una respuesta útil.\n"
        "2. Si el usuario responde con palabras cortas ('todas', 'ambas', 'los dos', 'sí', 'el primero', "
        "'la segunda opción', etc.) a una aclaración previa, interpreta su intención en contexto y usa search.\n"
        "   NUNCA vuelvas a preguntar ante respuestas como 'todas' o 'ambas'.\n"
        "3. Si los fragmentos cubren un tema relacionado aunque no exactamente lo pedido, usa search.\n"
        "4. No inventes módulos, menús, campos ni conceptos que no aparezcan en la pregunta o en los fragmentos.\n"
        "5. Corrige faltas obvias y completa abreviaturas cuando estén apoyadas por los fragmentos.\n"
        "6. Conserva el formato pedido ('paso a paso', 'dónde', 'cómo') en final_query.\n"
        "7. Si hay conversación previa, interpreta la nueva pregunta como continuación de esa conversación.\n"
        "8. La confidence debe reflejar tu certeza. Si puedes responder razonablemente, confidence >= 0.85.\n"
        "9. Devuelve EXCLUSIVAMENTE JSON válido, sin markdown ni texto adicional.\n\n"
        "Esquema obligatorio:\n"
        "{"
        '"action":"search|ask_clarification|no_answer",'
        '"corrected_query":"...",'
        '"final_query":"...",'
        '"retrieval_query":"...",'
        '"clarification_question":"...",'
        '"confidence":0.0,'
        '"reasoning_summary":"resumen breve",'
        '"is_followup_to_clarification":false'
        "}"
    )


def _format_prior_turns(prior_turns: list[PriorTurn]) -> str:
    if not prior_turns:
        return "(sin conversación previa)"
    lines: list[str] = []
    for idx, turn in enumerate(prior_turns[-3:], start=1):
        q = (turn.question or "").strip()
        a = (turn.answer or "").strip()
        lines.append(f"Turno {idx} usuario: {q}")
        if a:
            lines.append(f"Turno {idx} asistente: {a}")
    return "\n".join(lines)


def _format_chunks(chunks: list[AnswerChunk]) -> str:
    if not chunks:
        return "(sin fragmentos recuperados)"
    lines: list[str] = []
    for idx, chunk in enumerate(chunks[:5], start=1):
        text = re.sub(r"\s+", " ", (chunk.text or "").strip())
        if len(text) > 500:
            text = text[:497].rstrip() + "..."
        location = chunk.source_doc or "desconocido"
        section = f", sección={chunk.source_section}" if chunk.source_section else ""
        page = f", página={chunk.source_page}" if chunk.source_page else ""
        lines.append(
            f"Fragmento {idx} (score={chunk.score:.3f}, doc={location}{page}{section}):\n{text}"
        )
    return "\n\n".join(lines)


def _user_prompt(
    *,
    preprocessed: PreprocessedQuery,
    chunks: list[AnswerChunk],
    prior_turns: list[PriorTurn],
) -> str:
    scores = [float(c.score or 0.0) for c in chunks]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    margin = top1 - top2 if top2 else 1.0

    prior_block = _format_prior_turns(prior_turns)
    followup_note = ""
    if prior_turns:
        last_answer = (prior_turns[-1].answer or "").strip()
        if last_answer and ("?" in last_answer or "¿" in last_answer):
            followup_note = (
                "\nNOTA IMPORTANTE: El mensaje del usuario parece ser una RESPUESTA a una pregunta "
                "de aclaración que hiciste anteriormente. Interpreta su intención en contexto. "
                "Si responde con algo ambiguo ('todas', 'ambas', 'sí', 'el primero', etc.), "
                "combina todas las opciones mencionadas en la aclaración y usa search.\n"
            )

    return (
        f"Pregunta original:\n{preprocessed.original}\n\n"
        f"Pregunta corregida por preprocesado:\n{preprocessed.corrected}\n\n"
        f"Consulta usada en recuperación preliminar:\n{preprocessed.retrieval_query}\n\n"
        f"Términos de glosario detectados: {', '.join(preprocessed.matched_terms) or '(ninguno)'}\n"
        f"Expansiones: {', '.join(preprocessed.expansions) or '(ninguna)'}\n\n"
        f"Conversación previa:\n{prior_block}\n"
        f"{followup_note}\n"
        f"Señales de recuperación: top1={top1:.3f}, top2={top2:.3f}, margin={margin:.3f}\n\n"
        f"Fragmentos recuperados:\n{_format_chunks(chunks)}"
    )


def parse_planner_json(text: str) -> dict[str, Any] | None:
    raw = _THOUGHT_RE.sub("", str(text or "")).strip()
    if raw.startswith("```"):
        raw = "\n".join(line for line in raw.splitlines() if not line.strip().startswith("```")).strip()
    candidates = [raw]
    match = _JSON_BLOCK_RE.search(raw)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _decision_from_parsed(
    parsed: dict[str, Any],
    *,
    preprocessed: PreprocessedQuery,
    model_id: str,
    parent_query_id: str | None,
) -> QueryRefinementDecision:
    action_raw = str(parsed.get("action") or "search").strip().lower()
    action: PlannerAction = action_raw if action_raw in {"search", "ask_clarification", "no_answer"} else "search"  # type: ignore[assignment]
    corrected = str(parsed.get("corrected_query") or preprocessed.corrected or preprocessed.original).strip()
    final_query = str(parsed.get("final_query") or corrected or preprocessed.original).strip()
    retrieval_query = str(parsed.get("retrieval_query") or final_query or preprocessed.retrieval_query).strip()
    clarification = str(parsed.get("clarification_question") or "").strip()
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    if action == "ask_clarification":
        if not clarification:
            action = "search"
        elif confidence < _clarification_min_confidence():
            action = "search"

    if action == "no_answer" and final_query:
        # Prefer a grounded no-answer from the normal generator unless the
        # planner explicitly cannot form any useful query.
        action = "search"

    return QueryRefinementDecision(
        action=action,
        original_query=preprocessed.original,
        corrected_query=corrected,
        final_query=final_query or preprocessed.original,
        retrieval_query=retrieval_query or preprocessed.retrieval_query or preprocessed.original,
        clarification_question=clarification,
        confidence=confidence,
        reasoning_summary=str(parsed.get("reasoning_summary") or "").strip()[:500],
        is_followup_to_clarification=bool(parsed.get("is_followup_to_clarification")),
        parent_query_id=parent_query_id,
        model_id=model_id,
        used=True,
    )


def fallback_decision(
    preprocessed: PreprocessedQuery,
    *,
    reason: str = "",
    parent_query_id: str | None = None,
) -> QueryRefinementDecision:
    return QueryRefinementDecision(
        action="search",
        original_query=preprocessed.original,
        corrected_query=preprocessed.corrected,
        final_query=preprocessed.corrected or preprocessed.original,
        retrieval_query=preprocessed.retrieval_query or preprocessed.original,
        query_session_id=None,
        parent_query_id=parent_query_id,
        used=False,
        fallback_reason=reason,
    )


async def plan_query_intent(
    *,
    preprocessed: PreprocessedQuery,
    chunks: list[AnswerChunk],
    prior_turns: list[PriorTurn],
    parent_query_id: str | None = None,
) -> QueryRefinementDecision:
    model_id = _model_id()
    if not model_id:
        return fallback_decision(preprocessed, reason="no_model_id", parent_query_id=parent_query_id)
    try:
        resolved = resolve_generation_model_id(model_id)
        model_cfg = generation_catalog().get(resolved)
        if model_cfg is None:
            return fallback_decision(preprocessed, reason=f"unknown_model_id:{model_id}", parent_query_id=parent_query_id)
        backend = build_generation_backend(get_settings(), model_cfg)
        system_prompt = _system_prompt()
        user_prompt = _user_prompt(preprocessed=preprocessed, chunks=chunks, prior_turns=prior_turns)
        started = time.perf_counter()
        result = await asyncio.wait_for(
            backend.generate(prompt=system_prompt, chunks=[user_prompt], images=None),
            timeout=_timeout_sec(),
        )
        logger.info(
            "query_refinement model=%s elapsed_ms=%.0f",
            resolved, (time.perf_counter() - started) * 1000.0,
        )
        raw_response = result.text or ""
    except asyncio.TimeoutError:
        logger.warning("query_refinement_timeout planner_model=%s", model_id)
        return fallback_decision(preprocessed, reason="timeout", parent_query_id=parent_query_id)
    except Exception as exc:
        logger.warning("query_refinement_failed model=%s error=%s", model_id, exc)
        return fallback_decision(preprocessed, reason=f"error:{exc}", parent_query_id=parent_query_id)

    parsed = parse_planner_json(raw_response)
    if not parsed:
        return fallback_decision(preprocessed, reason="parse_error", parent_query_id=parent_query_id)
    return _decision_from_parsed(
        parsed,
        preprocessed=preprocessed,
        model_id=resolve_generation_model_id(model_id),
        parent_query_id=parent_query_id,
    )

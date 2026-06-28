"""LLM answer generation — synthesize an answer from retrieved chunks.

Generation is provider-agnostic: this module assembles the system prompt and
the user message, resolves the requested model id to a catalog entry, and
delegates to a :class:`GenerationBackend` (Gemini / OpenAI-compat / Ollama)
built by the factory. Adding a provider never touches this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from api.core.product import product_labels
from api.core.settings import get_settings
from api.core.models import AnswerChunk

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """Answer text plus generation telemetry (token units + estimated cost)."""

    text: str
    provider: str = ""
    model: str = ""
    units_in: float = 0.0
    units_out: float = 0.0
    units_thoughts: float = 0.0
    units_basis: str = "approx"
    estimated_cost: float = 0.0
    cost_currency: str = "USD"
    cost_breakdown: dict[str, Any] = field(default_factory=dict)


def _system_prompt() -> str:
    p = product_labels()
    return (
        f"You are a technical assistant for {p['short_name']}. "
        "Answer the user's question based ONLY on the provided documentation "
        "excerpts.\n\n"
        "Rules:\n"
        "- Answer in the same language as the user's question.\n"
        "- Cite sources as [Source: document_name, p.XX] when you use information "
        "from a specific excerpt.\n"
        "- If the information is not in the excerpts, say you do not have that "
        "information in the available documentation.\n"
        "- Be concise but complete. Use lists or steps when appropriate.\n"
        "- Do not invent information that is not in the excerpts."
    )


def _build_context(chunks: list[AnswerChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.source_doc or "unknown"
        page = f", p.{chunk.source_page}" if chunk.source_page else ""
        section = f", §{chunk.source_section}" if chunk.source_section else ""
        header = f"[Excerpt {i} — {source}{page}{section}]"
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


async def generate_answer(
    question: str,
    chunks: list[AnswerChunk],
    *,
    generation_model: str | None = None,
) -> GeneratedAnswer:
    """Synthesize an answer from chunks via the resolved generation backend.

    ``generation_model`` is a catalog id (e.g. ``gemini-2.5-flash``,
    ``gpt-4o-mini``, ``llama3.1-local``); when omitted or unknown the default
    from config/models.yaml is used. Returns a :class:`GeneratedAnswer` with
    empty text on any backend failure so the /query route can still serve the
    retrieved chunks.
    """
    from api.core.backends import build_generation_backend
    from api.core.generation_catalog import generation_model_config
    from api.core.cost_meter import price_generation, record_cost_event

    settings = get_settings()
    if not chunks:
        return GeneratedAnswer(text="")

    context = _build_context(chunks)
    user_message = (
        f"Documentation excerpts:\n\n{context}\n\n"
        f"---\n\nUser question: {question}"
    )
    system_prompt = _system_prompt()
    model_cfg = generation_model_config(generation_model)

    try:
        backend = build_generation_backend(settings, model_cfg)
        result = await backend.generate(prompt=system_prompt, chunks=[user_message], images=None)
        answer = (result.text or "").strip()
        priced = price_generation(
            provider=result.provider,
            model=result.model,
            units_in=result.units_in,
            units_out=result.units_out,
            units_thoughts=result.units_thoughts,
        )
        record_cost_event(priced)
        logger.info(
            "Generated answer: %d chars via %s/%s (units_in=%.0f units_out=%.0f basis=%s cost=%.6f %s)",
            len(answer),
            result.provider,
            result.model,
            result.units_in,
            result.units_out,
            result.units_basis,
            priced["estimated_cost"],
            priced["currency"],
        )
        return GeneratedAnswer(
            text=answer,
            provider=result.provider,
            model=result.model,
            units_in=result.units_in,
            units_out=result.units_out,
            units_thoughts=result.units_thoughts,
            units_basis=result.units_basis,
            estimated_cost=float(priced["estimated_cost"]),
            cost_currency=str(priced["currency"]),
            cost_breakdown={"generation": priced},
        )
    except Exception as e:
        logger.error("Answer generation failed (%s/%s): %s", model_cfg.provider, model_cfg.model, e)
        return GeneratedAnswer(text="")

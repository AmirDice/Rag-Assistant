"""LLM answer generation — synthesize an answer from retrieved chunks (Gemini or OpenAI)."""

from __future__ import annotations

import logging

from api.core.product import product_labels
from api.core.settings import get_settings
from api.core.models import AnswerChunk

logger = logging.getLogger(__name__)


def _system_prompt_es() -> str:
    p = product_labels()
    return (
        f"Eres un asistente técnico experto en {p['short_name']}. "
        f"{p['erp_context_es']} "
        "Responde la pregunta del usuario basándote ÚNICAMENTE "
        "en los fragmentos de documentación proporcionados.\n\n"
        "Reglas:\n"
        "- Responde en el mismo idioma que la pregunta del usuario.\n"
        "- Cita las fuentes con [Fuente: nombre_documento, p.XX] cuando uses información "
        "de un fragmento específico.\n"
        "- Si la información no está en los fragmentos, di que no tienes esa información "
        "en la documentación disponible.\n"
        "- Sé conciso pero completo. Usa listas o pasos cuando sea apropiado.\n"
        "- No inventes información que no esté en los fragmentos."
    )


def _build_context(chunks: list[AnswerChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.source_doc or "desconocido"
        page = f", p.{chunk.source_page}" if chunk.source_page else ""
        section = f", §{chunk.source_section}" if chunk.source_section else ""
        header = f"[Fragmento {i} — {source}{page}{section}]"
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


async def _generate_gemini(
    settings,
    model_name: str,
    system_prompt: str,
    user_message: str,
) -> str:
    from google import genai

    client = genai.Client(api_key=settings.google_api_key)
    response = await client.aio.models.generate_content(
        model=model_name,
        contents=[
            {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_message}]},
        ],
    )
    return (response.text or "").strip()


async def _generate_openai(
    settings,
    model_name: str,
    system_prompt: str,
    user_message: str,
) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    msg = response.choices[0].message
    return (msg.content or "").strip()


async def generate_answer(
    question: str,
    chunks: list[AnswerChunk],
    *,
    generation_model: str | None = None,
) -> str:
    """Synthesize an answer from chunks (Gemini or OpenAI, from arg or YAML/env default)."""
    from api.core.generation_catalog import uses_openai_api
    from api.core.model_names import gemini_generation_model

    settings = get_settings()
    if not chunks:
        return ""

    context = _build_context(chunks)
    user_message = (
        f"Fragmentos de documentación:\n\n{context}\n\n"
        f"---\n\nPregunta del usuario: {question}"
    )
    system_prompt = _system_prompt_es()
    model_name = (generation_model or "").strip() or gemini_generation_model()

    try:
        if uses_openai_api(model_name):
            if not settings.openai_api_key:
                logger.warning("OPENAI_API_KEY not set — skipping generation")
                return ""
            answer = await _generate_openai(
                settings, model_name, system_prompt, user_message
            )
        else:
            if not settings.google_api_key:
                logger.warning("GOOGLE_API_KEY not set — skipping generation")
                return ""
            answer = await _generate_gemini(
                settings, model_name, system_prompt, user_message
            )
        logger.info("Generated answer: %d chars", len(answer))
        return answer
    except Exception as e:
        logger.error("Answer generation failed: %s", e)
        return ""

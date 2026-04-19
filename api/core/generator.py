"""LLM answer generation — synthesize a natural-language answer from retrieved chunks."""

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


async def generate_answer(question: str, chunks: list[AnswerChunk]) -> str:
    """Generate a synthesized answer from retrieved chunks (Gemini model from config)."""
    settings = get_settings()
    if not settings.google_api_key:
        logger.warning("GOOGLE_API_KEY not set — skipping generation")
        return ""

    if not chunks:
        return ""

    context = _build_context(chunks)
    user_message = (
        f"Fragmentos de documentación:\n\n{context}\n\n"
        f"---\n\nPregunta del usuario: {question}"
    )

    try:
        from google import genai

        from api.core.model_names import gemini_generation_model

        client = genai.Client(api_key=settings.google_api_key)
        model_name = gemini_generation_model()

        response = await client.aio.models.generate_content(
            model=model_name,
            contents=[
                {"role": "user", "parts": [{"text": _system_prompt_es() + "\n\n" + user_message}]},
            ],
        )
        answer = response.text.strip()
        logger.info("Generated answer: %d chars", len(answer))
        return answer
    except Exception as e:
        logger.error("Answer generation failed: %s", e)
        return ""

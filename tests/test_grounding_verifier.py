"""Unit tests for the post-generation grounding verifier."""

from __future__ import annotations

from api.core.models import AnswerChunk
from api.core.grounding_verifier import verify_answer_grounding


def _chunk(text: str) -> AnswerChunk:
    return AnswerChunk(text=text, score=1.0, source_doc="doc.pdf")


def test_grounded_answer_passes():
    chunks = [
        _chunk("Para cerrar la caja, accede al menú de caja y pulsa cierre diario."),
    ]
    answer = "Accede al menú de caja y pulsa cierre diario para cerrar la caja."
    result = verify_answer_grounding(answer, chunks, min_grounded_ratio=0.6)
    assert result.grounded is True
    assert result.grounded_ratio == 1.0
    assert result.unsupported_sentences == 0


def test_hallucinated_answer_is_flagged():
    chunks = [
        _chunk("Para cerrar la caja, accede al menú de caja y pulsa cierre diario."),
    ]
    answer = (
        "Debes reiniciar el servidor central y reconfigurar el firewall corporativo. "
        "Después instala los controladores gráficos actualizados."
    )
    result = verify_answer_grounding(answer, chunks, min_grounded_ratio=0.6)
    assert result.grounded is False
    assert result.grounded_ratio < 0.6
    assert result.unsupported_sentences == result.checked_sentences


def test_empty_chunks_means_ungrounded():
    result = verify_answer_grounding("Cualquier respuesta sustancial aquí.", [])
    assert result.grounded is False
    assert result.grounded_ratio == 0.0


def test_empty_answer_is_trivially_grounded():
    chunks = [_chunk("texto de la documentación")]
    result = verify_answer_grounding("", chunks)
    assert result.grounded is True
    assert result.checked_sentences == 0


def test_partial_support_respects_threshold():
    chunks = [_chunk("La facturación de recetas se realiza desde el módulo de recetas.")]
    # One supported sentence, one unrelated -> ratio 0.5.
    answer = (
        "La facturación de recetas se realiza desde el módulo de recetas. "
        "Posteriormente conviene revisar las estadísticas meteorológicas regionales."
    )
    lenient = verify_answer_grounding(answer, chunks, min_grounded_ratio=0.5)
    strict = verify_answer_grounding(answer, chunks, min_grounded_ratio=0.8)
    assert lenient.grounded is True
    assert strict.grounded is False

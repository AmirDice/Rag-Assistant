"""Post-generation grounding verifier.

Heuristic verifier that checks whether answer sentences are supported by at
least one retrieved chunk. This is a lightweight, dependency-free safety net to
reduce confident hallucinations when retrieval is weak or ambiguous.

It runs after generation in the /query pipeline: every sentence of the
synthesized answer is tokenized and compared against the token sets of the
retrieved chunks. If too few sentences have lexical support, the answer is
flagged ungrounded so the route can replace it with a safe fallback.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from api.core.models import AnswerChunk

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñü]{4,}", re.IGNORECASE)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_MIN_TOKEN_OVERLAP = 2


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", (text or "").lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(_norm(text)))


def _candidate_sentences(answer: str) -> list[str]:
    out: list[str] = []
    for raw in _SENTENCE_SPLIT_RE.split(answer or ""):
        s = _LIST_PREFIX_RE.sub("", raw.strip())
        if len(s) < 8:
            continue
        out.append(s)
    return out


@dataclass
class GroundingResult:
    grounded: bool
    grounded_ratio: float
    checked_sentences: int
    unsupported_sentences: int


def verify_answer_grounding(
    answer: str,
    chunks: list[AnswerChunk],
    *,
    min_grounded_ratio: float = 0.6,
) -> GroundingResult:
    """Estimate whether generated answer is grounded in retrieved chunks."""
    sents = _candidate_sentences(answer)
    if not sents:
        return GroundingResult(True, 1.0, 0, 0)
    chunk_token_sets = [_tokens(c.text) for c in chunks if (c.text or "").strip()]
    if not chunk_token_sets:
        return GroundingResult(False, 0.0, len(sents), len(sents))

    supported = 0
    for sent in sents:
        st = _tokens(sent)
        if not st:
            supported += 1
            continue
        has_support = any(len(st & ct) >= _MIN_TOKEN_OVERLAP for ct in chunk_token_sets)
        if has_support:
            supported += 1

    ratio = supported / len(sents)
    grounded = ratio >= min_grounded_ratio
    return GroundingResult(
        grounded=grounded,
        grounded_ratio=ratio,
        checked_sentences=len(sents),
        unsupported_sentences=len(sents) - supported,
    )

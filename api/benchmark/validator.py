"""WP14 §14.1 — Dual-LLM validation of synthetic benchmark pairs (split from generator)."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from api.core.product import product_labels

logger = logging.getLogger(__name__)

VALIDATION_PROMPT = """You are validating a benchmark Q&A pair for {short_name} ({erp_context_es}).

Question: {question}
Answer (from chunk): {answer}
Source chunk text:
---
{chunk_text}
---

Evaluate:
1. Is the question clearly answerable from this chunk? (yes/no)
2. Is the answer accurate and supported by the chunk? (yes/no)
3. Confidence score 0.0 to 1.0

Respond in JSON format ONLY (no markdown):
{{"answerable": true/false, "accurate": true/false, "confidence": 0.0}}"""


async def validate_pair(
    pair: Any,
    chunk_text: str,
    provider: str,
    llm_call: Callable[[str, str | None], Awaitable[str]],
    parse_json: Callable[[str], dict],
) -> None:
    """Mutates pair: sets validator_confidence and validated."""
    try:
        pl = product_labels()
        prompt = VALIDATION_PROMPT.format(
            short_name=pl["short_name"],
            erp_context_es=pl["erp_context_es"],
            question=pair.question,
            answer=pair.answer,
            chunk_text=chunk_text,
        )
        raw = await llm_call(prompt, provider)
        result = parse_json(raw)

        pair.validator_confidence = float(result.get("confidence", 0))
        pair.validated = (
            result.get("answerable", False)
            and result.get("accurate", False)
            and pair.validator_confidence >= 0.6
        )
    except Exception as e:
        logger.warning("Validation failed for pair '%s': %s", pair.question[:50], e)
        pair.validator_confidence = 0.0
        pair.validated = False

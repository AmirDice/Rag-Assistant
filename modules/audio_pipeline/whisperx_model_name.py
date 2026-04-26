"""Map ``WHISPERX_MODEL`` / env to the string ``whisperx`` / ``faster_whisper`` expects.

The ASR backend is **faster-whisper** (CTranslate2), not Hugging Face Transformers directly.
Default uses ``large-v3`` for compatibility with older WhisperX Docker images.
You can still opt into turbo via ``WHISPERX_MODEL=large-v3-turbo`` or
``WHISPERX_MODEL=openai/whisper-large-v3-turbo``.

See: https://huggingface.co/openai/whisper-large-v3-turbo (source model card).
"""

from __future__ import annotations

import logging
import os
from typing import Final

logger = logging.getLogger(__name__)

# Transformers Hub id -> faster-whisper CLI ``--model`` value
_ALIASES: Final[dict[str, str]] = {
    "openai/whisper-large-v3-turbo": "large-v3-turbo",
}


def resolve_whisperx_model(env_value: str | None = None) -> str:
    """Return the ``--model`` argument for WhisperX (faster-whisper)."""
    raw = env_value if env_value is not None else os.getenv("WHISPERX_MODEL")
    raw = (raw or "").strip()
    if not raw:
        return "large-v3"
    if raw in _ALIASES:
        to = _ALIASES[raw]
        logger.info(
            "WHISPERX_MODEL=%r is a Hugging Face Transformers id; using faster-whisper name %r "
            "(CTranslate2 weights; same architecture as the OpenAI turbo checkpoint).",
            raw,
            to,
        )
        return to
    return raw

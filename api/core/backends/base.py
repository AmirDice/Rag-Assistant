"""Provider-agnostic generation backend interface.

Every LLM provider (Gemini, OpenAI-compatible gateways, Ollama) implements
:class:`GenerationBackend`. The query/answer pipeline talks only to this
interface, so adding a provider is a new subclass + a factory branch — no
changes to the callers. :class:`GenerationResult` carries token-usage units so
the cost meter can price a call without knowing which provider produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class VisionImagePart:
    label: str
    image: Image.Image


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    provider: str
    units_in: float
    units_out: float
    units_thoughts: float
    units_basis: str


class GenerationBackend(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        chunks: list[str],
        images: list[VisionImagePart] | None,
    ) -> GenerationResult:
        raise NotImplementedError

    async def generate_stream(
        self,
        prompt: str,
        chunks: list[str],
        images: list[VisionImagePart] | None,
    ) -> AsyncIterator[str]:
        """Best-effort streaming; default yields a single chunk from :meth:`generate`."""
        result = await self.generate(prompt, chunks, images)
        text = result.text or ""
        if text:
            yield text

    @abstractmethod
    def supports_vision(self) -> bool:
        """Return True if this backend can accept ``images`` in :meth:`generate`.

        Callers MUST check this before passing a non-empty images list; if False,
        the generator should skip multimodal paths and fall back to text-only.
        """
        raise NotImplementedError


def ensure_single_chunk(chunks: list[str]) -> str:
    if not chunks:
        raise ValueError("chunks must contain exactly one assembled user message")
    if len(chunks) != 1:
        raise ValueError("chunks must contain exactly one assembled user message")
    text = chunks[0].strip()
    if not text:
        raise ValueError("chunks[0] cannot be empty")
    return text

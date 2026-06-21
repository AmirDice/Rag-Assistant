from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator
from io import BytesIO

from openai import AsyncOpenAI

from api.core.backends.base import GenerationBackend, GenerationResult, VisionImagePart, ensure_single_chunk


logger = logging.getLogger(__name__)


class OpenAICompatBackend(GenerationBackend):
    """Any OpenAI-compatible Chat Completions endpoint (OpenAI, Together, Groq,
    Google's OpenAI-compat gateway, local vLLM, …). Selected by `base_url`."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        provider_name: str,
        vision_fallback_backend: GenerationBackend | None,
        max_completion_tokens: int | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._provider_name = provider_name
        self._vision_fallback_backend = vision_fallback_backend
        self._max_completion_tokens = max_completion_tokens

    def supports_vision(self) -> bool:
        return True

    def _completion_cap_kwargs(self) -> dict[str, int]:
        if self._max_completion_tokens is None:
            return {}
        # Chat Completions field name varies by gateway; max_tokens is widely accepted (incl. Google OpenAI-compat).
        return {"max_tokens": self._max_completion_tokens}

    def _image_to_data_url(self, image_part: VisionImagePart) -> str:
        buf = BytesIO()
        image_part.image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    async def generate(
        self,
        prompt: str,
        chunks: list[str],
        images: list[VisionImagePart] | None,
    ) -> GenerationResult:
        user_text = ensure_single_chunk(chunks)
        user_content: str | list[dict[str, object]]
        if images:
            parts: list[dict[str, object]] = [{"type": "text", "text": user_text}]
            for image in images:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_to_data_url(image)},
                    }
                )
            user_content = parts
        else:
            user_content = user_text

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
                **self._completion_cap_kwargs(),
            )
        except Exception as exc:
            if images and self._vision_fallback_backend is not None:
                logger.warning("OpenAI-compatible vision request failed", exc_info=exc)
                return await self._vision_fallback_backend.generate(
                    prompt=prompt,
                    chunks=chunks,
                    images=images,
                )
            raise
        text = (response.choices[0].message.content or "").strip()
        usage = response.usage
        units_in = float(getattr(usage, "prompt_tokens", len(user_text)) or len(user_text))
        units_out = float(getattr(usage, "completion_tokens", len(text)) or len(text))
        units_thoughts = 0.0
        units_basis = "real" if usage is not None else "approx"
        return GenerationResult(
            text=text,
            model=self._model,
            provider=self._provider_name,
            units_in=units_in,
            units_out=units_out,
            units_thoughts=units_thoughts,
            units_basis=units_basis,
        )

    async def generate_stream(
        self,
        prompt: str,
        chunks: list[str],
        images: list[VisionImagePart] | None,
    ) -> AsyncIterator[str]:
        user_text = ensure_single_chunk(chunks)
        if images:
            result = await self.generate(prompt, chunks, images)
            if result.text:
                yield result.text
            return

        user_content: str | list[dict[str, object]] = user_text

        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
                stream=True,
                **self._completion_cap_kwargs(),
            )
        except Exception as exc:
            logger.warning("OpenAI-compatible stream failed; falling back to non-streaming", exc_info=exc)
            result = await self.generate(prompt, chunks, images=None)
            if result.text:
                yield result.text
            return

        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            piece = getattr(delta, "content", None) if delta is not None else None
            if piece:
                yield piece

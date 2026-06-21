from __future__ import annotations

from google import genai

from api.core.backends.base import GenerationBackend, GenerationResult, VisionImagePart, ensure_single_chunk


class GeminiBackend(GenerationBackend):
    def __init__(self, api_key: str, text_model: str, vision_model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._text_model = text_model
        self._vision_model = vision_model

    def supports_vision(self) -> bool:
        return True

    async def generate(
        self,
        prompt: str,
        chunks: list[str],
        images: list[VisionImagePart] | None,
    ) -> GenerationResult:
        user_text = ensure_single_chunk(chunks)
        full_text = prompt + "\n\n" + user_text
        if images:
            contents: list[object] = [full_text]
            for part in images:
                contents.append(f"\n--- Imagen ({part.label}) ---\n")
                contents.append(part.image)
            response = await self._client.aio.models.generate_content(
                model=self._vision_model,
                contents=contents,
            )
            model = self._vision_model
        else:
            response = await self._client.aio.models.generate_content(
                model=self._text_model,
                contents=[{"role": "user", "parts": [{"text": full_text}]}],
            )
            model = self._text_model

        usage = getattr(response, "usage_metadata", None)
        units_in = float(len(user_text))
        units_out = float(len((response.text or "").strip()))
        units_thoughts = 0.0
        units_basis = "approx"
        if usage is not None:
            units_in = float(getattr(usage, "prompt_token_count", units_in) or units_in)
            units_out = float(getattr(usage, "candidates_token_count", units_out) or units_out)
            units_thoughts = float(getattr(usage, "thoughts_token_count", 0) or 0)
            units_basis = "real"

        return GenerationResult(
            text=(response.text or "").strip(),
            model=model,
            provider="google",
            units_in=units_in,
            units_out=units_out,
            units_thoughts=units_thoughts,
            units_basis=units_basis,
        )

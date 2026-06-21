from __future__ import annotations

import base64
import re
from io import BytesIO

import httpx

from api.core.backends.base import GenerationBackend, GenerationResult, VisionImagePart, ensure_single_chunk

THOUGHT_BLOCK_RE = re.compile(r"^\s*<\|channel>thought\n[\s\S]*?<channel\|>\s*")


class OllamaBackend(GenerationBackend):
    """Local / self-hosted models via an Ollama server's /api/chat endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        thinking: bool,
        api_key: str,
        num_predict: int | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._thinking = thinking
        self._api_key = api_key
        self._num_predict = num_predict

    def supports_vision(self) -> bool:
        return True

    def _system_prompt(self, prompt: str) -> str:
        if not self._thinking:
            return prompt
        return f"<|think|>\n{prompt}"

    def _strip_thought_blocks(self, text: str) -> str:
        return THOUGHT_BLOCK_RE.sub("", text).strip()

    def _image_base64(self, image_part: VisionImagePart) -> str:
        buf = BytesIO()
        image_part.image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _request_headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    async def generate(
        self,
        prompt: str,
        chunks: list[str],
        images: list[VisionImagePart] | None,
    ) -> GenerationResult:
        user_text = ensure_single_chunk(chunks)
        message: dict[str, object] = {"role": "user", "content": user_text}
        if images:
            message["images"] = [self._image_base64(image) for image in images]

        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": self._system_prompt(prompt)},
                message,
            ],
        }
        if self._num_predict is not None:
            payload["options"] = {"num_predict": self._num_predict}

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
                headers=self._request_headers(),
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama chat failed status={response.status_code} body={response.text[:1000]}"
            )

        data = response.json()
        msg = data.get("message") or {}
        text = self._strip_thought_blocks(str(msg.get("content") or ""))
        units_in = float(data.get("prompt_eval_count") or len(user_text))
        units_out = float(data.get("eval_count") or len(text))
        return GenerationResult(
            text=text,
            model=self._model,
            provider="ollama",
            units_in=units_in,
            units_out=units_out,
            units_thoughts=0.0,
            units_basis="real" if data.get("prompt_eval_count") is not None else "approx",
        )

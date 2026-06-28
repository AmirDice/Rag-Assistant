"""Embedding abstraction — Voyage (cloud ceiling) / Qwen3 (local baseline).

Both expose the same interface so the retriever doesn't care which is active.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import httpx
import tiktoken

from api.core.settings import get_settings

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")
VOYAGE_MAX_TOKENS = 16000
_MAX_RETRIES = 3
_RETRY_BACKOFF = [2, 5, 15]


def _truncate_to_tokens(text: str, max_tokens: int = VOYAGE_MAX_TOKENS) -> str:
    """Truncate text to fit within the token budget."""
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    logger.warning("Truncating text from %d to %d tokens", len(tokens), max_tokens)
    return _enc.decode(tokens[:max_tokens])


class Embedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of texts."""

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Return a single embedding for a search query."""


class VoyageEmbedder(Embedder):
    def __init__(self) -> None:
        settings = get_settings()
        cfg = settings.models_config()["embedding"]["voyage"]
        self.model = cfg["model"]
        self.batch_size = cfg.get("batch_size", 128)
        self.api_key = settings.voyage_api_key
        self.api_base = cfg.get("api_base", "https://api.voyageai.com/v1")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        safe_texts = [_truncate_to_tokens(t) for t in texts]
        all_embeddings: list[list[float]] = []
        for i in range(0, len(safe_texts), self.batch_size):
            batch = safe_texts[i:i + self.batch_size]
            try:
                resp = await self._call(batch, input_type="document")
                all_embeddings.extend(resp)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400 and len(batch) > 1:
                    logger.warning(
                        "Batch of %d failed with 400, falling back to one-by-one",
                        len(batch),
                    )
                    for j, single in enumerate(batch):
                        try:
                            r = await self._call([single], input_type="document")
                            all_embeddings.extend(r)
                        except httpx.HTTPStatusError:
                            logger.error(
                                "Skipping chunk %d (offset %d) — Voyage 400",
                                i + j, i + j,
                            )
                            all_embeddings.append([0.0] * 1024)
                else:
                    raise
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        result = await self._call([_truncate_to_tokens(query)], input_type="query")
        return result[0]

    async def _call(self, texts: list[str], input_type: str) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{self.api_base}/embeddings",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "input": texts,
                            "input_type": input_type,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return [item["embedding"] for item in data["data"]]
            except (httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout) as e:
                last_err = e
                is_retryable = isinstance(e, (httpx.ConnectError, httpx.ReadTimeout))
                if isinstance(e, httpx.HTTPStatusError):
                    is_retryable = e.response.status_code in (429, 500, 502, 503, 504)
                if not is_retryable:
                    raise
                wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                logger.warning(
                    "Voyage API error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, _MAX_RETRIES, e, wait,
                )
                await asyncio.sleep(wait)
        raise last_err  # type: ignore[misc]


class OpenAICompatibleEmbedder(Embedder):
    """Works with any OpenAI-compatible embedding API (Qwen3 via vLLM, SiliconFlow, etc.)."""

    def __init__(self) -> None:
        settings = get_settings()
        cfg = settings.models_config()["embedding"]["qwen3"]
        self.model = cfg["model"]
        self.batch_size = cfg.get("batch_size", 32)
        self.api_base = cfg["api_base"]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = await self._call(batch)
            all_embeddings.extend(resp)
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        result = await self._call([query])
        return result[0]

    async def _call(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.api_base}/embeddings",
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]


class FallbackEmbedder(Embedder):
    """Try a primary embedder; on failure use a fallback.

    Both providers MUST produce the same vector dimension — otherwise the
    fallback vectors are incompatible with the Qdrant collection. We log a loud
    warning if the configured dimensions differ.
    """

    def __init__(self, primary: Embedder, fallback: Embedder, *, primary_name: str, fallback_name: str) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_name = primary_name
        self._fallback_name = fallback_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self._primary.embed(texts)
        except Exception as e:
            logger.warning("Embedder '%s' failed (%s) — falling back to '%s'", self._primary_name, e, self._fallback_name)
            return await self._fallback.embed(texts)

    async def embed_query(self, query: str) -> list[float]:
        try:
            return await self._primary.embed_query(query)
        except Exception as e:
            logger.warning("Embedder '%s' failed (%s) — falling back to '%s'", self._primary_name, e, self._fallback_name)
            return await self._fallback.embed_query(query)


def _build_embedder(name: str) -> Embedder:
    if name == "voyage":
        return VoyageEmbedder()
    if name == "qwen3":
        return OpenAICompatibleEmbedder()
    raise ValueError(f"Unknown embedding provider: {name}")


_embedder_instance: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance

    cfg = get_settings().models_config()["embedding"]
    active = cfg["active"]
    primary = _build_embedder(active)

    fb_name = str(cfg.get("fallback") or "").strip()
    if fb_name and fb_name != active:
        # Same-dimension requirement: warn if the configured dims differ.
        dim_a = int(cfg.get(active, {}).get("dimensions", 1024))
        dim_b = int(cfg.get(fb_name, {}).get("dimensions", 1024))
        if dim_a != dim_b:
            logger.warning(
                "Embedding fallback '%s' (dim %d) differs from primary '%s' (dim %d) — "
                "vectors would be incompatible; fallback disabled.",
                fb_name, dim_b, active, dim_a,
            )
            _embedder_instance = primary
        else:
            _embedder_instance = FallbackEmbedder(
                primary, _build_embedder(fb_name), primary_name=active, fallback_name=fb_name
            )
            logger.info("Embedder initialized: %s (fallback: %s)", active, fb_name)
    else:
        _embedder_instance = primary
        logger.info("Embedder initialized: %s", active)
    return _embedder_instance


def get_active_embedding_dimensions(settings=None) -> int:
    """Return the configured embedding dimension for the active provider."""
    settings = settings or get_settings()
    cfg = settings.models_config()
    active = cfg["embedding"]["active"]
    dims = cfg["embedding"].get(active, {}).get("dimensions", 1024)
    try:
        return int(dims)
    except (TypeError, ValueError):
        return 1024

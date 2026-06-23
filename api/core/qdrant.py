"""Single place to construct the Qdrant client.

Centralizes connection config so a managed Qdrant (e.g. Qdrant Cloud, which
requires an API key over HTTPS) works everywhere by setting ``QDRANT_URL`` +
``QDRANT_API_KEY`` — no per-call-site changes.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from api.core.settings import Settings, get_settings


def make_qdrant_client(settings: Settings | None = None) -> AsyncQdrantClient:
    s = settings or get_settings()
    api_key = (s.qdrant_api_key or "").strip() or None
    return AsyncQdrantClient(url=s.qdrant_url, api_key=api_key)

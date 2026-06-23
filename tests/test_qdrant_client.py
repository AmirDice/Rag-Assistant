"""Qdrant client factory — centralized construction with optional API key."""

from __future__ import annotations

from types import SimpleNamespace

from qdrant_client import AsyncQdrantClient

from api.core.qdrant import make_qdrant_client


def test_make_client_without_api_key():
    s = SimpleNamespace(qdrant_url="http://localhost:6333", qdrant_api_key="")
    client = make_qdrant_client(s)
    assert isinstance(client, AsyncQdrantClient)


def test_make_client_with_api_key_for_cloud():
    # Qdrant Cloud style: https URL + API key. Construction must not raise.
    s = SimpleNamespace(
        qdrant_url="https://example.cloud.qdrant.io:6333",
        qdrant_api_key="secret-key",
    )
    client = make_qdrant_client(s)
    assert isinstance(client, AsyncQdrantClient)

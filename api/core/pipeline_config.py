"""WP12 deliverable: single JSON view of classifier + chunking config (pipeline_config v2)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from api.core.settings import get_settings


def build_pipeline_config() -> dict:
    """Merge doc_types + tenants + models slices relevant to the ingestion pipeline."""
    settings = get_settings()
    root = Path(settings.config_dir)

    def _load(name: str) -> dict:
        p = root / name
        if not p.exists():
            return {}
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    doc_types = _load("doc_types.yaml")
    tenants = _load("tenants.yaml")
    models = _load("models.yaml")
    product = _load("product.yaml")

    return {
        "version": 2,
        "doc_types": doc_types,
        "tenants": tenants,
        "product": product.get("product", product) if product else {},
        "models": {
            "embedding": models.get("embedding", {}),
            "reranker": models.get("reranker", {}),
            "retrieval": models.get("retrieval", {}),
            "generation": models.get("generation", {}),
            "vision": models.get("vision", {}),
            "cache": models.get("cache", {}),
        },
    }


def pipeline_config_json() -> str:
    return json.dumps(build_pipeline_config(), ensure_ascii=False, indent=2)

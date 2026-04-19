"""Load `config/product.yaml` for white-label prompts (merged with safe defaults)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from api.core.settings import get_settings

_DEFAULTS: dict[str, str] = {
    "short_name": "RAG Assistant",
    "erp_context_es": "Responde únicamente a partir de la documentación indexada que se te proporcione.",
    "reader_role_es": "un usuario profesional",
    "long_name_en": "RAG Assistant",
    "domain_en": "indexed organizational documentation and knowledge",
}


@lru_cache
def product_labels() -> dict[str, str]:
    merged = dict(_DEFAULTS)
    try:
        raw = get_settings().load_yaml("product.yaml")
        block = raw.get("product", {}) if isinstance(raw, dict) else {}
        for k, v in block.items():
            if k in _DEFAULTS and v is not None:
                merged[k] = str(v).strip() or _DEFAULTS[k]
    except (FileNotFoundError, OSError, TypeError, KeyError, ValueError):
        pass
    return merged


def clear_product_cache() -> None:
    """Call after hot-reloading config in tests or admin tooling."""
    product_labels.cache_clear()

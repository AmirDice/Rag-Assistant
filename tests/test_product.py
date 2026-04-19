from __future__ import annotations

import pytest

from api.core import product as prod


def test_product_labels_default():
    prod.clear_product_cache()
    p = prod.product_labels()
    assert p["short_name"] == "RAG Assistant"
    assert "document" in p["erp_context_es"].lower() or "index" in p["erp_context_es"].lower()


def test_product_labels_yaml_override(monkeypatch: pytest.MonkeyPatch):
    prod.clear_product_cache()

    class S:
        def load_yaml(self, name: str):
            if name == "product.yaml":
                return {"product": {"short_name": "AcmeERP"}}
            raise FileNotFoundError

    monkeypatch.setattr("api.core.product.get_settings", lambda: S())
    prod.clear_product_cache()
    try:
        assert prod.product_labels()["short_name"] == "AcmeERP"
    finally:
        prod.clear_product_cache()

from __future__ import annotations

from pathlib import Path

import pytest

from api.core import pipeline_config as pc


def test_build_pipeline_config_structure(monkeypatch: pytest.MonkeyPatch):
    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "config"

    class _S:
        config_dir = str(cfg_dir)

    monkeypatch.setattr(pc, "get_settings", lambda: _S())

    data = pc.build_pipeline_config()
    assert data.get("version") == 2
    assert "doc_types" in data
    assert "tenants" in data
    assert "product" in data
    assert "models" in data
    assert "embedding" in data["models"]

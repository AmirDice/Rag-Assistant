"""Tenant YAML + onboarding overlay merge."""

from __future__ import annotations

import json

import pytest

from api.core import tenant_state as ts


def test_merged_overlay_overrides_yaml(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ts, "load_onboarding_overlays", lambda: {"demo": {"erp_version": 99.0}})

    class S:
        def tenants_config(self):
            return {
                "tenants": {
                    "demo": {
                        "erp_version": 1.0,
                        "contracted_modules": ["rowa"],
                        "has_robot_integration": True,
                    }
                }
            }

    monkeypatch.setattr(ts, "get_settings", lambda: S())
    m = ts.merged_tenant_config("demo")
    assert m["erp_version"] == 99.0
    assert m["contracted_modules"] == ["rowa"]


def test_normalize_onboarding_legacy_erp_to_erp():
    p = ts.normalize_onboarding_patch({"legacy_erp_version": 4.2})
    assert p["erp_version"] == 4.2


def test_save_overlay_roundtrip(tmp_path, monkeypatch: pytest.MonkeyPatch):
    class S:
        data_dir = str(tmp_path)

    monkeypatch.setattr(ts, "get_settings", lambda: S())
    ts.save_tenant_overlay("demo", {"benchmark_lang": "ca"})
    path = tmp_path / "tenant_onboarding.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["demo"]["benchmark_lang"] == "ca"

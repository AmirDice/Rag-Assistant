"""Tenant YAML + optional `data/tenant_onboarding.json` overlay (operator onboarding)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from api.core.settings import get_settings

_ONBOARDING_FILENAME = "tenant_onboarding.json"


def _onboarding_path() -> Path:
    p = Path(get_settings().data_dir) / _ONBOARDING_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_onboarding_overlays() -> dict[str, dict[str, Any]]:
    path = _onboarding_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for tid, blob in raw.items():
        if isinstance(blob, dict):
            out[str(tid)] = dict(blob)
    return out


def save_tenant_overlay(tenant_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Merge `patch` into the stored overlay for `tenant_id` and persist file."""
    path = _onboarding_path()
    all_o = load_onboarding_overlays()
    cur = dict(all_o.get(tenant_id, {}))
    for k, v in patch.items():
        if v is None:
            cur.pop(k, None)
        else:
            cur[k] = v
    if cur:
        all_o[tenant_id] = cur
    else:
        all_o.pop(tenant_id, None)
    path.write_text(json.dumps(all_o, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return cur


def merged_tenant_config(tenant_id: str) -> dict[str, Any]:
    """Shallow merge: `config/tenants.yaml` tenant dict, then onboarding overlay."""
    settings = get_settings()
    tenants = settings.tenants_config().get("tenants", {})
    base: dict[str, Any] = dict(tenants.get(tenant_id, {}) or {})
    overlay = load_onboarding_overlays().get(tenant_id, {})
    merged = {**base, **overlay}
    return merged


def normalize_onboarding_patch(body: dict[str, Any]) -> dict[str, Any]:
    """Map API body to overlay keys; coalesce legacy_erp_version → erp_version."""
    patch: dict[str, Any] = {}
    if "erp_version" in body and body["erp_version"] is not None:
        patch["erp_version"] = body["erp_version"]
    if "legacy_erp_version" in body and body["legacy_erp_version"] is not None:
        if "erp_version" not in patch:
            patch["erp_version"] = body["legacy_erp_version"]
    if "contracted_modules" in body and body["contracted_modules"] is not None:
        patch["contracted_modules"] = list(body["contracted_modules"])
    if "has_robot_integration" in body and body["has_robot_integration"] is not None:
        patch["has_robot_integration"] = bool(body["has_robot_integration"])
    if "preferred_lang" in body and body["preferred_lang"] is not None:
        patch["preferred_lang"] = str(body["preferred_lang"]).strip()
    if "benchmark_lang" in body and body["benchmark_lang"] is not None:
        patch["benchmark_lang"] = str(body["benchmark_lang"]).strip().lower()
    return patch

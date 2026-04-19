"""Tenant profile + onboarding overlay API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Header

from api.core.models import TenantOnboardingUpdate
from api.core.settings import get_settings
from api.core.tenant_state import (
    load_onboarding_overlays,
    merged_tenant_config,
    normalize_onboarding_patch,
    save_tenant_overlay,
)

router = APIRouter(prefix="/tenant", tags=["tenant"])


@router.get("/{tenant_id}/profile")
async def tenant_profile(tenant_id: str) -> dict:
    settings = get_settings()
    yaml_tenants = settings.tenants_config().get("tenants", {})
    if tenant_id not in yaml_tenants and tenant_id not in load_onboarding_overlays():
        raise HTTPException(status_code=404, detail=f"Unknown tenant: {tenant_id}")
    overlay = load_onboarding_overlays().get(tenant_id, {})
    return {
        "tenant_id": tenant_id,
        "from_yaml": yaml_tenants.get(tenant_id, {}),
        "onboarding_overlay": overlay,
        "merged": merged_tenant_config(tenant_id),
    }


@router.put("/{tenant_id}/onboarding")
async def tenant_put_onboarding(
    tenant_id: str,
    body: TenantOnboardingUpdate,
    x_admin_token: str = Header(default="", alias="X-Admin-Token"),
) -> dict:
    settings = get_settings()
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    yaml_tenants = settings.tenants_config().get("tenants", {})
    if tenant_id not in yaml_tenants:
        raise HTTPException(
            status_code=404,
            detail=f"Tenant {tenant_id} not in config/tenants.yaml — add it before onboarding",
        )
    patch = normalize_onboarding_patch(body.model_dump(exclude_none=True))
    if not patch:
        raise HTTPException(status_code=400, detail="No fields to update")
    updated = save_tenant_overlay(tenant_id, patch)
    return {
        "tenant_id": tenant_id,
        "onboarding_overlay": updated,
        "merged": merged_tenant_config(tenant_id),
    }

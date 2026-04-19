"""Unit tests for tenant Qdrant filters and changelog_pure visibility."""

from __future__ import annotations

import pytest
from qdrant_client.models import FieldCondition, Filter, MinShould

from api.core.retriever import (
    _build_tenant_filter,
    _include_changelog_pure,
    _tenant_version_float,
    payload_visible_for_tenant,
)


def test_minimal_tenant_robot_filter_only():
    f = _build_tenant_filter({})
    assert isinstance(f, Filter)
    assert len(f.must) == 1
    assert isinstance(f.must[0], FieldCondition)
    assert f.must[0].key == "is_robot_doc"


def test_contracted_modules_uses_min_should():
    f = _build_tenant_filter(
        {
            "contracted_modules": ["rowa"],
            "has_robot_integration": True,
        }
    )
    nested = [c for c in f.must if isinstance(c, Filter)]
    assert nested
    inner = nested[0]
    assert inner.min_should is not None
    assert isinstance(inner.min_should, MinShould)
    assert inner.min_should.min_count == 1
    assert len(inner.min_should.conditions) == 2


def test_version_filters_float_conversion():
    f = _build_tenant_filter(
        {
            "legacy_erp_version": "5.2",
            "has_robot_integration": True,
        }
    )
    assert len(f.must) >= 2


def test_tenant_version_float_invalid_returns_none():
    assert _tenant_version_float({"legacy_erp_version": "x"}) is None
    assert _tenant_version_float({"legacy_erp_version": ""}) is None
    assert _tenant_version_float({}) is None


def test_erp_version_preferred_over_legacy_erp():
    assert _tenant_version_float({"erp_version": "5.0", "legacy_erp_version": "3.0"}) == 5.0
    assert _tenant_version_float({"erp_version": 4.2}) == 4.2
    assert _tenant_version_float({"legacy_erp_version": "3.1"}) == 3.1


@pytest.mark.parametrize(
    "payload,tv,expect",
    [
        ({"doc_type": "structured_manual"}, None, True),
        ({"doc_type": "changelog_pure", "version_min": 4.0}, None, False),
        ({"doc_type": "changelog_pure", "version_min": 4.0}, 5.0, True),
        ({"doc_type": "changelog_pure", "version_min": 6.0}, 5.0, False),
        (
            {"doc_type": "changelog_pure", "version_min": 4.0, "version_max": 5.0},
            5.0,
            True,
        ),
        (
            {"doc_type": "changelog_pure", "version_min": 4.0, "version_max": 4.5},
            5.0,
            False,
        ),
    ],
)
def test_include_changelog_pure(payload: dict, tv: float | None, expect: bool):
    assert _include_changelog_pure(payload, tv) is expect


def test_payload_visible_module_contract():
    tenant = {"contracted_modules": ["rowa"], "has_robot_integration": True}
    assert payload_visible_for_tenant({"module_id": "rowa", "doc_type": "x"}, tenant)
    assert payload_visible_for_tenant({"module_id": "", "doc_type": "x"}, tenant)
    assert payload_visible_for_tenant({"doc_type": "x"}, tenant)
    assert not payload_visible_for_tenant({"module_id": "other", "doc_type": "x"}, tenant)


def test_payload_visible_version_and_lang():
    tenant = {
        "erp_version": 5.0,
        "has_robot_integration": True,
        "preferred_lang": "es",
    }
    assert payload_visible_for_tenant(
        {"version_min": 4.0, "lang": "es", "doc_type": "x"}, tenant
    )
    assert not payload_visible_for_tenant(
        {"version_min": 6.0, "lang": "es", "doc_type": "x"}, tenant
    )
    assert not payload_visible_for_tenant(
        {"version_min": 4.0, "version_max": 4.5, "lang": "es", "doc_type": "x"}, tenant
    )
    assert not payload_visible_for_tenant(
        {"version_min": 4.0, "lang": "ca", "doc_type": "x"}, tenant
    )


def test_payload_visible_robot_doc():
    tenant = {"has_robot_integration": False}
    assert payload_visible_for_tenant({"is_robot_doc": False, "doc_type": "x"}, tenant)
    assert not payload_visible_for_tenant({"is_robot_doc": True, "doc_type": "x"}, tenant)

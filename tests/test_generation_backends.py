"""Provider-agnostic generation backend: catalog resolution + factory wiring.

These tests construct backends but never make network calls — they assert the
factory returns the right backend type and resolves provider details from the
config catalog.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from api.core.backends import build_generation_backend
from api.core.backends.gemini import GeminiBackend
from api.core.backends.ollama import OllamaBackend
from api.core.backends.openai_compat import OpenAICompatBackend
from api.core.generation_catalog import (
    GenerationModelConfig,
    default_generation_model_id,
    generation_model_config,
    generation_model_ids,
    resolve_generation_model_id,
)


def test_catalog_exposes_all_providers():
    ids = generation_model_ids()
    providers = {generation_model_config(mid).provider for mid in ids}
    # config/models.yaml defines gemini + openai_compat + ollama entries.
    assert {"gemini", "openai_compat", "ollama"} <= providers


def test_resolve_unknown_falls_back_to_default():
    assert resolve_generation_model_id("does-not-exist") == default_generation_model_id()
    assert resolve_generation_model_id(None) == default_generation_model_id()


def test_known_id_resolves_to_itself():
    assert resolve_generation_model_id("gpt-4o-mini") == "gpt-4o-mini"


def _fake_settings(**overrides):
    base = {
        "google_api_key": "",
        "openai_api_key": "",
        "ollama_base_url": "http://localhost:11434",
        "gemini_generation_model": "",
        "gemini_vision_model": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_factory_builds_openai_compat_backend():
    cfg = generation_model_config("gpt-4o-mini")
    settings = _fake_settings(openai_api_key="sk-test")  # google empty -> no vision fallback
    backend = build_generation_backend(settings, cfg)
    assert isinstance(backend, OpenAICompatBackend)


def test_factory_builds_ollama_backend():
    cfg = generation_model_config("llama3.1-local")
    backend = build_generation_backend(_fake_settings(), cfg)
    assert isinstance(backend, OllamaBackend)


def test_factory_builds_gemini_backend():
    cfg = generation_model_config("gemini-2.5-flash")
    backend = build_generation_backend(_fake_settings(google_api_key="dummy"), cfg)
    assert isinstance(backend, GeminiBackend)


def test_openai_compat_requires_api_key():
    cfg = generation_model_config("gpt-4o-mini")
    with pytest.raises(ValueError):
        build_generation_backend(_fake_settings(openai_api_key=""), cfg)


def test_unsupported_provider_raises():
    cfg = GenerationModelConfig(id="x", provider="mystery", model="m")
    with pytest.raises(ValueError):
        build_generation_backend(_fake_settings(), cfg)

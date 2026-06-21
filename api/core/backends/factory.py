from __future__ import annotations

from api.core.backends.base import GenerationBackend
from api.core.backends.gemini import GeminiBackend
from api.core.backends.ollama import OllamaBackend
from api.core.backends.openai_compat import OpenAICompatBackend
from api.core.generation_catalog import GenerationModelConfig
from api.core.model_names import gemini_generation_model, gemini_vision_model
from api.core.settings import Settings


def _resolve_ref(settings: Settings, ref_name: str) -> str:
    value = str(getattr(settings, ref_name, "") or "").strip()
    if not value:
        raise ValueError(f"Missing settings value for ref={ref_name}")
    return value


def build_generation_backend(settings: Settings, model_cfg: GenerationModelConfig) -> GenerationBackend:
    """Construct the concrete backend for a catalog entry.

    Provider details (api keys, base urls) come from the catalog entry, with
    `*_ref` fields pointing at a Settings attribute so secrets stay in env/.env
    rather than YAML.
    """
    provider = model_cfg.provider

    if provider == "gemini":
        # Env override (GEMINI_GENERATION_MODEL) wins as a global default;
        # otherwise the catalog entry's model selects the specific Gemini model.
        text_model = settings.gemini_generation_model.strip() or model_cfg.model
        return GeminiBackend(
            api_key=settings.google_api_key,
            text_model=text_model,
            vision_model=gemini_vision_model(),
        )

    if provider == "openai_compat":
        api_key = model_cfg.api_key or (
            _resolve_ref(settings, model_cfg.api_key_ref) if model_cfg.api_key_ref else settings.openai_api_key.strip()
        )
        if not api_key:
            raise ValueError("openai_compat requires api_key or OPENAI_API_KEY")
        base_url = model_cfg.base_url or (
            _resolve_ref(settings, model_cfg.base_url_ref) if model_cfg.base_url_ref else ""
        )
        if not base_url:
            raise ValueError("openai_compat requires base_url or base_url_ref")
        # If the deploy has a Google key, wire a Gemini vision fallback so a
        # transient vision-API failure on the primary model degrades gracefully.
        fallback_backend: GenerationBackend | None = None
        if settings.google_api_key.strip():
            fallback_backend = GeminiBackend(
                api_key=settings.google_api_key,
                text_model=gemini_generation_model(),
                vision_model=gemini_vision_model(),
            )
            if not fallback_backend.supports_vision():
                fallback_backend = None
        return OpenAICompatBackend(
            api_key=api_key,
            base_url=base_url,
            model=model_cfg.model,
            provider_name="openai_compat",
            vision_fallback_backend=fallback_backend,
            max_completion_tokens=model_cfg.max_completion_tokens,
        )

    if provider == "ollama":
        base_url = model_cfg.base_url or (
            _resolve_ref(settings, model_cfg.base_url_ref) if model_cfg.base_url_ref else settings.ollama_base_url.strip()
        )
        api_key = model_cfg.api_key or (_resolve_ref(settings, model_cfg.api_key_ref) if model_cfg.api_key_ref else "")
        return OllamaBackend(
            base_url=base_url,
            model=model_cfg.model,
            thinking=model_cfg.thinking,
            api_key=api_key,
            num_predict=model_cfg.max_completion_tokens,
        )

    raise ValueError(f"Unsupported generation provider: {provider}")

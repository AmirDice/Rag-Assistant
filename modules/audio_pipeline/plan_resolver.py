from __future__ import annotations

from dataclasses import dataclass

from api.core.settings import get_settings


@dataclass(frozen=True)
class ResolvedAudioPlan:
    transcription: dict[str, object]
    analysis: dict[str, object]


def resolve_for_tenant(tenant_id: str) -> ResolvedAudioPlan:
    del tenant_id
    settings = get_settings()
    transcription = {
        "provider": (getattr(settings, "whisperx_transcribe_mode", "") or "docker"),
        "model": "",
    }
    analysis = {"provider": "ollama", "model": (getattr(settings, "audio_pipeline_analysis_model", "") or "")}
    return ResolvedAudioPlan(transcription=transcription, analysis=analysis)

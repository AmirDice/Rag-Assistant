"""Ollama-based CallAnalysis extraction (Jinja2 prompt, JSON parse, retry)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
from jinja2 import Environment, FileSystemLoader, select_autoescape

from modules.audio_pipeline._env import load_repo_dotenv
from modules.audio_pipeline.json_utils import extract_json_object
from modules.audio_pipeline.schemas import CallAnalysis, TranscriptLine

load_repo_dotenv()

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_jinja = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape(enabled_extensions=()),
)

_OLLAMA_ALIASES: dict[str, str] = {
    "gemma4:4b": "gemma4:e4b",
    "gemma4:2b": "gemma4:e2b",
}


def _ollama_url() -> str:
    """Resolve Ollama base URL.

    ``OLLAMA_BASE_URL`` from the environment (``.env``) wins when set.

    When unset, defaults to ``http://127.0.0.1:11435``: that is the usual *local* end of
    ``ssh -L 11435:127.0.0.1:11434 user@remote`` so requests hit **remote** Ollama without
    colliding with a **separate** Ollama on this machine on :11434.

    If this PC runs only local Ollama (no tunnel), set ``OLLAMA_BASE_URL=http://127.0.0.1:11434``.
    If the pipeline runs **on the server** where Ollama listens on :11434, set
    ``OLLAMA_BASE_URL=http://127.0.0.1:11434`` there (do not use :11435 on the server).
    """
    raw = (os.getenv("OLLAMA_BASE_URL") or "").strip()
    if raw:
        return raw.rstrip("/")
    return "http://127.0.0.1:11435"


def get_ollama_base_url() -> str:
    """Public URL used for /api/tags and /api/generate (for logging and preflight)."""
    return _ollama_url()


def _ollama_headers() -> dict[str, str]:
    key = (os.getenv("OLLAMA_API_KEY") or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def fetch_ollama_model_names() -> list[str]:
    base = _ollama_url()
    url = f"{base}/api/tags"
    try:
        r = httpx.get(url, timeout=30, headers=_ollama_headers())
        r.raise_for_status()
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base} ({e}). "
            "Start the Ollama service on this machine, or set OLLAMA_BASE_URL in .env to a reachable "
            "host. If Ollama runs only on a remote server (e.g. after WhisperX SSH), open a tunnel in "
            "another terminal and point OLLAMA_BASE_URL at it, for example: "
            "ssh -L 11435:127.0.0.1:11434 user@server  then  OLLAMA_BASE_URL=http://127.0.0.1:11435"
        ) from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error at {url}: {e}") from e
    data = r.json()
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def resolve_ollama_model(requested: str, tags: list[str]) -> str | None:
    if requested in tags:
        return requested
    mapped = _OLLAMA_ALIASES.get(requested)
    if mapped and mapped in tags:
        return mapped
    return None


def diarized_block_from_segments(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for seg in segments:
        sp = seg.get("speaker") or "?"
        t0 = float(seg.get("start", 0))
        t1 = float(seg.get("end", t0))
        tx = (seg.get("text") or "").strip()
        lines.append(f"[{t0:.1f}s–{t1:.1f}s] {sp}: {tx}")
    return "\n".join(lines)


def _schema_hint_text() -> str:
    return (
        '{ "call_id", "source_file", "source_file_hash", "timestamp_start", "timestamp_end", '
        '"farmacia", "llamante", "agent" (nullable strings), '
        '"problema_corto", "descripcion_problema", "causa_raiz", '
        '"resolucion", "resolucion_exitosa", "resumen", '
        '"rag_qa": [{"question","answer","category","confidence"}], '
        '"software_features", "error_codes", "tags", '
        '"transcript": [{"start","end","speaker","text"}], '
        '"processing_metadata" }'
    )


def render_prompt(
    *,
    diarized_block: str,
    call_id: str,
    source_file: str,
    source_file_hash: str,
    timestamp_start: str,
    timestamp_end: str,
) -> str:
    tpl = _jinja.get_template("call_analysis.jinja2")
    return tpl.render(
        diarized_block=diarized_block,
        call_id=call_id,
        source_file=source_file,
        source_file_hash=source_file_hash,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        schema_hint=_schema_hint_text(),
    )


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r — using default %d", name, raw, default)
        return default


def _ollama_generate_json(model: str, prompt: str, *, timeout_s: float) -> str:
    url = f"{_ollama_url()}/api/generate"
    num_predict = _int_env("AUDIO_PIPELINE_OLLAMA_NUM_PREDICT", 16384)
    num_ctx = _int_env("AUDIO_PIPELINE_OLLAMA_NUM_CTX", 8192)
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
        "think": False,
    }
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=payload, headers=_ollama_headers())
        if r.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:1200]}")
        data = r.json()
        if data.get("error"):
            raise RuntimeError(f"Ollama: {data['error']}")
        text = (data.get("response") or "").strip()
        done_reason = data.get("done_reason")
        if done_reason and done_reason != "stop":
            logger.warning(
                "Ollama stopped early (done_reason=%s, response_len=%d, num_predict=%d). "
                "JSON may be truncated — raise AUDIO_PIPELINE_OLLAMA_NUM_PREDICT.",
                done_reason,
                len(text),
                num_predict,
            )
        return text


def _transcript_lines_from_segments(segments: list[dict[str, Any]]) -> list[TranscriptLine]:
    out: list[TranscriptLine] = []
    for seg in segments:
        try:
            out.append(
                TranscriptLine(
                    start=float(seg.get("start", 0)),
                    end=float(seg.get("end", 0)),
                    speaker=str(seg.get("speaker") or ""),
                    text=str(seg.get("text") or ""),
                )
            )
        except Exception:
            continue
    return out


def fallback_call_analysis(
    *,
    call_id: str,
    source_file: str,
    source_file_hash: str,
    timestamp_start: str,
    timestamp_end: str,
    segments: list[dict[str, Any]],
    error_message: str,
    attempts: int,
) -> CallAnalysis:
    """Minimal valid analysis after repeated JSON / validation failure."""
    lines = _transcript_lines_from_segments(segments)
    meta: dict[str, Any] = {
        "phase": 3,
        "analysis_fallback": True,
        "analysis_error": error_message[:2000],
        "analysis_attempts": attempts,
        "ollama_model_requested": os.getenv("AUDIO_PIPELINE_ANALYSIS_MODEL") or "",
    }
    return CallAnalysis(
        call_id=call_id,
        source_file=source_file,
        source_file_hash=source_file_hash,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        farmacia=None,
        llamante=None,
        agent=None,
        problema_corto="Fallo al analizar la conversación automáticamente",
        descripcion_problema=(
            "El modelo no devolvió JSON válido tras varios intentos. "
            f"Detalle técnico: {error_message[:800]}"
        ),
        causa_raiz=None,
        resolucion="",
        resolucion_exitosa=False,
        resumen="",
        rag_qa=[],
        software_features=[],
        error_codes=[],
        tags=["analysis-error", "needs-review"],
        transcript=lines,
        processing_metadata=meta,
    )


def analyze_conversation(
    *,
    segments_relative: list[dict[str, Any]],
    call_id: str,
    source_file: str,
    source_file_hash: str,
    timestamp_start: str,
    timestamp_end: str,
    tenant_id: str,
    ollama_timeout_s: float = 600.0,
) -> tuple[CallAnalysis, dict[str, Any]]:
    """Run Ollama analysis with one retry on parse/validation failure.

    Returns ``(CallAnalysis, stats)`` where stats include ``model``, ``gen_s``, ``attempts``.
    """
    del tenant_id  # audio analysis currently uses global env model selection.
    model_req = (os.getenv("AUDIO_PIPELINE_ANALYSIS_MODEL") or "gemma3:12b").strip()
    model = model_req

    block = diarized_block_from_segments(segments_relative)
    prompt = render_prompt(
        diarized_block=block,
        call_id=call_id,
        source_file=source_file,
        source_file_hash=source_file_hash,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
    )

    last_err: str | None = None
    gen_s = 0.0
    text: str = ""
    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            text = _ollama_generate_json(model, prompt, timeout_s=ollama_timeout_s)
            obj = extract_json_object(text)
            ca = CallAnalysis.model_validate(obj)
            elapsed = time.perf_counter() - t0
            gen_s += elapsed
            meta = dict(ca.processing_metadata) if ca.processing_metadata else {}
            meta.setdefault("ollama_model", model)
            meta.setdefault("ollama_model_requested", model_req)
            meta.setdefault("analysis_attempts", attempt + 1)
            meta.setdefault("analysis_gen_s", round(gen_s, 3))
            ca = ca.model_copy(update={"processing_metadata": meta})
            return ca, {
                "model_requested": model_req,
                "model": model,
                "gen_s": gen_s,
                "attempts": attempt + 1,
                "parse_ok": True,
                "error": None,
            }
        except Exception as exc:
            gen_s += time.perf_counter() - t0
            last_err = f"{type(exc).__name__}: {exc}"
            head = text[:400].replace("\n", "\\n")
            tail = text[-400:].replace("\n", "\\n") if len(text) > 400 else ""
            logger.warning(
                "CallAnalysis attempt %d failed for %s: %s (response_len=%d)\n  head: %s\n  tail: %s",
                attempt + 1,
                call_id,
                last_err,
                len(text),
                head,
                tail,
            )

    logger.warning(
        "All analysis attempts failed for %s — using fallback (resolucion_exitosa=false).",
        call_id,
    )
    fb = fallback_call_analysis(
        call_id=call_id,
        source_file=source_file,
        source_file_hash=source_file_hash,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        segments=segments_relative,
        error_message=last_err or "unknown",
        attempts=2,
    )
    meta = dict(fb.processing_metadata)
    meta["ollama_model"] = model
    meta["ollama_model_requested"] = model_req
    meta["analysis_gen_s"] = round(gen_s, 3)
    fb = fb.model_copy(update={"processing_metadata": meta})
    return fb, {
        "model_requested": model_req,
        "model": model,
        "gen_s": gen_s,
        "attempts": 2,
        "parse_ok": False,
        "error": last_err,
    }

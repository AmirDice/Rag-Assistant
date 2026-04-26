#!/usr/bin/env python3
"""Phase 3 — end-to-end audio pipeline (WhisperX -> segment -> Ollama analysis).

Run from repo root::

    python -m modules.audio_pipeline.run --audio path/to/file.mp3

Or from this directory (``modules/audio_pipeline``)::

    python run.py --audio path/to/file.mp3

Outputs under ``modules/audio_pipeline/output/{sha256}/``:

- ``whisperx_raw/`` — WhisperX JSON + timing
- ``CALL-XXX.json`` — one ``CallAnalysis`` per detected conversation
- ``manifest.json`` — run summary

After WhisperX has succeeded once, re-run analysis only with ``--resume`` (reuses
``whisperx_raw/<stem>.json``) or ``--from-whisperx-json PATH``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Allow `python run.py` from this folder without setting PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from modules.audio_pipeline._env import load_repo_dotenv
from modules.audio_pipeline.analyzer import analyze_conversation, get_ollama_base_url
from modules.audio_pipeline.segmenter import format_mm_ss, split_conversations
from modules.audio_pipeline.transcriber import ffprobe_duration_seconds, transcribe

load_repo_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_OUT = Path(__file__).resolve().parent / "output"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _load_whisperx_json(path: Path, duration_sec: float) -> dict[str, Any]:
    """Build the same shape as ``transcribe()`` from a saved WhisperX JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        segments = raw
        merged: dict[str, Any] = {}
    else:
        merged = raw
        segments = merged.get("segments") or []
    d = float(merged.get("duration_sec", duration_sec) or duration_sec)
    return {
        "segments": segments,
        "word_segments": merged.get("word_segments", []),
        "duration_sec": d,
        "processing_time_sec": float(merged.get("processing_time_sec", 0.0) or 0.0),
        "processing_ratio": float(merged.get("processing_ratio", 0.0) or 0.0),
        "model": str(merged.get("model", "") or ""),
        "json_path": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audio pipeline: transcribe -> segment -> analyze")
    parser.add_argument("--audio", type=Path, required=True, help="Path to audio file (e.g. .mp3)")
    parser.add_argument("--tenant-id", type=str, required=True, help="Tenant identifier")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output root (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=600.0,
        help="Ollama HTTP timeout per attempt (seconds)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output/<hash>/whisperx_raw/<audio_stem>.json exists, skip WhisperX and reuse it (analysis only).",
    )
    parser.add_argument(
        "--from-whisperx-json",
        type=Path,
        default=None,
        help="Explicit WhisperX JSON path; skips transcription. Overrides --resume.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    audio = args.audio.resolve()
    if not audio.is_file():
        logger.error("Audio file not found: %s", audio)
        return 1

    duration_sec = ffprobe_duration_seconds(audio)
    if duration_sec is None:
        logger.error("ffprobe failed — cannot read duration. Skipping %s", audio)
        return 2

    file_hash = _sha256_file(audio)
    out_root: Path = args.out_dir.resolve() / file_hash
    out_root.mkdir(parents=True, exist_ok=True)
    whisperx_dir = out_root / "whisperx_raw"
    whisperx_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Source %s | SHA-256 %s | duration %.1fs", audio.name, file_hash[:16], duration_sec)

    candidate_json = whisperx_dir / f"{audio.stem}.json"
    explicit_json = args.from_whisperx_json

    if explicit_json is not None:
        wx_path = explicit_json.resolve()
        if not wx_path.is_file():
            logger.error("WhisperX JSON not found: %s", wx_path)
            return 2
        logger.info("Using --from-whisperx-json (skip WhisperX): %s", wx_path)
        tr = _load_whisperx_json(wx_path, duration_sec)
        tr_wall_s = 0.0
    elif args.resume and candidate_json.is_file():
        logger.info("Resume: reusing existing WhisperX output (skip transcribe): %s", candidate_json)
        tr = _load_whisperx_json(candidate_json, duration_sec)
        tr_wall_s = 0.0
    else:
        t_tr0 = time.perf_counter()
        try:
            tr = transcribe(audio, output_dir=whisperx_dir, tenant_id=args.tenant_id)
        except Exception as exc:
            logger.exception("WhisperX transcription failed: %s", exc)
            _write_json(
                out_root / "manifest.json",
                {
                    "source_file": audio.name,
                    "source_path": str(audio),
                    "source_file_hash": file_hash,
                    "audio_duration_sec": duration_sec,
                    "status": "transcribe_failed",
                    "error": str(exc),
                },
            )
            return 3

        tr_wall_s = time.perf_counter() - t_tr0
    segments: list[dict[str, Any]] = tr.get("segments") or []
    if not segments:
        logger.error("WhisperX returned no segments.")
        _write_json(
            out_root / "manifest.json",
            {
                "source_file": audio.name,
                "source_path": str(audio),
                "source_file_hash": file_hash,
                "audio_duration_sec": duration_sec,
                "status": "empty_transcript",
                "whisperx": {k: tr[k] for k in tr if k != "segments"},
            },
        )
        return 4

    chunks = split_conversations(segments)
    whisperx_image = (os.getenv("WHISPERX_IMAGE") or "thomasvvugt/whisperx:cpu").strip()

    ollama_url = get_ollama_base_url()
    logger.info(
        "Analysis uses Ollama at %s - it must accept HTTP (test: curl %s/api/tags)",
        ollama_url,
        ollama_url,
    )

    manifest_conversations: list[dict[str, Any]] = []
    try:
        for i, ch in enumerate(chunks):
            call_id = f"CALL-{i + 1:03d}"
            rel = ch.segments_relative
            ts_start = format_mm_ss(0.0)
            ts_end = format_mm_ss(rel[-1]["end"]) if rel else "00:00"

            ca, stats = analyze_conversation(
                segments_relative=rel,
                call_id=call_id,
                source_file=audio.name,
                source_file_hash=file_hash,
                timestamp_start=ts_start,
                timestamp_end=ts_end,
                tenant_id=args.tenant_id,
                ollama_timeout_s=args.ollama_timeout,
            )

            pm: dict[str, Any] = dict(ca.processing_metadata) if ca.processing_metadata else {}
            pm.setdefault("pipeline", "modules.audio_pipeline.run")
            pm.setdefault("whisperx_model", tr.get("model"))
            pm.setdefault("whisperx_image", whisperx_image)
            pm.setdefault("whisperx_processing_time_sec", tr.get("processing_time_sec"))
            pm.setdefault("whisperx_processing_ratio", tr.get("processing_ratio"))
            pm.setdefault("whisperx_wall_clock_transcribe_sec", round(tr_wall_s, 3))
            pm.setdefault("diarization_used", tr.get("diarization_used"))
            pm.setdefault("diarization_status", tr.get("diarization_status"))
            pm.setdefault("audio_duration_sec", duration_sec)
            pm.setdefault("conversation_abs_start_sec", ch.abs_start_sec)
            pm.setdefault("conversation_abs_end_sec", ch.abs_end_sec)
            pm.setdefault("segmenter_min_duration_s", 30.0)

            ca = ca.model_copy(
                update={
                    "call_id": call_id,
                    "source_file": audio.name,
                    "source_file_hash": file_hash,
                    "timestamp_start": ts_start,
                    "timestamp_end": ts_end,
                    "processing_metadata": pm,
                }
            )

            out_path = out_root / f"{call_id}.json"
            _write_json(out_path, ca.model_dump(mode="json"))

            manifest_conversations.append(
                {
                    "call_id": call_id,
                    "json_path": str(out_path),
                    "abs_start_sec": ch.abs_start_sec,
                    "abs_end_sec": ch.abs_end_sec,
                    "duration_sec": ch.duration_sec,
                    "skipped": False,
                    "analysis_parse_ok": stats.get("parse_ok"),
                    "ollama_model": stats.get("model"),
                    "ollama_gen_s": round(stats.get("gen_s") or 0.0, 3),
                }
            )
            logger.info(
                "Wrote %s (parse_ok=%s model=%s)",
                out_path.name,
                stats.get("parse_ok"),
                stats.get("model"),
            )

    except RuntimeError as exc:
        logger.error("%s", exc)
        return 5

    manifest: dict[str, Any] = {
        "source_file": audio.name,
        "source_path": str(audio),
        "source_file_hash": file_hash,
        "audio_duration_sec": duration_sec,
        "status": "ok",
        "output_directory": str(out_root),
        "whisperx": {
            "model": tr.get("model"),
            "processing_time_sec": tr.get("processing_time_sec"),
            "processing_ratio": tr.get("processing_ratio"),
            "json_path": tr.get("json_path"),
            "wall_clock_sec": round(tr_wall_s, 3),
            "segment_count": len(segments),
        },
        "conversations_detected": len(chunks),
        "conversations": manifest_conversations,
    }
    _write_json(out_root / "manifest.json", manifest)
    logger.info("Done — manifest: %s", out_root / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

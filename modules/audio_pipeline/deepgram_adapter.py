from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"


def _content_type_for_audio(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".ogg":
        return "audio/ogg"
    if suffix == ".flac":
        return "audio/flac"
    return "audio/mpeg"


def _normalize_deepgram_to_segments(raw: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    results = raw.get("results")
    if not isinstance(results, dict):
        return [], []
    utterances = results.get("utterances")
    if isinstance(utterances, list) and utterances:
        segments = [
            {
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", 0.0)),
                "text": str(item.get("transcript") or "").strip(),
                "speaker": f"SPEAKER_{int(item.get('speaker', 0)):02d}",
            }
            for item in utterances
            if isinstance(item, dict)
        ]
        return segments, []

    channels = results.get("channels")
    if not isinstance(channels, list) or not channels:
        return [], []
    first_channel = channels[0]
    if not isinstance(first_channel, dict):
        return [], []
    alternatives = first_channel.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        return [], []
    first_alt = alternatives[0]
    if not isinstance(first_alt, dict):
        return [], []
    words = first_alt.get("words")
    if not isinstance(words, list) or not words:
        return [], []

    grouped_segments: list[dict[str, object]] = []
    word_segments: list[dict[str, object]] = []

    current_speaker = int(words[0].get("speaker", 0)) if isinstance(words[0], dict) else 0
    segment_start = float(words[0].get("start", 0.0)) if isinstance(words[0], dict) else 0.0
    current_words: list[str] = []
    previous_end = segment_start

    for raw_word in words:
        if not isinstance(raw_word, dict):
            continue
        start = float(raw_word.get("start", 0.0))
        end = float(raw_word.get("end", start))
        speaker = int(raw_word.get("speaker", current_speaker))
        token = str(raw_word.get("punctuated_word") or raw_word.get("word") or "").strip()

        word_segments.append(
            {
                "start": start,
                "end": end,
                "text": token,
                "word": token,
                "speaker": f"SPEAKER_{speaker:02d}",
            }
        )

        if speaker != current_speaker and current_words:
            grouped_segments.append(
                {
                    "start": segment_start,
                    "end": previous_end,
                    "text": " ".join(current_words).strip(),
                    "speaker": f"SPEAKER_{current_speaker:02d}",
                }
            )
            segment_start = start
            current_words = []
            current_speaker = speaker

        current_words.append(token)
        previous_end = end

    if current_words:
        grouped_segments.append(
            {
                "start": segment_start,
                "end": previous_end,
                "text": " ".join(current_words).strip(),
                "speaker": f"SPEAKER_{current_speaker:02d}",
            }
        )

    return grouped_segments, word_segments


def transcribe_deepgram_normalized(audio_path: Path, output_dir: Path, cfg) -> dict[str, object]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    api_key = ""
    if isinstance(cfg, dict):
        api_key = str(cfg.get("api_key") or "").strip()
    else:
        api_key = str(getattr(cfg, "api_key", "") or "").strip()
    if not api_key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY is required for Deepgram transcription."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "model": str(getattr(cfg, "model", "") or "nova-3"),
        "language": str(getattr(cfg, "language", "") or "es"),
        "diarize": "true" if bool(getattr(cfg, "diarize", True)) else "false",
        "punctuate": "true",
        "utterances": "true",
    }
    content_type = _content_type_for_audio(audio_path)

    with audio_path.open("rb") as reader:
        payload = reader.read()

    t0 = time.perf_counter()
    with httpx.Client(timeout=300) as client:
        response = client.post(
            DEEPGRAM_URL,
            params=params,
            headers={"Authorization": f"Token {api_key}", "Content-Type": content_type},
            content=payload,
        )
    elapsed = time.perf_counter() - t0

    if response.status_code != 200:
        raise RuntimeError(
            f"Deepgram transcription failed: status={response.status_code}, body={response.text[:1000]}"
        )

    raw = response.json()
    segments, word_segments = _normalize_deepgram_to_segments(raw)
    duration_sec = float((raw.get("metadata") or {}).get("duration") or 0.0)
    ratio = elapsed / duration_sec if duration_sec else 0.0

    json_path = output_dir / f"{audio_path.stem}.json"
    merged = {
        "segments": segments,
        "word_segments": word_segments,
        "duration_sec": duration_sec,
        "processing_time_sec": round(elapsed, 2),
        "processing_ratio": round(ratio, 3),
        "model": params["model"],
        "diarization_used": params["diarize"] == "true",
        "diarization_status": "complete" if params["diarize"] == "true" else "disabled_by_config",
        "provider": "deepgram",
        "raw_response": raw,
    }
    json_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "segments": segments,
        "word_segments": word_segments,
        "duration_sec": duration_sec,
        "processing_time_sec": round(elapsed, 2),
        "processing_ratio": round(ratio, 3),
        "model": params["model"],
        "diarization_used": params["diarize"] == "true",
        "diarization_status": "complete" if params["diarize"] == "true" else "disabled_by_config",
        "json_path": str(json_path),
    }

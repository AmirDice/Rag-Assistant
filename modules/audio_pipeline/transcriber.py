"""WhisperX transcription: Docker (default) or SSH remote venv (no Docker on server).

Usage (standalone):
    python -m modules.audio_pipeline.transcriber path/to/audio.mp3

Set ``WHISPERX_TRANSCRIBE_MODE=ssh`` and ``WHISPERX_REMOTE_*`` to offload to a Linux host.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from modules.audio_pipeline._env import load_repo_dotenv
from modules.audio_pipeline.whisperx_model_name import resolve_whisperx_model

load_repo_dotenv()

logger = logging.getLogger(__name__)

# Docker / ffprobe may emit bytes that are not valid Windows default encoding (cp1252);
# without UTF-8 + replace, subprocess's reader thread raises UnicodeDecodeError.
_SUBPROC_TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}


def _env_bool(name: str) -> bool | None:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return None
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    logger.warning("Ignoring invalid boolean for %s=%r", name, raw)
    return None


def _resolve_whisperx_image() -> str:
    """Docker Hub image for WhisperX CPU.

    The correct repository is ``thomasvvugt/whisperx:cpu`` (two ``v``s). A common
    typo ``thomasvugt`` (one ``v``) does not exist on Docker Hub; we fix it so
    a stale ``WHISPERX_IMAGE`` in ``.env`` does not break runs.
    """
    raw = (os.getenv("WHISPERX_IMAGE") or "thomasvvugt/whisperx:cpu").strip()
    if raw.startswith("thomasvugt/") and not raw.startswith("thomasvvugt/"):
        fixed = "thomasvvugt/" + raw[len("thomasvugt/") :]
        logger.warning(
            "WHISPERX_IMAGE points at non-existent repo %r; using %r instead "
            "(update your .env to WHISPERX_IMAGE=%s).",
            raw,
            fixed,
            fixed,
        )
        return fixed
    return raw


def _resolve_whisperx_model() -> str:
    """ASR model name passed to ``whisperx --model`` (faster-whisper / CTranslate2).

    Default is ``large-v3`` for compatibility with older WhisperX Docker images;
    set ``WHISPERX_MODEL`` to override (e.g. ``large-v3-turbo``, ``medium``, or a full Hub id
    for a CTranslate2 repo).
    """
    return resolve_whisperx_model()


def _should_use_diarization(hf_token: str) -> bool:
    override = _env_bool("WHISPERX_DIARIZE")
    if override is not None:
        return override
    return bool(hf_token)


WHISPERX_COMPUTE_TYPE = os.getenv("WHISPERX_COMPUTE_TYPE", "int8")
# Lower = less RAM (important on 8GB laptops). Try 2, then 1 if Docker OOMs.
WHISPERX_BATCH_SIZE = int(os.getenv("WHISPERX_BATCH_SIZE", "2"))
WHISPERX_LANGUAGE = os.getenv("WHISPERX_LANGUAGE", "es")
WHISPERX_MIN_SPEAKERS = int(os.getenv("WHISPERX_MIN_SPEAKERS", "2"))
WHISPERX_MAX_SPEAKERS = int(os.getenv("WHISPERX_MAX_SPEAKERS", "3"))
WHISPERX_CACHE_VOLUME = os.getenv("WHISPERX_CACHE_VOLUME", "whisperx_cache")
# Docker default /dev/shm is 64MB — PyTorch/Lightning + long audio often OOM or crash with
# "error waiting for container: unexpected EOF" (exit 125). 2g is a safe default.
WHISPERX_DOCKER_SHM_SIZE = (os.getenv("WHISPERX_DOCKER_SHM_SIZE") or "2g").strip()
# Optional hard cap for container RAM (e.g. 5g on an 8GB host — set in .env).
WHISPERX_DOCKER_MEMORY = (os.getenv("WHISPERX_DOCKER_MEMORY") or "").strip()

# WhisperX stores the Silero VAD weights here (inside the container: /root/.cache/...).
VAD_FILE_IN_CACHE = "torch/whisperx-vad-segmentation.bin"

# Old S3 URL in whisperx.vad raises HTTP 301; urllib in Python 3.10 does not follow it.
# Upstream ships the same bytes in-repo (see whisperX#1044).
VAD_EXPECTED_SHA256 = "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea"
VAD_DOWNLOAD_URL = os.getenv(
    "WHISPERX_VAD_URL",
    "https://github.com/m-bain/whisperX/raw/main/whisperx/assets/pytorch_model.bin",
)


def _ensure_vad_model_in_cache() -> None:
    """Place the VAD binary on the Docker cache volume before running WhisperX.

    The CPU Docker image still calls ``urllib.request.urlopen`` on a broken S3 URL
    (301 without redirect follow).  We seed ``~/.cache/torch/whisperx-vad-segmentation.bin``
    from GitHub; SHA256 must match what ``whisperx.vad.load_vad_model`` expects.
    """
    check = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{WHISPERX_CACHE_VOLUME}:/root/.cache",
            "alpine",
            "sh",
            "-c",
            (
                f"test -f /root/.cache/{VAD_FILE_IN_CACHE} && "
                f"test \"$(sha256sum /root/.cache/{VAD_FILE_IN_CACHE} | awk '{{print $1}}')\" = "
                f"\"{VAD_EXPECTED_SHA256}\""
            ),
        ],
        capture_output=True,
        **_SUBPROC_TEXT,
    )
    if check.returncode == 0:
        return

    logger.info(
        "Seeding VAD model into Docker volume %s (fixes S3 301 / urllib in old WhisperX image) ...",
        WHISPERX_CACHE_VOLUME,
    )
    dl = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{WHISPERX_CACHE_VOLUME}:/root/.cache",
            "alpine",
            "sh",
            "-c",
            (
                "apk add --no-cache curl >/dev/null 2>&1 && "
                f"mkdir -p /root/.cache/torch && "
                f"curl -fSL -o /root/.cache/{VAD_FILE_IN_CACHE} '{VAD_DOWNLOAD_URL}' && "
                f"test \"$(sha256sum /root/.cache/{VAD_FILE_IN_CACHE} | awk '{{print $1}}')\" = "
                f"\"{VAD_EXPECTED_SHA256}\""
            ),
        ],
        capture_output=True,
        **_SUBPROC_TEXT,
    )
    if dl.returncode != 0:
        raise RuntimeError(
            "Could not seed whisperx-vad-segmentation.bin into the Docker cache volume. "
            f"Check network, WHISPERX_VAD_URL, and volume {WHISPERX_CACHE_VOLUME}. "
            f"stderr: {dl.stderr[-1500:]}"
        )
    logger.info("VAD model present in cache with expected SHA256.")


def _docker_remove_vad_cache_file() -> None:
    """Delete cached VAD model so WhisperX can re-download (fixes SHA256 mismatch)."""
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{WHISPERX_CACHE_VOLUME}:/root/.cache",
            "alpine",
            "rm",
            "-f",
            f"/root/.cache/{VAD_FILE_IN_CACHE}",
        ],
        capture_output=True,
        **_SUBPROC_TEXT,
    )


def ffprobe_duration_seconds(audio_path: str | Path) -> float | None:
    """Return audio duration in seconds via ffprobe, or None on failure."""
    return _probe_duration(Path(audio_path).resolve())


def _probe_duration(audio_path: Path) -> float | None:
    """Return audio duration in seconds via ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            **_SUBPROC_TEXT,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", audio_path, exc)
        return None


def _transcribe_via_docker(
    audio_path: Path,
    output_dir: Path,
    duration_sec: float,
    hf_token: str,
    diarize: bool,
    model: str,
    language: str,
    compute_type: str,
    batch_size: int,
) -> dict:
    if (os.getenv("WHISPERX_CLEAR_VAD_CACHE") or "").strip().lower() in ("1", "true", "yes"):
        logger.info("WHISPERX_CLEAR_VAD_CACHE set — removing cached VAD file before re-seed.")
        _docker_remove_vad_cache_file()

    _ensure_vad_model_in_cache()
    audio_dir = str(audio_path.parent)
    audio_filename = audio_path.name
    container_output = "/app/output"
    image = _resolve_whisperx_image()
    cmd = ["docker", "run", "--rm", "--shm-size", WHISPERX_DOCKER_SHM_SIZE]
    if WHISPERX_DOCKER_MEMORY:
        cmd.extend(["-m", WHISPERX_DOCKER_MEMORY])
    cmd.extend(
        [
            "-v",
            f"{audio_dir}:/app",
            "-v",
            f"{str(output_dir)}:{container_output}",
            "-v",
            f"{WHISPERX_CACHE_VOLUME}:/root/.cache",
            image,
            f"/app/{audio_filename}",
            "--model",
            model,
            "--language",
            language,
            "--compute_type",
            compute_type,
            "--batch_size",
            str(batch_size),
        ]
    )
    if diarize:
        cmd.extend(
            [
                "--diarize",
                "--hf_token",
                hf_token,
                "--min_speakers",
                str(WHISPERX_MIN_SPEAKERS),
                "--max_speakers",
                str(WHISPERX_MAX_SPEAKERS),
            ]
        )
    cmd.extend(["--output_format", "json", "--output_dir", container_output])

    logger.info(
        "Running WhisperX: %s (%.1f min audio) batch_size=%d diarization=%s shm=%s%s",
        audio_filename,
        duration_sec / 60,
        batch_size,
        "on" if diarize else "off",
        WHISPERX_DOCKER_SHM_SIZE,
        f" memory={WHISPERX_DOCKER_MEMORY}" if WHISPERX_DOCKER_MEMORY else "",
    )

    proc: subprocess.CompletedProcess[str]
    elapsed = 0.0
    stderr_combined = ""
    for attempt in range(2):
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, **_SUBPROC_TEXT)
        elapsed = time.perf_counter() - t0
        err = proc.stderr or ""
        stderr_combined = err
        if proc.returncode == 0:
            break
        if attempt == 0 and (
            "SHA256 checksum" in err
            or "HTTP Error 301" in err
            or "Moved Permanently" in err
        ):
            logger.warning(
                "WhisperX VAD issue (checksum or S3 301). Refreshing cache in %s and retrying once.",
                WHISPERX_CACHE_VOLUME,
            )
            _docker_remove_vad_cache_file()
            _ensure_vad_model_in_cache()
            continue
        break

    if proc.returncode != 0:
        if proc.returncode == 125 or "unexpected eof" in stderr_combined.lower():
            logger.warning(
                "Docker exit 125 / 'unexpected EOF' usually means the WhisperX container died "
                "(OOM or too little shared memory). This run uses --shm-size=%s. Try "
                "WHISPERX_DOCKER_SHM_SIZE=2g–4g, increase Docker Desktop → Settings → Resources → Memory, "
                "WHISPERX_BATCH_SIZE=1, WHISPERX_DOCKER_MEMORY=5g (on 8GB hosts), or a smaller "
                "WHISPERX_MODEL. Turn off WHISPERX_CLEAR_VAD_CACHE unless debugging.",
                WHISPERX_DOCKER_SHM_SIZE,
            )
        if proc.returncode == 137:
            logger.warning(
                "Docker exit 137 means the container was SIGKILL'd (often out-of-memory). "
                "Try WHISPERX_MODEL=medium or small, unset WHISPERX_DOCKER_MEMORY if the cap is too low, "
                "raise Docker Desktop memory, or keep WHISPERX_BATCH_SIZE=1.",
            )
        raise RuntimeError(
            f"WhisperX failed (exit {proc.returncode}):\n{stderr_combined[-2000:]}"
        )

    stem = audio_path.stem
    json_candidates = list(output_dir.glob(f"{stem}*.json"))
    if not json_candidates:
        raise FileNotFoundError(
            f"WhisperX produced no JSON in {output_dir}. "
            f"stdout={proc.stdout[-500:]}"
        )
    json_path = sorted(json_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    ratio = elapsed / duration_sec if duration_sec else 0
    logger.info(
        "WhisperX done: %.1f s (ratio=%.2fx, segments=%d)",
        elapsed, ratio, len(data.get("segments", [])),
    )

    merged = {
        **data,
        "duration_sec": duration_sec,
        "processing_time_sec": round(elapsed, 2),
        "processing_ratio": round(ratio, 3),
        "model": model,
        "diarization_used": bool(diarize),
        "diarization_status": "complete" if diarize else ("disabled_by_config" if hf_token else "skipped_no_hf_token"),
    }
    with open(json_path, "w", encoding="utf-8") as wf:
        json.dump(merged, wf, indent=2, ensure_ascii=False)

    return {
        "segments": data.get("segments", []),
        "word_segments": data.get("word_segments", []),
        "duration_sec": duration_sec,
        "processing_time_sec": round(elapsed, 2),
        "processing_ratio": round(ratio, 3),
        "model": model,
        "diarization_used": bool(diarize),
        "diarization_status": "complete" if diarize else ("disabled_by_config" if hf_token else "skipped_no_hf_token"),
        "json_path": str(json_path),
    }


def transcribe(audio_path: str | Path, output_dir: str | Path | None = None, *, tenant_id: str = "default") -> dict:
    audio_path = Path(audio_path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    output = Path(output_dir).resolve() if output_dir is not None else (audio_path.parent / "whisperx_output").resolve()
    output.mkdir(parents=True, exist_ok=True)
    duration_sec = _probe_duration(audio_path)
    if duration_sec is None:
        raise RuntimeError(f"Cannot read audio duration (ffprobe failed): {audio_path}")

    del tenant_id
    hf = (os.getenv("AUDIO_PIPELINE_HF_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
    diarize = _should_use_diarization(hf)
    mode = (os.getenv("WHISPERX_TRANSCRIBE_MODE") or "docker").strip().lower()
    if diarize and not hf:
        raise RuntimeError("WhisperX diarization was requested but AUDIO_PIPELINE_HF_TOKEN/HF_TOKEN is empty.")

    model = _resolve_whisperx_model()
    language = WHISPERX_LANGUAGE
    compute_type = WHISPERX_COMPUTE_TYPE
    batch_size = int(WHISPERX_BATCH_SIZE)

    if mode == "docker":
        return _transcribe_via_docker(audio_path, output, float(duration_sec), hf, diarize, model, language, compute_type, batch_size)
    if mode == "ssh":
        from modules.audio_pipeline.remote_transcribe import transcribe_via_ssh

        return transcribe_via_ssh(
            audio_path=audio_path,
            output_dir=output,
            duration_sec=float(duration_sec),
            hf_token=hf,
            diarize=diarize,
            model=model,
            language=language,
            compute_type=compute_type,
            batch_size=batch_size,
        )
    raise RuntimeError(f"Unsupported WHISPERX_TRANSCRIBE_MODE: {mode!r} (expected 'docker' or 'ssh')")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m modules.audio_pipeline.transcriber <audio_file>")
        sys.exit(1)
    result = transcribe(sys.argv[1], None, tenant_id="default")
    print(json.dumps(result, indent=2, ensure_ascii=False))

"""WhisperX on a remote Linux host over SSH (venv + pip — no Docker on server).

Requires OpenSSH client (``ssh`` / ``scp``) on the machine running this repo.

Authentication: **SSH keys** (recommended). Do not put server passwords in ``.env``.

Environment:

  WHISPERX_TRANSCRIBE_MODE=ssh
  WHISPERX_REMOTE_HOST=157.180.56.174
  WHISPERX_REMOTE_USER=vdev
  WHISPERX_REMOTE_VENV=~/whisperx-venv
  WHISPERX_REMOTE_JOBS_DIR=~/whisperx_jobs

Optional:

  WHISPERX_SSH_OPTS=-i C:/Users/Me/.ssh/id_ed25519
  WHISPERX_REMOTE_BATCH_SIZE=2   (lower = less RAM on server)
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from modules.audio_pipeline._env import load_repo_dotenv

load_repo_dotenv()

logger = logging.getLogger(__name__)

# Avoid UnicodeDecodeError when capturing SSH/Docker output on Windows (cp1252 default).
_SUBPROC_TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}

VAD_EXPECTED_SHA256 = "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea"
VAD_DOWNLOAD_URL = os.getenv(
    "WHISPERX_VAD_URL",
    "https://github.com/m-bain/whisperX/raw/main/whisperx/assets/pytorch_model.bin",
)


def _ssh_opts() -> list[str]:
    raw = os.getenv("WHISPERX_SSH_OPTS", "")
    return shlex.split(raw) if raw.strip() else []


def _remote_config() -> tuple[str, str, str, str]:
    host = (os.getenv("WHISPERX_REMOTE_HOST") or "").strip()
    user = (os.getenv("WHISPERX_REMOTE_USER") or "").strip()
    venv = (os.getenv("WHISPERX_REMOTE_VENV") or "~/whisperx-venv").strip()
    jobs = (os.getenv("WHISPERX_REMOTE_JOBS_DIR") or "~/whisperx_jobs").strip()
    if not host or not user:
        raise RuntimeError(
            "Set WHISPERX_REMOTE_HOST and WHISPERX_REMOTE_USER for remote WhisperX "
            "(and WHISPERX_TRANSCRIBE_MODE=ssh)."
        )
    return host, user, venv, jobs


def _expand_remote_path(home: str, path: str) -> str:
    """Turn ``~/foo`` or absolute paths into an absolute path on the server."""
    p = path.strip()
    if p.startswith("~/"):
        return f"{home.rstrip('/')}/{p[2:]}"
    if p.startswith("/"):
        return p
    return f"{home.rstrip('/')}/{p}"


def transcribe_via_ssh(
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
    """Upload audio, run ``whisperx`` on the server, download JSON, return same shape as Docker."""
    host, user, venv, jobs_base = _remote_config()
    target = f"{user}@{host}"
    job_id = uuid.uuid4().hex
    ext = audio_path.suffix if audio_path.suffix else ".mp3"
    rin = f"input{ext}"

    lang = language
    compute = compute_type
    batch = batch_size
    mn = int(os.getenv("WHISPERX_MIN_SPEAKERS") or "2")
    mx = int(os.getenv("WHISPERX_MAX_SPEAKERS") or "3")
    if diarize and not hf_token:
        raise RuntimeError(
            "WhisperX diarization was requested for the remote host, but no HuggingFace token is set."
        )

    ssh_exe = shutil.which("ssh")
    scp_exe = shutil.which("scp")
    if not ssh_exe or not scp_exe:
        raise RuntimeError("ssh and scp must be on PATH (install OpenSSH Client on Windows).")

    opts = _ssh_opts()

    def ssh_cmd(remote_bash: str) -> list[str]:
        # Pass as ONE arg: SSH joins post-target args into a single string for sshd.
        # "bash -lc <quoted>" must stay together so bash -c gets the full command.
        return [ssh_exe, *opts, target, f"bash -lc {shlex.quote(remote_bash)}"]

    home_run = subprocess.run(
        ssh_cmd("echo $HOME"),
        capture_output=True,
        **_SUBPROC_TEXT,
    )
    if home_run.returncode != 0:
        raise RuntimeError(
            f"Cannot read remote $HOME (SSH failed). Use ssh {target} to verify login. "
            f"{home_run.stderr[-500:]}"
        )
    home = home_run.stdout.strip()
    if not home.startswith("/"):
        raise RuntimeError(f"Unexpected remote HOME: {home!r}")

    venv_abs = _expand_remote_path(home, venv)
    jobs_root = _expand_remote_path(home, jobs_base)
    rdir = f"{jobs_root.rstrip('/')}/{job_id}"
    rout = f"{rdir}/out"

    # 1) Prepare remote dir
    prep = ssh_cmd(f"mkdir -p {shlex.quote(rout)}")
    r = subprocess.run(prep, capture_output=True, **_SUBPROC_TEXT)
    if r.returncode != 0:
        raise RuntimeError(f"SSH mkdir failed: {r.stderr[-800:]}")

    # 2) Upload audio
    remote_in = f"{rdir}/{rin}"
    scp_target = f"{target}:{remote_in}"
    up = subprocess.run(
        [scp_exe, *opts, str(audio_path), scp_target],
        capture_output=True,
        **_SUBPROC_TEXT,
    )
    if up.returncode != 0:
        raise RuntimeError(f"scp upload failed: {up.stderr[-800:]}")

    # 3) Seed VAD + run whisperx (same GitHub asset as local Docker helper)
    hf_q = shlex.quote(hf_token)
    act = shlex.quote(f"{venv_abs}/bin/activate")
    vad_url_q = shlex.quote(VAD_DOWNLOAD_URL)
    in_abs = shlex.quote(f"{rdir}/{rin}")
    out_abs = shlex.quote(rout)
    diarize_args = ""
    if diarize:
        diarize_args = (
            f" --diarize --hf_token {hf_q} "
            f"--min_speakers {mn} --max_speakers {mx}"
        )
    remote_script = (
        "set -euo pipefail\n"
        "export TORCH_HOME=\"${HOME}/.cache/torch\"\n"
        "mkdir -p \"$TORCH_HOME\"\n"
        f"VADF=\"$TORCH_HOME/whisperx-vad-segmentation.bin\"\n"
        "if [ ! -f \"$VADF\" ] || "
        f"[ \"$(sha256sum \"$VADF\" 2>/dev/null | awk '{{print $1}}')\" != '{VAD_EXPECTED_SHA256}' ]; then\n"
        f"  curl -fSL -o \"$VADF\" {vad_url_q}\n"
        f"  test \"$(sha256sum \"$VADF\" | awk '{{print $1}}')\" = '{VAD_EXPECTED_SHA256}'\n"
        "fi\n"
        f"source {act}\n"
        "command -v whisperx >/dev/null 2>&1 || { echo 'whisperx not found — run scripts/whisperx_remote_server_setup.sh on the server' >&2; exit 127; }\n"
        f"whisperx {in_abs} "
        f"--model {shlex.quote(model)} "
        f"--language {shlex.quote(lang)} "
        f"--compute_type {shlex.quote(compute)} "
        f"--batch_size {batch} "
        f"{diarize_args}"
        f"--output_format json "
        f"--output_dir {out_abs}\n"
    )

    logger.info("Remote WhisperX on %s — job %s (%.1f min audio)", target, job_id, duration_sec / 60)
    t0 = time.perf_counter()
    run = subprocess.run(ssh_cmd(remote_script), capture_output=True, **_SUBPROC_TEXT)
    elapsed = time.perf_counter() - t0
    if run.returncode != 0:
        raise RuntimeError(
            f"Remote whisperx failed (exit {run.returncode}):\n"
            f"{run.stderr[-3000:]}\n{run.stdout[-1000:]}"
        )

    # 4) Download JSON — whisperx names output after input stem → input.json
    stem = Path(rin).stem
    remote_json = f"{rout}/{stem}.json"
    local_json = output_dir / f"{audio_path.stem}.json"
    down = subprocess.run(
        [scp_exe, *opts, f"{target}:{remote_json}", str(local_json)],
        capture_output=True,
        **_SUBPROC_TEXT,
    )
    if down.returncode != 0:
        # Fallback: list directory
        ls = subprocess.run(
            ssh_cmd(f"ls -1 {shlex.quote(rout)}"),
            capture_output=True,
            **_SUBPROC_TEXT,
        )
        raise RuntimeError(
            f"scp download failed for {remote_json}: {down.stderr}. "
            f"Remote dir listing:\n{ls.stdout}"
        )

    if (os.getenv("WHISPERX_REMOTE_CLEANUP") or "").strip().lower() in ("1", "true", "yes"):
        subprocess.run(ssh_cmd(f"rm -rf {shlex.quote(rdir)}"), capture_output=True)

    with open(local_json, encoding="utf-8") as f:
        data = json.load(f)

    ratio = elapsed / duration_sec if duration_sec else 0.0
    logger.info(
        "Remote WhisperX done: %.1f s (ratio=%.2fx, segments=%d)",
        elapsed,
        ratio,
        len(data.get("segments", [])),
    )

    merged = {
        **data,
        "duration_sec": duration_sec,
        "processing_time_sec": round(elapsed, 2),
        "processing_ratio": round(ratio, 3),
        "model": model,
        "remote_host": host,
        "remote_job_id": job_id,
        "diarization_used": bool(diarize),
        "diarization_status": "complete" if diarize else ("disabled_by_config" if hf_token else "skipped_no_hf_token"),
    }
    with open(local_json, "w", encoding="utf-8") as wf:
        json.dump(merged, wf, indent=2, ensure_ascii=False)

    return {
        "segments": data.get("segments", []),
        "word_segments": data.get("word_segments", []),
        "duration_sec": duration_sec,
        "processing_time_sec": round(elapsed, 2),
        "processing_ratio": round(ratio, 3),
        "model": model,
        "json_path": str(local_json),
        "remote_host": host,
        "remote_job_id": job_id,
        "diarization_used": bool(diarize),
        "diarization_status": "complete" if diarize else ("disabled_by_config" if hf_token else "skipped_no_hf_token"),
    }

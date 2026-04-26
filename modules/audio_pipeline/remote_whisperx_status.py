"""Easiest path: verify remote WhisperX (SSH + venv) before running the pipeline.

Usage:
    python -m modules.audio_pipeline.remote_whisperx_status
    python -m modules.audio_pipeline.remote_whisperx_status --verify

Reads WHISPERX_REMOTE_* from `.env` (via load_repo_dotenv).
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys

from modules.audio_pipeline._env import load_repo_dotenv

load_repo_dotenv()

_SUBPROC_TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}


def _remote_venv_path_expr(venv: str) -> str:
    """Path used in remote bash ``test`` / ``ls`` — always quoted, no ``$V`` (avoids empty V over SSH)."""
    v = venv.strip()
    if v.startswith("~/"):
        return f"$HOME/{v[2:]}"
    return shlex.quote(v)


def _ssh_base() -> tuple[str, str, list[str]]:
    host = (os.getenv("WHISPERX_REMOTE_HOST") or "").strip()
    user = (os.getenv("WHISPERX_REMOTE_USER") or "").strip()
    raw = os.getenv("WHISPERX_SSH_OPTS", "")
    opts = shlex.split(raw) if raw.strip() else []
    if not host or not user:
        print(
            "Set in .env: WHISPERX_REMOTE_HOST and WHISPERX_REMOTE_USER "
            "(and WHISPERX_TRANSCRIBE_MODE=ssh when you run the pipeline).",
            file=sys.stderr,
        )
        sys.exit(1)
    return host, user, opts


def print_instructions() -> None:
    print(
        """
=== Remote WhisperX (you have sudo on the server) ===

1) From your PC (repo root), copy the setup script — must be LF line endings (not Windows CRLF):
   scp scripts/whisperx_remote_server_setup.sh USER@HOST:~/

2) Run the full installer on the server (apt + venv + CPU torch + whisperx). Use -t so sudo can prompt:
   ssh -t USER@HOST "chmod +x whisperx_remote_server_setup.sh && ./whisperx_remote_server_setup.sh"

   Or: ssh USER@HOST  then  chmod +x whisperx_remote_server_setup.sh && ./whisperx_remote_server_setup.sh

   No sudo? Use ./whisperx_remote_server_setup.sh --user-only after an admin installed packages.

3) Accept HuggingFace licenses for pyannote. Locally set AUDIO_PIPELINE_HF_TOKEN or HF_TOKEN.

4) SSH keys on Windows (avoid password on every scp/ssh):
   WHISPERX_SSH_OPTS=-i C:/Users/YOU/.ssh/id_ed25519

5) In .env on this machine:
   WHISPERX_TRANSCRIBE_MODE=ssh
   WHISPERX_REMOTE_HOST=your.server.ip
   WHISPERX_REMOTE_USER=youruser

6) Verify, then run your pipeline:
   python -m modules.audio_pipeline.remote_whisperx_status --verify
"""
    )


def verify() -> int:
    host, user, opts = _ssh_base()
    ssh_exe = shutil.which("ssh")
    if not ssh_exe:
        print("ssh not on PATH (install OpenSSH Client on Windows).", file=sys.stderr)
        return 1

    target = f"{user}@{host}"
    venv = (os.getenv("WHISPERX_REMOTE_VENV") or "~/whisperx-venv").strip()
    rp = _remote_venv_path_expr(venv)
    inner = (
        f'test -x "{rp}/bin/whisperx" && "{rp}/bin/whisperx" --help >/dev/null '
        f'&& echo OK_whisperx_in_venv || {{ echo "FAIL: missing whisperx in {rp}/bin"; exit 1; }}'
    )
    # Must pass as ONE arg so SSH sends the full command string to sshd intact.
    cmd = [ssh_exe, *opts, target, f"bash -lc {shlex.quote(inner)}"]
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=True, **_SUBPROC_TEXT)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip())
    if r.returncode != 0:
        opts_raw = os.getenv("WHISPERX_SSH_OPTS", "")
        if "/USER/" in opts_raw or "\\USER\\" in opts_raw:
            print(
                "\nHint: WHISPERX_SSH_OPTS looks like a template — replace USER with your Windows "
                "username (e.g. C:/Users/YourName/.ssh/id_ed25519).",
                file=sys.stderr,
            )
        print(
            "\nIf SSH works but you see 'missing whisperx', WhisperX is not installed in the "
            "remote venv yet. On the server run (once):\n"
            f"  scp scripts/whisperx_remote_server_setup.sh {target}:~/\n"
            f'  ssh -t {target} "chmod +x whisperx_remote_server_setup.sh && '
            f'./whisperx_remote_server_setup.sh"\n'
            "Or paste the pip one-liner from `python -m modules.audio_pipeline.remote_whisperx_status`.\n"
            f"If the venv lives elsewhere, set WHISPERX_REMOTE_VENV in .env (currently: {venv!r}).",
            file=sys.stderr,
        )
        diag_inner = (
            f'ls -la "{rp}/bin/whisperx" 2>&1; echo ---; ls -la "{rp}/bin" 2>&1 | tail -20'
        )
        diag = subprocess.run(
            [ssh_exe, *opts, target, f"bash -lc {shlex.quote(diag_inner)}"],
            capture_output=True,
            **_SUBPROC_TEXT,
        )
        d = (diag.stdout or "") + (diag.stderr or "")
        if d.strip():
            print("\nRemote paths (diagnostic):\n" + d.strip(), file=sys.stderr)
        return r.returncode
    print("\nRemote WhisperX is reachable — set WHISPERX_TRANSCRIBE_MODE=ssh and run your pipeline.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Remote WhisperX setup hints and SSH check.")
    p.add_argument(
        "--verify",
        action="store_true",
        help="SSH to the server and check that whisperx exists in the venv.",
    )
    args = p.parse_args()
    if args.verify:
        raise SystemExit(verify())
    print_instructions()


if __name__ == "__main__":
    main()

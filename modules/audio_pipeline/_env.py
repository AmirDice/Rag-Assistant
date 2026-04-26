"""Load repo-root `.env` for standalone CLI scripts (same pattern as tests/benchmark)."""

from __future__ import annotations

from pathlib import Path

_loaded = False


def load_repo_dotenv() -> None:
    """Find the project root (directory containing `.env`) and load it."""
    global _loaded
    if _loaded:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    for directory in Path(__file__).resolve().parents:
        env_file = directory / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)
            _loaded = True
            return
    load_dotenv(override=False)
    _loaded = True

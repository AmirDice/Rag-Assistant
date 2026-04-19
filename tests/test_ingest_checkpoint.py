from __future__ import annotations

from pathlib import Path

import pytest

from api.pipeline import ingest as ing


@pytest.fixture
def patch_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _S:
        data_dir = str(tmp_path)

    monkeypatch.setattr(ing, "get_settings", lambda: _S())
    yield tmp_path


def test_checkpoint_append_upserts_same_path(patch_data_dir: Path):
    a = patch_data_dir / "a.pdf"
    a.write_bytes(b"x")
    ing._append_failed("/root", a, "err1")
    ing._append_failed("/root", a, "err2")
    ck = ing.load_checkpoint()
    assert ck and len(ck["failed"]) == 1
    assert ck["failed"][0]["error"] == "err2"


def test_checkpoint_remove_success_clears_when_empty(patch_data_dir: Path):
    a = patch_data_dir / "a.pdf"
    b = patch_data_dir / "b.pdf"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    ing._append_failed("/root", a, "e")
    ing._append_failed("/root", b, "e")
    ing._checkpoint_remove_success(a)
    ck = ing.load_checkpoint()
    assert ck and len(ck["failed"]) == 1
    ing._checkpoint_remove_success(b)
    assert ing.load_checkpoint() is None

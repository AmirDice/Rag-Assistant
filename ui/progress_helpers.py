"""Percentage progress for blocking work (thread + animated bar) and image loops."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Callable, TypeVar

import streamlit as st

T = TypeVar("T")


def progress_bar_text(label: str, pct: int) -> str:
    """Unified 'Label — N%' line for progress bars."""
    return f"{label} — {pct}%"


def run_with_progress(
    label: str,
    fn: Callable[[], T],
    *,
    cap_while_waiting: float = 0.92,
    step: float = 0.018,
    interval_s: float = 0.12,
) -> T:
    """Run `fn` in a worker thread; advance a bar 0% → ~92% until done, then 100% and clear.

    Use for long httpx calls where the API does not stream progress (benchmark, query, etc.).
    """
    slot = st.empty()
    slot.progress(0.0, text=progress_bar_text(label, 0))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn)
        p = 0.0
        while not fut.done():
            time.sleep(interval_s)
            p = min(p + step, cap_while_waiting)
            slot.progress(p, text=progress_bar_text(label, int(p * 100)))
        try:
            result = fut.result(timeout=0)
        except Exception:
            slot.empty()
            raise
    slot.progress(1.0, text=progress_bar_text(label, 100))
    time.sleep(0.1)
    slot.empty()
    return result


def image_fetch_progress(slot, label: str, index_0_based: int, total: int) -> None:
    """Update bar while loading each image (real fraction of gallery)."""
    if total <= 0:
        return
    frac = (index_0_based + 1) / total
    pct = int(frac * 100)
    slot.progress(frac, text=progress_bar_text(label, pct))

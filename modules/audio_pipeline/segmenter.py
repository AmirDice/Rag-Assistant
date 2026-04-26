"""Split a diarized WhisperX transcript into conversation-sized chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Primary split: silence between segments longer than this (seconds).
DEFAULT_GAP_SPLIT_S = 3.0
# Secondary: "speaker label reset" — first speaker of the file re-appears after another
# speaker was active, with a shorter gap (new caller / new session heuristic).
RESET_GAP_MIN_S = 1.0
MIN_CONVERSATION_DURATION_S = 30.0


def _seg_start_end(seg: dict[str, Any]) -> tuple[float, float]:
    s = float(seg.get("start", 0.0))
    e = float(seg.get("end", s))
    return s, e


def _norm_text(seg: dict[str, Any]) -> str:
    return (seg.get("text") or "").strip()


def _speaker(seg: dict[str, Any]) -> str:
    return str(seg.get("speaker") or "UNKNOWN").strip() or "UNKNOWN"


@dataclass
class ConversationChunk:
    """One detected conversation with timestamps relative to chunk start (from 0)."""

    call_index: int
    abs_start_sec: float
    abs_end_sec: float
    segments_relative: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.abs_end_sec - self.abs_start_sec)


def _to_relative_segments(raw: list[dict[str, Any]], origin: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in raw:
        s, e = _seg_start_end(seg)
        out.append(
            {
                "start": round(s - origin, 3),
                "end": round(e - origin, 3),
                "text": _norm_text(seg),
                "speaker": _speaker(seg),
            }
        )
    return out


def split_conversations(
    segments: list[dict[str, Any]],
    *,
    gap_split_s: float = DEFAULT_GAP_SPLIT_S,
    min_duration_s: float = MIN_CONVERSATION_DURATION_S,
    reset_gap_min_s: float = RESET_GAP_MIN_S,
) -> list[ConversationChunk]:
    """Split diarized segments into conversations.

    Rules:
    - Sort by start time; drop segments with no text (logged).
    - Split when gap (next.start - prev.end) > ``gap_split_s``.
    - Additionally split on *speaker reset pattern*: gap > ``reset_gap_min_s``, the next
      segment's speaker equals the **first speaker in the whole recording**, and the
      current block already contains at least two distinct speakers (not the opening line).
    - Each chunk gets ``segments_relative`` with times reset so the first line starts at 0.
    - Chunks shorter than ``min_duration_s`` are **dropped** (not returned).
    """
    if not segments:
        logger.warning("No segments to split.")
        return []

    sorted_segs = sorted(segments, key=lambda s: _seg_start_end(s)[0])

    nonempty: list[dict[str, Any]] = []
    for seg in sorted_segs:
        if not _norm_text(seg):
            s, e = _seg_start_end(seg)
            logger.debug("Skipping empty text segment [%.2f–%.2f]", s, e)
            continue
        nonempty.append(seg)

    if not nonempty:
        logger.warning("All segments were empty after filtering.")
        return []

    first_spk = _speaker(nonempty[0])

    def should_split_after(
        prev: dict[str, Any],
        nxt: dict[str, Any],
        current_block: list[dict[str, Any]],
    ) -> bool:
        _, pe = _seg_start_end(prev)
        ns, _ = _seg_start_end(nxt)
        gap = ns - pe
        if gap > gap_split_s:
            return True
        spks = {_speaker(x) for x in current_block}
        if (
            gap > reset_gap_min_s
            and _speaker(nxt) == first_spk
            and len(spks) >= 2
            and len(current_block) >= 2
        ):
            return True
        return False

    blocks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = [nonempty[0]]
    for i in range(1, len(nonempty)):
        prev, nxt = nonempty[i - 1], nonempty[i]
        if should_split_after(prev, nxt, cur):
            blocks.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)
    blocks.append(cur)

    chunks: list[ConversationChunk] = []
    for idx, block in enumerate(blocks):
        abs_start = _seg_start_end(block[0])[0]
        abs_end = _seg_start_end(block[-1])[1]
        rel = _to_relative_segments(block, abs_start)
        ch = ConversationChunk(
            call_index=idx,
            abs_start_sec=abs_start,
            abs_end_sec=abs_end,
            segments_relative=rel,
        )
        if ch.duration_sec < min_duration_s:
            logger.info(
                "Skipping short conversation block #%d (%.1fs < %.1fs): %.1f–%.1f",
                idx + 1,
                ch.duration_sec,
                min_duration_s,
                abs_start,
                abs_end,
            )
            continue
        chunks.append(ch)

    return chunks


def format_mm_ss(seconds: float) -> str:
    """Format seconds as MM:SS for CallAnalysis timestamp fields."""
    s = max(0.0, float(seconds))
    m = int(s // 60)
    sec = int(round(s % 60))
    if sec >= 60:
        m += 1
        sec = 0
    return f"{m:02d}:{sec:02d}"

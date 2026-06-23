"""Cheap geometric / layout filters for embedded PDF images before pixmap decode.

Pipeline (metadata + bbox first, decode last):
  1. implausible metadata size / extreme aspect ratio → skip
  2. full-page, header/footer strips, margin bullets, arrow-like strips → skip
  3. small images with no nearby text in the PDF text layer → skip
Decode + logo hash only for survivors (handled in converter).
"""

from __future__ import annotations

import logging
from typing import Any

import pymupdf

logger = logging.getLogger(__name__)

# --- Tunables (conservative defaults; full pixmap still does final size / logo checks) ---

MIN_META_W = 15
MIN_META_H = 10
MAX_WH_ASPECT = 22.0  # strip / arrow / divider-like

# Skip bitmaps that cover almost the whole page (scanned background / master image)
MAX_PAGE_COVER_FRAC = 0.88

# Header / footer bands (fraction of page height from top / bottom)
HEADER_BAND_FRAC = 0.11
FOOTER_BAND_FRAC = 0.10

# Narrow full-width horizontal bars (progress, separators)
FULL_WIDTH_MIN_W_FRAC = 0.68
FULL_HEIGHT_BAR_MAX_FRAC = 0.075

# Tall thin vertical bars
FULL_HEIGHT_MIN_H_FRAC = 0.68
FULL_WIDTH_BAR_MAX_FRAC = 0.075

# “Logo corner”: small object wholly in top band and narrow
HEADER_LOGO_MAX_AREA_FRAC = 0.06
HEADER_LOGO_MAX_W_FRAC = 0.42

# Side margin decorative (small + centered in margin column)
SIDE_MARGIN_FRAC = 0.075
MARGIN_DECOR_MAX_AREA_FRAC = 0.04
MARGIN_DECOR_MAX_DIM_FRAC = 0.12

# Text proximity: only for images smaller than this area fraction of the page
SMALL_IMG_AREA_FRAC = 0.12
TEXT_NEARBY_PAD_PT = 40.0


def should_skip_pdf_image_metadata(meta_w: int, meta_h: int) -> str | None:
    if meta_w > 0 and meta_h > 0:
        if meta_w < MIN_META_W or meta_h < MIN_META_H:
            return "tiny_metadata"
        ar = max(meta_w, meta_h) / max(min(meta_w, meta_h), 1)
        if ar > MAX_WH_ASPECT:
            return "extreme_aspect_metadata"
    return None


def should_skip_pdf_image_layout(
    rect: Any | None,
    page_rect: Any,
    meta_w: int,
    meta_h: int,
) -> str | None:
    """Use placement rectangle and page size; no pixmap decode.

    ``rect`` may be None if get_image_rects returned nothing (then only wide rules apply).
    """
    pw = float(page_rect.width)
    ph = float(page_rect.height)
    page_a = max(pw * ph, 1.0)

    rw = float(rect.width) if rect is not None else 0.0
    rh = float(rect.height) if rect is not None else 0.0

    area_frac = 0.0
    if rect is not None and rw > 0 and rh > 0:
        area_frac = (rw * rh) / page_a
        if area_frac >= MAX_PAGE_COVER_FRAC:
            return "near_fullpage_background"

        if rw >= pw * FULL_WIDTH_MIN_W_FRAC and rh <= ph * FULL_HEIGHT_BAR_MAX_FRAC:
            return "full_width_thin_bar"

        if rh >= ph * FULL_HEIGHT_MIN_H_FRAC and rw <= pw * FULL_WIDTH_BAR_MAX_FRAC:
            return "full_height_thin_bar"

    eff_w = meta_w if meta_w > 0 else rw
    eff_h = meta_h if meta_h > 0 else rh
    if eff_w > 0 and eff_h > 0:
        ar = max(eff_w, eff_h) / max(min(eff_w, eff_h), 1)
        if ar > MAX_WH_ASPECT:
            return "extreme_aspect_effective"

    if rect is None:
        return None

    cy = (float(rect.y0) + float(rect.y1)) / 2.0
    cx = (float(rect.x0) + float(rect.x1)) / 2.0

    in_header = cy < ph * HEADER_BAND_FRAC
    in_footer = cy > ph * (1.0 - FOOTER_BAND_FRAC)
    left_margin = cx < pw * SIDE_MARGIN_FRAC
    right_margin = cx > pw * (1.0 - SIDE_MARGIN_FRAC)

    if in_header and area_frac <= HEADER_LOGO_MAX_AREA_FRAC and rw <= pw * HEADER_LOGO_MAX_W_FRAC:
        return "header_band_small"

    if in_footer and area_frac <= HEADER_LOGO_MAX_AREA_FRAC * 1.2:
        return "footer_band_small"

    if (left_margin or right_margin) and area_frac <= MARGIN_DECOR_MAX_AREA_FRAC:
        if rw <= pw * MARGIN_DECOR_MAX_DIM_FRAC and rh <= ph * MARGIN_DECOR_MAX_DIM_FRAC:
            return "margin_decor"

    return None


def should_skip_pdf_image_text_proximity(
    rect: Any | None,
    page_rect: Any,
    words: list[tuple],
) -> str | None:
    """Small images far from any PDF text word are often bullets / ornaments."""
    if rect is None or not words:
        return None

    pw = float(page_rect.width)
    ph = float(page_rect.height)
    page_a = max(pw * ph, 1.0)
    rw = float(rect.width)
    rh = float(rect.height)
    area_frac = (rw * rh) / page_a

    if area_frac >= SMALL_IMG_AREA_FRAC:
        return None

    try:
        expanded = rect + (
            -TEXT_NEARBY_PAD_PT,
            -TEXT_NEARBY_PAD_PT,
            TEXT_NEARBY_PAD_PT,
            TEXT_NEARBY_PAD_PT,
        )
    except Exception:
        return None

    for w in words:
        if len(w) < 4:
            continue
        wr = pymupdf.Rect(w[0], w[1], w[2], w[3])
        if expanded.intersects(wr):
            return None

    return "no_nearby_text"

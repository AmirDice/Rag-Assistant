"""Geometric PDF-image prefilter — drops junk images before pixmap decode."""

from __future__ import annotations

import pymupdf

from api.pipeline.pdf_image_prefilter import (
    should_skip_pdf_image_layout,
    should_skip_pdf_image_metadata,
    should_skip_pdf_image_text_proximity,
)

# A4-ish page in points.
PAGE = pymupdf.Rect(0, 0, 595, 842)


def test_metadata_tiny_is_skipped():
    assert should_skip_pdf_image_metadata(10, 5) == "tiny_metadata"


def test_metadata_extreme_aspect_is_skipped():
    # Both dims clear the min size, but the aspect ratio is strip-like (>22).
    assert should_skip_pdf_image_metadata(400, 15) == "extreme_aspect_metadata"


def test_metadata_normal_kept():
    assert should_skip_pdf_image_metadata(200, 150) is None


def test_layout_near_fullpage_background_skipped():
    rect = pymupdf.Rect(5, 5, 590, 837)  # ~covers the page
    assert should_skip_pdf_image_layout(rect, PAGE, 580, 832) == "near_fullpage_background"


def test_layout_header_logo_skipped():
    # Small image high in the header band.
    rect = pymupdf.Rect(20, 10, 120, 50)
    assert should_skip_pdf_image_layout(rect, PAGE, 100, 40) == "header_band_small"


def test_layout_content_image_kept():
    # Reasonable mid-page figure.
    rect = pymupdf.Rect(150, 350, 450, 600)
    assert should_skip_pdf_image_layout(rect, PAGE, 300, 250) is None


def test_text_proximity_orphan_small_image_skipped():
    rect = pymupdf.Rect(100, 400, 130, 430)  # small, far from any word
    words = [(500, 800, 540, 815, "pie")]  # word in a far corner
    assert should_skip_pdf_image_text_proximity(rect, PAGE, words) == "no_nearby_text"


def test_text_proximity_image_near_text_kept():
    rect = pymupdf.Rect(100, 400, 130, 430)
    words = [(135, 405, 180, 420, "etiqueta")]  # adjacent word
    assert should_skip_pdf_image_text_proximity(rect, PAGE, words) is None


def test_text_proximity_large_image_kept():
    rect = pymupdf.Rect(100, 100, 500, 700)  # big → proximity rule doesn't apply
    assert should_skip_pdf_image_text_proximity(rect, PAGE, []) is None

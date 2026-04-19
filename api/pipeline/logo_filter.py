"""Perceptual-hash-based logo detection.

Compares extracted images against known logo reference hashes using pHash.
Images whose Hamming distance to any reference is below the configured
threshold are classified as logos and skipped during extraction.

New logos can be added by appending entries to config/logo_hashes.json.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "logo_hashes.json"

_logo_hashes: list[tuple[str, imagehash.ImageHash]] = []
_threshold: int = 14


def _load_config() -> None:
    global _logo_hashes, _threshold
    if _logo_hashes:
        return
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        _threshold = data.get("threshold", 14)
        for entry in data.get("logos", []):
            h = imagehash.hex_to_hash(entry["phash"])
            _logo_hashes.append((entry["name"], h))
        logger.info(
            "Loaded %d logo reference hashes (threshold=%d)",
            len(_logo_hashes), _threshold,
        )
    except Exception as e:
        logger.warning("Could not load logo hashes from %s: %s", _CONFIG_PATH, e)


def is_logo_pil(img: Image.Image) -> bool:
    """Check if a PIL Image matches any known logo via pHash."""
    _load_config()
    if not _logo_hashes:
        return False
    try:
        h = imagehash.phash(img)
        for name, ref_hash in _logo_hashes:
            dist = h - ref_hash
            if dist <= _threshold:
                logger.debug("Logo match: %s (distance=%d)", name, dist)
                return True
    except Exception as e:
        logger.debug("pHash check failed: %s", e)
    return False


def is_logo_bytes(png_bytes: bytes) -> bool:
    """Check if raw PNG bytes match any known logo via pHash."""
    try:
        img = Image.open(io.BytesIO(png_bytes))
        return is_logo_pil(img)
    except Exception:
        return False


def is_logo_file(img_path: Path) -> bool:
    """Check if an image file matches any known logo via pHash."""
    try:
        img = Image.open(img_path)
        result = is_logo_pil(img)
        img.close()
        return result
    except Exception:
        return False

#!/usr/bin/env python3
"""Compare Gemini model latency (rough): one text reply + one tiny image caption.

Use the same env vars as the API:
  GOOGLE_API_KEY
  GEMINI_GENERATION_MODEL  (default: gemini-2.5-flash)
  GEMINI_VISION_MODEL      (default: gemini-2.5-flash)

Compare full flash vs flash-lite:

  python scripts/model_latency_smoke.py
  set GEMINI_GENERATION_MODEL=gemini-2.5-flash-lite
  set GEMINI_VISION_MODEL=gemini-2.5-flash-lite
  python scripts/model_latency_smoke.py
"""

from __future__ import annotations

import os
import sys
import time

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def main() -> int:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        print("GOOGLE_API_KEY not set — nothing to run.")
        return 0

    gen_model = os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash").strip()
    vis_model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash").strip()

    from google import genai
    from PIL import Image

    client = genai.Client(api_key=key)

    t0 = time.perf_counter()
    r1 = client.models.generate_content(
        model=gen_model,
        contents="Reply with exactly one word: OK.",
    )
    t1 = time.perf_counter() - t0
    print(f"[generation] model={gen_model!r} latency_s={t1:.2f} preview={r1.text[:80]!r}")

    img = Image.new("RGB", (64, 64), color=(210, 200, 190))
    t0 = time.perf_counter()
    r2 = client.models.generate_content(
        model=vis_model,
        contents=["Describe this solid-color square in one short phrase.", img],
    )
    t2 = time.perf_counter() - t0
    print(f"[vision]     model={vis_model!r} latency_s={t2:.2f} preview={r2.text[:120]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

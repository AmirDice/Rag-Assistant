"""LLM document-type classification.

Self-contained replacement for the optional top-level ``classification/``
package (which is not shipped in the API image and is usually missing locally —
so ``api.pipeline.classifier`` always fell back to filename heuristics). Calls
the LLM directly from the configured API keys (Gemini, then OpenAI) and returns
a normalized ``{type_id, confidence, pipeline_hints}`` dict, or None.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import httpx

from api.core.model_names import gemini_generation_model
from api.core.settings import get_settings

logger = logging.getLogger(__name__)

_MIN_TEXT_CHARS = 40

_CLASSIFICATION_PROMPT = """You classify product / software documentation.

From the FILE NAME and the TEXT SAMPLE, pick exactly ONE type_id from this list (copy the id string exactly):
- structured_manual — comprehensive user manual with chapters or table of contents
- module_manual — installation/configuration for one product module (robots, receta electrónica, etc.)
- operational_guide — step-by-step procedures, workflows, numbered instructions
- changelog_pure — release notes that are mostly version lists without how-to procedures
- changelog_as_manual — release notes that include real configuration or usage steps
- faq_document — questions and answers format
- technical_spec — APIs, parameters, technical reference sheets
- training_material — slides, courses, presentations for training
- technical_note — short technical bulletin (will be stored as technical_spec)
- presentation — slide deck (stored as training_material)

Respond with JSON ONLY (no markdown fences):
{{"type_id":"<id>","confidence":<0.0-1.0>,"pipeline_hints":{{"discard":false}}}}

File name: {filename}
Text sample:
---
{text}
---
"""


def _parse_json_response(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        out = json.loads(text)
        if isinstance(out, dict):
            return out
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        out = json.loads(m.group(0))
        if isinstance(out, dict):
            return out
    raise json.JSONDecodeError("No JSON object in model response", text, 0)


def _call_openai_chat(*, url: str, api_key: str, model: str, prompt: str) -> str:
    with httpx.Client(timeout=60) as client:
        r = client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
        )
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"]).strip()


def _call_gemini(prompt: str) -> str:
    from google import genai

    s = get_settings()
    key = (s.google_api_key or "").strip()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY / google_api_key empty")
    client = genai.Client(api_key=key)
    model = gemini_generation_model()
    response = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
    )
    return (response.text or "").strip()


def classify_document_text(*, filename: str, text: str) -> Optional[dict[str, Any]]:
    """Run LLM classification. Returns {type_id, confidence, pipeline_hints} or None.

    None means "couldn't classify" (no API key, text too short, or a failure) —
    the caller should fall back to filename heuristics.
    """
    text = (text or "").strip()
    if len(text) < _MIN_TEXT_CHARS:
        logger.debug("doc_type LLM skipped (%s): text sample too short (%d chars)", filename, len(text))
        return None

    sample = text[:4000]
    prompt = _CLASSIFICATION_PROMPT.format(filename=filename, text=sample)
    settings = get_settings()
    raw: Optional[str] = None

    try:
        if (settings.google_api_key or "").strip():
            raw = _call_gemini(prompt)
        elif (settings.openai_api_key or "").strip():
            raw = _call_openai_chat(
                url="https://api.openai.com/v1/chat/completions",
                api_key=settings.openai_api_key.strip(),
                model="gpt-4o-mini",
                prompt=prompt,
            )
        else:
            return None

        if not raw:
            return None
        data = _parse_json_response(raw)
        tid = str(data.get("type_id") or "").strip()
        if not tid or tid.lower() == "unknown":
            return None
        conf = data.get("confidence", 0.7)
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            conf_f = 0.7
        hints = data.get("pipeline_hints") or {}
        if not isinstance(hints, dict):
            hints = {}
        return {
            "type_id": tid,
            "confidence": max(0.0, min(1.0, conf_f)),
            "pipeline_hints": hints,
        }
    except Exception as exc:
        logger.warning("doc_type LLM classification failed for %s: %s", filename, exc)
        return None

"""Extract a JSON object from LLM text (supports ```json fences)."""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _json_object_span(s: str, start: int) -> tuple[int, int] | None:
    """First top-level ``{...}`` slice; braces inside JSON strings are ignored."""
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    i = start
    while i < len(s):
        c = s[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return (start, i + 1)
        i += 1
    return None


def _escape_controls_inside_json_strings(slice_: str) -> str:
    """Escape raw newlines/tabs/control chars inside ``"..."`` so :func:`json.loads` accepts the slice."""
    out: list[str] = []
    in_string = False
    escape = False
    for c in slice_:
        if not in_string:
            out.append(c)
            if c == '"':
                in_string = True
            continue
        if escape:
            out.append(c)
            escape = False
            continue
        if c == "\\":
            out.append(c)
            escape = True
            continue
        if c == '"':
            out.append(c)
            in_string = False
            continue
        if c == "\n":
            out.append("\\n")
            continue
        if c == "\r":
            out.append("\\r")
            continue
        if c == "\t":
            out.append("\\t")
            continue
        o = ord(c)
        if o < 32:
            out.append(f"\\u{o:04x}")
            continue
        out.append(c)
    return "".join(out)


def _strip_trailing_commas(s: str) -> str:
    """Remove ``,`` immediately before ``}`` / ``]`` (outside strings). LLMs often emit these."""
    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(s):
        c = s[i]
        if not in_string:
            if c == '"':
                in_string = True
                out.append(c)
                i += 1
                continue
            if c == ",":
                j = i + 1
                while j < len(s) and s[j] in " \t\n\r":
                    j += 1
                if j < len(s) and s[j] in "}]":
                    i += 1
                    continue
            out.append(c)
            i += 1
            continue
        if escape:
            out.append(c)
            escape = False
            i += 1
            continue
        if c == "\\":
            out.append(c)
            escape = True
            i += 1
            continue
        if c == '"':
            out.append(c)
            in_string = False
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object in ``text``.

    Uses :func:`json.JSONDecoder.raw_decode` so ``{`` / ``}`` inside JSON **strings**
    (e.g. UI paths, prose) do not break parsing — naive brace-counting fails on those.

    If strict parsing fails (common when the model puts **literal line breaks or tabs**
    inside string values), applies a small repair: string-aware object bounds, escape
    illegal control characters inside strings, strip trailing commas, then parse again.
    """
    raw = text.strip()
    m = _JSON_FENCE.search(raw)
    if m:
        raw = m.group(1).strip()
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    decoder = json.JSONDecoder()

    def _parse_from(idx: int) -> dict[str, Any]:
        obj, _end = decoder.raw_decode(raw, idx)
        if not isinstance(obj, dict):
            raise ValueError("First JSON value is not an object")
        return obj

    try:
        return _parse_from(start)
    except json.JSONDecodeError:
        pass

    span = _json_object_span(raw, start)
    if span is None:
        raise ValueError("Could not locate balanced JSON object in model output")
    a, b = span[0], span[1]
    candidate = _strip_trailing_commas(_escape_controls_inside_json_strings(raw[a:b]))
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in model output: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("First JSON value is not an object")
    return obj

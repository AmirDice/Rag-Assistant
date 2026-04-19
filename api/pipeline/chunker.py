"""WP13 §13.4 — Config-driven chunking by document type.

Splits Markdown text into chunks based on doc_types.yaml rules.
Each chunk carries its section breadcrumb and metadata.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

import tiktoken

from api.core.models import DocumentMeta, ChunkPayload
from api.core.settings import get_settings

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


@dataclass
class RawChunk:
    text: str
    section_path: str = ""
    page_num: int | None = None
    has_image_caption: bool = False


def chunk_document(
    markdown: str,
    meta: DocumentMeta,
) -> list[ChunkPayload]:
    """Split markdown into chunks according to the doc type config."""
    settings = get_settings()
    dt_config = settings.doc_types_config()
    type_cfg = dt_config.get("doc_types", {}).get(meta.doc_type, dt_config.get("default", {}))
    max_tokens = type_cfg.get("max_tokens", 1500)
    chunk_unit = type_cfg.get("chunk_unit", "section")

    if chunk_unit == "single":
        raw_chunks = _chunk_single(markdown, max_tokens)
    elif chunk_unit == "qa_pair":
        raw_chunks = _chunk_qa_pairs(markdown, max_tokens)
    elif chunk_unit == "procedure":
        raw_chunks = _chunk_procedures(markdown, max_tokens)
    else:
        raw_chunks = _chunk_by_sections(markdown, max_tokens)

    # Apply config flags
    payloads: list[ChunkPayload] = []
    for i, rc in enumerate(raw_chunks):
        text = rc.text.strip()
        if not text:
            continue

        if type_cfg.get("prepend_breadcrumb") and rc.section_path:
            text = f"Sección: {rc.section_path}\n\n{text}"
        if type_cfg.get("prepend_module_name") and meta.module_id:
            text = f"Módulo: {meta.module_id}\n\n{text}"
        if type_cfg.get("include_procedure_title") and rc.section_path:
            text = f"Procedimiento: {rc.section_path}\n\n{text}"

        payloads.append(ChunkPayload(
            doc_id=meta.doc_id,
            chunk_index=i,
            doc_type=meta.doc_type,
            module_id=meta.module_id,
            version_min=meta.version_min,
            version_max=meta.version_max,
            is_robot_doc=meta.is_robot_doc,
            has_image_caption=rc.has_image_caption,
            source_doc=meta.source_file,
            source_page=rc.page_num,
            source_section=rc.section_path or None,
            lang=meta.lang,
            text=text,
        ))

    logger.info(
        "Chunked %s → %d chunks (type=%s, unit=%s)",
        meta.doc_id, len(payloads), meta.doc_type, chunk_unit,
    )
    return payloads


# ── Section-based chunking (default) ─────────────────────────

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


def _chunk_by_sections(markdown: str, max_tokens: int) -> list[RawChunk]:
    sections = _split_on_headings(markdown)
    chunks: list[RawChunk] = []

    for section_path, text in sections:
        is_image = "[Imagen:" in text
        if _count_tokens(text) <= max_tokens:
            chunks.append(RawChunk(text=text, section_path=section_path, has_image_caption=is_image))
        else:
            for sub in _split_by_token_limit(text, max_tokens):
                chunks.append(RawChunk(text=sub, section_path=section_path, has_image_caption=is_image))

    return chunks


def _split_on_headings(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into (heading_path, body) tuples at H2/H3 boundaries."""
    lines = markdown.split("\n")
    sections: list[tuple[str, str]] = []
    current_path_parts: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            if current_lines:
                path = " > ".join(current_path_parts) if current_path_parts else ""
                sections.append((path, "\n".join(current_lines)))
                current_lines = []

            level = len(m.group(1))
            title = m.group(2).strip()
            # Maintain hierarchical path
            if level <= len(current_path_parts):
                current_path_parts = current_path_parts[:level - 1]
            current_path_parts.append(title)
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        path = " > ".join(current_path_parts) if current_path_parts else ""
        sections.append((path, "\n".join(current_lines)))

    return sections


def _split_by_token_limit(text: str, max_tokens: int) -> list[str]:
    """Split text into chunks that fit within the token budget."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ── Single-chunk mode (changelog_pure) ───────────────────────

def _chunk_single(markdown: str, max_tokens: int) -> list[RawChunk]:
    if _count_tokens(markdown) <= max_tokens:
        return [RawChunk(text=markdown)]
    return [RawChunk(text=sub) for sub in _split_by_token_limit(markdown, max_tokens)]


# ── Q&A pair chunking (faq_document) ─────────────────────────

_QA_RE = re.compile(r"(?:^|\n)(?:#{1,4}\s*)?(?:P(?:regunta)?|Q)\s*[:\.\-]\s*", re.IGNORECASE)


def _chunk_qa_pairs(markdown: str, max_tokens: int) -> list[RawChunk]:
    parts = _QA_RE.split(markdown)
    chunks: list[RawChunk] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _count_tokens(part) <= max_tokens:
            chunks.append(RawChunk(text=part))
        else:
            for sub in _split_by_token_limit(part, max_tokens):
                chunks.append(RawChunk(text=sub))
    return chunks if chunks else _chunk_by_sections(markdown, max_tokens)


# ── Procedure chunking (operational_guide) ───────────────────

_STEP_RE = re.compile(r"(?:^|\n)(?:Paso|Step)\s+\d+", re.IGNORECASE)


def _chunk_procedures(markdown: str, max_tokens: int) -> list[RawChunk]:
    # Split on headings first; each section is a procedure
    sections = _split_on_headings(markdown)
    chunks: list[RawChunk] = []
    for path, text in sections:
        if _count_tokens(text) <= max_tokens:
            chunks.append(RawChunk(text=text, section_path=path))
        else:
            for sub in _split_by_token_limit(text, max_tokens):
                chunks.append(RawChunk(text=sub, section_path=path))
    return chunks

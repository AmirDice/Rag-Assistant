"""WP14 — Synthetic benchmark generation.

Pipeline: chunk → LLM generates Q/A + hard negatives → LLM validates → save.
Spec refs: §14.1 generation, §14.2 human spot-check, §14.3 target distribution.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from api.benchmark import validator as bench_validator
from api.core.product import product_labels
from api.core.retriever import payload_visible_for_tenant
from api.core.settings import get_settings
from api.core.tenant_state import merged_tenant_config

logger = logging.getLogger(__name__)


class BenchmarkPair(BaseModel):
    question: str
    answer: str
    difficulty: str  # L1 | L2 | L3
    doc_type: str
    source_doc: str
    source_section: Optional[str] = None
    source_page: Optional[int] = None
    chunk_id: str = ""
    has_image_source: bool = False
    hard_neg_a: Optional[str] = None
    hard_neg_b: Optional[str] = None
    validator_confidence: float = 0.0
    validated: bool = False
    chunk_text_preview: str = ""


class BenchmarkStats(BaseModel):
    total_pairs: int = 0
    by_difficulty: dict[str, int] = {}
    by_doc_type: dict[str, int] = {}
    validated_count: int = 0


GENERATION_PROMPT = """You are generating benchmark Q&A pairs for {short_name}. Context: {erp_context_es}.

Given the following text chunk from the indexed documentation, generate ONE question-answer pair.

Rules:
- The question must be answerable ONLY from this chunk
- The question must be in Spanish
- The answer must be a direct excerpt or close paraphrase from the chunk (1-3 sentences)
- Make the question specific and practical for someone using this product in their day-to-day work
- Do NOT ask meta-questions about the document itself

Chunk text:
---
{chunk_text}
---

Source document: {source_doc}
Section: {source_section}
Document type: {doc_type}

Respond in JSON format ONLY (no markdown):
{{"question": "...", "answer": "..."}}"""


def _language_rule(benchmark_lang: str) -> str:
    bl = (benchmark_lang or "es").strip().lower()
    if bl.startswith("ca"):
        return (
            "- The question must be in Catalan\n"
            "- The answer must be a direct excerpt or close paraphrase from the chunk (1-3 sentences)"
        )
    return (
        "- The question must be in Spanish\n"
        "- The answer must be a direct excerpt or close paraphrase from the chunk (1-3 sentences)"
    )

HARD_NEGATIVE_PROMPT = """Given this question about {short_name} ({erp_context_es}) and the CORRECT answer chunk, select the 2 chunks from the candidates below that look most plausible as answers but are actually WRONG.

Question: {question}
Correct answer chunk: {answer_chunk}

Candidate chunks (pick 2 that are most misleadingly similar):
{candidates}

Respond in JSON format ONLY (no markdown):
{{"neg_a_index": <int>, "neg_b_index": <int>}}"""


async def _get_llm_client():
    """Get an OpenAI-compatible client for generation."""
    import os
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek"
    return None


async def _llm_call(prompt: str, provider: str = None) -> str:
    """Call the LLM and return the raw response text."""
    import os

    if provider is None:
        provider = await _get_llm_client()
    if provider is None:
        raise RuntimeError("No LLM API key available for benchmark generation")

    if provider == "gemini":
        from google import genai
        from api.core.model_names import gemini_generation_model

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = await client.aio.models.generate_content(
            model=gemini_generation_model(),
            contents=prompt,
        )
        return response.text.strip()

    elif provider == "openai":
        import httpx
        resp = await httpx.AsyncClient().post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=30,
        )
        return resp.json()["choices"][0]["message"]["content"].strip()

    elif provider == "deepseek":
        import httpx
        resp = await httpx.AsyncClient().post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=30,
        )
        return resp.json()["choices"][0]["message"]["content"].strip()

    raise ValueError(f"Unknown provider: {provider}")


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


async def _fetch_all_chunks() -> list[dict]:
    """Pull all chunks from Qdrant for benchmark generation."""
    from qdrant_client import AsyncQdrantClient
    settings = get_settings()
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        all_chunks = []
        offset = None
        while True:
            results, next_offset = await qdrant.scroll(
                collection_name=settings.qdrant_collection,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for pt in results:
                payload = pt.payload or {}
                payload["_point_id"] = str(pt.id)
                all_chunks.append(payload)
            if next_offset is None:
                break
            offset = next_offset
        return all_chunks
    finally:
        await qdrant.close()


def _assign_difficulty(chunk: dict) -> str:
    if chunk.get("has_image_caption"):
        return "L3"
    text = chunk.get("text", "").lower()
    if any(kw in text for kw in ["configuración", "paso", "procedimiento", "cómo",
                                  "instrucciones", "pasos para"]):
        return "L2"
    return "L1"


async def generate_benchmark(
    max_pairs: int = 300,
    validate: bool = True,
    tenant_id: str = "demo",
    append: bool = False,
) -> tuple[list[BenchmarkPair], BenchmarkStats]:
    """Full WP14 pipeline: generate Q/A pairs from indexed chunks."""

    merged = merged_tenant_config(tenant_id)
    benchmark_lang = merged.get("benchmark_lang") or merged.get("preferred_lang") or "es"

    logger.info(
        "Starting benchmark generation (max_pairs=%d, tenant_id=%s, benchmark_lang=%s)",
        max_pairs, tenant_id, benchmark_lang,
    )

    chunks = await _fetch_all_chunks()
    if not chunks:
        raise RuntimeError("No chunks in Qdrant — ingest documents first")

    n_all = len(chunks)
    chunks = [c for c in chunks if payload_visible_for_tenant(c, merged)]
    logger.info(
        "Loaded %d chunks from Qdrant, %d visible for tenant %s (module/version/lang/robot/changelog rules)",
        n_all,
        len(chunks),
        tenant_id,
    )
    if not chunks:
        raise RuntimeError(
            f"No chunks visible for tenant {tenant_id!r} — widen contracted_modules, erp_version, "
            "or preferred_lang in config/tenants.yaml, or re-ingest."
        )

    by_doc_type: dict[str, list[dict]] = {}
    for c in chunks:
        dt = c.get("doc_type", "unknown")
        by_doc_type.setdefault(dt, []).append(c)

    selected = _select_chunks(chunks, by_doc_type, max_pairs)
    logger.info("Selected %d chunks for Q/A generation", len(selected))

    provider = await _get_llm_client()
    pairs: list[BenchmarkPair] = []
    errors = 0

    for i, chunk in enumerate(selected):
        try:
            pair = await _generate_one_pair(chunk, by_doc_type, provider)
            if pair:
                pairs.append(pair)
                if (i + 1) % 10 == 0:
                    logger.info("Generated %d/%d pairs", len(pairs), len(selected))
        except Exception as e:
            errors += 1
            logger.warning("Generation failed for chunk %d: %s", i, e)
            if errors > len(selected) * 0.3:
                logger.error("Too many errors (>30%%), stopping generation")
                break

    if validate and pairs:
        pairs = await _validate_pairs(pairs, chunks, provider)

    stats = _compute_stats(pairs)
    _save_artifacts(pairs, stats, append=append)

    logger.info(
        "Benchmark complete: %d pairs, %d validated, %d errors (append=%s)",
        len(pairs), stats.validated_count, errors, append,
    )
    return pairs, stats


def _select_chunks(
    all_chunks: list[dict],
    by_doc_type: dict[str, list[dict]],
    max_pairs: int,
) -> list[dict]:
    """Select chunks for generation following spec distribution (§14.3)."""
    target_dist = {
        "structured_manual": 0.25,
        "operational_guide": 0.20,
        "module_manual": 0.20,
        "changelog_as_manual": 0.15,
    }
    remaining_share = 1.0 - sum(target_dist.values())

    selected: list[dict] = []
    for dt, share in target_dist.items():
        pool = by_doc_type.get(dt, [])
        n = min(int(max_pairs * share), len(pool))
        if pool:
            selected.extend(random.sample(pool, n))

    other_types = [dt for dt in by_doc_type if dt not in target_dist]
    other_pool = []
    for dt in other_types:
        other_pool.extend(by_doc_type[dt])
    n_other = min(int(max_pairs * remaining_share), len(other_pool))
    if other_pool:
        selected.extend(random.sample(other_pool, n_other))

    random.shuffle(selected)
    return selected[:max_pairs]


async def _generate_one_pair(
    chunk: dict,
    by_doc_type: dict[str, list[dict]],
    provider: str,
    benchmark_lang: str = "es",
) -> Optional[BenchmarkPair]:
    """Generate a Q/A pair + hard negatives for one chunk."""
    text = chunk.get("text", "")
    if len(text.strip()) < 50:
        return None

    pl = product_labels()
    prompt = GENERATION_PROMPT.format(
        short_name=pl["short_name"],
        erp_context_es=pl["erp_context_es"],
        chunk_text=text[:2000],
        source_doc=chunk.get("source_doc", ""),
        source_section=chunk.get("source_section", ""),
        doc_type=chunk.get("doc_type", ""),
    )

    raw = await _llm_call(prompt, provider)
    result = _parse_json_response(raw)

    if not result.get("question") or not result.get("answer"):
        return None

    difficulty = _assign_difficulty(chunk)

    pair = BenchmarkPair(
        question=result["question"],
        answer=result["answer"],
        difficulty=difficulty,
        doc_type=chunk.get("doc_type", "unknown"),
        source_doc=chunk.get("source_doc", ""),
        source_section=chunk.get("source_section"),
        source_page=chunk.get("source_page"),
        chunk_id=chunk.get("_point_id", ""),
        has_image_source=chunk.get("has_image_caption", False),
        chunk_text_preview=text[:1500],
    )

    hard_negs = await _find_hard_negatives(pair, chunk, by_doc_type, provider)
    if hard_negs:
        pair.hard_neg_a = hard_negs[0]
        pair.hard_neg_b = hard_negs[1] if len(hard_negs) > 1 else None

    return pair


async def _find_hard_negatives(
    pair: BenchmarkPair,
    correct_chunk: dict,
    by_doc_type: dict[str, list[dict]],
    provider: str,
) -> list[str]:
    """Find 2 misleadingly similar but wrong chunks."""
    same_type = by_doc_type.get(pair.doc_type, [])
    candidates = [
        c for c in same_type
        if c.get("_point_id") != correct_chunk.get("_point_id")
        and len(c.get("text", "")) > 50
    ]

    if len(candidates) < 2:
        return []

    sample = random.sample(candidates, min(6, len(candidates)))
    candidate_texts = "\n\n".join(
        f"[{i}] {c.get('text', '')[:300]}"
        for i, c in enumerate(sample)
    )

    try:
        pl = product_labels()
        prompt = HARD_NEGATIVE_PROMPT.format(
            short_name=pl["short_name"],
            erp_context_es=pl["erp_context_es"],
            question=pair.question,
            answer_chunk=correct_chunk.get("text", "")[:500],
            candidates=candidate_texts,
        )
        raw = await _llm_call(prompt, provider)
        result = _parse_json_response(raw)

        neg_a_idx = result.get("neg_a_index", 0)
        neg_b_idx = result.get("neg_b_index", 1)

        negs = []
        if 0 <= neg_a_idx < len(sample):
            negs.append(sample[neg_a_idx].get("text", "")[:500])
        if 0 <= neg_b_idx < len(sample):
            negs.append(sample[neg_b_idx].get("text", "")[:500])
        return negs
    except Exception as e:
        logger.warning("Hard negative mining failed: %s", e)
        return [c.get("text", "")[:500] for c in sample[:2]]


async def _validate_pairs(
    pairs: list[BenchmarkPair],
    all_chunks: list[dict],
    provider: str,
) -> list[BenchmarkPair]:
    """LLM validation pass: score each pair for quality."""
    chunk_map = {c.get("_point_id", ""): c for c in all_chunks}

    for pair in pairs:
        chunk = chunk_map.get(pair.chunk_id, {})
        text = chunk.get("text", pair.answer)[:2000]
        await bench_validator.validate_pair(
            pair, text, provider, _llm_call, _parse_json_response
        )

    return pairs


def _compute_stats(pairs: list[BenchmarkPair]) -> BenchmarkStats:
    by_diff: dict[str, int] = {}
    by_type: dict[str, int] = {}
    validated = 0

    for p in pairs:
        by_diff[p.difficulty] = by_diff.get(p.difficulty, 0) + 1
        by_type[p.doc_type] = by_type.get(p.doc_type, 0) + 1
        if p.validated:
            validated += 1

    return BenchmarkStats(
        total_pairs=len(pairs),
        by_difficulty=by_diff,
        by_doc_type=by_type,
        validated_count=validated,
    )


def _load_existing_benchmark_pairs(path: Path) -> list[BenchmarkPair]:
    if not path.exists():
        return []
    out: list[BenchmarkPair] = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            try:
                out.append(BenchmarkPair.model_validate_json(line))
            except Exception:
                continue
    return out


def _save_artifacts(
    pairs: list[BenchmarkPair],
    stats: BenchmarkStats,
    *,
    append: bool = False,
) -> None:
    settings = get_settings()
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = data_dir / "benchmark_v1.jsonl"
    to_write = list(pairs)
    if append and benchmark_path.exists():
        existing = _load_existing_benchmark_pairs(benchmark_path)
        to_write = existing + pairs
        stats = _compute_stats(to_write)
    with open(benchmark_path, "w", encoding="utf-8") as f:
        for pair in to_write:
            f.write(pair.model_dump_json() + "\n")
    logger.info("Saved %d pairs to %s", len(to_write), benchmark_path)

    stats_path = data_dir / "benchmark_stats.json"
    stats_path.write_text(stats.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved stats to %s", stats_path)

"""Pydantic request / response schemas — matches spec tables 24-28."""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Document metadata (WP12 §12.2) ──────────────────────────

class DocumentMeta(BaseModel):
    doc_id: str
    doc_type: str                          # 8 canonical types
    module_id: Optional[str] = None        # None = core product
    version_min: float = 0.0
    version_max: Optional[float] = None    # None = still current
    is_robot_doc: bool = False
    ocr_needed: bool = False
    format: str = "pdf"                    # pdf | docx | pptx | xlsx
    lang: str = "es"                       # es | ca | other
    discard: bool = False
    image_count: int = 0
    source_file: str = ""


# ── Chunk payload stored in Qdrant (WP15 §15.1) ─────────────

class ChunkPayload(BaseModel):
    doc_id: str
    chunk_index: int
    doc_type: str
    module_id: Optional[str] = None
    version_min: float = 0.0
    version_max: Optional[float] = None
    is_robot_doc: bool = False
    has_image_caption: bool = False
    source_doc: str
    source_page: Optional[int] = None
    source_section: Optional[str] = None
    lang: str = "es"
    text: str


# ── Query API (WP15 §15.2) ──────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    tenant_id: str = "demo"
    top_k: int = Field(default=5, ge=1, le=20)
    lang: Optional[str] = None
    generate: bool = False

class AnswerChunk(BaseModel):
    text: str
    score: float
    source_doc: str
    source_page: Optional[int] = None
    source_section: Optional[str] = None
    has_image_caption: bool = False
    chunk_id: str = ""  # Qdrant point id — for benchmark eval / debugging

class RelatedDoc(BaseModel):
    doc: str
    relevance: str

class QueryResponse(BaseModel):
    answer_chunks: list[AnswerChunk]
    synthesized_answer: Optional[str] = None
    related_docs: list[RelatedDoc] = []
    reranked: bool = False
    cache_hit: bool = False
    query_id: str = Field(default_factory=lambda: str(uuid4()))


# ── Feedback API (WP15 §15.2) ────────────────────────────────

class FeedbackRequest(BaseModel):
    query_id: str
    rating: str  # "ok" | "not_ok"
    reason: Optional[str] = None
    correction: Optional[str] = None
    tenant_id: str = "demo"

class FeedbackResponse(BaseModel):
    status: str = "stored"
    query_id: str


# ── Ingest API ───────────────────────────────────────────────

class IngestRequest(BaseModel):
    path: str = ""                         # file or folder (optional if resume-only)
    force: bool = False                    # re-ingest even if already indexed
    resume: bool = False                   # retry files listed in data/ingest_checkpoint.json
    workers: Optional[int] = Field(default=None, ge=1, le=32)

class IngestResponse(BaseModel):
    status: str
    docs_processed: int = 0
    chunks_created: int = 0
    errors: list[str] = []


# ── Stats / Health ───────────────────────────────────────────

class StatsResponse(BaseModel):
    total_docs: int = 0
    total_chunks: int = 0
    by_type: dict[str, int] = {}
    by_module: dict[str, int] = {}
    last_ingestion: Optional[str] = None
    approximate_index_mb: float = 0.0

class HealthResponse(BaseModel):
    api: str = "ok"
    qdrant: str = "unknown"
    redis: str = "unknown"


# ── Tenant onboarding (data/tenant_onboarding.json overlay) ──

class TenantOnboardingUpdate(BaseModel):
    erp_version: Optional[float] = None
    legacy_erp_version: Optional[float] = None
    contracted_modules: Optional[list[str]] = None
    has_robot_integration: Optional[bool] = None
    preferred_lang: Optional[str] = None
    benchmark_lang: Optional[str] = None  # es | ca — benchmark question language


class BenchmarkReviewRequest(BaseModel):
    pair_index: int = Field(ge=0)
    action: str  # accept | reject
    edited_answer: Optional[str] = None
    notes: Optional[str] = None

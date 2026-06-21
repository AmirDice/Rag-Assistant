"""Pydantic request / response schemas — matches spec tables 24-28."""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from api.core.generation_catalog import generation_model_ids


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

class PriorTurn(BaseModel):
    """One prior conversation turn, for multi-turn / clarification context."""
    question: str = ""
    answer: str = ""


class QueryRequest(BaseModel):
    question: str
    tenant_id: str = "demo"
    top_k: int = Field(default=5, ge=1, le=20)
    lang: Optional[str] = None
    generate: bool = False
    reranker: Optional[str] = Field(
        default=None,
        description="Override reranker: voyage | cohere | bge | none (omit = use config/models.yaml)",
    )
    generation_model: Optional[str] = Field(
        default=None,
        description="Override generation model id (Gemini or GPT; omit = use YAML / env)",
    )
    prior_turns: list[PriorTurn] = Field(
        default_factory=list,
        description="Recent conversation turns for multi-turn / clarification context",
    )

    @field_validator("reranker")
    @classmethod
    def _norm_reranker(cls, v: Optional[str]) -> Optional[str]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        low = v.strip().lower()
        allowed = {"voyage", "cohere", "bge", "none"}
        if low not in allowed:
            raise ValueError(f"reranker must be one of {sorted(allowed)}")
        return low

    @field_validator("generation_model")
    @classmethod
    def _norm_gen(cls, v: Optional[str]) -> Optional[str]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        s = v.strip()
        if len(s) > 128:
            raise ValueError("generation_model too long")
        allowed = generation_model_ids()
        if s not in allowed:
            raise ValueError(f"generation_model must be one of {list(allowed)}")
        return s

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
    # Post-generation grounding check (None when no answer was synthesized).
    grounded: Optional[bool] = None
    grounding_ratio: Optional[float] = None
    # Query preprocessing transparency (what was actually embedded for retrieval).
    original_query: Optional[str] = None
    retrieval_query: Optional[str] = None
    matched_terms: list[str] = []
    # Estimated cost of the synthesized answer (0 when no generation / unpriced).
    total_estimated_cost: float = 0.0
    cost_currency: str = "USD"
    cost_breakdown: dict = {}
    # Query intent planner (clarification flow).
    needs_clarification: bool = False
    message_type: str = "final_answer"   # final_answer | clarification_question | no_answer
    final_query: Optional[str] = None


# ── Feedback API (WP15 §15.2) ────────────────────────────────

class FeedbackRequest(BaseModel):
    query_id: str
    rating: str  # "ok" | "not_ok"
    stars: Optional[int] = Field(default=None, ge=0, le=5)
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
    total_docs_bytes: int = 0

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

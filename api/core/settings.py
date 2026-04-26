from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "default"
    redis_url: str = "redis://localhost:6379/0"

    voyage_api_key: str = ""
    cohere_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""

    config_dir: str = "./config"
    data_dir: str = "./data"
    corpus_dir: str = "./corpus"
    docs_dir: str = Field(
        default="./docs",
        description="Directory for uploaded / staged source files (env: DOCS_DIR)",
    )

    admin_token: str = "changeme"
    enforce_admin_auth: bool = Field(
        default=False,
        description="Require X-Admin-Token on protected API routes (env: ENFORCE_ADMIN_AUTH)",
    )
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable in-memory API rate limiting middleware (env: RATE_LIMIT_ENABLED)",
    )
    rate_limit_requests_per_min: int = Field(
        default=120,
        ge=10,
        le=5000,
        description="Default requests/minute per client IP (env: RATE_LIMIT_REQUESTS_PER_MIN)",
    )
    rate_limit_heavy_requests_per_min: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Requests/minute for heavy endpoints (env: RATE_LIMIT_HEAVY_REQUESTS_PER_MIN)",
    )

    gemini_generation_model: str = Field(
        default="",
        description="Override generation model ID (env: GEMINI_GENERATION_MODEL)",
    )
    gemini_vision_model: str = Field(
        default="",
        description="Override vision/captioning model ID (env: GEMINI_VISION_MODEL)",
    )
    ingestion_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Parallel document workers for background ingest (env: INGESTION_WORKERS)",
    )
    benchmark_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Concurrent retrieval calls for benchmark eval/analyze (env: BENCHMARK_WORKERS)",
    )
    log_level: str = Field(
        default="INFO",
        description="Root log level: DEBUG, INFO, WARNING (env: LOG_LEVEL)",
    )
    audio_rag_qdrant_collection: str = Field(
        default="audio_calls",
        description="Qdrant collection used by audio pipeline outputs",
    )
    audio_index_sqlite_path: str = Field(
        default="",
        description="Optional custom path for audio index dedup sqlite",
    )
    audio_calls_catalog_sqlite_path: str = Field(
        default="",
        description="Optional custom path for audio calls catalog sqlite",
    )
    audio_output_dir: str = Field(
        default="",
        description="Optional output directory for analyzed calls JSON files",
    )
    audio_uploads_dir: str = Field(
        default="",
        description="Optional upload directory for call audio files",
    )
    audio_pipeline_dedup_threshold: float = Field(
        default=0.92,
        description="Dedup similarity threshold for RAG Q/A points",
    )
    audio_issue_dedup_threshold: float = Field(
        default=0.94,
        description="Dedup similarity threshold for issue-summary points",
    )

    model_config = {"env_file": ".env", "extra": "ignore"}

    # ── helpers ──────────────────────────────────────────────

    @property
    def config_path(self) -> Path:
        return Path(self.config_dir)

    def load_yaml(self, name: str) -> dict:
        with open(self.config_path / name, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def doc_types_config(self) -> dict:
        return self.load_yaml("doc_types.yaml")

    def tenants_config(self) -> dict:
        return self.load_yaml("tenants.yaml")

    def models_config(self) -> dict:
        return self.load_yaml("models.yaml")


@lru_cache
def get_settings() -> Settings:
    return Settings()

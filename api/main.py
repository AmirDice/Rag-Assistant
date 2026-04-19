"""RAG Assistant — FastAPI entry point."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import query, ingest, feedback, admin, benchmark, tenant, ui_config
from api.core.cache import get_cache
from api.core.settings import get_settings

_lvl = os.getenv("LOG_LEVEL", "INFO").upper()
_root_level = getattr(logging, _lvl, logging.INFO)
logging.basicConfig(
    level=_root_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
for _name in ("httpx", "httpcore", "hpack", "google_genai", "google.auth"):
    logging.getLogger(_name).setLevel(max(logging.WARNING, _root_level))

app = FastAPI(
    title="RAG Assistant",
    version="0.1.0",
    description="Retrieval-augmented Q&A over your indexed documentation (multi-tenant API + optional UI).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, tags=["query"])
app.include_router(ingest.router, tags=["ingest"])
app.include_router(feedback.router, tags=["feedback"])
app.include_router(admin.router, tags=["admin"])
app.include_router(benchmark.router, tags=["benchmark"])
app.include_router(tenant.router)
app.include_router(ui_config.router, tags=["config"])

settings = get_settings()
from pathlib import Path
_corpus = Path(settings.corpus_dir)
_corpus.mkdir(parents=True, exist_ok=True)
app.mount("/corpus", StaticFiles(directory=str(_corpus)), name="corpus")


@app.on_event("startup")
async def startup():
    await get_cache()


@app.on_event("shutdown")
async def shutdown():
    cache = await get_cache()
    await cache.close()

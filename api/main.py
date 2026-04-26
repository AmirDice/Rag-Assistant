"""RAG Assistant — FastAPI entry point."""

from __future__ import annotations

import logging
import os

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import query, ingest, feedback, admin, benchmark, tenant, ui_config, calls
from api.core.cache import get_cache
from api.core.settings import get_settings
from api.core.security import InMemoryRateLimiter, require_admin_token

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

_settings = get_settings()
_limiter = InMemoryRateLimiter(
    default_limit_per_min=_settings.rate_limit_requests_per_min,
    heavy_limit_per_min=_settings.rate_limit_heavy_requests_per_min,
)
_HEAVY_PREFIXES = (
    "/benchmark/generate",
    "/benchmark/run",
    "/benchmark/analyze",
    "/ingest/upload",
    "/ingest/start",
    "/calls/upload",
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not _settings.rate_limit_enabled:
        return await call_next(request)
    path = request.url.path
    # keep docs/static/health free and predictable
    if path.startswith("/docs") or path.startswith("/openapi") or path.startswith("/corpus") or path in {"/health", "/readyz"}:
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    heavy = any(path.startswith(p) for p in _HEAVY_PREFIXES)
    if not _limiter.check(client_key=client_ip, heavy=heavy):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please retry shortly."},
        )
    return await call_next(request)


app.include_router(query.router, tags=["query"])
app.include_router(ingest.router, tags=["ingest"])
app.include_router(feedback.router, tags=["feedback"])
app.include_router(admin.router, tags=["admin"])
app.include_router(benchmark.router, tags=["benchmark"], dependencies=[Depends(require_admin_token)])
app.include_router(tenant.router)
app.include_router(ui_config.router, tags=["config"], dependencies=[Depends(require_admin_token)])
app.include_router(calls.router, tags=["calls"])

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


@app.get("/readyz")
async def readyz() -> dict:
    """Lightweight readiness endpoint for probes and smoke checks."""
    return {"status": "ready"}

"""POST /feedback — Store user corrections (WP15 §15.2)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from api.core.models import FeedbackRequest, FeedbackResponse
from api.core.settings import get_settings

router = APIRouter()


class FeedbackReviewRequest(BaseModel):
    feedback_id: str
    action: str  # open | acknowledged | resolved | dismissed
    notes: str | None = None


def _feedback_path() -> Path:
    settings = get_settings()
    p = Path(settings.data_dir) / "feedback.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _feedback_reviews_path() -> Path:
    settings = get_settings()
    p = Path(settings.data_dir) / "feedback_reviews.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _require_admin(x_admin_token: str) -> None:
    settings = get_settings()
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")


def _safe_read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except Exception:
            continue
    return out


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(req: FeedbackRequest) -> FeedbackResponse:
    feedback_path = _feedback_path()

    entry = {
        "feedback_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **req.model_dump(),
    }

    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return FeedbackResponse(status="stored", query_id=req.query_id)


@router.get("/feedback/list")
async def feedback_list_endpoint(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    rating: str | None = Query(default=None),
    status: str | None = Query(default=None),
    x_admin_token: str = Header(default="", alias="X-Admin-Token"),
) -> dict:
    _require_admin(x_admin_token)
    feedback_rows = _safe_read_jsonl(_feedback_path())
    review_rows = _safe_read_jsonl(_feedback_reviews_path())

    latest_review_by_id: dict[str, dict] = {}
    for rr in review_rows:
        fid = str(rr.get("feedback_id") or "").strip()
        if not fid:
            continue
        latest_review_by_id[fid] = rr

    enriched: list[dict] = []
    for idx, row in enumerate(feedback_rows):
        ts = str(row.get("timestamp") or "")
        qid = str(row.get("query_id") or "")
        fallback_id = f"{qid}:{ts}:{idx}"
        fid = str(row.get("feedback_id") or fallback_id)
        current_review = latest_review_by_id.get(fid) or {}
        item = {
            "feedback_id": fid,
            "timestamp": ts,
            "query_id": qid,
            "tenant_id": row.get("tenant_id"),
            "rating": row.get("rating"),
            "stars": row.get("stars"),
            "reason": row.get("reason"),
            "correction": row.get("correction"),
            "review_status": current_review.get("action", "open"),
            "review_notes": current_review.get("notes"),
            "reviewed_at": current_review.get("timestamp"),
        }
        enriched.append(item)

    if rating:
        enriched = [r for r in enriched if str(r.get("rating")) == rating]
    if status:
        enriched = [r for r in enriched if str(r.get("review_status")) == status]

    enriched = sorted(enriched, key=lambda r: str(r.get("timestamp") or ""), reverse=True)
    total = len(enriched)
    chunk = enriched[offset : offset + limit]
    return {"items": chunk, "total": total, "offset": offset, "limit": limit}


@router.post("/feedback/review")
async def feedback_review_endpoint(
    body: FeedbackReviewRequest,
    x_admin_token: str = Header(default="", alias="X-Admin-Token"),
) -> dict:
    _require_admin(x_admin_token)
    action = str(body.action or "").strip().lower()
    if action not in {"open", "acknowledged", "resolved", "dismissed"}:
        raise HTTPException(status_code=400, detail="action must be open|acknowledged|resolved|dismissed")

    entry = {
        "feedback_id": body.feedback_id,
        "action": action,
        "notes": body.notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(_feedback_reviews_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"status": "stored", "review": entry}

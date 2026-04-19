"""POST /feedback — Store user corrections (WP15 §15.2)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from api.core.models import FeedbackRequest, FeedbackResponse
from api.core.settings import get_settings

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(req: FeedbackRequest) -> FeedbackResponse:
    settings = get_settings()
    feedback_path = Path(settings.data_dir) / "feedback.jsonl"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **req.model_dump(),
    }

    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return FeedbackResponse(status="stored", query_id=req.query_id)

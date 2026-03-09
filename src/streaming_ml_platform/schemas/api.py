from typing import Any

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = Field(default=10, ge=1, le=50)
    context: dict[str, Any] = Field(default_factory=dict)


class Recommendation(BaseModel):
    item_id: str
    score: float
    reason_codes: list[str]


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list[Recommendation]
    latency_ms: int

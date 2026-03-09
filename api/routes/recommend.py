from fastapi import APIRouter, Depends

from api.dependencies import get_service
from streaming_ml_platform.schemas.api import RecommendRequest, RecommendResponse

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest, service=Depends(get_service)) -> RecommendResponse:
    result = service.recommend(payload.user_id, payload.top_k, payload.context)
    return RecommendResponse(user_id=payload.user_id, recommendations=result["recommendations"], latency_ms=result["latency_ms"])

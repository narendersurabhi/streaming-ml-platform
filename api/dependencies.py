from functools import lru_cache

from streaming_ml_platform.inference.service import RecommendationService
from streaming_ml_platform.paths import ARTIFACT_DIR, PROCESSED_DIR


@lru_cache
def get_service() -> RecommendationService:
    return RecommendationService(
        candidate_model_path=ARTIFACT_DIR / "candidate_model.joblib",
        ranking_model_path=ARTIFACT_DIR / "ranking_model.joblib",
        item_features_path=PROCESSED_DIR / "item_features.csv",
    )

from streaming_ml_platform.paths import ARTIFACT_DIR
from streaming_ml_platform.registry.model_registry import LocalModelRegistry


if __name__ == "__main__":
    registry = LocalModelRegistry(ARTIFACT_DIR / "model_registry.json")
    registry.register("candidate", "1.0.0", "production", str(ARTIFACT_DIR / "candidate_model.joblib"), {"precision_at_10": 0.12}, {"owner": "ml-platform"})
    registry.register("ranking", "1.0.0", "production", str(ARTIFACT_DIR / "ranking_model.joblib"), {"auc": 0.73}, {"owner": "ml-platform"})
    print("Models registered")

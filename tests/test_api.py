import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from streaming_ml_platform.paths import ARTIFACT_DIR, PROCESSED_DIR


def test_health() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


def test_fallback_recommendation(monkeypatch, tmp_path) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"item_id": ["m1", "m2"], "recent_popularity": [0.9, 0.8], "duration": [90, 80], "avg_completion_ratio_item": [0.7, 0.5]})
    df.to_csv(PROCESSED_DIR / "item_features.csv", index=False)
    from api.dependencies import get_service
    get_service.cache_clear()
    # no model files should trigger startup issues only on recommend call; service fallback should work for unknown user when model exists not loaded
    client = TestClient(app)
    # monkeypatch service retrieval call path by forcing dependency override
    from streaming_ml_platform.inference.service import RecommendationService

    class StubService:
        def __init__(self):
            self.metrics = type("M", (), {"snapshot": lambda _: {}})()

        def recommend(self, user_id, top_k=10, context=None):
            svc = RecommendationService.__new__(RecommendationService)
            svc.item_features = df
            svc.metrics = type("MM", (), {"inc": lambda *a, **k: None, "set_gauge": lambda *a, **k: None})()
            return RecommendationService.fallback(svc, top_k)

    app.dependency_overrides[get_service] = lambda: StubService()
    resp = client.post("/recommend", json={"user_id": "cold", "top_k": 2, "context": {}})
    assert resp.status_code == 200
    assert len(resp.json()["recommendations"]) == 2
    app.dependency_overrides.clear()

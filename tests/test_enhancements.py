import pandas as pd

from streaming_ml_platform.inference.ranking import build_candidate_frame
from streaming_ml_platform.inference.retrieval import ShardedRetrievalOrchestrator
from streaming_ml_platform.monitoring.drift import compute_drift_report
from streaming_ml_platform.registry.model_registry import ManagedModelRegistry


class _StubBackend:
    def __init__(self, values):
        self.values = values

    def retrieve(self, user_id: str, top_n: int):
        return self.values[:top_n]


def test_candidate_frame_session_aware_boost() -> None:
    item_features = pd.DataFrame(
        {
            "item_id": ["m1", "m2"],
            "genre": ["action", "drama"],
            "is_franchise": [1, 0],
            "recent_popularity": [0.8, 0.4],
            "duration": [120, 80],
            "avg_completion_ratio_item": [0.6, 0.7],
        }
    )
    frame = build_candidate_frame(
        ["m1", "m2"],
        item_features,
        user_features={"preferred_genre": "action"},
        session_events=[{"item_id": "m1", "dwell_ms": 50000}],
    )
    row = frame[frame["item_id"] == "m1"].iloc[0]
    assert row["genre_affinity"] == 1.0
    assert row["search_match_score"] > 0.0


def test_drift_report_contains_psi_jsd() -> None:
    ref = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    cur = pd.DataFrame({"x": [2, 3, 4, 5, 6]})
    report = compute_drift_report(ref, cur, ["x"])
    assert "x" in report
    assert "psi" in report["x"]
    assert "jsd" in report["x"]


def test_managed_registry_fallback_without_mlflow() -> None:
    registry = ManagedModelRegistry(tracking_uri="http://localhost:5000")
    record = registry.register("ranking", "1", "production", "artifact.joblib", {"auc": 0.7}, {"team": "ml"})
    assert record.name == "ranking"


def test_sharded_retrieval_orchestrator_dedupes() -> None:
    shard = ShardedRetrievalOrchestrator([_StubBackend(["m1", "m2"]), _StubBackend(["m2", "m3"])])
    assert shard.retrieve("u1", 3) == ["m1", "m2", "m3"]

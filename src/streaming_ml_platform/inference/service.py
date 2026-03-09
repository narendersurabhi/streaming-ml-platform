from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from streaming_ml_platform.inference.ranking import build_candidate_frame, reason_codes
from streaming_ml_platform.inference.retrieval import retrieve_candidates
from streaming_ml_platform.models.ranking.infer import rank_candidates
from streaming_ml_platform.monitoring.metrics import MetricsCollector


class RecommendationService:
    def __init__(self, candidate_model_path: Path, ranking_model_path: Path, item_features_path: Path):
        self.candidate_model_path = candidate_model_path
        self.ranking_model_path = ranking_model_path
        self.item_features = pd.read_csv(item_features_path)
        self.metrics = MetricsCollector()

    def fallback(self, top_k: int, region: str | None = None) -> list[dict]:
        self.metrics.inc("fallback_rate", 1)
        frame = self.item_features.sort_values("recent_popularity", ascending=False).head(top_k)
        return [{"item_id": r.item_id, "score": float(r.recent_popularity), "reason_codes": ["fallback_trending"]} for r in frame.itertuples()]

    def recommend(self, user_id: str, top_k: int = 10, context: dict | None = None) -> dict:
        start = time.perf_counter()
        self.metrics.inc("request_count", 1)
        candidates = retrieve_candidates(self.candidate_model_path, user_id, top_n=max(50, top_k))
        if not candidates:
            recs = self.fallback(top_k)
            latency_ms = int((time.perf_counter() - start) * 1000)
            self.metrics.set_gauge("request_latency_ms", latency_ms)
            return {"recommendations": recs, "latency_ms": latency_ms}
        candidate_df = build_candidate_frame(candidates, self.item_features)
        if candidate_df.empty:
            recs = self.fallback(top_k)
            latency_ms = int((time.perf_counter() - start) * 1000)
            return {"recommendations": recs, "latency_ms": latency_ms}
        ranked = rank_candidates(self.ranking_model_path, candidate_df).head(top_k)
        merged = ranked.merge(candidate_df, on="item_id", how="left")
        recs = [{"item_id": r.item_id, "score": float(r.score), "reason_codes": reason_codes(r._asdict())} for r in merged.itertuples()]
        latency_ms = int((time.perf_counter() - start) * 1000)
        self.metrics.set_gauge("request_latency_ms", latency_ms)
        self.metrics.inc("recommendation_count", len(recs))
        return {"recommendations": recs, "latency_ms": latency_ms}

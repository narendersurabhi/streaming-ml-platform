from __future__ import annotations

import pandas as pd

from streaming_ml_platform.models.ranking.model import FEATURE_COLS


def build_candidate_frame(
    candidates: list[str],
    item_features: pd.DataFrame,
    user_features: dict | None = None,
    session_events: list[dict] | None = None,
) -> pd.DataFrame:
    base = item_features[item_features["item_id"].isin(candidates)].copy()
    if base.empty:
        return pd.DataFrame(columns=["item_id", *FEATURE_COLS])

    user_features = user_features or {}
    session_events = session_events or []
    preferred_genre = user_features.get("preferred_genre")
    recent_item_ids = {e.get("item_id") for e in session_events if e.get("item_id")}
    dwell_ms = [float(e.get("dwell_ms", 0.0)) for e in session_events]
    avg_dwell = (sum(dwell_ms) / len(dwell_ms)) if dwell_ms else 0.0

    base["genre_affinity"] = (base["genre"] == preferred_genre).astype(float) if preferred_genre else 0.0
    base["search_match_score"] = base["item_id"].isin(recent_item_ids).astype(float)
    base["franchise_flag"] = base.get("is_franchise", 0).astype(float) if "is_franchise" in base.columns else 0.0
    base["popularity_overlap"] = base["recent_popularity"]
    base["watch_duration"] = base["duration"].clip(upper=120)
    base["completion_ratio"] = base["avg_completion_ratio_item"].clip(upper=1.0)

    # Session-aware reranking boost.
    base["session_boost"] = base["item_id"].isin(recent_item_ids).astype(float) * min(avg_dwell / 60_000.0, 1.0)
    base["search_match_score"] = (base["search_match_score"] + base["session_boost"]).clip(upper=1.0)

    return base[["item_id", *FEATURE_COLS]]


def reason_codes(row: pd.Series) -> list[str]:
    reasons = []
    if row.get("genre_affinity", 0) > 0:
        reasons.append("genre_affinity")
    if row.get("search_match_score", 0) > 0.5:
        reasons.append("session_intent")
    if row.get("popularity_overlap", 0) > 0.2:
        reasons.append("recent_popularity")
    reasons.append("similar_users")
    return reasons[:3]

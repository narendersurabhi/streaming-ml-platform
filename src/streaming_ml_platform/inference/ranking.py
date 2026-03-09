import pandas as pd

from streaming_ml_platform.models.ranking.model import FEATURE_COLS


def build_candidate_frame(candidates: list[str], item_features: pd.DataFrame) -> pd.DataFrame:
    base = item_features[item_features["item_id"].isin(candidates)].copy()
    if base.empty:
        return pd.DataFrame(columns=["item_id", *FEATURE_COLS])
    base["genre_affinity"] = 0
    base["search_match_score"] = 0
    base["popularity_overlap"] = base["recent_popularity"]
    base["watch_duration"] = base["duration"].clip(upper=120)
    base["completion_ratio"] = base["avg_completion_ratio_item"]
    return base[["item_id", *FEATURE_COLS]]


def reason_codes(row: pd.Series) -> list[str]:
    reasons = []
    if row.get("genre_affinity", 0) > 0:
        reasons.append("genre_affinity")
    if row.get("popularity_overlap", 0) > 0.2:
        reasons.append("recent_popularity")
    reasons.append("similar_users")
    return reasons[:3]

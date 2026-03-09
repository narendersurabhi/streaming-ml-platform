import pandas as pd


def build_interaction_features(events: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    merged = events.merge(users[["user_id", "preferred_genres"]], on="user_id", how="left").merge(items[["item_id", "genre", "franchise_flag", "popularity_score"]], on="item_id", how="left")
    merged["genre_affinity"] = (merged["preferred_genres"] == merged["genre"]).astype(int)
    merged["search_match_score"] = merged["search_query"].str.contains(merged["genre"], case=False, regex=False, na=False).astype(int)
    merged["popularity_overlap"] = merged["popularity_score"] * merged["genre_affinity"]
    merged["engagement_label"] = ((merged["event_type"].isin(["click", "like"])) | (merged["completion_ratio"] >= 0.7)).astype(int)
    return merged[["user_id", "item_id", "genre_affinity", "franchise_flag", "search_match_score", "popularity_overlap", "completion_ratio", "watch_duration", "engagement_label"]]

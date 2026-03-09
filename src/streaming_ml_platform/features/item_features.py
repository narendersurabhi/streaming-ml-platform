import pandas as pd


def build_item_features(events: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    by_item = events.groupby("item_id")
    stats = by_item.agg(
        recent_popularity=("event_type", "count"),
        avg_completion_ratio_item=("completion_ratio", "mean"),
    ).reset_index()
    ctr = events.assign(is_click=(events["event_type"] == "click").astype(int)).groupby("item_id").agg(ctr_approx=("is_click", "mean")).reset_index()
    return items.merge(stats, on="item_id", how="left").merge(ctr, on="item_id", how="left").fillna(0)

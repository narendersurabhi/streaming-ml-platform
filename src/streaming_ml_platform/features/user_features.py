import pandas as pd


def build_user_features(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])
    max_ts = ev["timestamp"].max()
    last7 = ev[ev["timestamp"] >= max_ts - pd.Timedelta(days=7)]
    last30 = ev[ev["timestamp"] >= max_ts - pd.Timedelta(days=30)]

    base = ev.groupby("user_id").agg(avg_completion_ratio=("completion_ratio", "mean")).reset_index()
    watch = last7.groupby("user_id").agg(total_watch_time_7d=("watch_duration", "sum")).reset_index()
    active = last30.assign(day=last30["timestamp"].dt.date).groupby("user_id").agg(active_days_30d=("day", "nunique")).reset_index()
    devices = ev.groupby("user_id").agg(device_diversity=("session_id", "nunique")).reset_index()

    out = base.merge(watch, on="user_id", how="left").merge(active, on="user_id", how="left").merge(devices, on="user_id", how="left")
    return out.fillna(0)

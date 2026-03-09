import pandas as pd


REQUIRED = {
    "users": {"user_id", "region", "preferred_genres"},
    "items": {"item_id", "genre", "popularity_score"},
    "events": {"user_id", "item_id", "event_type", "timestamp"},
}


def validate_dataframes(frames: dict[str, pd.DataFrame]) -> None:
    for name, cols in REQUIRED.items():
        missing = cols - set(frames[name].columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

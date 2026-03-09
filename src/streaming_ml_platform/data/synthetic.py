from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from streaming_ml_platform.utils.ids import prefixed_id
from streaming_ml_platform.utils.time import utc_now


def generate_synthetic_data(
    n_users: int = 10000,
    n_items: int = 3000,
    n_events: int = 150000,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    genres = np.array(["drama", "comedy", "action", "sports", "kids", "documentary", "sci-fi"])
    regions = np.array(["us", "eu", "apac", "latam"])
    devices = np.array(["tv", "mobile", "web", "tablet"])

    users = pd.DataFrame(
        {
            "user_id": [prefixed_id("u", i) for i in range(n_users)],
            "age_bucket": rng.choice(["18-24", "25-34", "35-49", "50+"], n_users),
            "region": rng.choice(regions, n_users),
            "subscription_tier": rng.choice(["basic", "standard", "premium"], n_users, p=[0.3, 0.5, 0.2]),
            "preferred_genres": rng.choice(genres, n_users),
            "device_type": rng.choice(devices, n_users),
            "activity_level": rng.choice(["low", "medium", "high"], n_users, p=[0.25, 0.5, 0.25]),
        }
    )

    items = pd.DataFrame(
        {
            "item_id": [prefixed_id("m", i) for i in range(n_items)],
            "title": [f"Title {i}" for i in range(n_items)],
            "genre": rng.choice(genres, n_items),
            "release_year": rng.integers(1980, 2025, n_items),
            "language": rng.choice(["en", "es", "fr", "hi", "jp"], n_items),
            "popularity_score": rng.uniform(0.0, 1.0, n_items),
            "duration": rng.integers(20, 180, n_items),
            "maturity_rating": rng.choice(["G", "PG", "PG-13", "R"], n_items),
            "franchise_flag": rng.integers(0, 2, n_items),
        }
    )

    user_ids = users["user_id"].values
    item_ids = items["item_id"].values
    event_types = np.array(["impression", "click", "watch", "search", "like", "skip"])
    probs = np.array([0.45, 0.14, 0.22, 0.09, 0.05, 0.05])
    now = utc_now()

    timestamps = [now - timedelta(minutes=int(v)) for v in rng.integers(0, 60 * 24 * 30, n_events)]
    events = pd.DataFrame(
        {
            "user_id": rng.choice(user_ids, n_events),
            "item_id": rng.choice(item_ids, n_events),
            "event_type": rng.choice(event_types, n_events, p=probs),
            "timestamp": pd.to_datetime(timestamps),
            "session_id": [prefixed_id("s", int(i)) for i in rng.integers(0, n_users * 4, n_events)],
            "watch_duration": rng.uniform(0, 180, n_events),
            "completion_ratio": rng.uniform(0, 1, n_events),
            "search_query": rng.choice(["", "sports", "family", "thriller", "marvel"], n_events),
        }
    )
    events["recency_hours"] = ((now - events["timestamp"]).dt.total_seconds() / 3600.0).round(2)

    return {"users": users, "items": items, "events": events}

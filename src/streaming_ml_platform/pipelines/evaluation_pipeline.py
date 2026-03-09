import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from streaming_ml_platform.models.candidate.infer import infer_candidates


def precision_recall_at_k(actual: set[str], predicted: list[str], k: int) -> tuple[float, float]:
    pred_k = predicted[:k]
    hits = len(set(pred_k) & actual)
    return hits / max(k, 1), hits / max(len(actual), 1)


def ndcg_at_k(actual: set[str], predicted: list[str], k: int) -> float:
    dcg = 0.0
    for i, p in enumerate(predicted[:k], 1):
        if p in actual:
            dcg += 1 / (np.log2(i + 1))
    ideal = sum(1 / (np.log2(i + 1)) for i in range(1, min(len(actual), k) + 1))
    return float(dcg / ideal) if ideal else 0.0


def evaluate(candidate_model_path: Path, events_path: Path, output_path: Path, k: int = 10) -> dict:
    start = time.perf_counter()
    events = pd.read_csv(events_path)
    users = events["user_id"].drop_duplicates().head(200)
    precs, recs, ndcgs = [], [], []
    for user_id in users:
        user_events = events[events["user_id"] == user_id]
        actual = set(user_events[user_events["event_type"].isin(["click", "watch", "like"])]["item_id"].tail(k))
        predicted = infer_candidates(candidate_model_path, user_id, top_n=k)
        p, r = precision_recall_at_k(actual, predicted, k)
        precs.append(p)
        recs.append(r)
        ndcgs.append(ndcg_at_k(actual, predicted, k))
    report = {
        "precision_at_k": float(sum(precs) / len(precs)),
        "recall_at_k": float(sum(recs) / len(recs)),
        "ndcg_at_k": float(sum(ndcgs) / len(ndcgs)),
        "candidate_generation_time_ms": int((time.perf_counter() - start) * 1000),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

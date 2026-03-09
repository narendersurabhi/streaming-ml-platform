from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class CandidateArtifacts:
    model: NearestNeighbors
    user_item: pd.DataFrame


def build_implicit_matrix(events: pd.DataFrame) -> pd.DataFrame:
    score_map = {"impression": 1.0, "click": 2.0, "watch": 3.0, "like": 4.0, "skip": -1.0, "search": 0.5}
    ev = events.copy()
    ev["score"] = ev["event_type"].map(score_map).fillna(0.0)
    pivot = ev.pivot_table(index="user_id", columns="item_id", values="score", aggfunc="sum", fill_value=0.0)
    return pivot


def train_candidate_model(events: pd.DataFrame, n_neighbors: int = 50) -> CandidateArtifacts:
    matrix = build_implicit_matrix(events)
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(n_neighbors, len(matrix)))
    model.fit(matrix.values)
    return CandidateArtifacts(model=model, user_item=matrix)


def recommend_candidates(artifacts: CandidateArtifacts, user_id: str, top_n: int = 50) -> list[str]:
    if user_id not in artifacts.user_item.index:
        return []
    idx = artifacts.user_item.index.get_loc(user_id)
    user_vec = artifacts.user_item.iloc[[idx]].values
    distances, neighbors = artifacts.model.kneighbors(user_vec)
    neighbor_ids = artifacts.user_item.index[neighbors[0]].tolist()
    scores = artifacts.user_item.loc[neighbor_ids].sum(axis=0).sort_values(ascending=False)
    watched = set(artifacts.user_item.loc[user_id][artifacts.user_item.loc[user_id] > 0].index)
    candidates = [item for item in scores.index if item not in watched]
    return candidates[:top_n]

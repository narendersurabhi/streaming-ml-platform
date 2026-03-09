from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


FEATURE_COLS = [
    "genre_affinity",
    "franchise_flag",
    "search_match_score",
    "popularity_overlap",
    "completion_ratio",
    "watch_duration",
]


@dataclass
class RankingArtifacts:
    model: GradientBoostingClassifier


def train_ranking_model(interactions: pd.DataFrame) -> RankingArtifacts:
    X = interactions[FEATURE_COLS]
    y = interactions["engagement_label"]
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)
    return RankingArtifacts(model=model)

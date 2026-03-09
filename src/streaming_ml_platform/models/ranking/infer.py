from pathlib import Path

import joblib
import pandas as pd

from streaming_ml_platform.models.ranking.model import FEATURE_COLS, RankingArtifacts


def load_ranking_artifacts(path: Path) -> RankingArtifacts:
    return joblib.load(path)


def rank_candidates(path: Path, candidate_df: pd.DataFrame) -> pd.DataFrame:
    artifacts = load_ranking_artifacts(path)
    scores = artifacts.model.predict_proba(candidate_df[FEATURE_COLS])[:, 1]
    output = candidate_df[["item_id"]].copy()
    output["score"] = scores
    return output.sort_values("score", ascending=False)

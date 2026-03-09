from pathlib import Path

import joblib
import pandas as pd

from streaming_ml_platform.models.ranking.model import train_ranking_model


def train_and_save_ranking_model(interactions_path: Path, artifact_dir: Path) -> Path:
    interactions = pd.read_csv(interactions_path)
    artifacts = train_ranking_model(interactions)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output = artifact_dir / "ranking_model.joblib"
    joblib.dump(artifacts, output)
    return output

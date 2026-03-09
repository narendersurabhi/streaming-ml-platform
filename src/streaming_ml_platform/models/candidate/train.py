from pathlib import Path

import joblib
import pandas as pd

from streaming_ml_platform.models.candidate.model import train_candidate_model


def train_and_save_candidate_model(events_path: Path, artifact_dir: Path) -> Path:
    events = pd.read_csv(events_path)
    artifacts = train_candidate_model(events)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output = artifact_dir / "candidate_model.joblib"
    joblib.dump(artifacts, output)
    return output

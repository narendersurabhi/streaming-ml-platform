from pathlib import Path

from streaming_ml_platform.models.candidate.train import train_and_save_candidate_model
from streaming_ml_platform.models.ranking.train import train_and_save_ranking_model


def run(processed_dir: Path, raw_dir: Path, artifact_dir: Path) -> tuple[Path, Path]:
    c = train_and_save_candidate_model(raw_dir / "events.csv", artifact_dir)
    r = train_and_save_ranking_model(processed_dir / "interaction_features.csv", artifact_dir)
    return c, r

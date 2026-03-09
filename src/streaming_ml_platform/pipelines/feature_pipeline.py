from pathlib import Path

from streaming_ml_platform.features.batch_features import run_batch_features


def run(raw_dir: Path, processed_dir: Path) -> None:
    run_batch_features(raw_dir, processed_dir)

from pathlib import Path

from streaming_ml_platform.data.loaders import load_raw_data
from streaming_ml_platform.features.interaction_features import build_interaction_features
from streaming_ml_platform.features.item_features import build_item_features
from streaming_ml_platform.features.user_features import build_user_features
from streaming_ml_platform.utils.io import write_csv


def run_batch_features(raw_dir: Path, processed_dir: Path) -> None:
    data = load_raw_data(raw_dir)
    user_features = build_user_features(data["events"])
    item_features = build_item_features(data["events"], data["items"])
    interaction_features = build_interaction_features(data["events"], data["users"], data["items"])
    write_csv(user_features, processed_dir / "user_features.csv")
    write_csv(item_features, processed_dir / "item_features.csv")
    write_csv(interaction_features, processed_dir / "interaction_features.csv")

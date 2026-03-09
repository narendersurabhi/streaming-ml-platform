from streaming_ml_platform.models.ranking.train import train_and_save_ranking_model
from streaming_ml_platform.paths import ARTIFACT_DIR, PROCESSED_DIR


if __name__ == "__main__":
    out = train_and_save_ranking_model(PROCESSED_DIR / "interaction_features.csv", ARTIFACT_DIR)
    print(f"Saved ranking model to {out}")

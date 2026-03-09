from streaming_ml_platform.models.candidate.train import train_and_save_candidate_model
from streaming_ml_platform.paths import ARTIFACT_DIR, RAW_DIR


if __name__ == "__main__":
    out = train_and_save_candidate_model(RAW_DIR / "events.csv", ARTIFACT_DIR)
    print(f"Saved candidate model to {out}")

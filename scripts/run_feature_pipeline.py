from streaming_ml_platform.paths import PROCESSED_DIR, RAW_DIR
from streaming_ml_platform.pipelines.feature_pipeline import run


if __name__ == "__main__":
    run(RAW_DIR, PROCESSED_DIR)
    print("Features generated")

from streaming_ml_platform.paths import ARTIFACT_DIR, RAW_DIR
from streaming_ml_platform.pipelines.evaluation_pipeline import evaluate


if __name__ == "__main__":
    report = evaluate(ARTIFACT_DIR / "candidate_model.joblib", RAW_DIR / "events.csv", ARTIFACT_DIR / "evaluation_report.json")
    print(report)

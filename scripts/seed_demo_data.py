import subprocess


if __name__ == "__main__":
    commands = [
        ["python", "scripts/generate_synthetic_data.py"],
        ["python", "scripts/run_feature_pipeline.py"],
        ["python", "scripts/train_candidate_model.py"],
        ["python", "scripts/train_ranking_model.py"],
        ["python", "scripts/evaluate_models.py"],
        ["python", "scripts/register_models.py"],
    ]
    for cmd in commands:
        subprocess.check_call(cmd)

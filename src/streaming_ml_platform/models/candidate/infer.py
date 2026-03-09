from pathlib import Path

import joblib

from streaming_ml_platform.models.candidate.model import CandidateArtifacts, recommend_candidates


def load_candidate_artifacts(path: Path) -> CandidateArtifacts:
    return joblib.load(path)


def infer_candidates(path: Path, user_id: str, top_n: int = 50) -> list[str]:
    artifacts = load_candidate_artifacts(path)
    return recommend_candidates(artifacts, user_id, top_n)

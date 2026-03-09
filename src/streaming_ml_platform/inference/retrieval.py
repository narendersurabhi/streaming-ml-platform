from pathlib import Path

from streaming_ml_platform.models.candidate.infer import infer_candidates


def retrieve_candidates(candidate_model_path: Path, user_id: str, top_n: int = 50) -> list[str]:
    return infer_candidates(candidate_model_path, user_id, top_n)

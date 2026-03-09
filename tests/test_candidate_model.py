from streaming_ml_platform.data.synthetic import generate_synthetic_data
from streaming_ml_platform.models.candidate.model import recommend_candidates, train_candidate_model


def test_candidate_model_retrieves() -> None:
    frames = generate_synthetic_data(n_users=50, n_items=40, n_events=500, seed=3)
    artifacts = train_candidate_model(frames["events"])
    user_id = frames["users"]["user_id"].iloc[0]
    out = recommend_candidates(artifacts, user_id, top_n=10)
    assert isinstance(out, list)

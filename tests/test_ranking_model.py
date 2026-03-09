from streaming_ml_platform.data.synthetic import generate_synthetic_data
from streaming_ml_platform.features.interaction_features import build_interaction_features
from streaming_ml_platform.models.ranking.model import train_ranking_model


def test_ranking_model_trains() -> None:
    frames = generate_synthetic_data(n_users=40, n_items=30, n_events=400, seed=4)
    feats = build_interaction_features(frames["events"], frames["users"], frames["items"])
    artifacts = train_ranking_model(feats)
    assert artifacts.model is not None

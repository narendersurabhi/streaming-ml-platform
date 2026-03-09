from streaming_ml_platform.data.synthetic import generate_synthetic_data
from streaming_ml_platform.features.interaction_features import build_interaction_features


def test_interaction_features_has_label() -> None:
    frames = generate_synthetic_data(n_users=20, n_items=10, n_events=200, seed=2)
    out = build_interaction_features(frames["events"], frames["users"], frames["items"])
    assert "engagement_label" in out.columns

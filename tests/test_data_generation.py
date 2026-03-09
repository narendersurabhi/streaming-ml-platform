from streaming_ml_platform.data.synthetic import generate_synthetic_data


def test_synthetic_shapes() -> None:
    frames = generate_synthetic_data(n_users=100, n_items=80, n_events=1000, seed=1)
    assert len(frames["users"]) == 100
    assert len(frames["items"]) == 80
    assert len(frames["events"]) == 1000

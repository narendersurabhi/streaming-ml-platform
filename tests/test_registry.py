from streaming_ml_platform.registry.model_registry import LocalModelRegistry


def test_registry_roundtrip(tmp_path) -> None:
    reg = LocalModelRegistry(tmp_path / "registry.json")
    reg.register("candidate", "1", "production", "a.joblib", {"p@10": 0.2}, {})
    latest = reg.latest("candidate")
    assert latest is not None
    assert latest["version"] == "1"

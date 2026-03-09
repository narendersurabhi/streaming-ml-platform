from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from streaming_ml_platform.paths import CONFIG_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SMP_", env_file=".env", extra="ignore")

    env: str = "dev"
    data_dir: str = "data"
    artifact_dir: str = "data/artifacts"
    config_dir: str = "configs"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    retrieval_backend: str = "default"
    retrieval_shards: int = 1
    enable_online_feature_store: bool = True
    managed_registry_backend: str = "local"
    mlflow_tracking_uri: str | None = None


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_named_config(name: str) -> dict[str, Any]:
    return load_yaml_config(CONFIG_DIR / f"{name}.yaml")

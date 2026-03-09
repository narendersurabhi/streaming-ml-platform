from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ModelRecord:
    name: str
    version: str
    stage: str
    artifact_path: str
    metrics: dict
    metadata: dict
    created_at: str


class LocalModelRegistry:
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text("[]", encoding="utf-8")

    def _read(self) -> list[dict]:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write(self, records: list[dict]) -> None:
        self.registry_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def register(self, name: str, version: str, stage: str, artifact_path: str, metrics: dict, metadata: dict) -> ModelRecord:
        record = ModelRecord(name, version, stage, artifact_path, metrics, metadata, datetime.now(timezone.utc).isoformat())
        records = self._read()
        records.append(asdict(record))
        self._write(records)
        return record

    def latest(self, name: str, stage: str = "production") -> dict | None:
        items = [r for r in self._read() if r["name"] == name and r["stage"] == stage]
        return items[-1] if items else None


class ManagedModelRegistry:
    """Managed registry adapter with MLflow integration when available."""

    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri

    def register(self, name: str, version: str, stage: str, artifact_path: str, metrics: dict, metadata: dict) -> ModelRecord:
        record = ModelRecord(name, version, stage, artifact_path, metrics, metadata, datetime.now(timezone.utc).isoformat())
        try:
            import mlflow
        except ImportError:
            # Graceful fallback keeps workflow operational without optional dependency.
            return record

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_name=f"{name}_{version}"):
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.log_params({"model_name": name, "version": version, "stage": stage, **{f"meta_{k}": str(v) for k, v in metadata.items()}})
            mlflow.log_artifact(artifact_path)
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{Path(artifact_path).name}"
            try:
                mv = mlflow.register_model(model_uri=model_uri, name=name)
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(name=name, version=mv.version, stage=stage.capitalize())
            except Exception:
                # Keep request non-fatal when registry permissions/policies differ.
                pass
        return record

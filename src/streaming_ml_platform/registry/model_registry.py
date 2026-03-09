from __future__ import annotations

import json
from dataclasses import dataclass, asdict
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

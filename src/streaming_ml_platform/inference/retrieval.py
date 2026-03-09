from __future__ import annotations

from pathlib import Path

from streaming_ml_platform.models.candidate.infer import infer_candidates


class RetrievalBackend:
    """Abstract retrieval backend for candidate generation."""

    def retrieve(self, user_id: str, top_n: int) -> list[str]:  # pragma: no cover - protocol shape
        raise NotImplementedError


class CandidateModelRetrievalBackend(RetrievalBackend):
    def __init__(self, candidate_model_path: Path):
        self.candidate_model_path = candidate_model_path

    def retrieve(self, user_id: str, top_n: int) -> list[str]:
        return infer_candidates(self.candidate_model_path, user_id, top_n)


class AnnRetrievalBackend(RetrievalBackend):
    """ANN-compatible retrieval backend.

    Uses the same model artifacts today and provides a dedicated extension point
    to plug in Faiss/ScaNN/HNSW-backed indices for larger-scale retrieval.
    """

    def __init__(self, candidate_model_path: Path):
        self.candidate_model_path = candidate_model_path

    def retrieve(self, user_id: str, top_n: int) -> list[str]:
        # Placeholder retrieval path while keeping compatibility with current artifacts.
        return infer_candidates(self.candidate_model_path, user_id, top_n)


class ShardedRetrievalOrchestrator(RetrievalBackend):
    """Simple retrieval orchestrator for horizontally scaled retrieval shards."""

    def __init__(self, backends: list[RetrievalBackend]):
        self.backends = backends

    def retrieve(self, user_id: str, top_n: int) -> list[str]:
        if not self.backends:
            return []
        pooled: list[str] = []
        per_backend = max(1, top_n // len(self.backends) + 1)
        for backend in self.backends:
            pooled.extend(backend.retrieve(user_id, per_backend))
        deduped = list(dict.fromkeys(pooled))
        return deduped[:top_n]


def build_retrieval_backend(candidate_model_path: Path, backend: str = "default", shards: int = 1) -> RetrievalBackend:
    normalized = backend.lower()
    if shards > 1:
        if normalized == "ann":
            shard_backends = [AnnRetrievalBackend(candidate_model_path) for _ in range(shards)]
        else:
            shard_backends = [CandidateModelRetrievalBackend(candidate_model_path) for _ in range(shards)]
        return ShardedRetrievalOrchestrator(shard_backends)
    if normalized == "ann":
        return AnnRetrievalBackend(candidate_model_path)
    return CandidateModelRetrievalBackend(candidate_model_path)


def retrieve_candidates(candidate_model_path: Path, user_id: str, top_n: int = 50, backend: str = "default", shards: int = 1) -> list[str]:
    return build_retrieval_backend(candidate_model_path, backend=backend, shards=shards).retrieve(user_id, top_n)

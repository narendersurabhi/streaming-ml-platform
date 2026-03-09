# AGENTS Change Log

## Repository status
- Initialized full production-style repository scaffold for `streaming-ml-platform`.
- Implemented synthetic data generation, feature engineering, model training, evaluation, local registry, inference service, and FastAPI endpoints.
- Added containerization, deployment manifests, CI workflow, tests, configs, and docs.

## Recent changes
- Added ANN-ready retrieval backend abstraction and sharded retrieval orchestration in inference layer.
- Added an online feature store module and session-aware reranking inputs in recommendation serving.
- Expanded monitoring with drift report utilities (PSI + JSD) and online model-performance metrics.
- Added managed registry adapter with optional MLflow integration and configurable registration script behavior.
- Updated API response schema to surface online performance snapshot metadata.
- Extended tests for retrieval, reranking, monitoring, and registry enhancements.
- Updated README and inference config to document and configure new capabilities.

## Agent notes
- Prefer `make` targets for standard workflows.
- Keep model artifacts in `data/artifacts`.
- Keep this file updated when making repository changes.

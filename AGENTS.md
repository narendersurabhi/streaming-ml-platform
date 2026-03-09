# AGENTS Change Log

## Repository status
- Initialized full production-style repository scaffold for `streaming-ml-platform`.
- Implemented synthetic data generation, feature engineering, model training, evaluation, local registry, inference service, and FastAPI endpoints.
- Added containerization, deployment manifests, CI workflow, tests, configs, and docs.

## Recent changes
- Added Python package under `src/streaming_ml_platform` with modular components for data, features, models, pipelines, inference, monitoring, and utilities.
- Added runnable scripts in `scripts/` to execute end-to-end local workflows.
- Added API app and routes in `api/`.
- Added deployment assets in `docker/` and `deployment/`.
- Added pytest suite and lint setup.
- Added README and architecture documentation.

## Agent notes
- Prefer `make` targets for standard workflows.
- Keep model artifacts in `data/artifacts`.
- Keep this file updated when making repository changes.

# Architecture

## High-level flow
1. Synthetic events are generated into `data/raw`.
2. Batch feature pipeline builds user, item, and interaction feature sets in `data/processed`.
3. Candidate model trains on implicit feedback from events.
4. Ranking model trains on interaction-level labels and feature vectors.
5. Evaluation pipeline computes ranking and operational metrics.
6. Model registry stores metadata, metrics, and stage assignment.
7. FastAPI serves online inference using retrieval plus reranking with fallback logic.

## Components
- Ingestion simulation: `scripts/generate_synthetic_data.py`
- Feature store simulation: CSV artifacts in `data/processed`
- Candidate retrieval: nearest-neighbor collaborative filtering
- Ranking: gradient boosting classifier
- Serving: FastAPI endpoints for health, recommend, and metrics

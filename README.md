# streaming-ml-platform

Production-style Python repository for a two-stage streaming recommendation system inspired by platforms like Disney+, Hulu, ESPN+, and Netflix.

## What this project demonstrates
- Two-stage recommendation architecture
- Synthetic event pipeline design
- ML feature engineering
- Model training and evaluation
- Local model registry patterns
- Real-time inference API
- Observability and fallback logic
- Containerized deployment

## Project overview
This project simulates an end-to-end ML platform with ingestion, feature pipelines, model training, evaluation, registry, and low-latency serving.

## Repository structure
See `docs/architecture.md` and `docs/system_design.md` for design details.

## Local setup
```bash
python -m venv .venv
source .venv/bin/activate
make install
```

## Run end-to-end
```bash
make generate-data
make features
make train-candidate
make train-ranking
make evaluate
make register
make serve
```

## API usage
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u_00001","top_k":10,"context":{"device_type":"tv","time_of_day":"evening"}}'
```

## Testing and linting
```bash
make test
make lint
```

## Future improvements
- ANN retrieval backend with Faiss or ScaNN
- Online feature store integration
- Model registry backed by MLflow
- Session-aware reranking
- Expanded drift monitoring

## Real-world mapping
- Candidate generation mirrors collaborative filtering retrieval tiers in large recommender stacks.
- Ranking service mirrors feature-rich reranking used by content feeds.
- API and observability patterns align with production inference services.

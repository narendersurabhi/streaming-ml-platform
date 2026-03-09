.PHONY: install generate-data features train-candidate train-ranking evaluate register serve test lint

install:
	python -m pip install -e .[dev]

generate-data:
	python scripts/generate_synthetic_data.py

features:
	python scripts/run_feature_pipeline.py

train-candidate:
	python scripts/train_candidate_model.py

train-ranking:
	python scripts/train_ranking_model.py

evaluate:
	python scripts/evaluate_models.py

register:
	python scripts/register_models.py

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check .

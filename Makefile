.PHONY: setup prepare train test api dashboard cv-data
setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
prepare:
	python scripts/run_pipeline.py --raw-dir data/raw --skip-training
train:
	python scripts/train_models.py
test:
	pytest -q
api:
	uvicorn api.main:app --reload
dashboard:
	streamlit run app.py
cv-data:
	python scripts/generate_cv_dataset.py

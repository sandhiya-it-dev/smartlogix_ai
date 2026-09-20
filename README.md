# SmartLogix AI

An end-to-end, modular logistics intelligence platform built from the supplied SmartLogix datasets. 

## Included modules

- Data cleaning and feature engineering for all 11 supplied datasets
- SQLite development database plus PostgreSQL-compatible schema
- Logistics transport-mode classification
- Delivery ETA regression
- Drone predictive-maintenance classification
- Constraint-aware route and payload optimization
- YOLO-compatible synthetic drone-damage dataset generator and training entry point
- Review sentiment analysis, product recommendations, order tracking, and local RAG FAQ retrieval
- Agent router for tracking, products, reviews, logistics, and support
- FastAPI REST service
- Streamlit monitoring dashboard
- Automated tests, metrics, documentation, and reproducible scripts

## Important CV disclosure

No drone inspection images were supplied. `scripts/generate_cv_dataset.py` creates a clearly labelled synthetic development dataset with YOLO annotations for five damage classes; healthy images have empty label files. Replace `data/cv/images` and `data/cv/labels` with real annotated inspection images before production use.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py --raw-dir data/raw
streamlit run app.py
```

In a second terminal:

```bash
uvicorn api.main:app --reload
```

API documentation: `http://127.0.0.1:8000/docs`.

## Train models separately

```bash
python scripts/train_models.py
python scripts/generate_cv_dataset.py --samples-per-class 80
# Optional after installing ultralytics:
python scripts/train_yolo.py --epochs 25
```

## Project layout

```text
smartlogix_ai/
  api/              FastAPI endpoints
  config/           settings and YOLO data config
  data/raw/         supplied source datasets
  docs/             architecture, data dictionary, evaluation guide
  models/           trained artifacts and metrics
  scripts/          runnable pipelines
  sql/              relational schema
  src/              reusable application modules
  tests/            automated tests
  app.py            Streamlit dashboard
```

## Notes

- The development database is SQLite so the whole solution runs locally. Set `DATABASE_URL` to a PostgreSQL SQLAlchemy URL for PostgreSQL.
- The chatbot works offline through retrieval and structured tools. If `OPENAI_API_KEY` is provided, the architecture can be extended with an external LLM without changing data access tools.
- AWS EC2, RDS, S3, Lambda, and email notification deployment are listed only in `docs/CLOUD_FUTURE_WORK.md` and are not implemented.


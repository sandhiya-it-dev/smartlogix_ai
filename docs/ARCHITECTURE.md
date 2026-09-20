# Architecture and requirement traceability

## Flow

1. Raw CSV/JSON data enters the cleaning layer.
2. Validated tables are written to processed CSV and a local relational database.
3. Independent ML, optimization, CV, NLP/RAG, and recommendation modules consume curated tables.
4. FastAPI exposes core services; Streamlit provides operational dashboards.
5. The agent router selects tracking, review, product, or FAQ tools based on the request.

| PDF requirement | Implementation |
|---|---|
| Data cleaning, missing values, transformation | `src/cleaning.py`, `src/data_utils.py` |
| SQL database design | `src/database.py`, `sql/schema_postgresql.sql` |
| Mode classification | `src/modeling.py` |
| ETA prediction | `src/modeling.py` |
| Predictive maintenance | `src/modeling.py` |
| VRP, battery and payload constraints | `src/optimizer.py` |
| YOLO damage detection | `scripts/generate_cv_dataset.py`, `scripts/train_yolo.py` |
| Sentiment analysis | `src/modeling.py` |
| Order tracking and recommendations | `src/services.py` |
| RAG and multi-agent assistant | `src/rag.py`, `src/services.py` |
| REST API | `api/main.py` |
| Dashboards | `app.py` |
| Evaluation metrics | `models/metrics.json` after training |
| Cloud | Deferred per user request |

## Modeling safeguards

- Fixed random seeds provide reproducible splits.
- Models are fitted only on training folds.
- Numeric imputation and scaling and categorical imputation/encoding live inside pipelines, preventing preprocessing leakage.
- Classification uses stratified splits and weighted metrics.
- Mode labels come from historical decisions; performance measures imitation of recorded choices, not proof of causal optimality.
- ETA evaluation uses MAE, MSE, RMSE, and R-squared.
- Maintenance evaluation emphasizes recall and F1 because missed failures are costly.

## Known limitations

- The supplied data is synthetic/simulated and contains deliberate quality defects.
- No real drone images were supplied; synthetic CV images validate the pipeline but cannot establish production accuracy.
- Nearest-neighbor routing is an explainable baseline, not an exact large-scale VRP solver.
- Offline TF-IDF retrieval is a RAG baseline. A generative LLM can be connected later while retaining the same verified data tools.


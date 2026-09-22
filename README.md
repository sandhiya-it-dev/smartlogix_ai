# SmartLogix AI

SmartLogix AI is an end-to-end intelligent logistics platform designed to support faster, safer and more efficient product delivery. It combines data engineering, PostgreSQL, machine learning, route optimization, FAQ retrieval, computer vision, FastAPI, Streamlit and AWS deployment in one modular project.

## Project objectives

- Recommend a suitable transport mode for an order.
- Predict the estimated delivery time (ETA).
- Identify drones that may require maintenance.
- Classify customer-review sentiment.
- Arrange delivery stops using a distance-based route optimizer.
- Support order tracking, product recommendations and customer FAQs.
- Demonstrate drone-damage detection using computer vision and YOLOv8.
- Expose backend capabilities through FastAPI and visualize them in Streamlit.

## End-to-end workflow

```text
Raw CSV and JSON files
        ↓
Cleaning and preprocessing
        ↓
PostgreSQL database (12 tables)
        ↓
Model training and evaluation
        ↓
Route optimization and drone feasibility
        ↓
FAQ retrieval and application services
        ↓
Computer vision and YOLOv8
        ↓
FastAPI backend
        ↓
Streamlit interface
        ↓
AWS deployment
```

## Data cleaning and preprocessing

The cleaning pipeline handles common data-quality problems in CSV and JSON data:

- Duplicate-record removal
- Missing-value handling
- Median imputation for numerical columns
- Most-frequent-value imputation for categorical columns
- Datetime standardization
- Currency and numeric-string conversion
- Category and text standardization
- Boolean conversion to `0` and `1`
- Weight conversion from grams to kilograms
- JSON flattening and serialization of nested lists and dictionaries
- Feature engineering such as logarithmic and eligibility features

Run the complete local pipeline with:

```bash
python scripts/run_pipeline.py --raw-dir data/raw
```

## Database layer

The cleaned datasets are persisted as 12 relational tables:

| Table | Purpose |
|---|---|
| `orders` | Order, delivery and transport information |
| `customers` | Customer information |
| `fleet` | Vehicle and ownership information |
| `drone_telemetry` | Drone flight and sensor measurements |
| `maintenance` | Maintenance history |
| `weather` | Weather observations |
| `traffic` | Traffic information |
| `delivery_logs` | Order-tracking events |
| `products` | Product catalogue |
| `reviews` | Customer reviews and sentiment labels |
| `routes` | Route-level information |
| `waypoints` | Route waypoint coordinates |

The project supports SQLite for simple local development and PostgreSQL for deployment. Indexes are created on frequently searched identifiers such as `order_id` and `product_id`.

Configure the database through `.env`:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/smartlogix
MODEL_DIR=models
```

Do not commit `.env` or database passwords to Git.

## Machine-learning models

| Task | Problem type | Model |
|---|---|---|
| Transport-mode prediction | Multiclass classification | Random Forest Classifier |
| ETA prediction | Regression | HistGradientBoosting Regressor |
| Predictive maintenance | Binary classification | Random Forest Classifier |
| Review sentiment | Multiclass text classification | TF-IDF + Logistic Regression |

The model pipeline includes train/test splitting, numerical imputation and scaling, categorical imputation and one-hot encoding, model training, evaluation and Joblib serialization.

Train the models separately:

```bash
python scripts/train_models.py
```

Generated artifacts include:

```text
models/mode_classifier.joblib
models/eta_regressor.joblib
models/maintenance_classifier.joblib
models/sentiment_classifier.joblib
models/metrics.json
```

Evaluation metrics are regenerated in `models/metrics.json` during training. Near-perfect sentiment results should be interpreted cautiously because the available review dataset contains highly repeated synthetic text patterns.

## Route optimization

The route optimizer uses:

- The Haversine formula to calculate approximate geographic distance between latitude/longitude coordinates.
- A nearest-neighbour greedy algorithm to select the next closest unvisited stop.
- An optional return-to-origin calculation.
- Drone feasibility checks based on distance, payload, maximum range, capacity, starting battery and reserve battery.

This baseline is fast and explainable, but it does not guarantee a globally optimal route.

## FAQ retrieval and application services

`src/rag.py` contains a lightweight FAQ retriever:

1. FAQ questions and answers are converted into character-level TF-IDF vectors.
2. Character n-grams of length 3–5 handle variations such as `track`, `tracking` and minor spelling mistakes.
3. A customer query is converted using the same vectorizer.
4. Similarity scores are calculated against the stored FAQs.
5. The highest-scoring reliable answer is returned with its source.

This module performs retrieval and grounded answer selection. It does not currently use an external generative LLM.

Application services also support order tracking, product recommendations and review summaries using database queries.

## Computer vision and YOLOv8

No real drone-inspection photographs were supplied. Therefore, `scripts/generate_cv_dataset.py` generates a clearly labelled synthetic dataset using OpenCV.

The dataset represents:

- Healthy drone
- Broken propeller
- Frame damage
- Motor damage
- Landing-gear damage
- Battery damage

YOLO detects the five damage classes; healthy images contain empty label files. Each damage annotation uses:

```text
class_id x_center y_center width height
```

Generate the synthetic dataset and train YOLOv8n:

```bash
python scripts/generate_cv_dataset.py --samples-per-class 20
python scripts/train_yolo.py --epochs 25
```

The best weights are saved under:

```text
models/yolo/drone_damage/weights/best.pt
```

The observed synthetic validation result was approximately `0.855 mAP50`. Landing-gear damage performed poorly and requires additional, more diverse training examples. Real annotated inspection images are required before production use.

YOLO training and evaluation currently run as a separate module. Live YOLO image upload and inference are not yet connected to the Streamlit interface.

## FastAPI backend

FastAPI exposes predictions and logistics services as JSON endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | API health check |
| `POST` | `/predict/mode` | Transport-mode prediction |
| `POST` | `/predict/eta` | ETA prediction |
| `POST` | `/predict/maintenance` | Maintenance prediction |
| `GET` | `/orders/{order_id}` | Order tracking |
| `GET` | `/products/recommend` | Product recommendations |
| `GET` | `/reviews/{product_id}/summary` | Review summary |
| `POST` | `/routes/optimize` | Delivery-route optimization |
| `POST` | `/drones/feasibility` | Drone feasibility check |
| `POST` | `/chat` | Application agent/FAQ request |

Start the API from the project root:

```bash
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- Health check: `http://127.0.0.1:8000/health`
- Swagger documentation: `http://127.0.0.1:8000/docs`

## Streamlit application

The Streamlit interface provides pages for:

- Dashboard and order monitoring
- Order tracking
- Delivery-mode recommendation
- Fleet management
- Route optimization
- Drone telemetry and feasibility
- Logistics analytics
- AI assistant
- Products, reviews and notifications
- Model-performance evaluation

Start Streamlit in a second terminal while FastAPI remains running:

```bash
streamlit run app.py
```

## AWS deployment

The project has been demonstrated with:

- **Amazon EC2:** Runs FastAPI and Streamlit.
- **Amazon RDS for PostgreSQL:** Stores the 12 application tables.
- **Amazon S3:** Privately stores trained Joblib models and evaluation metrics.
- **AWS IAM:** Gives the EC2 instance temporary, limited permission to retrieve required model artifacts from S3.

The S3 bucket remains private. The application accesses it through an EC2 IAM role rather than embedding permanent AWS access keys in source code.

AWS Lambda and email notification services are planned future enhancements and are not part of the current working deployment.

## Local setup

### 1. Clone and enter the project

```bash
git clone <repository-url>
cd SmartLogix_AI_Project
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For YOLO training, install Ultralytics if it is not already installed:

```bash
pip install ultralytics
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Update `DATABASE_URL` in `.env` for the required local or PostgreSQL database.

### 5. Prepare data and models

```bash
python scripts/run_pipeline.py --raw-dir data/raw
```

### 6. Run FastAPI

```bash
python -m uvicorn api.main:app --reload --port 8000
```

### 7. Run Streamlit in another terminal

```bash
streamlit run app.py
```

## Testing

```bash
pytest -q
```

## Project structure

```text
SmartLogix_AI_Project/
├── api/                       FastAPI application
├── config/                    YOLO dataset configuration
├── data/
│   ├── raw/                   Source CSV and JSON files
│   ├── processed/             Cleaned data and local database
│   └── cv/                    Synthetic YOLO images and labels
├── docs/                      Architecture and evaluation notes
├── models/                    Joblib models, metrics and YOLO weights
├── scripts/                   Pipeline, training and data-generation scripts
├── sql/                       PostgreSQL schema
├── src/
│   ├── cleaning.py            Cleaning and preprocessing
│   ├── config.py              Paths and environment configuration
│   ├── database.py            Persistence and SQL query helpers
│   ├── modeling.py            ML training and evaluation
│   ├── optimizer.py           Routes and drone feasibility
│   ├── rag.py                 FAQ retrieval
│   └── services.py            Application service layer
├── tests/                     Automated tests
├── app.py                     Streamlit interface
├── requirements.txt           Python dependencies
└── README.md
```

## Limitations and future improvements

- Collect real drone-damage images and improve landing-gear detection.
- Integrate YOLO image upload and inference into Streamlit.
- Expand the FAQ knowledge base and optionally add a grounded generative LLM.
- Replace the greedy route baseline with a constrained vehicle-routing solver.
- Add authentication, HTTPS, structured logging and monitoring.
- Run FastAPI and Streamlit as managed background services.
- Add CI/CD, containerization and automated cloud deployment.
- Add AWS Lambda and email/SNS/SES notification workflows.

## Responsible-use note

This project is an educational prototype. Model predictions, synthetic computer-vision results and optimized routes should be validated before operational use in a real logistics environment.

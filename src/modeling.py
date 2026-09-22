from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from .config import MODEL_DIR
from .features import add_mode_features as build_mode_features


# ==========================================================
# COMMON PREPROCESSING
# ==========================================================

def _preprocessor(
    numeric: list[str],
    categorical: list[str],
) -> ColumnTransformer:
    """
    Create preprocessing pipelines for numerical and
    categorical columns.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(strategy="median"),
            ),
            (
                "scale",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ]
    )


def _model_frame(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    """
    Convert columns into sklearn-compatible datatypes.
    """

    output = df[numeric + categorical].copy()

    for column in numeric:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    for column in categorical:
        output[column] = (
            output[column]
            .astype(object)
            .where(output[column].notna(), np.nan)
        )

    return output


# ==========================================================
# CLASSIFICATION METRICS
# ==========================================================

def _classification_metrics(
    model,
    X_test,
    y_test,
) -> dict:
    """
    Calculate classification evaluation metrics.
    """

    predictions = model.predict(X_test)

    labels = [str(label) for label in model.classes_]

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_,
        ).tolist(),
        "labels": labels,
    }

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_test)

            if len(model.classes_) == 2:
                results["roc_auc"] = roc_auc_score(
                    y_test,
                    probabilities[:, 1],
                )
            else:
                results["roc_auc_ovr_weighted"] = roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                    labels=model.classes_,
                )

        except (ValueError, IndexError):
            pass

    return results


# ==========================================================
# TRANSPORT MODE CLASSIFIER
# ==========================================================

def train_mode_classifier(
    orders: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    """
    Train a Random Forest model to predict transportation mode.
    """

    raw_features = [
        "distance_km",
        "package_weight_kg",
        "quantity",
        "is_fragile",
        "is_hazmat",
        "cold_chain_required",
        "order_value_inr",
        "delivery_priority",
        "weather_condition_at_dest",
        "origin_city",
        "destination_city",
    ]

    derived_features = [
        "distance_log",
        "weight_log",
        "value_log",
        "same_city",
        "drone_eligible",
        "long_distance",
        "heavy_package",
    ]

    numeric_features = raw_features[:7] + derived_features
    categorical_features = raw_features[7:]

    data = orders.dropna(
        subset=["transport_mode"]
    ).copy()

    X = data[raw_features].copy()
    y = data["transport_mode"].astype(str)

    for column in raw_features[:7]:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    for column in categorical_features:
        X[column] = (
            X[column]
            .astype(object)
            .where(X[column].notna(), np.nan)
        )

    class_counts = y.value_counts()

    stratify_target = (
        y if class_counts.min() >= 2 else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_target,
    )

    model = Pipeline(
        steps=[
            (
                "features",
                FunctionTransformer(
                    build_mode_features,
                    validate=False,
                ),
            ),
            (
                "prep",
                _preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    metrics = _classification_metrics(
        model,
        X_test,
        y_test,
    )

    return model, metrics


# ==========================================================
# ETA REGRESSION MODEL
# ==========================================================

def train_eta_regressor(
    orders: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    """
    Train a HistGradientBoostingRegressor for ETA prediction.
    """

    features = [
        "distance_km",
        "package_weight_kg",
        "quantity",
        "is_fragile",
        "is_hazmat",
        "cold_chain_required",
        "delivery_cost_inr",
        "transport_mode",
        "delivery_priority",
        "weather_condition_at_dest",
    ]

    numeric_features = features[:7]
    categorical_features = features[7:]

    data = orders.dropna(
        subset=["actual_delivery_hours"]
    ).copy()

    X = _model_frame(
        data,
        numeric_features,
        categorical_features,
    )

    y = pd.to_numeric(
        data["actual_delivery_hours"],
        errors="coerce",
    )

    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    preprocessing = _preprocessor(
        numeric_features,
        categorical_features,
    )

    # HistGradientBoostingRegressor requires dense input.
    try:
        preprocessing.set_params(
            cat__onehot__sparse_output=False
        )
    except ValueError:
        # Compatibility with older scikit-learn versions.
        preprocessing.set_params(
            cat__onehot__sparse=False
        )

    model = Pipeline(
        steps=[
            ("prep", preprocessing),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=180,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(
            y_test,
            predictions,
        ),
        "mse": mean_squared_error(
            y_test,
            predictions,
        ),
        "rmse": mean_squared_error(
            y_test,
            predictions,
        ) ** 0.5,
        "r2": r2_score(
            y_test,
            predictions,
        ),
    }

    return model, metrics


# ==========================================================
# MAINTENANCE CLASSIFIER
# ==========================================================

def train_maintenance_classifier(
    telemetry: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    """
    Train a Random Forest model to predict maintenance need.
    """

    numeric_features = [
        "flight_duration_min",
        "cumulative_flight_hours",
        "battery_cycles",
        "battery_start_pct",
        "battery_end_pct",
        "battery_health_pct",
        "motor_temp_c",
        "vibration_rms",
        "payload_kg",
        "max_altitude_m",
        "wind_speed_kmph",
        "rotor_rpm_avg",
        "route_deviation_m",
    ]

    categorical_features = [
        "gps_signal_quality",
    ]

    required_columns = (
        numeric_features
        + categorical_features
        + ["maintenance_required"]
    )

    data = telemetry[required_columns].dropna(
        subset=["maintenance_required"]
    ).copy()

    X = _model_frame(
        data,
        numeric_features,
        categorical_features,
    )

    y = pd.to_numeric(
        data["maintenance_required"],
        errors="coerce",
    )

    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    stratify_target = (
        y if y.value_counts().min() >= 2 else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_target,
    )

    model = Pipeline(
        steps=[
            (
                "prep",
                _preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=180,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    metrics = _classification_metrics(
        model,
        X_test,
        y_test,
    )

    return model, metrics


# ==========================================================
# SENTIMENT CLASSIFIER
# ==========================================================

def normalize_review_text(value: object) -> str:
    """
    Normalize review text to identify repeated text patterns.
    """

    text = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    text = text.lower().strip()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text,
    )

    text = re.sub(
        r"\d+",
        " number ",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _create_sentiment_pipeline() -> Pipeline:
    """
    Create the TF-IDF and Logistic Regression pipeline.
    """

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=25000,
                    sublinear_tf=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_sentiment(
    reviews: pd.DataFrame,
) -> tuple[Pipeline, dict]:
    """
    Train and validate the customer-review sentiment model.

    Duplicate normalized review texts are removed before the
    train/test split to reduce duplicate-text leakage.
    """

    data = reviews.dropna(
        subset=[
            "review_text",
            "sentiment_label",
        ]
    ).copy()

    data["review_text"] = (
        data["review_text"]
        .astype(str)
        .str.strip()
    )

    data["sentiment_label"] = (
        data["sentiment_label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data = data[
        data["review_text"].ne("")
        & data["sentiment_label"].ne("")
    ].copy()

    data["normalized_text"] = data[
        "review_text"
    ].map(normalize_review_text)

    data = data[
        data["normalized_text"].ne("")
    ].copy()

    original_row_count = len(data)

    label_counts_per_text = (
        data.groupby("normalized_text")[
            "sentiment_label"
        ]
        .nunique()
    )

    conflicting_texts = set(
        label_counts_per_text[
            label_counts_per_text > 1
        ].index
    )

    conflicting_row_count = int(
        data["normalized_text"]
        .isin(conflicting_texts)
        .sum()
    )

    if conflicting_texts:
        data = data[
            ~data["normalized_text"].isin(
                conflicting_texts
            )
        ].copy()

    unique_data = data.drop_duplicates(
        subset=["normalized_text"],
        keep="first",
    ).copy()

    unique_review_count = len(unique_data)

    duplicate_text_count = max(
        original_row_count
        - unique_review_count
        - conflicting_row_count,
        0,
    )

    duplicate_text_rate = (
        duplicate_text_count / original_row_count
        if original_row_count
        else 0.0
    )

    if unique_review_count < 10:
        raise ValueError(
            "Not enough unique review texts to train "
            "the sentiment classifier."
        )

    X = unique_data["normalized_text"]
    y = unique_data["sentiment_label"]

    class_counts = y.value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "Sentiment training requires at least two classes."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Each sentiment class requires at least two "
            "unique review texts."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    evaluation_model = _create_sentiment_pipeline()

    evaluation_model.fit(
        X_train,
        y_train,
    )

    metrics = _classification_metrics(
        evaluation_model,
        X_test,
        y_test,
    )

    number_of_folds = min(
        5,
        int(class_counts.min()),
    )

    if number_of_folds >= 2:
        cross_validation = StratifiedKFold(
            n_splits=number_of_folds,
            shuffle=True,
            random_state=42,
        )

        cv_results = cross_validate(
            _create_sentiment_pipeline(),
            X,
            y,
            cv=cross_validation,
            scoring={
                "accuracy": "accuracy",
                "f1_macro": "f1_macro",
                "f1_weighted": "f1_weighted",
            },
            n_jobs=-1,
        )

        metrics["cv_folds"] = number_of_folds
        metrics["cv_accuracy_mean"] = float(
            np.mean(cv_results["test_accuracy"])
        )
        metrics["cv_accuracy_std"] = float(
            np.std(cv_results["test_accuracy"])
        )
        metrics["cv_f1_macro_mean"] = float(
            np.mean(cv_results["test_f1_macro"])
        )
        metrics["cv_f1_macro_std"] = float(
            np.std(cv_results["test_f1_macro"])
        )
        metrics["cv_f1_weighted_mean"] = float(
            np.mean(cv_results["test_f1_weighted"])
        )
        metrics["cv_f1_weighted_std"] = float(
            np.std(cv_results["test_f1_weighted"])
        )

    metrics["original_usable_reviews"] = (
        original_row_count
    )

    metrics["unique_review_texts"] = (
        unique_review_count
    )

    metrics["duplicate_text_count"] = (
        duplicate_text_count
    )

    metrics["duplicate_text_rate"] = (
        duplicate_text_rate
    )

    metrics["conflicting_text_rows_removed"] = (
        conflicting_row_count
    )

    warnings = []

    if duplicate_text_rate >= 0.50:
        warnings.append(
            "The review dataset contains many repeated text "
            "patterns. Internal scores may overestimate "
            "real-world performance."
        )

    if unique_review_count < 500:
        warnings.append(
            "The dataset contains fewer than 500 unique "
            "normalized review texts. More diverse real-world "
            "reviews are recommended."
        )

    if metrics["accuracy"] >= 0.99:
        warnings.append(
            "Near-perfect holdout accuracy should be validated "
            "using an independent external review dataset."
        )

    metrics["warnings"] = warnings

    # Train the saved model on all unique review texts after
    # evaluation has been completed.
    final_model = _create_sentiment_pipeline()

    final_model.fit(
        X,
        y,
    )

    return final_model, metrics


# ==========================================================
# TRAIN AND SAVE ALL MODELS
# ==========================================================

def train_all(
    tables: dict[str, pd.DataFrame],
    model_dir: str | Path = MODEL_DIR,
) -> dict:
    """
    Train all SmartLogix models and save them using joblib.
    """

    output_directory = Path(model_dir)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_jobs = {
        "mode_classifier": train_mode_classifier(
            tables["orders"]
        ),
        "eta_regressor": train_eta_regressor(
            tables["orders"]
        ),
        "maintenance_classifier":
            train_maintenance_classifier(
                tables["drone_telemetry"]
            ),
        "sentiment_classifier": train_sentiment(
            tables["reviews"]
        ),
    }

    all_metrics = {}

    for model_name, (
        trained_model,
        model_metrics,
    ) in training_jobs.items():

        model_path = (
            output_directory
            / f"{model_name}.joblib"
        )

        joblib.dump(
            trained_model,
            model_path,
        )

        all_metrics[model_name] = model_metrics

        print(
            f"Saved {model_name}: "
            f"{model_path.resolve()}"
        )

    metrics_path = (
        output_directory
        / "metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            all_metrics,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved evaluation metrics: "
        f"{metrics_path.resolve()}"
    )

    return all_metrics


# ==========================================================
# CONSOLE OUTPUT
# ==========================================================

def print_main_results(
    results: dict,
) -> None:
    """
    Print important evaluation results in the terminal.
    """

    print("\nMain evaluation results")
    print("-" * 60)

    display_metrics = [
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "roc_auc",
        "roc_auc_ovr_weighted",
        "r2",
        "mae",
        "rmse",
        "cv_accuracy_mean",
        "cv_f1_macro_mean",
        "cv_f1_weighted_mean",
        "unique_review_texts",
        "duplicate_text_rate",
    ]

    percentage_metrics = {
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "roc_auc",
        "roc_auc_ovr_weighted",
        "r2",
        "cv_accuracy_mean",
        "cv_f1_macro_mean",
        "cv_f1_weighted_mean",
        "duplicate_text_rate",
    }

    integer_metrics = {
        "unique_review_texts",
    }

    for model_name, scores in results.items():
        display_name = (
            model_name
            .replace("_", " ")
            .title()
        )

        print(f"\n{display_name}")

        for metric in display_metrics:
            if metric not in scores:
                continue

            value = scores[metric]

            if metric in percentage_metrics:
                print(
                    f"  {metric:<26}: "
                    f"{float(value):.2%}"
                )

            elif metric in integer_metrics:
                print(
                    f"  {metric:<26}: "
                    f"{int(value)}"
                )

            else:
                print(
                    f"  {metric:<26}: "
                    f"{float(value):.3f}"
                )

        for warning in scores.get(
            "warnings",
            [],
        ):
            print(f"  Warning: {warning}")


# ==========================================================
# RUN MODEL TRAINING
# ==========================================================

if __name__ == "__main__":
    from .cleaning import clean_all
    from .config import RAW_DIR

    print(
        "Loading and cleaning SmartLogix datasets..."
    )

    cleaned_tables = clean_all(RAW_DIR)

    print(
        "Training classification and regression models..."
    )

    training_results = train_all(
        cleaned_tables
    )

    print(
        "\nModel training completed successfully."
    )

    print(
        f"Saved models in: "
        f"{Path(MODEL_DIR).resolve()}"
    )

    print_main_results(training_results)
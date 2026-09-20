from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import MODEL_DIR


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def _model_frame(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Return sklearn-friendly dtypes; pandas StringDtype's pd.NA confuses older imputers."""
    out = df[numeric + categorical].copy()
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in categorical:
        out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
    return out


def add_mode_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create nonlinear, pre-delivery features for transport-mode selection."""
    data = frame.copy()
    numeric_sources = [
        "distance_km",
        "package_weight_kg",
        "quantity",
        "is_fragile",
        "is_hazmat",
        "cold_chain_required",
        "order_value_inr",
    ]
    for column in numeric_sources:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["distance_log"] = np.log1p(data["distance_km"].clip(lower=0))
    data["weight_log"] = np.log1p(data["package_weight_kg"].clip(lower=0))
    data["value_log"] = np.log1p(data["order_value_inr"].clip(lower=0))
    data["same_city"] = (
        data["origin_city"].astype(str).str.lower()
        == data["destination_city"].astype(str).str.lower()
    ).astype(int)
    data["drone_eligible"] = (
        (data["distance_km"] <= 25)
        & (data["package_weight_kg"] <= 5)
        & (data["is_hazmat"] == 0)
        & (data["cold_chain_required"] == 0)
    ).astype(int)
    data["long_distance"] = (data["distance_km"] > 500).astype(int)
    data["heavy_package"] = (data["package_weight_kg"] > 100).astype(int)
    return data


def _classification_metrics(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    result = {"accuracy":accuracy_score(y_test,pred), "f1_weighted":f1_score(y_test,pred,average="weighted"),
              "classification_report":classification_report(y_test,pred,output_dict=True,zero_division=0),
              "confusion_matrix":confusion_matrix(y_test,pred).tolist()}
    if hasattr(model, "predict_proba"):
        try: result["roc_auc_ovr_weighted"] = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="weighted")
        except ValueError: pass
    return result


def train_mode_classifier(orders: pd.DataFrame) -> tuple[Pipeline, dict]:
    raw_features = [
        "distance_km", "package_weight_kg", "quantity", "is_fragile",
        "is_hazmat", "cold_chain_required", "order_value_inr",
        "delivery_priority", "weather_condition_at_dest", "origin_city",
        "destination_city",
    ]
    derived_features = [
        "distance_log", "weight_log", "value_log", "same_city",
        "drone_eligible", "long_distance", "heavy_package",
    ]
    data = orders.dropna(subset=["transport_mode"])
    numeric = raw_features[:7] + derived_features
    categorical = raw_features[7:]
    X = data[raw_features].copy()
    for column in raw_features[:7]:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in categorical:
        X[column] = X[column].astype(object).where(X[column].notna(), np.nan)
    y = data["transport_mode"]
    stratify = y if y.value_counts().min() >= 2 else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=stratify)
    model = Pipeline([
        ("features", FunctionTransformer(add_mode_features, validate=False)),
        ("prep", _preprocessor(numeric, categorical)),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(Xtr,ytr)
    return model,_classification_metrics(model,Xte,yte)


def train_eta_regressor(orders: pd.DataFrame) -> tuple[Pipeline, dict]:
    features=["distance_km","package_weight_kg","quantity","is_fragile","is_hazmat","cold_chain_required","delivery_cost_inr","transport_mode","delivery_priority","weather_condition_at_dest"]
    data=orders.dropna(subset=["actual_delivery_hours"])
    num,cat=features[:7],features[7:]
    X=_model_frame(data,num,cat)
    Xtr,Xte,ytr,yte=train_test_split(X,data["actual_delivery_hours"],test_size=.2,random_state=42)
    model=Pipeline([("prep",_preprocessor(num,cat)),("model",HistGradientBoostingRegressor(max_iter=180,random_state=42))])
    # HGB cannot consume sparse one-hot output; request dense output for this modest dataset.
    model.named_steps["prep"].set_params(cat__onehot__sparse_output=False)
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    return model,{"mae":mean_absolute_error(yte,pred),"mse":mean_squared_error(yte,pred),"rmse":mean_squared_error(yte,pred)**.5,"r2":r2_score(yte,pred)}


def train_maintenance_classifier(telemetry: pd.DataFrame) -> tuple[Pipeline, dict]:
    num=["flight_duration_min","cumulative_flight_hours","battery_cycles","battery_start_pct","battery_end_pct","battery_health_pct","motor_temp_c","vibration_rms","payload_kg","max_altitude_m","wind_speed_kmph","rotor_rpm_avg","route_deviation_m"]
    cat=["gps_signal_quality"]
    X=_model_frame(telemetry,num,cat)
    Xtr,Xte,ytr,yte=train_test_split(X,telemetry["maintenance_required"],test_size=.2,random_state=42,stratify=telemetry["maintenance_required"])
    model=Pipeline([("prep",_preprocessor(num,cat)),("model",RandomForestClassifier(n_estimators=180,class_weight="balanced",random_state=42,n_jobs=-1,min_samples_leaf=2))])
    model.fit(Xtr,ytr)
    return model,_classification_metrics(model,Xte,yte)


def train_sentiment(reviews: pd.DataFrame) -> tuple[Pipeline, dict]:
    data=reviews.dropna(subset=["review_text","sentiment_label"])
    Xtr,Xte,ytr,yte=train_test_split(data["review_text"].astype(str),data["sentiment_label"],test_size=.2,random_state=42,stratify=data["sentiment_label"])
    model=Pipeline([("tfidf",TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=2,max_features=25000)),("model",LogisticRegression(max_iter=1000,class_weight="balanced"))])
    model.fit(Xtr,ytr)
    return model,_classification_metrics(model,Xte,yte)


def train_all(tables: dict[str,pd.DataFrame], model_dir: str | Path=MODEL_DIR) -> dict:
    out=Path(model_dir); out.mkdir(parents=True,exist_ok=True)
    jobs={"mode_classifier":train_mode_classifier(tables["orders"]),"eta_regressor":train_eta_regressor(tables["orders"]),"maintenance_classifier":train_maintenance_classifier(tables["drone_telemetry"]),"sentiment_classifier":train_sentiment(tables["reviews"])}
    metrics={}
    for name,(model,score) in jobs.items():
        joblib.dump(model,out/f"{name}.joblib"); metrics[name]=score
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2,default=float))
    return metrics


if __name__ == "__main__":
    from .cleaning import clean_all
    from .config import RAW_DIR

    print("Loading and cleaning SmartLogix datasets...")
    cleaned_tables = clean_all(RAW_DIR)
    print("Training classification and regression models...")
    results = train_all(cleaned_tables)

    print("\nModel training completed successfully.")
    print(f"Saved models in: {MODEL_DIR}")
    print("\nMain evaluation results")
    print("-" * 52)
    for model_name, scores in results.items():
        print(f"\n{model_name.replace('_', ' ').title()}")
        for metric in ("accuracy", "f1_weighted", "roc_auc_ovr_weighted", "r2", "mae", "rmse"):
            if metric in scores:
                value = float(scores[metric])
                if metric in {"accuracy", "f1_weighted", "roc_auc_ovr_weighted", "r2"}:
                    print(f"  {metric:<24}: {value:.2%}")
                else:
                    print(f"  {metric:<24}: {value:.3f}")

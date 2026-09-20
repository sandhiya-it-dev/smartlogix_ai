from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_DIR
from src.optimizer import drone_feasibility, nearest_neighbor_route
from src.services import agent_chat, recommend_products, summarize_reviews, track_order

app=FastAPI(title="SmartLogix AI API",version="1.0.0",description="Local, non-cloud SmartLogix platform API")


class PredictionRequest(BaseModel):
    features: dict[str,Any]


class RouteRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    stops: list[dict[str,Any]]
    return_to_origin: bool=True


class DroneCheck(BaseModel):
    distance_km: float=Field(ge=0)
    payload_kg: float=Field(ge=0)
    max_range_km: float=Field(gt=0)
    capacity_kg: float=Field(gt=0)
    battery_start_pct: float=Field(default=100,ge=0,le=100)
    reserve_pct: float=Field(default=20,ge=0,le=100)


class ChatRequest(BaseModel): message: str


def _predict(name: str, payload: PredictionRequest):
    path=MODEL_DIR/f"{name}.joblib"
    if not path.exists(): raise HTTPException(503,"Model is not trained. Run scripts/train_models.py")
    model=joblib.load(path); frame=pd.DataFrame([payload.features]); pred=model.predict(frame)[0]
    result={"prediction":pred.item() if hasattr(pred,"item") else pred}
    if hasattr(model,"predict_proba"):
        result["probabilities"]={str(k):round(float(v),4) for k,v in zip(model.classes_,model.predict_proba(frame)[0])}
    return result


@app.get("/health")
def health(): return {"status":"ok","service":"SmartLogix AI"}

@app.post("/predict/mode")
def predict_mode(body:PredictionRequest): return _predict("mode_classifier",body)

@app.post("/predict/eta")
def predict_eta(body:PredictionRequest): return _predict("eta_regressor",body)

@app.post("/predict/maintenance")
def predict_maintenance(body:PredictionRequest): return _predict("maintenance_classifier",body)

@app.get("/orders/{order_id}")
def order_tracking(order_id:str): return track_order(order_id)

@app.get("/products/recommend")
def products(category:str|None=None,max_price:float|None=None,limit:int=5): return recommend_products(category,max_price,min(limit,20))

@app.get("/reviews/{product_id}/summary")
def reviews(product_id:str): return summarize_reviews(product_id)

@app.post("/routes/optimize")
def route(body:RouteRequest): return nearest_neighbor_route((body.origin_lat,body.origin_lon),body.stops,body.return_to_origin)

@app.post("/drones/feasibility")
def drone(body:DroneCheck): return drone_feasibility(**body.model_dump())

@app.post("/chat")
def chat(body:ChatRequest): return agent_chat(body.message)


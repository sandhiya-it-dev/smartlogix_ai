from __future__ import annotations

import joblib
import pandas as pd

from .config import MODEL_DIR
from .database import query_df
from .rag import FAQRetriever


def track_order(order_id: str) -> dict:
    order=query_df("SELECT order_id, order_status, transport_mode, promised_eta_hours, actual_delivery_hours, destination_city FROM orders WHERE order_id=:id",{"id":order_id.upper()})
    if order.empty: return {"found":False,"message":"Order not found"}
    logs=query_df("SELECT event_type,event_timestamp,location_city,remarks FROM delivery_logs WHERE order_id=:id ORDER BY event_timestamp",{"id":order_id.upper()})
    return {"found":True,"order":order.iloc[0].to_dict(),"events":logs.fillna("").to_dict("records")}


def recommend_products(category: str|None=None,max_price: float|None=None,limit: int=5) -> list[dict]:
    where=["stock_qty > 0"]; params={"limit":limit}
    if category: where.append("LOWER(category)=LOWER(:category)"); params["category"]=category
    if max_price is not None: where.append("price_inr<=:max_price"); params["max_price"]=max_price
    sql=f"SELECT product_id,product_name,category,price_inr,avg_rating,stock_qty FROM products WHERE {' AND '.join(where)} ORDER BY COALESCE(avg_rating,0) DESC, price_inr ASC LIMIT :limit"
    return query_df(sql,params).fillna("").to_dict("records")


def summarize_reviews(product_id: str) -> dict:
    df=query_df("SELECT rating,review_text,sentiment_label FROM reviews WHERE product_id=:id",{"id":product_id.upper()})
    if df.empty:return {"found":False,"message":"No reviews found"}
    counts=df["sentiment_label"].value_counts().to_dict()
    return {"found":True,"review_count":len(df),"average_rating":round(df["rating"].mean(),2),"sentiment_counts":counts,"sample_comments":df["review_text"].dropna().head(3).tolist()}


def agent_chat(message: str) -> dict:
    import re
    order=re.search(r"ORD-\d+",message.upper()); product=re.search(r"PRD-\d+",message.upper())
    low=message.lower()
    if order:return {"agent":"order_tracking","result":track_order(order.group())}
    if product and any(x in low for x in ["review","sentiment","rating"]):return {"agent":"review_analysis","result":summarize_reviews(product.group())}
    if any(x in low for x in ["recommend","product","buy"]):return {"agent":"product_recommendation","result":recommend_products(limit=5)}
    return {"agent":"support_rag","result":FAQRetriever().answer(message)}


if __name__ == "__main__":
    import json

    print("SmartLogix service-layer test")
    try:
        orders = query_df("SELECT order_id FROM orders LIMIT 1")
        if orders.empty:
            print("The database contains no orders.")
        else:
            sample_order_id = str(orders.iloc[0]["order_id"])
            print(f"Testing order tracking with: {sample_order_id}")
            print(json.dumps(track_order(sample_order_id), indent=2, default=str))

        print("\nTesting the support agent:")
        print(json.dumps(agent_chat("Why is my delivery delayed?"), indent=2, default=str))
    except Exception as error:
        print(f"Service test could not access the database: {error}")
        print("Run this first: python -m src.database")

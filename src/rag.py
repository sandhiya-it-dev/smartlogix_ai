from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

FAQS = [
    ("How can I track my order?", "Enter the order ID in Order Tracking. The latest delivery event and status will be shown."),
    ("Why is my delivery delayed?", "Weather, traffic, vehicle availability, address issues, or hub exceptions can delay a shipment."),
    ("What is drone delivery?", "Drone delivery is used for eligible lightweight packages within payload, range, weather, and safety limits."),
    ("Can I change my delivery address?", "Address changes depend on shipment status. Contact support before the package reaches the destination hub."),
    ("How are products recommended?", "Recommendations use category, price, rating, stock, product attributes, and review sentiment."),
]


class FAQRetriever:
    def __init__(self, documents=FAQS):
        self.documents=documents
        # Character n-grams handle natural variations such as track/tracking.
        self.vectorizer=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5))
        self.matrix=self.vectorizer.fit_transform([q+" "+a for q,a in documents])

    def retrieve(self, query: str, k: int=3) -> list[dict]:
        vector=self.vectorizer.transform([query]); scores=(self.matrix@vector.T).toarray().ravel()
        idx=np.argsort(scores)[::-1][:k]
        return [{"question":self.documents[i][0],"answer":self.documents[i][1],"score":round(float(scores[i]),4)} for i in idx]

    def answer(self, query: str) -> dict:
        matches=self.retrieve(query,1); best=matches[0]
        if best["score"]<0.05: return {"answer":"I could not find a reliable answer in the current knowledge base.","sources":matches}
        return {"answer":best["answer"],"sources":matches}


if __name__ == "__main__":
    import json

    sample_question = "How can I track my delivery?"
    print("SmartLogix RAG retrieval test")
    print(f"Question: {sample_question}")
    response = FAQRetriever().answer(sample_question)
    print(json.dumps(response, indent=2))

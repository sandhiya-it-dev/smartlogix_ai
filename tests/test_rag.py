from src.rag import FAQRetriever

def test_tracking_retrieval():
    result=FAQRetriever().answer("How do I track my package?")
    assert "order ID" in result["answer"]
    assert result["sources"][0]["score"]>0


# Evaluation guide

Run `python scripts/run_pipeline.py --raw-dir data/raw`, then inspect `models/metrics.json`.

- Mode classification: accuracy, weighted F1, per-class precision/recall/F1, confusion matrix, multiclass ROC-AUC when calculable.
- ETA: MAE, MSE, RMSE, R-squared.
- Maintenance: accuracy, weighted F1, class-level recall, confusion matrix, ROC-AUC.
- Sentiment: accuracy, weighted F1, per-class results, confusion matrix, ROC-AUC.
- YOLO: use Ultralytics validation output for precision, recall, mAP@50 and mAP@50-95. Treat synthetic results only as pipeline validation.
- RAG: create a labelled question set and record top-1 retrieval precision and human context-relevance scores.
- Business: on-time percentage, cost per kilometre, fleet utilization proxy, drone utilization, and review rating/sentiment.

For a stronger final evaluation, use temporal holdout for ETA and mode models and replace synthetic drone images with independently labelled real images.


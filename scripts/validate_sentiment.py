from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODEL_DIR


def main():
    validation_path = (
        ROOT
        / "data"
        / "validation"
        / "sentiment_holdout.csv"
    )

    model_path = MODEL_DIR / "sentiment_classifier.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "Sentiment model was not found. Run:\n"
            "python -m src.modeling"
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            "External validation dataset was not found.\n"
            "Create: data/validation/sentiment_holdout.csv"
        )

    validation = pd.read_csv(validation_path)

    required_columns = {
        "review_text",
        "sentiment_label",
    }

    missing_columns = (
        required_columns - set(validation.columns)
    )

    if missing_columns:
        raise ValueError(
            "Validation file is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    validation = validation.dropna(
        subset=["review_text", "sentiment_label"]
    ).copy()

    X_test = validation["review_text"].astype(str)
    y_test = validation["sentiment_label"].astype(str)

    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    print("Independent sentiment validation")
    print("=" * 50)
    print(f"Reviews tested: {len(validation)}")
    print(
        f"Accuracy: "
        f"{accuracy_score(y_test, predictions):.2%}"
    )

    labels = sorted(y_test.unique())

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        pd.DataFrame(
            confusion_matrix(
                y_test,
                predictions,
                labels=labels,
            ),
            index=[
                f"Actual {label}"
                for label in labels
            ],
            columns=[
                f"Predicted {label}"
                for label in labels
            ],
        )
    )

    incorrect = validation[
        y_test.to_numpy() != predictions
    ].copy()

    incorrect["predicted_sentiment"] = predictions[
        y_test.to_numpy() != predictions
    ]

    print(
        f"\nIncorrect predictions: "
        f"{len(incorrect)}"
    )

    if not incorrect.empty:
        print(
            incorrect[
                [
                    "review_text",
                    "sentiment_label",
                    "predicted_sentiment",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
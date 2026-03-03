"""Evaluation helpers: metrics and confusion matrix for saved models."""
from __future__ import annotations
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

MODELS_DIR = "models"


def evaluate_model(model_path: str, test_csv: str) -> None:
    model = joblib.load(model_path)
    df = pd.read_csv(test_csv)
    X = df.drop(columns=["risk_zone"])
    y = df["risk_zone"]

    preds = model.predict(X)
    proba = model.predict_proba(X)

    print(classification_report(y, preds))

    cm = confusion_matrix(y, preds, labels=model.classes_)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    out = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.title("Confusion Matrix")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved confusion matrix to {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate saved model on holdout CSV")
    parser.add_argument("--model", default=os.path.join(MODELS_DIR, "best_model.joblib"))
    parser.add_argument("--test", default=os.path.join(MODELS_DIR, "test_holdout.csv"))
    args = parser.parse_args()
    evaluate_model(args.model, args.test)

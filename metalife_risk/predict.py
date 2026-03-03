"""Prediction utilities: load model, predict, and format risk + confidence."""
from __future__ import annotations
from typing import Dict
import joblib
import pandas as pd
import numpy as np


DISCLAIMER = "This tool does not provide medical advice. Do not change medication or treatment based on these results."


def load_model(path: str):
    return joblib.load(path)


def predict_df(model, df: pd.DataFrame) -> pd.DataFrame:
    # Drop label column if present (e.g., using holdout CSV)
    X = df.copy()
    if "risk_zone" in X.columns:
        X = X.drop(columns=["risk_zone"])

    # If the preprocessing step recorded feature names, ensure all exist and in correct order
    try:
        imputer = model.named_steps.get("preproc").named_steps.get("imputer")
        feat_names = getattr(imputer, "feature_names_in_", None)
        if feat_names is not None:
            # Add any missing columns with NaN (imputer will fill with median)
            for col in feat_names:
                if col not in X.columns:
                    X[col] = np.nan
            # Reorder to match training feature order exactly
            X = X[list(feat_names)]
    except Exception:
        pass

    probs = model.predict_proba(X)
    preds = model.classes_[probs.argmax(axis=1)]

    # Confidence: margin between top two probabilities
    top_idxs = probs.argsort(axis=1)[:, ::-1]
    top1 = probs[np.arange(len(probs)), top_idxs[:, 0]]
    # handle case where only one class present
    top2 = np.where(probs.shape[1] > 1, probs[np.arange(len(probs)), top_idxs[:, 1]], 0.0)
    margin = top1 - top2

    # Confidence label per rules
    conf_labels = []
    for t1, m in zip(top1, margin):
        if t1 >= 0.75 or m >= 0.25:
            conf_labels.append("HIGH")
        elif t1 >= 0.60 or m >= 0.15:
            conf_labels.append("MEDIUM")
        else:
            conf_labels.append("LOW")

    out = df.reset_index(drop=True).copy()
    out["predicted_risk"] = preds
    out["confidence"] = top1
    out["confidence_label"] = conf_labels
    out["margin"] = margin
    out["disclaimer"] = DISCLAIMER
    return out


def predict_single(model, x: Dict) -> Dict:
    df = pd.DataFrame([x])
    out = predict_df(model, df)
    row = out.iloc[0].to_dict()
    probs = model.predict_proba(df)[0]
    top_idx = probs.argmax()
    top1 = float(probs[top_idx])
    sorted_idxs = probs.argsort()[::-1]
    top2 = float(probs[sorted_idxs[1]]) if len(probs) > 1 else 0.0
    margin = top1 - top2

    # Confidence label per rules
    if top1 >= 0.75 or margin >= 0.25:
        conf_label = "HIGH"
    elif top1 >= 0.60 or margin >= 0.15:
        conf_label = "MEDIUM"
    else:
        conf_label = "LOW"

    return {
        "risk_zone": row["predicted_risk"],
        "probabilities": dict(zip(model.classes_, probs)),
        "confidence": float(top1),
        "confidence_label": conf_label,
        "margin": float(margin),
        "disclaimer": row["disclaimer"],
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Predict using saved MetaLife Risk model")
    parser.add_argument("--model", default="models/best_model.joblib")
    parser.add_argument("--input", help="Path to CSV of feature rows")
    parser.add_argument("--out", default="predictions.csv")
    args = parser.parse_args()

    model = load_model(args.model)
    df = pd.read_csv(args.input)
    preds = predict_df(model, df)
    preds.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")

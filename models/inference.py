from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from utils.config import (
    MODEL_ARTIFACT_PATH,
    CONFIDENCE_HIGH_MARGIN,
    CONFIDENCE_HIGH_PROB,
    CONFIDENCE_MEDIUM_MARGIN,
    CONFIDENCE_MEDIUM_PROB,
)


@lru_cache(maxsize=1)
def load_artifact(path: str = str(MODEL_ARTIFACT_PATH)) -> dict:
    return joblib.load(path)


def _confidence_label(top_prob: float, margin: float) -> str:
    if top_prob >= CONFIDENCE_HIGH_PROB or margin >= CONFIDENCE_HIGH_MARGIN:
        return "HIGH"
    if top_prob >= CONFIDENCE_MEDIUM_PROB or margin >= CONFIDENCE_MEDIUM_MARGIN:
        return "MEDIUM"
    return "LOW"


def predict_risk(feature_df: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    art = artifact or load_artifact()
    model = art["model"]
    label_encoder = art["label_encoder"]
    feature_columns = art["feature_columns"]

    X = feature_df.copy()
    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_columns]

    pred_enc = model.predict(X)
    prob = model.predict_proba(X)
    pred_label = label_encoder.inverse_transform(pred_enc)

    top_idx = np.argmax(prob, axis=1)
    top_prob = prob[np.arange(len(prob)), top_idx]
    sorted_prob = np.sort(prob, axis=1)
    margins = top_prob - sorted_prob[:, -2]

    out = feature_df.copy().reset_index(drop=True)
    out["predicted_risk"] = pred_label
    out["confidence"] = top_prob
    out["confidence_margin"] = margins
    out["confidence_label"] = [
        _confidence_label(float(p), float(m)) for p, m in zip(top_prob, margins)
    ]
    return out

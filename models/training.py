from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from features.timeseries import add_lag_features, add_next_day_target
from utils.config import (
    ARTIFACT_DIR,
    MODEL_ARTIFACT_PATH,
    METADATA_PATH,
    ROC_CURVE_PATH,
    LOW_RISK_MAX,
    MODERATE_RISK_MAX,
)

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    XGBClassifier = None
    _XGB_IMPORT_ERROR = exc


@dataclass
class TrainingOutput:
    best_model_name: str
    metrics: dict
    feature_columns: list[str]


def assign_risk_label(pct_time_above_140: float) -> str:
    if pct_time_above_140 <= LOW_RISK_MAX:
        return "Low"
    if pct_time_above_140 <= MODERATE_RISK_MAX:
        return "Moderate"
    return "High"


def build_training_frame(features_df: pd.DataFrame, use_time_series: bool = True, lag_days: int = 7) -> pd.DataFrame:
    df = features_df.copy().sort_values("date").reset_index(drop=True)
    if "risk_label" not in df.columns:
        df["risk_label"] = df["pct_time_above_140"].apply(assign_risk_label)

    if use_time_series:
        df = add_lag_features(df, lag_days=lag_days)
        df = add_next_day_target(df, target_col="risk_label")
        df = df.rename(columns={"next_day_risk_label": "target_label"})
    else:
        df["target_label"] = df["risk_label"]

    df = df.dropna(subset=["target_label"]).reset_index(drop=True)
    return df


def _split_data(df: pd.DataFrame, target_col: str = "target_label"):
    feature_cols = [c for c in df.columns if c not in {"date", "risk_label", "target_label"}]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def _build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), feature_cols)],
        remainder="drop",
    )


def _evaluate(model, X: pd.DataFrame, y: pd.Series, label_encoder: LabelEncoder) -> dict:
    y_true = label_encoder.transform(y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc_ovr": float(roc_auc),
    }


def _plot_roc_binary(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    if y_prob.shape[1] != 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH)
    plt.close()


def train_models(features_df: pd.DataFrame, use_time_series: bool = True, lag_days: int = 7) -> TrainingOutput:
    if XGBClassifier is None:
        raise ImportError(f"xgboost is required for training: {_XGB_IMPORT_ERROR}")

    train_df = build_training_frame(features_df, use_time_series=use_time_series, lag_days=lag_days)
    if len(train_df) < 10:
        raise ValueError("Need at least 10 daily rows for training.")
    if len(train_df) < 30:
        warnings.warn("Training with fewer than 30 rows may produce unstable metrics.")

    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = _split_data(train_df)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    preproc = _build_preprocessor(feature_cols)

    rf_model = Pipeline(
        steps=[
            ("preproc", preproc),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgb_model = Pipeline(
        steps=[
            ("preproc", preproc),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=42,
                ),
            ),
        ]
    )

    rf_model.fit(X_train, y_train_enc)
    xgb_model.fit(X_train, y_train_enc)

    rf_metrics_val = _evaluate(rf_model, X_val, y_val, label_encoder)
    xgb_metrics_val = _evaluate(xgb_model, X_val, y_val, label_encoder)

    best_name = "xgboost" if xgb_metrics_val["f1_macro"] >= rf_metrics_val["f1_macro"] else "random_forest"
    best_model = xgb_model if best_name == "xgboost" else rf_model

    test_metrics = _evaluate(best_model, X_test, y_test, label_encoder)
    y_prob_test = best_model.predict_proba(X_test)
    _plot_roc_binary(y_test_enc, y_prob_test)

    artifact = {
        "model": best_model,
        "label_encoder": label_encoder,
        "feature_columns": feature_cols,
    }
    joblib.dump(artifact, MODEL_ARTIFACT_PATH)

    metadata = {
        "best_model": best_name,
        "validation_metrics": {
            "random_forest": rf_metrics_val,
            "xgboost": xgb_metrics_val,
        },
        "test_metrics": test_metrics,
        "feature_columns": feature_cols,
        "use_time_series": use_time_series,
        "lag_days": lag_days,
        "rows_used": len(train_df),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrainingOutput(best_model_name=best_name, metrics=metadata, feature_columns=feature_cols)

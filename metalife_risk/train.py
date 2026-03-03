"""Train Logistic Regression (multinomial) and Random Forest models.

Saves trained models into `models/` directory and prints evaluation metrics.
"""
from __future__ import annotations
import os
from typing import Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from metalife_risk.preprocessing import build_preprocessing_pipeline, split_features_target
from metalife_risk.feature_engineering import prepare_features


MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def train_models(df: pd.DataFrame, random_state: int = 42) -> Dict[str, str]:
    X_raw, y = split_features_target(df)
    X = prepare_features(X_raw, use_wearables=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Logistic Regression pipeline (needs scaling)
    log_pipe = Pipeline([
        ("preproc", build_preprocessing_pipeline(scale=True)),
        (
            "clf",
            LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000, random_state=random_state),
        ),
    ])

    # Random Forest pipeline (scaling optional)
    rf_pipe = Pipeline([
        ("preproc", build_preprocessing_pipeline(scale=False)),
        (
            "clf",
            RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
        ),
    ])

    # Train
    log_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)

    # Evaluate on test set
    models = {"logistic": log_pipe, "random_forest": rf_pipe}
    reports = {}
    best_model_name = None
    best_f1 = -1.0

    for name, model in models.items():
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        f1_macro = float(report["macro avg"]["f1-score"]) if "macro avg" in report else 0.0
        reports[name] = report

        # Save individual model
        filepath = os.path.join(MODELS_DIR, f"{name}_model.joblib")
        joblib.dump(model, filepath)

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_name = name

    # Save best model alias
    if best_model_name:
        best_path = os.path.join(MODELS_DIR, f"{best_model_name}_model.joblib")
        best_alias = os.path.join(MODELS_DIR, "best_model.joblib")
        joblib.dump(joblib.load(best_path), best_alias)

    # Print concise summary
    print("Model training complete. Selected best model:", best_model_name)
    for name, rpt in reports.items():
        print(f"\n=== {name} ===")
        print(classification_report(y_test, models[name].predict(X_test)))

    # Save test split for reproducible evaluation
    test_df = X_test.copy()
    test_df["risk_zone"] = y_test.values
    test_df.to_csv(os.path.join(MODELS_DIR, "test_holdout.csv"), index=False)

    return {"best_model": best_model_name, "best_f1": best_f1}


if __name__ == "__main__":
    # expects path data/simulated_metaflife_risk.csv or similar
    import argparse

    parser = argparse.ArgumentParser(description="Train MetaLife Risk models")
    parser.add_argument("--data", default="data/simulated_metaflife_risk.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    train_models(df)

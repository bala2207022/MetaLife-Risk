"""Train models from real data in data/ folder.

This script:
1. Loads all Dexcom Clarity CSVs and WHOOP exports from data/
2. Converts raw data to daily aggregated features
3. Creates rule-based labels (Low/Moderate/High based on pct_time_above_140)
4. Trains Logistic Regression and Random Forest
5. Saves best model to models/best_model.joblib

Usage:
    python -m metalife_risk.train_from_data
"""
from __future__ import annotations
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from metalife_risk.parsers import parse_clarity_csv, cgm_to_daily_features, parse_whoop_export
from metalife_risk.preprocessing import build_preprocessing_pipeline


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def find_cgm_files(data_dir: Path) -> list:
    """Find all potential CGM CSV files in data directory."""
    patterns = ["**/*clarity*.csv", "**/*cgm*.csv", "**/*glucose*.csv", "**/*CGMacros*.csv"]
    files = []
    for pattern in patterns:
        files.extend(data_dir.glob(pattern))
    # Also check for any CSV that might contain glucose data
    for csv_file in data_dir.glob("**/*.csv"):
        if csv_file not in files:
            try:
                df = pd.read_csv(csv_file, nrows=5)
                cols_lower = [c.lower() for c in df.columns]
                if any("glucose" in c or "sgv" in c for c in cols_lower):
                    files.append(csv_file)
            except Exception:
                pass
    return list(set(files))


def find_whoop_files(data_dir: Path) -> list:
    """Find WHOOP export files (ZIP or CSV with sleep/recovery data)."""
    files = []
    files.extend(data_dir.glob("**/*whoop*.zip"))
    files.extend(data_dir.glob("**/*whoop*.csv"))
    files.extend(data_dir.glob("**/*sleep*.csv"))
    files.extend(data_dir.glob("**/*recovery*.csv"))
    return list(set(files))


def load_all_cgm_data(data_dir: Path) -> pd.DataFrame:
    """Load and combine all CGM files into daily features."""
    cgm_files = find_cgm_files(data_dir)
    print(f"Found {len(cgm_files)} potential CGM files")
    
    all_daily = []
    for f in cgm_files:
        try:
            print(f"  Parsing: {f.name}")
            cgm_rows = parse_clarity_csv(str(f))
            if cgm_rows.empty:
                continue
            daily = cgm_to_daily_features(cgm_rows)
            if not daily.empty:
                daily["source_file"] = f.name
                all_daily.append(daily)
                print(f"    -> {len(daily)} days extracted")
        except Exception as e:
            print(f"    -> Failed: {e}")
    
    if not all_daily:
        return pd.DataFrame()
    
    combined = pd.concat(all_daily, ignore_index=True)
    # Remove duplicate dates (keep first)
    combined = combined.drop_duplicates(subset=["date"], keep="first")
    return combined


def load_all_whoop_data(data_dir: Path) -> pd.DataFrame:
    """Load and combine WHOOP data."""
    whoop_files = find_whoop_files(data_dir)
    print(f"Found {len(whoop_files)} potential WHOOP files")
    
    all_whoop = []
    for f in whoop_files:
        try:
            print(f"  Parsing: {f.name}")
            whoop = parse_whoop_export(str(f))
            if not whoop.empty:
                all_whoop.append(whoop)
                print(f"    -> {len(whoop)} days extracted")
        except Exception as e:
            print(f"    -> Failed: {e}")
    
    if not all_whoop:
        return pd.DataFrame()
    
    combined = pd.concat(all_whoop, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="first")
    return combined


def create_labels(df: pd.DataFrame) -> pd.Series:
    """Create rule-based labels based on pct_time_above_140.
    
    - Low: pct_time_above_140 < 10%
    - Moderate: 10% - 25%
    - High: > 25%
    """
    labels = pd.cut(
        df["pct_time_above_140"],
        bins=[-np.inf, 10, 25, np.inf],
        labels=["Low", "Moderate", "High"]
    )
    return labels


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """Train Logistic Regression and Random Forest, select best model."""
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Logistic Regression pipeline
    log_pipe = Pipeline([
        ("preproc", build_preprocessing_pipeline(scale=True)),
        ("clf", LogisticRegression(
            multi_class="multinomial", 
            solver="lbfgs", 
            max_iter=1000, 
            random_state=random_state
        )),
    ])
    
    # Random Forest pipeline
    rf_pipe = Pipeline([
        ("preproc", build_preprocessing_pipeline(scale=False)),
        ("clf", RandomForestClassifier(
            n_estimators=200, 
            random_state=random_state, 
            n_jobs=-1
        )),
    ])
    
    models = {
        "logistic_regression": log_pipe,
        "random_forest": rf_pipe
    }
    
    results = {}
    best_model_name = None
    best_f1 = -1.0
    
    for name, model in models.items():
        print(f"\n--- {name.upper()} ---")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
        print(f"CV F1 (macro): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Fit and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"Test F1 (macro): {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_path = MODELS_DIR / f"{name}_model.joblib"
        joblib.dump(model, model_path)
        print(f"Saved to: {model_path}")
        
        results[name] = {"model": model, "f1": f1, "y_pred": y_pred}
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
    
    # Save best model
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name} (F1={best_f1:.4f})")
    print(f"{'='*60}")
    
    best_model = results[best_model_name]["model"]
    best_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_path)
    print(f"Saved best model to: {best_path}")
    
    # Save confusion matrix
    y_pred_best = results[best_model_name]["y_pred"]
    cm = confusion_matrix(y_test, y_pred_best, labels=best_model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_model_name}")
    cm_path = MODELS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    print(f"Saved confusion matrix to: {cm_path}")
    
    return best_model, results


def main():
    print("="*60)
    print("MetaLife Risk - Training from data/ folder")
    print("="*60)
    
    # Load CGM data
    print("\n[1/4] Loading CGM data...")
    cgm_daily = load_all_cgm_data(DATA_DIR)
    
    if cgm_daily.empty:
        print("ERROR: No CGM data found. Please add Clarity CSV files to data/ folder.")
        return
    
    print(f"Total CGM days: {len(cgm_daily)}")
    
    # Load WHOOP data (optional)
    print("\n[2/4] Loading WHOOP data...")
    whoop_daily = load_all_whoop_data(DATA_DIR)
    
    # Merge datasets
    print("\n[3/4] Merging datasets...")
    merged = cgm_daily.copy()
    if not whoop_daily.empty:
        merged = merged.merge(whoop_daily, on="date", how="left")
        print(f"Merged with WHOOP data")
    
    # Drop non-feature columns
    feature_cols = [
        "glucose_mean", "glucose_std", "pct_time_above_140", "spike_freq", "gvi",
        "total_sleep_mins", "deep_sleep_pct", "hrv", "resting_hr", "daily_strain", "recovery"
    ]
    existing_features = [c for c in feature_cols if c in merged.columns]
    print(f"Features available: {existing_features}")
    
    X = merged[existing_features].copy()
    
    # Add derived feature
    if "spike_freq" in X.columns and "glucose_mean" in X.columns:
        X["spike_rate_norm"] = X["spike_freq"] / (1 + X["glucose_mean"]) * 100
    
    # Create labels
    y = create_labels(merged)
    print(f"Label distribution:\n{y.value_counts()}")
    
    # Remove rows with NaN labels
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(X) < 10:
        print(f"ERROR: Not enough data ({len(X)} samples). Need at least 10 days.")
        return
    
    # Save training data
    train_df = X.copy()
    train_df["risk_zone"] = y.values
    train_df["date"] = merged.loc[valid_idx, "date"].values
    train_df.to_csv(MODELS_DIR / "training_data.csv", index=False)
    print(f"Saved training data to: {MODELS_DIR / 'training_data.csv'}")
    
    # Train models
    print("\n[4/4] Training models...")
    best_model, results = train_and_evaluate(X, y)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

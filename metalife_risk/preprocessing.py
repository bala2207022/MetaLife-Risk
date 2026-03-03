"""Preprocessing utilities: loading, imputation, and scaling pipelines."""
from __future__ import annotations
from typing import Iterable, Tuple
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_preprocessing_pipeline(scale: bool = True) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def split_features_target(df: pd.DataFrame, target_col: str = "risk_zone") -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

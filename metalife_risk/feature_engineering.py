"""Feature engineering helpers for MetaLife Risk.

The module assumes input rows are already aggregated daily feature rows.
It provides selection of primary features and optional wearable features.
"""
from __future__ import annotations
from typing import List
import pandas as pd


PRIMARY_FEATURES: List[str] = [
    "glucose_mean",
    "glucose_std",
    "pct_time_above_140",
    "spike_freq",
    "gvi",
]

WEARABLE_FEATURES: List[str] = [
    "total_sleep_mins",
    "deep_sleep_pct",
    "hrv",
    "resting_hr",
    "daily_strain",
    "recovery",
]


def prepare_features(df: pd.DataFrame, use_wearables: bool = True) -> pd.DataFrame:
    """Select and prepare feature columns for modeling.

    Accepts either already-aggregated rows or a merged dataframe produced from parsers.
    Ensures column order matches the PRIMARY_FEATURES first, followed by wearables.
    """
    cols = list(PRIMARY_FEATURES)
    if use_wearables:
        cols += [c for c in WEARABLE_FEATURES if c in df.columns]

    # keep only existing columns in requested order
    existing = [c for c in cols if c in df.columns]
    if not existing:
        raise KeyError(f"Input dataframe does not contain required feature columns. Expected one of: {cols}")

    X = df[existing].copy()

    # Derived example: normalized spike rate
    if "spike_freq" in X.columns and "glucose_mean" in X.columns:
        X["spike_rate_norm"] = X["spike_freq"] / (1 + X["glucose_mean"]) * 100

    return X

from __future__ import annotations

import pandas as pd


def add_lag_features(df: pd.DataFrame, lag_days: int = 7) -> pd.DataFrame:
    if lag_days < 1:
        return df.copy()

    out = df.sort_values("date").reset_index(drop=True).copy()
    base_cols = [
        "glucose_mean",
        "glucose_std",
        "pct_time_above_140",
        "spike_freq",
        "gvi",
        "total_sleep_mins",
        "hrv",
        "resting_hr",
        "daily_strain",
        "recovery",
    ]
    existing_cols = [c for c in base_cols if c in out.columns]

    for col in existing_cols:
        for lag in range(1, lag_days + 1):
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)

    return out


def add_next_day_target(df: pd.DataFrame, target_col: str = "risk_label") -> pd.DataFrame:
    out = df.sort_values("date").reset_index(drop=True).copy()
    if target_col in out.columns:
        out["next_day_risk_label"] = out[target_col].shift(-1)
    return out

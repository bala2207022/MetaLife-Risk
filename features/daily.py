from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.schema import WEARABLE_COLUMN_MAP, find_column


def _spike_frequency(glucose_values: np.ndarray, threshold: float = 180.0) -> int:
    if len(glucose_values) < 2:
        return 0
    prev = glucose_values[:-1]
    curr = glucose_values[1:]
    return int(((prev <= threshold) & (curr > threshold)).sum())


def cgm_daily_features(cgm_df: pd.DataFrame) -> pd.DataFrame:
    if cgm_df.empty:
        return pd.DataFrame()

    df = cgm_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
    df = df.dropna(subset=["date", "glucose_mgdl"])

    rows = []
    for date_key, group in df.groupby("date"):
        values = group["glucose_mgdl"].astype(float).values
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        pct_above_140 = float((values > 140.0).mean() * 100.0)
        spike_freq = _spike_frequency(values, threshold=180.0)
        gvi = float((std / mean) * 100.0) if mean != 0 else 0.0
        rows.append(
            {
                "date": date_key,
                "glucose_mean": mean,
                "glucose_std": std,
                "pct_time_above_140": pct_above_140,
                "spike_freq": spike_freq,
                "gvi": gvi,
            }
        )

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def wearable_daily_features(wearable_df: pd.DataFrame) -> pd.DataFrame:
    if wearable_df is None or wearable_df.empty:
        return pd.DataFrame(columns=["date", "total_sleep_mins", "hrv", "resting_hr", "daily_strain", "recovery"])

    df = wearable_df.copy()
    mapped = {"date": "date"}
    for target, aliases in WEARABLE_COLUMN_MAP.items():
        col = find_column(df, aliases)
        if col:
            mapped[target] = col

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    if "sleep_mins" in mapped:
        out["total_sleep_mins"] = pd.to_numeric(df[mapped["sleep_mins"]], errors="coerce")
    else:
        out["total_sleep_mins"] = np.nan

    out["hrv"] = pd.to_numeric(df[mapped["hrv"]], errors="coerce") if "hrv" in mapped else np.nan
    out["resting_hr"] = pd.to_numeric(df[mapped["resting_hr"]], errors="coerce") if "resting_hr" in mapped else np.nan
    out["daily_strain"] = pd.to_numeric(df[mapped["strain"]], errors="coerce") if "strain" in mapped else np.nan
    out["recovery"] = pd.to_numeric(df[mapped["recovery"]], errors="coerce") if "recovery" in mapped else np.nan

    grouped = out.groupby("date", as_index=False).mean(numeric_only=True)
    grouped = grouped.sort_values("date").reset_index(drop=True)
    return grouped


def merge_daily_features(cgm_daily: pd.DataFrame, wearable_daily: pd.DataFrame | None) -> pd.DataFrame:
    if wearable_daily is None or wearable_daily.empty:
        return cgm_daily.copy()
    merged = cgm_daily.merge(wearable_daily, on="date", how="left")
    return merged.sort_values("date").reset_index(drop=True)

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.schema import find_column, CGM_TIMESTAMP_CANDIDATES, CGM_GLUCOSE_CANDIDATES, DATE_CANDIDATES


def _to_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any() and (parsed.dropna().dt.year < 2000).mean() > 0.5:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            unit = "ms" if float(numeric.max()) > 1e12 else "s"
            parsed = pd.to_datetime(numeric, unit=unit, errors="coerce")
    return parsed


def _remove_iqr_outliers(df: pd.DataFrame, col: str, k: float = 3.0) -> pd.DataFrame:
    if col not in df.columns or df[col].dropna().empty:
        return df
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return df
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return df[(df[col].isna()) | ((df[col] >= lo) & (df[col] <= hi))]


def clean_cgm(df: pd.DataFrame) -> pd.DataFrame:
    ts_col = find_column(df, CGM_TIMESTAMP_CANDIDATES) or df.columns[0]
    gl_col = find_column(df, CGM_GLUCOSE_CANDIDATES)
    if gl_col is None:
        raise ValueError("Could not find glucose column in CGM file.")

    out = pd.DataFrame({
        "timestamp": _to_datetime(df[ts_col]),
        "glucose_mgdl": pd.to_numeric(df[gl_col], errors="coerce"),
    })
    out = out.dropna(subset=["timestamp", "glucose_mgdl"])
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    out = _remove_iqr_outliers(out, "glucose_mgdl", k=3.0)

    # Fill small gaps while preserving sequence; larger gaps remain missing and are dropped.
    out["glucose_mgdl"] = out["glucose_mgdl"].ffill(limit=3)
    out = out.dropna(subset=["glucose_mgdl"])
    return out.reset_index(drop=True)


def clean_wearable(df: pd.DataFrame) -> pd.DataFrame:
    date_col = find_column(df, DATE_CANDIDATES) or df.columns[0]
    out = df.copy()
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.date
    out = out.dropna(subset=["date"])
    out = out.drop_duplicates().sort_values("date")

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        out[col] = out[col].ffill(limit=2)
        out = _remove_iqr_outliers(out, col, k=3.5)

    return out.reset_index(drop=True)

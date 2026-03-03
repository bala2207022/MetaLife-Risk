"""Parsers for raw CGM (Dexcom Clarity) and WHOOP exports.

Functions:
- parse_clarity_csv(file) -> DataFrame(datetime, glucose_mgdl, date)
- cgm_to_daily_features(df) -> DataFrame(date, glucose_mean, glucose_std, pct_time_above_140, spike_freq, gvi)
- parse_whoop_export(file_or_zip) -> DataFrame(date, total_sleep_mins, deep_sleep_pct, hrv, resting_hr, daily_strain, recovery)

The parsers are forgiving about column name variants.
"""
from __future__ import annotations
import io
import zipfile
from typing import Union
import pandas as pd
import numpy as np


def _find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # try lowercased match
    lc = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lc:
            return lc[c.lower()]
    return None


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """Parse timestamps handling both string formats and epoch seconds/milliseconds."""
    # First try direct datetime parsing
    parsed = pd.to_datetime(series, errors="coerce")
    
    # Check if many values are in 1970 (likely epoch misparse)
    if parsed.notna().any():
        years = parsed.dropna().dt.year
        if len(years) > 0 and (years < 2000).mean() > 0.5:
            # Likely epoch - try as seconds first
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                # If values are very large (>1e12), treat as milliseconds
                if numeric.max() > 1e12:
                    parsed = pd.to_datetime(numeric, unit="ms", errors="coerce")
                else:
                    parsed = pd.to_datetime(numeric, unit="s", errors="coerce")
    
    # If still many NaT or 1970, try string parsing with multiple formats
    if parsed.isna().mean() > 0.5 or (parsed.dt.year < 2000).mean() > 0.5:
        for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M"]:
            try:
                alt = pd.to_datetime(series.astype(str), format=fmt, errors="coerce")
                if alt.notna().mean() > parsed.notna().mean():
                    parsed = alt
            except Exception:
                pass
    
    return parsed


def parse_clarity_csv(uploaded_file: Union[str, io.BytesIO]) -> pd.DataFrame:
    """Read a Dexcom Clarity-like CSV and return standardized CGM rows.

    Expected output columns: `datetime` (pd.Timestamp), `glucose_mgdl` (float), `date` (date)
    """
    # Accept file paths or uploaded bytes-like
    if hasattr(uploaded_file, "read"):
        raw = uploaded_file.read()
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception:
            df = pd.read_csv(io.StringIO(raw.decode("utf-8", errors="ignore")))
    else:
        df = pd.read_csv(uploaded_file)

    # Possible timestamp columns
    ts_col = _find_col(df, ["Timestamp", "timestamp", "Time", "time", "Date", "date", "Timestamp (YYYY-MM-DDThh:mm:ss)"]) or df.columns[0]

    # possible glucose columns
    glucose_col = _find_col(
        df,
        [
            "Glucose (mg/dL)",
            "GlucoseValue",
            "Glucose Value (mg/dL)",
            "glucose",
            "sgv",
            "Sensor Glucose (mg/dL)",
            "Value",
            "Glucose Value",
        ],
    )
    if glucose_col is None:
        # try numeric columns besides timestamp
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        glucose_col = numeric_cols[0] if numeric_cols else None

    # Parse datetime with epoch handling
    df["datetime"] = _parse_timestamp(df[ts_col])

    if glucose_col is None:
        raise ValueError("Could not find glucose column in uploaded CGM file.")

    # Standardize glucose column
    df["glucose_mgdl"] = pd.to_numeric(df[glucose_col], errors="coerce")
    df = df.dropna(subset=["datetime", "glucose_mgdl"]).sort_values("datetime")
    df["date"] = df["datetime"].dt.date
    return df[["datetime", "glucose_mgdl", "date"]]


def cgm_to_daily_features(df: pd.DataFrame, above_thresh: float = 140.0) -> pd.DataFrame:
    """Aggregate CGM rows to daily features.

    Returns `date` + features where `pct_time_above_140` is 0-100 range.
    spike_freq: count of transitions from <=thresh to >thresh within the day.
    gvi: glucose_std / glucose_mean * 100
    """
    if df.empty:
        return pd.DataFrame()

    def daily_stats(g):
        gm = g["glucose_mgdl"].mean()
        gs = g["glucose_mgdl"].std()
        pct = (g["glucose_mgdl"] > above_thresh).mean() * 100.0
        vals = g["glucose_mgdl"].values
        # spike: rising crossings from <=thresh to >thresh
        spikes = 0
        if len(vals) > 1:
            prev = vals[:-1]
            curr = vals[1:]
            spikes = int(((prev <= above_thresh) & (curr > above_thresh)).sum())
        gvi = (gs / gm * 100.0) if gm and not np.isnan(gs) and gm != 0 else 0.0
        return pd.Series({
            "glucose_mean": float(gm),
            "glucose_std": float(gs) if not np.isnan(gs) else 0.0,
            "pct_time_above_140": float(pct),
            "spike_freq": int(spikes),
            "gvi": float(gvi),
        })

    out = df.groupby("date").apply(daily_stats).reset_index()
    return out


def parse_whoop_export(uploaded_file: Union[str, io.BytesIO]) -> pd.DataFrame:
    """Parse WHOOP export or ZIP with CSVs into daily wearable summary.

    Returns date + wearable features (if found). This function heuristically maps columns.
    """
    # If ZIP, search inside for CSVs
    if hasattr(uploaded_file, "read"):
        raw = uploaded_file.read()
        try:
            z = zipfile.ZipFile(io.BytesIO(raw))
            # try to find a file with 'sleep' or 'recovery' in its name first
            csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
            dfs = []
            for name in csv_names:
                try:
                    dfs.append(pd.read_csv(z.open(name)))
                except Exception:
                    pass
            if not dfs:
                return pd.DataFrame()
            df = pd.concat(dfs, ignore_index=True, sort=False)
        except zipfile.BadZipFile:
            # not a zip; treat as CSV
            try:
                df = pd.read_csv(io.BytesIO(raw))
            except Exception:
                df = pd.read_csv(io.StringIO(raw.decode('utf-8', errors='ignore')))
    else:
        # path
        if str(uploaded_file).lower().endswith('.zip'):
            with zipfile.ZipFile(uploaded_file) as z:
                csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
                dfs = [pd.read_csv(z.open(n)) for n in csv_names]
                df = pd.concat(dfs, ignore_index=True, sort=False)
        else:
            df = pd.read_csv(uploaded_file)

    # Heuristics: find a date column
    date_col = _find_col(df, ["day", "date", "Date", "Start Date", "start_date"]) or df.columns[0]
    try:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    except Exception:
        df["date"] = df[date_col]

    # Map possible columns
    mapping = {
        "total_sleep_mins": ["Total Sleep (min)", "total_sleep", "sleep_minutes", "total_sleep_mins"],
        "deep_sleep_pct": ["Deep Sleep %", "deep_sleep_pct", "deep_sleep_percent"],
        "hrv": ["HRV", "hrv"],
        "resting_hr": ["Resting HR", "resting_heart_rate", "resting_hr"],
        "daily_strain": ["Strain", "daily_strain"],
        "recovery": ["Recovery", "Recovery Score", "recovery"],
    }

    out = df.groupby("date").agg({})
    # We'll build per-date rows
    rows = []
    for d, g in df.groupby("date"):
        row = {"date": d}
        for key, candidates in mapping.items():
            col = _find_col(g, candidates)
            if col is not None:
                vals = pd.to_numeric(g[col], errors="coerce")
                if key == "deep_sleep_pct":
                    # assume percent values already
                    row[key] = float(vals.mean())
                else:
                    row[key] = float(vals.mean()) if not vals.dropna().empty else None
            else:
                row[key] = None
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    out_df = pd.DataFrame(rows)
    return out_df

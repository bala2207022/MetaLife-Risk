from __future__ import annotations

import pandas as pd


def apply_what_if(
    feature_df: pd.DataFrame,
    spike_reduction_pct: float = 0.0,
    sleep_increase_mins: float = 0.0,
    hrv_increase_pct: float = 0.0,
) -> pd.DataFrame:
    out = feature_df.copy()

    if "spike_freq" in out.columns and spike_reduction_pct != 0:
        factor = max(0.0, 1.0 - (spike_reduction_pct / 100.0))
        out["spike_freq"] = out["spike_freq"] * factor

    if "pct_time_above_140" in out.columns and spike_reduction_pct != 0:
        factor = max(0.0, 1.0 - (spike_reduction_pct / 120.0))
        out["pct_time_above_140"] = out["pct_time_above_140"] * factor

    if "total_sleep_mins" in out.columns and sleep_increase_mins != 0:
        out["total_sleep_mins"] = out["total_sleep_mins"] + sleep_increase_mins

    if "hrv" in out.columns and hrv_increase_pct != 0:
        out["hrv"] = out["hrv"] * (1.0 + hrv_increase_pct / 100.0)

    return out

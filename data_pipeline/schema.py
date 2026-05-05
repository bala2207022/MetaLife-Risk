from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


CGM_TIMESTAMP_CANDIDATES = [
    "Timestamp",
    "timestamp",
    "Time",
    "time",
    "Date",
    "date",
    "Timestamp (YYYY-MM-DDThh:mm:ss)",
]

CGM_GLUCOSE_CANDIDATES = [
    "Glucose (mg/dL)",
    "Glucose Value (mg/dL)",
    "GlucoseValue",
    "glucose",
    "sgv",
    "Sensor Glucose (mg/dL)",
    "Value",
    "Glucose Value",
]

WEARABLE_COLUMN_MAP = {
    "sleep_mins": ["total_sleep_mins", "Asleep duration (min)", "Total Sleep (min)"],
    "hrv": ["hrv", "HRV", "Heart rate variability (ms)"],
    "resting_hr": ["resting_hr", "Resting HR", "Resting heart rate (bpm)"],
    "strain": ["daily_strain", "Strain", "Day Strain", "Activity Strain"],
    "recovery": ["recovery", "Recovery", "Recovery Score", "Recovery score %"],
}

DATE_CANDIDATES = [
    "date",
    "Date",
    "day",
    "Day",
    "start_date",
    "Start Date",
    "Cycle start time",
    "Sleep onset",
]


@dataclass
class ValidationResult:
    is_valid: bool
    message: str


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def validate_non_empty(df: pd.DataFrame, dataset_name: str) -> ValidationResult:
    if df is None or df.empty:
        return ValidationResult(False, f"{dataset_name} is empty after parsing.")
    return ValidationResult(True, "ok")

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from data_pipeline.cleaning import clean_cgm, clean_wearable
from data_pipeline.schema import validate_non_empty


def _read_csv_from_bytes(raw: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        return pd.read_csv(io.StringIO(raw.decode("utf-8", errors="ignore")))


def read_any_table(file_obj_or_path: str | Path | BinaryIO) -> pd.DataFrame:
    if hasattr(file_obj_or_path, "read"):
        raw = file_obj_or_path.read()
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                frames = []
                for name in zf.namelist():
                    if name.lower().endswith(".csv"):
                        try:
                            frames.append(pd.read_csv(zf.open(name)))
                        except Exception:
                            continue
                if frames:
                    return pd.concat(frames, ignore_index=True, sort=False)
        except zipfile.BadZipFile:
            pass
        return _read_csv_from_bytes(raw)

    path = Path(file_obj_or_path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            frames = [pd.read_csv(zf.open(name)) for name in zf.namelist() if name.lower().endswith(".csv")]
        if not frames:
            raise ValueError("ZIP did not contain any CSV files.")
        return pd.concat(frames, ignore_index=True, sort=False)

    return pd.read_csv(path)


def load_cgm(file_obj_or_path: str | Path | BinaryIO) -> pd.DataFrame:
    raw = read_any_table(file_obj_or_path)
    cleaned = clean_cgm(raw)
    result = validate_non_empty(cleaned, "CGM data")
    if not result.is_valid:
        raise ValueError(result.message)
    return cleaned


def load_wearable(file_obj_or_path: str | Path | BinaryIO) -> pd.DataFrame:
    raw = read_any_table(file_obj_or_path)
    cleaned = clean_wearable(raw)
    result = validate_non_empty(cleaned, "Wearable data")
    if not result.is_valid:
        raise ValueError(result.message)
    return cleaned

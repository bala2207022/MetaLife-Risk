from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from data_pipeline.loaders import load_cgm, load_wearable
from features.daily import cgm_daily_features, wearable_daily_features, merge_daily_features
from models.explainability import compute_shap_importance
from models.inference import load_artifact, predict_risk
from models.simulation import apply_what_if

app = FastAPI(title="MetaLife Risk API", version="2.0.0")


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]]


class SimulateRequest(BaseModel):
    rows: list[dict[str, Any]]
    spike_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    sleep_increase_mins: float = 0.0
    hrv_increase_pct: float = Field(default=0.0, ge=-100.0, le=200.0)


def _file_to_buffer(upload: UploadFile) -> io.BytesIO:
    content = upload.file.read()
    return io.BytesIO(content)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload-data")
def upload_data(cgm_file: UploadFile = File(...), wearable_file: UploadFile | None = File(default=None)) -> dict:
    try:
        cgm_df = load_cgm(_file_to_buffer(cgm_file))
        cgm_daily = cgm_daily_features(cgm_df)

        wearable_daily = None
        if wearable_file is not None:
            wearable_df = load_wearable(_file_to_buffer(wearable_file))
            wearable_daily = wearable_daily_features(wearable_df)

        merged = merge_daily_features(cgm_daily, wearable_daily)
        return {
            "rows": merged.to_dict(orient="records"),
            "summary": {
                "cgm_rows": int(len(cgm_df)),
                "days": int(len(merged)),
                "wearable_used": wearable_daily is not None and not wearable_daily.empty,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    if not payload.rows:
        raise HTTPException(status_code=400, detail="rows cannot be empty")

    try:
        df = pd.DataFrame(payload.rows)
        artifact = load_artifact()
        pred_df = predict_risk(df, artifact=artifact)
        shap_imp = compute_shap_importance(artifact, df).head(10)
        return {
            "predictions": pred_df.to_dict(orient="records"),
            "top_features": shap_imp.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/simulate")
def simulate(payload: SimulateRequest) -> dict:
    if not payload.rows:
        raise HTTPException(status_code=400, detail="rows cannot be empty")

    try:
        df = pd.DataFrame(payload.rows)
        simulated = apply_what_if(
            df,
            spike_reduction_pct=payload.spike_reduction_pct,
            sleep_increase_mins=payload.sleep_increase_mins,
            hrv_increase_pct=payload.hrv_increase_pct,
        )
        artifact = load_artifact()
        pred_df = predict_risk(simulated, artifact=artifact)
        return {"simulated_predictions": pred_df.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

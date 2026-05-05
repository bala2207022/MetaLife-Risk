from __future__ import annotations

import streamlit as st
import pandas as pd
import altair as alt

from data_pipeline.loaders import load_cgm, load_wearable
from features.daily import cgm_daily_features, wearable_daily_features, merge_daily_features
from models.explainability import compute_shap_importance
from models.inference import load_artifact, predict_risk
from models.simulation import apply_what_if

st.set_page_config(page_title="MetaLife Risk Pro", page_icon="📈", layout="wide")
st.title("MetaLife Risk Pro")
st.caption("AI-powered health analytics for CGM + wearable metabolic risk forecasting")

with st.sidebar:
    st.header("Data Upload")
    cgm_file = st.file_uploader("Dexcom Clarity CSV", type=["csv"])
    wearable_file = st.file_uploader("Wearable Data (optional)", type=["csv", "zip"])
    run = st.button("Run Pipeline", type="primary", use_container_width=True)

if not run:
    st.info("Upload CGM data and click Run Pipeline.")
    st.stop()

if cgm_file is None:
    st.error("CGM file is required.")
    st.stop()

try:
    cgm_raw = load_cgm(cgm_file)
    cgm_daily = cgm_daily_features(cgm_raw)

    wearable_daily = None
    if wearable_file is not None:
        wearable_raw = load_wearable(wearable_file)
        wearable_daily = wearable_daily_features(wearable_raw)

    features = merge_daily_features(cgm_daily, wearable_daily)

    artifact = load_artifact()
    pred = predict_risk(features, artifact=artifact)

except Exception as exc:
    st.error(f"Pipeline failed: {exc}")
    st.stop()

latest = pred.iloc[-1]
col1, col2, col3 = st.columns(3)
col1.metric("Latest Risk", str(latest["predicted_risk"]))
col2.metric("Confidence", str(latest["confidence_label"]))
col3.metric("Days", len(pred))

st.subheader("Data Summary")
st.dataframe(pred.head(30), use_container_width=True, hide_index=True)

st.subheader("Glucose Trend")
glucose_chart = alt.Chart(cgm_raw).mark_line().encode(
    x=alt.X("timestamp:T", title="Timestamp"),
    y=alt.Y("glucose_mgdl:Q", title="Glucose (mg/dL)"),
).properties(height=280)
st.altair_chart(glucose_chart, use_container_width=True)

st.subheader("Risk Over Time")
risk_color = alt.Scale(domain=["Low", "Moderate", "High"], range=["#2ca02c", "#ffbb00", "#d62728"])
risk_chart = alt.Chart(pred).mark_circle(size=100).encode(
    x=alt.X("date:T", title="Date"),
    y=alt.Y("confidence:Q", title="Confidence"),
    color=alt.Color("predicted_risk:N", scale=risk_color),
    tooltip=["date:T", "predicted_risk:N", "confidence:Q", "confidence_label:N"],
).properties(height=280)
st.altair_chart(risk_chart, use_container_width=True)

st.subheader("Feature Analysis")
try:
    imp = compute_shap_importance(artifact, features).head(12)
    bar = alt.Chart(imp).mark_bar().encode(
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        x=alt.X("importance:Q", title="Mean |SHAP|"),
    ).properties(height=320)
    st.altair_chart(bar, use_container_width=True)
except Exception as exc:
    st.warning(f"SHAP not available: {exc}")

st.subheader("What-If Simulation")
col1, col2, col3 = st.columns(3)
with col1:
    spike_reduction = st.slider("Reduce spikes (%)", 0, 60, 10)
with col2:
    sleep_delta = st.slider("Increase sleep (min)", 0, 120, 30)
with col3:
    hrv_delta = st.slider("Increase HRV (%)", 0, 40, 10)

sim_df = apply_what_if(
    features,
    spike_reduction_pct=float(spike_reduction),
    sleep_increase_mins=float(sleep_delta),
    hrv_increase_pct=float(hrv_delta),
)
sim_pred = predict_risk(sim_df, artifact=artifact)

latest_sim = sim_pred.iloc[-1]
st.info(
    f"Simulated next risk: {latest_sim['predicted_risk']} "
    f"(confidence {latest_sim['confidence_label']})"
)

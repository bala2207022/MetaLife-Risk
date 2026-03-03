"""MetaLife Risk — Streamlit App for Lifestyle-Based Metabolic Risk Prediction

Upload raw Dexcom Clarity CGM CSV and optional WHOOP export.
The app parses, aggregates daily features, predicts risk, and shows graphs.

Run with:
    streamlit run metalife_risk/app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

from metalife_risk.feature_engineering import prepare_features
from metalife_risk.predict import predict_df, DISCLAIMER
from metalife_risk.parsers import parse_clarity_csv, cgm_to_daily_features, parse_whoop_export


# Page config
st.set_page_config(
    page_title="MetaLife Risk",
    page_icon="🩺",
    layout="wide"
)

# Header
st.title("🩺 MetaLife Risk")
st.markdown("### Lifestyle-Based Metabolic Risk Forecasting (Awareness Only)")
st.warning(f"**Disclaimer:** {DISCLAIMER}")

st.markdown("---")

# Sidebar for uploads
with st.sidebar:
    st.header("📤 Upload Data")
    
    uploaded_cgm = st.file_uploader(
        "Dexcom Clarity CSV (required)",
        type=["csv"],
        help="Raw CGM export from Dexcom Clarity"
    )
    
    uploaded_whoop = st.file_uploader(
        "WHOOP Export (optional)",
        type=["csv", "zip", "xls", "xlsx"],
        help="Sleep, recovery, and strain data"
    )
    
    st.markdown("---")
    st.header("⚙️ Settings")
    model_path = st.text_input("Model path", value="models/best_model.joblib")
    
    run_btn = st.button("🚀 Run Predictions", type="primary", use_container_width=True)

# Main content
if not run_btn:
    st.info("👈 Upload your CGM data and click **Run Predictions** to get started.")
    
    with st.expander("📋 How to use this app"):
        st.markdown("""
        1. **Upload CGM Data**: Export your glucose data from Dexcom Clarity as CSV
        2. **Upload WHOOP (optional)**: Add sleep/recovery data for better insights
        3. **Run Predictions**: Click the button to analyze your data
        4. **Review Results**: See your risk zone, trends, and key drivers
        """)
    
    with st.expander("📊 What features are analyzed"):
        st.markdown("""
        **CGM Features (per day):**
        - Glucose mean & standard deviation
        - % time above 140 mg/dL
        - Glucose spike frequency
        - Glucose variability index (GVI)
        
        **WHOOP Features (if available):**
        - Total sleep & deep sleep %
        - HRV (Heart Rate Variability)
        - Resting heart rate
        - Daily strain & recovery score
        """)
    st.stop()

# Process uploads
if uploaded_cgm is None:
    st.error("❌ Please upload a Dexcom Clarity CSV file.")
    st.stop()

# Parse CGM
with st.spinner("Parsing CGM data..."):
    try:
        uploaded_cgm.seek(0)  # Reset file pointer
        cgm_rows = parse_clarity_csv(uploaded_cgm)
    except Exception as e:
        st.error(f"❌ Failed to parse CGM file: {e}")
        st.stop()

if cgm_rows.empty:
    st.error("❌ No valid glucose readings found in the uploaded file.")
    st.stop()

# Check for 1970 date bug
min_year = cgm_rows["datetime"].dt.year.min()
if min_year < 2000:
    st.warning(f"⚠️ Detected timestamps from year {min_year}. Check your data format.")

cgm_daily = cgm_to_daily_features(cgm_rows)
if cgm_daily.empty:
    st.error("❌ Could not compute daily features from CGM data.")
    st.stop()

st.success(f"✅ Parsed {len(cgm_rows):,} glucose readings across {len(cgm_daily)} days")

# Parse WHOOP
whoop_daily = None
if uploaded_whoop is not None:
    with st.spinner("Parsing WHOOP data..."):
        try:
            uploaded_whoop.seek(0)
            whoop_daily = parse_whoop_export(uploaded_whoop)
            if whoop_daily is not None and not whoop_daily.empty:
                st.success(f"✅ Parsed WHOOP data for {len(whoop_daily)} days")
            else:
                st.warning("⚠️ No WHOOP features could be extracted")
                whoop_daily = None
        except Exception as e:
            st.warning(f"⚠️ Failed to parse WHOOP: {e}")

# Merge datasets
merged = cgm_daily.copy()
if whoop_daily is not None and not whoop_daily.empty:
    merged = merged.merge(whoop_daily, on="date", how="left")

# Load model
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model from {model_path}: {e}")
    st.stop()

# Prepare features and predict
try:
    X = prepare_features(merged, use_wearables=True)
except Exception as e:
    st.error(f"❌ Feature preparation failed: {e}")
    st.stop()

preds = predict_df(model, X)

# Build output dataframe
out_df = merged.copy().reset_index(drop=True)
pred_cols = ["predicted_risk", "confidence", "confidence_label", "margin"]
preds = preds.reset_index(drop=True)
for col in pred_cols:
    if col in preds.columns:
        out_df[col] = preds[col]

# Sort by date
out_df["date"] = pd.to_datetime(out_df["date"])
out_df = out_df.sort_values("date").reset_index(drop=True)

# ============================================================
# RESULTS
# ============================================================

st.markdown("---")
st.header("📊 Results")

# Overall Risk Card
latest = out_df.iloc[-1]
overall_risk = latest["predicted_risk"]
overall_conf = latest.get("confidence_label", "MEDIUM")

risk_colors = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
risk_emoji = risk_colors.get(overall_risk, "⚪")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Overall Risk Zone",
        f"{risk_emoji} {overall_risk}",
        help="Based on most recent day's prediction"
    )
with col2:
    st.metric(
        "Confidence",
        overall_conf,
        help="HIGH: prob≥75% or margin≥25%, MEDIUM: prob≥60% or margin≥15%, LOW: otherwise"
    )
with col3:
    st.metric(
        "Days Analyzed",
        len(out_df),
        help="Total number of days with CGM data"
    )

# ============================================================
# 120-DAY LIFESTYLE PROJECTION
# ============================================================

st.markdown("---")
st.header("🔮 120-Day Lifestyle Projection")

st.markdown("*If current glucose and lifestyle patterns continue for the next ~120 days, the projected metabolic stress continuation is:*")

# Projection mapping based on current risk zone
projection_map = {
    "Low": ("🟢", "Low Metabolic Stress Continuation"),
    "Moderate": ("🟡", "Moderate Metabolic Stress Continuation"),
    "High": ("🔴", "High Metabolic Stress Continuation")
}

proj_emoji, proj_label = projection_map.get(overall_risk, ("⚪", "Unknown"))

# Display projection card
st.markdown(f"""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            padding: 1.5rem; 
            border-radius: 12px; 
            border-left: 4px solid {'#2ca02c' if overall_risk == 'Low' else '#ffbb00' if overall_risk == 'Moderate' else '#d62728'};
            margin: 1rem 0;">
    <h3 style="margin: 0 0 0.5rem 0; color: #ffffff;">Projected Status</h3>
    <p style="font-size: 1.5rem; margin: 0; font-weight: bold; color: {'#2ca02c' if overall_risk == 'Low' else '#ffbb00' if overall_risk == 'Moderate' else '#d62728'};">
        {proj_emoji} {proj_label}
    </p>
    <p style="font-size: 0.85rem; color: #aaaaaa; margin-top: 0.75rem;">
        Confidence: <strong>{overall_conf}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Supporting explanation
st.caption("Projection is based on sustained glucose variability, exposure time above 140 mg/dL, and lifestyle recovery trends.")

# Safe disclaimer
st.info("""
**Disclaimer:** This projection reflects sustained metabolic pattern continuation assuming current habits remain unchanged.  
It does not diagnose, predict, or confirm diabetes.  
This tool does not provide medical advice.
""")

# ============================================================
# GRAPHS
# ============================================================

st.markdown("---")
st.header("📈 Trends")

# Create tabs for different graphs
tab1, tab2, tab3, tab4 = st.tabs(["🩸 Raw Glucose", "📊 Daily Features", "📉 Risk Trend", "💤 WHOOP Data"])

with tab1:
    st.subheader("Raw Glucose Trend")
    chart_df = cgm_rows.copy()
    chart_df["datetime"] = pd.to_datetime(chart_df["datetime"])
    chart_df = chart_df.sort_values("datetime")
    
    # Altair chart for raw glucose
    glucose_chart = alt.Chart(chart_df).mark_line(
        strokeWidth=1,
        opacity=0.7
    ).encode(
        x=alt.X("datetime:T", title="Date/Time"),
        y=alt.Y("glucose_mgdl:Q", title="Glucose (mg/dL)", scale=alt.Scale(domain=[40, 300])),
        tooltip=["datetime:T", "glucose_mgdl:Q"]
    ).properties(height=300)
    
    # Add threshold line at 140
    threshold = alt.Chart(pd.DataFrame({"y": [140]})).mark_rule(
        color="red",
        strokeDash=[5, 5]
    ).encode(y="y:Q")
    
    st.altair_chart(glucose_chart + threshold, use_container_width=True)
    st.caption("Red dashed line = 140 mg/dL threshold")

with tab2:
    st.subheader("Daily Aggregated Features")
    
    daily_df = out_df.copy()
    
    # Glucose Mean
    col1, col2 = st.columns(2)
    with col1:
        chart = alt.Chart(daily_df).mark_line(point=True, color="#1f77b4").encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("glucose_mean:Q", title="Glucose Mean (mg/dL)"),
            tooltip=["date:T", "glucose_mean:Q"]
        ).properties(height=200, title="Daily Glucose Mean")
        st.altair_chart(chart, use_container_width=True)
    
    with col2:
        chart = alt.Chart(daily_df).mark_line(point=True, color="#ff7f0e").encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("glucose_std:Q", title="Glucose Std Dev"),
            tooltip=["date:T", "glucose_std:Q"]
        ).properties(height=200, title="Daily Glucose Variability")
        st.altair_chart(chart, use_container_width=True)
    
    # % Time Above 140 and Spike Freq
    col1, col2 = st.columns(2)
    with col1:
        chart = alt.Chart(daily_df).mark_bar(color="#d62728").encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("pct_time_above_140:Q", title="% Time > 140 mg/dL"),
            tooltip=["date:T", "pct_time_above_140:Q"]
        ).properties(height=200, title="% Time Above 140 mg/dL")
        st.altair_chart(chart, use_container_width=True)
    
    with col2:
        chart = alt.Chart(daily_df).mark_bar(color="#9467bd").encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("spike_freq:Q", title="Spike Count"),
            tooltip=["date:T", "spike_freq:Q"]
        ).properties(height=200, title="Daily Glucose Spikes")
        st.altair_chart(chart, use_container_width=True)

with tab3:
    st.subheader("Predicted Risk Over Time")
    
    # Map risk to numeric for visualization
    risk_map = {"Low": 0, "Moderate": 1, "High": 2}
    plot_df = out_df.copy()
    plot_df["risk_numeric"] = plot_df["predicted_risk"].map(risk_map)
    
    chart = alt.Chart(plot_df).mark_circle(size=100).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("risk_numeric:Q", title="Risk Level", scale=alt.Scale(domain=[-0.5, 2.5])),
        color=alt.Color("predicted_risk:N", scale=alt.Scale(
            domain=["Low", "Moderate", "High"],
            range=["#2ca02c", "#ffbb00", "#d62728"]
        )),
        tooltip=["date:T", "predicted_risk:N", "confidence:Q", "confidence_label:N"]
    ).properties(height=250, title="Risk Zone by Day")
    
    st.altair_chart(chart, use_container_width=True)
    st.caption("0 = Low, 1 = Moderate, 2 = High")

with tab4:
    st.subheader("WHOOP Wearable Data")
    
    if whoop_daily is None or whoop_daily.empty:
        st.info("No WHOOP data uploaded or parsed.")
    else:
        wdf = out_df.copy()
        
        # Sleep
        if "total_sleep_mins" in wdf.columns:
            col1, col2 = st.columns(2)
            with col1:
                chart = alt.Chart(wdf.dropna(subset=["total_sleep_mins"])).mark_bar(color="#17becf").encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("total_sleep_mins:Q", title="Sleep (mins)"),
                    tooltip=["date:T", "total_sleep_mins:Q"]
                ).properties(height=200, title="Total Sleep Duration")
                st.altair_chart(chart, use_container_width=True)
            
            with col2:
                if "deep_sleep_pct" in wdf.columns:
                    chart = alt.Chart(wdf.dropna(subset=["deep_sleep_pct"])).mark_bar(color="#bcbd22").encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("deep_sleep_pct:Q", title="Deep Sleep %"),
                        tooltip=["date:T", "deep_sleep_pct:Q"]
                    ).properties(height=200, title="Deep Sleep Percentage")
                    st.altair_chart(chart, use_container_width=True)
        
        # HRV and HR
        if "hrv" in wdf.columns or "resting_hr" in wdf.columns:
            col1, col2 = st.columns(2)
            with col1:
                if "hrv" in wdf.columns:
                    chart = alt.Chart(wdf.dropna(subset=["hrv"])).mark_line(point=True, color="#e377c2").encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("hrv:Q", title="HRV (ms)"),
                        tooltip=["date:T", "hrv:Q"]
                    ).properties(height=200, title="Heart Rate Variability")
                    st.altair_chart(chart, use_container_width=True)
            
            with col2:
                if "resting_hr" in wdf.columns:
                    chart = alt.Chart(wdf.dropna(subset=["resting_hr"])).mark_line(point=True, color="#7f7f7f").encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("resting_hr:Q", title="Resting HR (bpm)"),
                        tooltip=["date:T", "resting_hr:Q"]
                    ).properties(height=200, title="Resting Heart Rate")
                    st.altair_chart(chart, use_container_width=True)

# ============================================================
# DATA TABLE & KEY DRIVERS
# ============================================================

st.markdown("---")
st.header("📋 Details")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Per-Day Predictions")
    display_cols = ["date", "glucose_mean", "glucose_std", "pct_time_above_140", 
                    "spike_freq", "gvi", "predicted_risk", "confidence_label"]
    display_df = out_df[[c for c in display_cols if c in out_df.columns]].copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Key Drivers")
    st.markdown("*Features with largest deviation from median:*")
    
    feature_cols = [c for c in X.columns if c in out_df.columns]
    diffs = []
    med = X.median()
    for f in feature_cols:
        try:
            val = float(latest.get(f, 0))
            m = float(med.get(f, 0))
            if m != 0:
                pct_diff = abs(val - m) / abs(m) * 100
            else:
                pct_diff = abs(val - m)
            diffs.append((f, val, m, pct_diff))
        except Exception:
            pass
    
    diffs = sorted(diffs, key=lambda x: x[3], reverse=True)[:5]
    for f, val, m, pct in diffs:
        direction = "↑" if val > m else "↓"
        st.write(f"**{f}**: {val:.1f} ({direction} {pct:.0f}% from median)")

# Download button
st.markdown("---")
st.download_button(
    "📥 Download Full Results CSV",
    out_df.to_csv(index=False),
    "metalife_risk_predictions.csv",
    "text/csv",
    use_container_width=True
)

# Footer
st.markdown("---")
st.caption("MetaLife Risk v1.0 — For awareness only, not medical advice.")


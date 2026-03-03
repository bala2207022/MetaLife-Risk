# MetaLife Risk — Lifestyle-Based Metabolic Risk Forecasting

This package provides an end-to-end pipeline and Streamlit app to parse raw CGM (Dexcom Clarity)
and WHOOP exports, aggregate daily features, run interpretable ML models, and present
daily lifestyle-based metabolic risk forecasts (Awareness Only — Non-Medical).

Quick start
1) Create & activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2) Install dependencies:
```bash
pip install -r metalife_risk/requirements.txt
```
3) (Optional) Generate simulated data and train models:
```bash
python -m metalife_risk.data_simulation
python -m metalife_risk.train --data data/simulated_metaflife_risk.csv
```
4) Run Streamlit UI and upload your files:
```bash
# from project root
PYTHONPATH="$PWD" streamlit run "metalife_risk/app.py"
```

User uploads (app):
- Dexcom Clarity CSV (raw CGM; required)
- WHOOP export (optional; CSV or ZIP containing CSVs)

Outputs:
- Daily aggregated features
- Per-day predicted risk zone (Low / Moderate / High)
- Class probabilities and confidence label (HIGH/MEDIUM/LOW)
- Interactive graphs for glucose and wearable trends

Disclaimer: This tool does not provide medical advice. Do not change medication or treatment based on these results.

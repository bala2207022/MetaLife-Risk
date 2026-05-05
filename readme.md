# MetaLife Risk

Lifestyle-Based Metabolic Risk Forecasting (awareness only).

This repository now supports a production-style flow with real-world data:

RAW DATA -> CLEANING -> FEATURE ENGINEERING -> MODEL -> PREDICTION -> DASHBOARD/API

## End-to-End Capabilities

- Real Dexcom Clarity CSV ingestion with column auto-detection
- Optional wearable ingestion (WHOOP or combined health exports)
- Validation and cleaning pipeline (datetime parsing, sort, deduplicate, missing handling, outlier filtering)
- Daily feature engineering for glucose and wearables
- Time-series lag features for next-day risk prediction
- Model training with Random Forest and XGBoost comparison
- Metrics: accuracy, precision, recall, F1, ROC-AUC (+ ROC curve artifact)
- Prediction output: Low / Moderate / High + confidence score/label
- SHAP-based explainability of top features
- What-if simulation for spikes/sleep/HRV changes
- FastAPI service with upload, predict, and simulate endpoints
- Professional Streamlit dashboard

## Project Structure

```
data_pipeline/
	__init__.py
	schema.py
	cleaning.py
	loaders.py

features/
	__init__.py
	daily.py
	timeseries.py

models/
	__init__.py
	training.py
	inference.py
	explainability.py
	simulation.py
	*.joblib / metadata artifacts

api/
	main.py

dashboard/
	app.py

utils/
	__init__.py
	config.py

data/sample_real/
	sample_cgm.csv
	sample_wearable.csv

train_real_world.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train with Real-World Pipeline

Use your own files by adapting `train_real_world.py`, or start with sample files:

```bash
python train_real_world.py
```

Artifacts are saved under `models/`.

## Run Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

## Run FastAPI Backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

- `POST /upload-data`
- `POST /predict`
- `POST /simulate`

## Data Requirements

### CGM input

- Timestamp column (multiple aliases supported)
- Glucose value column (multiple aliases supported)

### Wearable input (optional)

- Date or cycle timestamp
- Sleep duration
- HRV
- Resting heart rate
- Strain
- Recovery

## Important Notes

- The system prioritizes real-world data handling and robust parsing.
- If your training dataset has no explicit label, `risk_label` is generated from `% time above 140` thresholds.
- For reliable model training, use larger longitudinal datasets (multiple weeks/months).

## Disclaimer

This tool does not provide medical advice and is not for diagnosis.
Do not change medication or treatment based on model output.

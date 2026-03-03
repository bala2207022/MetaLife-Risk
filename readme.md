# MetaLife Risk

> **Lifestyle-Based Metabolic Risk Forecasting (Awareness Only)**

MetaLife Risk analyzes CGM (Continuous Glucose Monitor) data and optional wearable metrics to estimate daily metabolic risk zones (**Low** / **Moderate** / **High**) and provides a **120-Day Lifestyle Projection**. This is for **awareness only** — not medical advice.

---

## 🎯 What It Does

1. **Analyzes your CGM data** — Parses Dexcom Clarity exports and computes daily glucose metrics
2. **Predicts risk zones** — Classifies each day as Low, Moderate, or High metabolic risk
3. **Projects forward** — Shows 120-day lifestyle continuation outlook based on current patterns
4. **Visualizes trends** — Interactive charts for glucose, features, and risk over time

---

## ✨ Features

- **Dexcom Clarity CGM support**: Upload raw CSV exports directly
- **WHOOP wearable support** (optional): Add sleep, recovery, and strain data
- **Daily risk predictions**: Low / Moderate / High risk zones per day
- **120-Day Lifestyle Projection**: Pattern continuation outlook assuming current habits
- **Confidence scoring**: HIGH / MEDIUM / LOW confidence based on model probability
- **Interactive visualizations**: Raw glucose, daily features, risk timeline, WHOOP data
- **Key driver analysis**: See which features contribute most to predictions

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r metalife_risk/requirements.txt
```

### 2. Train a model

```bash
# Train using CGMacros data from data/ folder
python -m metalife_risk.train_from_data

# Or train with simulated data
python -m metalife_risk.train
```

### 3. Run the Streamlit app

```bash
streamlit run metalife_risk/app.py
```

Then:
1. Upload your Dexcom Clarity CSV
2. (Optional) Upload WHOOP export
3. Click **Run Predictions**
4. View your risk zones and 120-day projection

---

## 📁 Project Structure

```
metalife_risk/
├── app.py               # Streamlit UI with visualizations
├── parsers.py           # Parse Dexcom Clarity, WHOOP exports
├── feature_engineering.py
├── preprocessing.py
├── train.py             # Train with simulated data
├── train_from_data.py   # Train from data/ folder
├── predict.py           # Load model, predict, compute confidence
├── evaluate.py          # Classification report, confusion matrix
└── data_simulation.py   # Generate synthetic data

models/
├── best_model.joblib    # Best trained model (Random Forest)
├── logistic_regression_model.joblib
├── random_forest_model.joblib
└── confusion_matrix.png

data/
└── cgmacros-.../        # CGMacros dataset
```

---

## 📊 CGM Features (Computed Daily)

| Feature | Description |
|---------|-------------|
| `glucose_mean` | Average glucose (mg/dL) |
| `glucose_std` | Standard deviation of glucose |
| `pct_time_above_140` | % of readings > 140 mg/dL |
| `spike_freq` | Number of glucose spikes (>180 mg/dL) |
| `gvi` | Glucose Variability Index |

## 💤 WHOOP Features (Optional)

| Feature | Description |
|---------|-------------|
| `total_sleep_mins` | Total sleep duration |
| `deep_sleep_pct` | Percentage of deep sleep |
| `hrv` | Heart Rate Variability (ms) |
| `resting_hr` | Resting heart rate (bpm) |
| `daily_strain` | WHOOP daily strain score |
| `recovery` | WHOOP recovery percentage |

---

## 🏷️ Risk Zone Classification

Risk zones are determined by `pct_time_above_140`:

| Zone | % Time > 140 mg/dL | Color |
|------|-------------------|-------|
| **Low** | < 10% | 🟢 Green |
| **Moderate** | 10% – 25% | 🟡 Yellow |
| **High** | > 25% | 🔴 Red |

---

## 🔮 120-Day Lifestyle Projection

The app shows a **pattern continuation projection** based on your current risk zone:

| Current Risk | Projected Status |
|--------------|------------------|
| Low | 🟢 Low Metabolic Stress Continuation |
| Moderate | 🟡 Moderate Metabolic Stress Continuation |
| High | 🔴 High Metabolic Stress Continuation |

**Important:** This projection assumes current habits remain unchanged. It does **not** diagnose, predict, or confirm diabetes.

---

## 📈 Confidence Levels

| Level | Criteria |
|-------|----------|
| **HIGH** | Probability ≥ 75% or margin ≥ 25% |
| **MEDIUM** | Probability ≥ 60% or margin ≥ 15% |
| **LOW** | Otherwise |

---

## 📥 Supported Input Formats

### Dexcom Clarity CSV
- Export from Dexcom Clarity app/web
- Must contain glucose values and timestamps
- Auto-detects various column name formats

### WHOOP Export
- CSV, XLSX, or ZIP export from WHOOP app
- Extracts sleep, HRV, strain, and recovery data

---

## ⚠️ Disclaimer

**MetaLife Risk is for educational and awareness purposes only.**

- This tool does **not** provide medical advice
- Do **not** change insulin or medication based on these insights
- Not a diagnostic tool — consult a healthcare professional
- Results assume patterns continue unchanged
- Does **not** diagnose, predict, or confirm diabetes

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Data**: pandas, numpy
- **ML**: scikit-learn, joblib
- **UI**: Streamlit, Altair
- **Parsing**: openpyxl (Excel support)

---

## 📄 License

For educational use only. See dataset-specific licenses in `data/` folder.

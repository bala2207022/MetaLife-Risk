from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR.mkdir(exist_ok=True)

MODEL_ARTIFACT_PATH = ARTIFACT_DIR / "metabolic_risk_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metabolic_risk_metadata.json"
ROC_CURVE_PATH = ARTIFACT_DIR / "roc_curve.png"

RISK_LABELS = ["Low", "Moderate", "High"]

# Rule-based defaults used when real labels are unavailable.
LOW_RISK_MAX = 10.0
MODERATE_RISK_MAX = 25.0

CONFIDENCE_HIGH_PROB = 0.75
CONFIDENCE_MEDIUM_PROB = 0.60
CONFIDENCE_HIGH_MARGIN = 0.25
CONFIDENCE_MEDIUM_MARGIN = 0.15

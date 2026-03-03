"""Generate simulated dataset for MetaLife Risk.

This script produces aggregated daily features per user for supervised
classification into Low/Moderate/High lifestyle-based metabolic risk.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def simulate(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # Primary glucose features (realistic ranges)
    glucose_mean = rng.normal(110, 25, size=n_samples).clip(60, 300)
    glucose_std = rng.normal(25, 10, size=n_samples).clip(5, 120)
    pct_time_above_140 = rng.beta(1.5, 6, size=n_samples) * 100
    spike_freq = rng.poisson(lam=np.clip((glucose_std - 10) / 10, 0.1, 10), size=n_samples)

    # Glucose variability index (derived-like, but simulated)
    gvi = (glucose_std / glucose_mean) * 100 + rng.normal(0, 2, size=n_samples)

    # Optional wearable features
    total_sleep_mins = rng.normal(420, 80, size=n_samples).clip(180, 720)
    deep_sleep_pct = rng.beta(2, 8, size=n_samples) * 100
    hrv = rng.normal(50, 15, size=n_samples).clip(10, 200)
    resting_hr = rng.normal(65, 8, size=n_samples).clip(40, 120)
    daily_strain = rng.normal(30, 12, size=n_samples).clip(0, 100)
    recovery = rng.normal(60, 20, size=n_samples).clip(0, 100)

    df = pd.DataFrame(
        {
            "glucose_mean": glucose_mean,
            "glucose_std": glucose_std,
            "pct_time_above_140": pct_time_above_140,
            "spike_freq": spike_freq,
            "gvi": gvi,
            "total_sleep_mins": total_sleep_mins,
            "deep_sleep_pct": deep_sleep_pct,
            "hrv": hrv,
            "resting_hr": resting_hr,
            "daily_strain": daily_strain,
            "recovery": recovery,
        }
    )

    # Simple rule-based label generation for simulation (not medical)
    risk_score = (
        0.4 * (df["glucose_mean"] - 90) / 60
        + 0.3 * (df["gvi"] - 10) / 40
        + 0.2 * (df["pct_time_above_140"] / 100)
        + 0.1 * (df["spike_freq"] / (1 + df["spike_freq"]))
    )

    # Adjust by protective wearable signals
    risk_score -= 0.1 * (df["hrv"] - 50) / 50
    risk_score -= 0.05 * (df["deep_sleep_pct"] - 15) / 50

    # Map to 3 classes
    labels = pd.cut(
        risk_score,
        bins=[-999, 0.1, 0.6, 999],
        labels=["Low", "Moderate", "High"],
    )

    df["risk_zone"] = labels.astype(str)
    return df


if __name__ == "__main__":
    df = simulate(3000)
    df.to_csv("data/simulated_metaflife_risk.csv", index=False)
    print("Saved simulated dataset to data/simulated_metaflife_risk.csv")

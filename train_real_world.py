from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.loaders import load_cgm, load_wearable
from features.daily import cgm_daily_features, wearable_daily_features, merge_daily_features
from models.training import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MetaLife Risk with real-world files")
    parser.add_argument("--cgm", default="data/sample_real/sample_cgm.csv", help="Path to Dexcom CSV")
    parser.add_argument("--wearable", default="data/sample_real/sample_wearable.csv", help="Path to wearable CSV/ZIP")
    parser.add_argument("--no-time-series", action="store_true", help="Disable lag features for small datasets")
    parser.add_argument("--lag-days", type=int, default=3, help="Lag window for time-series features")
    args = parser.parse_args()

    cgm_path = Path(args.cgm)
    wearable_path = Path(args.wearable)

    cgm = load_cgm(cgm_path)
    cgm_daily = cgm_daily_features(cgm)

    wearable_daily = None
    if wearable_path.exists():
        wearable = load_wearable(wearable_path)
        wearable_daily = wearable_daily_features(wearable)

    features = merge_daily_features(cgm_daily, wearable_daily)
    output = train_models(
        features,
        use_time_series=not args.no_time_series,
        lag_days=args.lag_days,
    )

    print("Training complete")
    print(f"Best model: {output.best_model_name}")
    print(f"Metrics saved: {output.metrics.get('test_metrics')}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import pandas as pd


def compute_shap_importance(artifact: dict, feature_df: pd.DataFrame, max_rows: int = 200) -> pd.DataFrame:
    try:
        import shap
    except Exception as exc:  # pragma: no cover
        raise ImportError(f"shap is required for explainability: {exc}")

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    X = feature_df.copy()
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_columns].head(max_rows)

    preproc = model.named_steps["preproc"]
    clf = model.named_steps["clf"]

    Xt = preproc.transform(X)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xt)

    if isinstance(shap_values, list):
        abs_mean = sum(abs(v).mean(axis=0) for v in shap_values) / len(shap_values)
    else:
        abs_mean = abs(shap_values).mean(axis=0)

    imp = pd.DataFrame({"feature": feature_columns, "importance": abs_mean})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp

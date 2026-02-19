import pandas as pd
import numpy as np

from .metadata import infer_feature_roles


# --------------------------------------------------
# PRE CLEAN
# --------------------------------------------------

def pre_clean_checks(df: pd.DataFrame, target=None):

    report = []

    roles = infer_feature_roles(df, target)
    numeric_cols = roles["numeric"]

    # ---------- missing ----------
    for col in df.columns:

        ratio = df[col].isna().mean()
        if ratio == 0:
            continue

        action = (
            "drop_feature" if ratio >= 0.40
            else "impute_numeric_median" if col in numeric_cols
            else None
        )

        if action:
            report.append({
                "column": col,
                "action": action,
                "missing_ratio": float(ratio)
            })

    # ---------- skew ONLY TRUE NUMERIC ----------
    for col in numeric_cols:

        skew_val = df[col].skew()
        if abs(skew_val) > 2:
            report.append({
                "column": col,
                "action": "apply_power_transform",
                "skew": float(skew_val)
            })

    return report


# --------------------------------------------------
# POST CLEAN
# --------------------------------------------------

def post_clean_checks(df: pd.DataFrame):
    return {
        "remaining_missing": float(df.isna().mean().mean()),
        "n_features": df.shape[1],
        "mean_variance": float(df.var(numeric_only=True).mean())
    }
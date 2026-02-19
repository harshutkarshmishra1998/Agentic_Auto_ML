import pandas as pd
import numpy as np


# ---------- PRE CLEAN DIAGNOSTICS ----------

def pre_clean_checks(df: pd.DataFrame):
    report = []

    # missing values
    for col in df.columns:
        ratio = df[col].isna().mean()

        if ratio == 0:
            continue

        action = (
            "drop_feature" if ratio >= 0.40
            else "impute_numeric_median"
            if pd.api.types.is_numeric_dtype(df[col])
            else None
        )

        if action:
            report.append({
                "column": col,
                "action": action,
                "missing_ratio": float(ratio)
            })

    # skew detection
    for col in df.select_dtypes(include=np.number):
        if abs(df[col].skew()) > 2: #type: ignore
            report.append({
                "column": col,
                "action": "apply_power_transform"
            })

    return report


# ---------- POST CLEAN VALIDATION ----------

def post_clean_checks(df: pd.DataFrame):
    return {
        "remaining_missing": float(df.isna().mean().mean()),
        "n_features": df.shape[1],
        "mean_variance": float(df.var(numeric_only=True).mean())
    }
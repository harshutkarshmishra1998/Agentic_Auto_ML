import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor #type: ignore

from .diagnostic_types import DiagnosticResult


# --------------------------------------------------
# AUTO FIXABLE DETECTORS
# --------------------------------------------------

def detect_duplicate_columns(df):
    results = []
    seen = {}
    for col in df.columns:
        vals = tuple(df[col].values)
        if vals in seen:
            results.append(DiagnosticResult(
                "duplicate_columns", col, None, "high",
                auto_fixable=True,
                recommended_action="drop_column"
            ))
        seen[vals] = col
    return results


def detect_duplicate_rows(df):
    ratio = df.duplicated().mean()
    if ratio > 0:
        return [DiagnosticResult(
            "duplicate_rows", None, ratio, "moderate",
            auto_fixable=True,
            recommended_action="drop_duplicate_rows"
        )]
    return []


def detect_near_constant(df):
    results = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            results.append(DiagnosticResult(
                "near_constant", col, None, "high",
                auto_fixable=True,
                recommended_action="drop_column"
            ))
    return results


def detect_extreme_kurtosis(df):
    results = []
    for col in df.select_dtypes(include=np.number):
        k = kurtosis(df[col].dropna())
        if abs(k) > 10:
            results.append(DiagnosticResult(
                "extreme_kurtosis", col, k, "moderate",
                auto_fixable=True,
                recommended_action="power_transform"
            ))
    return results


def detect_impossible_values(df):
    results = []
    for col in df.select_dtypes(include=np.number):
        if "age" in col.lower():
            if (df[col] < 0).any():
                results.append(DiagnosticResult(
                    "impossible_values", col, "negative", "high",
                    auto_fixable=True,
                    recommended_action="clip_zero"
                ))
    return results


# --------------------------------------------------
# POLICY REQUIRED
# --------------------------------------------------

def detect_class_imbalance(df, target):
    if not target or target not in df.columns:
        return []

    dist = df[target].value_counts(normalize=True)
    if dist.max() > 0.9:
        return [DiagnosticResult(
            "class_imbalance", target, dist.to_dict(), "high",
            policy_required=True,
            recommended_action="resampling_strategy_required"
        )]
    return []


def detect_multicollinearity(df):
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2:
        return []

    vif_data = []
    for i in range(num.shape[1]):
        vif = variance_inflation_factor(num.values, i)
        if vif > 10:
            vif_data.append(DiagnosticResult(
                "multicollinearity_index",
                num.columns[i],
                float(vif),
                "high",
                policy_required=True
            ))
    return vif_data


def detect_high_feature_to_sample_ratio(df):
    ratio = df.shape[1] / max(df.shape[0], 1)
    if ratio > 0.5:
        return [DiagnosticResult(
            "high_feature_to_sample_ratio",
            None,
            ratio,
            "high",
            policy_required=True
        )]
    return []


# --------------------------------------------------
# INFORMATIONAL
# --------------------------------------------------

def detect_text_columns(df):
    results = []
    for col in df.columns:
        if df[col].dtype == "O" and df[col].str.len().mean() > 30:
            results.append(DiagnosticResult(
                "text_column_detection", col, None, "info"
            ))
    return results


def detect_boolean_columns(df):
    results = []
    for col in df.columns:
        if df[col].dropna().isin([0, 1, True, False]).all():
            results.append(DiagnosticResult(
                "boolean_detection", col, None, "info"
            ))
    return results


def detect_small_sample(df):
    if len(df) < 500:
        return [DiagnosticResult(
            "small_sample_warning", None, len(df), "info"
        )]
    return []


# --------------------------------------------------
# MASTER REGISTRY
# --------------------------------------------------

AUTO_FIX_DETECTORS = [
    detect_duplicate_columns,
    detect_duplicate_rows,
    detect_near_constant,
    detect_extreme_kurtosis,
    detect_impossible_values,
]

POLICY_DETECTORS = [
    detect_class_imbalance,
    detect_multicollinearity,
    detect_high_feature_to_sample_ratio,
]

INFO_DETECTORS = [
    detect_text_columns,
    detect_boolean_columns,
    detect_small_sample,
]
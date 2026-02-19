import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, ks_2samp
from scipy.signal import find_peaks
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from .diagnostic_types import DiagnosticResult


# =====================================================
# AUTO FIX
# =====================================================

def detect_duplicate_columns(df):
    results = []
    seen = {}
    for col in df.columns:
        key = tuple(df[col].values)
        if key in seen:
            results.append(DiagnosticResult(
                "duplicate_columns", col, None, "high",
                auto_fixable=True, recommended_action="drop_feature"
            ))
        seen[key] = col
    return results


def detect_duplicate_rows(df):
    if df.duplicated().any():
        return [DiagnosticResult(
            "duplicate_rows", None, None, "high",
            auto_fixable=True, recommended_action="drop_duplicate_rows"
        )]
    return []


def detect_text_missing(df):
    res = []
    for c in df.select_dtypes(include="object"):
        if df[c].isna().any():
            res.append(DiagnosticResult(
                "text_missing", c, None, "moderate",
                auto_fixable=True, recommended_action="impute_text_missing"
            ))
    return res


def detect_near_constant(df):
    res = []
    for c in df.columns:
        if df[c].nunique(dropna=False) <= 1:
            res.append(DiagnosticResult(
                "near_constant", c, None, "high",
                auto_fixable=True, recommended_action="drop_feature"
            ))
    return res


def detect_extreme_kurtosis(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        k = kurtosis(df[c].dropna())
        if abs(k) > 10:
            res.append(DiagnosticResult(
                "extreme_kurtosis", c, k, "moderate",
                auto_fixable=True, recommended_action="apply_power_transform"
            ))
    return res


def detect_heavy_tail(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        if abs(skew(df[c].dropna())) > 2:
            res.append(DiagnosticResult(
                "heavy_tail", c, None, "moderate",
                auto_fixable=True, recommended_action="apply_power_transform"
            ))
    return res


def detect_zero_inflation(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        if (df[c] == 0).mean() > 0.7:
            res.append(DiagnosticResult(
                "zero_inflation", c, None, "moderate",
                auto_fixable=True, recommended_action="apply_power_transform"
            ))
    return res


def detect_impossible_values(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        if "age" in c.lower() and (df[c] < 0).any():
            res.append(DiagnosticResult(
                "impossible_values", c, None, "high",
                auto_fixable=True, recommended_action="clip_zero"
            ))
    return res


def detect_encoding_inconsistency(df):
    res = []
    for c in df.select_dtypes(include="object"):
        vals = df[c].dropna().astype(str)
        if len(vals) > 0 and any(v.strip().lower() != v for v in vals):
            res.append(DiagnosticResult(
                "encoding_inconsistency", c, None, "moderate",
                auto_fixable=True, recommended_action="normalize_encoding"
            ))
    return res


def detect_rare_categories(df):
    res = []
    for c in df.select_dtypes(include="object"):
        if (df[c].value_counts(normalize=True) < 0.01).any():
            res.append(DiagnosticResult(
                "rare_categories", c, None, "moderate",
                auto_fixable=True, recommended_action="group_rare_categories"
            ))
    return res


def detect_high_cardinality(df):
    res = []
    for c in df.select_dtypes(include="object"):
        if df[c].nunique() > 100:
            res.append(DiagnosticResult(
                "high_cardinality", c, None, "moderate",
                auto_fixable=True, recommended_action="reduce_cardinality"
            ))
    return res


def detect_outlier_clusters(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        z = (df[c] - df[c].mean()) / df[c].std()
        if (np.abs(z) > 5).any():
            res.append(DiagnosticResult(
                "outlier_clusters", c, None, "moderate",
                auto_fixable=True, recommended_action="clip_outliers"
            ))
    return res


# =====================================================
# POLICY
# =====================================================

def detect_class_imbalance(df, target=None):
    if not target or target not in df.columns:
        return []
    dist = df[target].value_counts(normalize=True)
    if dist.max() > 0.9:
        return [DiagnosticResult(
            "class_imbalance", target, dist.to_dict(),
            "high", policy_required=True
        )]
    return []


def detect_multicollinearity(df):
    num = df.select_dtypes(include=np.number).dropna()
    if num.shape[1] < 2:
        return []
    res = []
    try:
        for i in range(num.shape[1]):
            vif = variance_inflation_factor(num.values, i)
            if vif > 10:
                res.append(DiagnosticResult(
                    "multicollinearity_index",
                    num.columns[i], float(vif),
                    "high", policy_required=True
                ))
    except:
        pass
    return res


def detect_correlation_groups(df):
    corr = df.select_dtypes(include=np.number).corr().abs()
    if (corr > 0.95).sum().sum() > len(corr):
        return [DiagnosticResult("correlation_groups", None, None, "moderate", policy_required=True)]
    return []


def detect_feature_redundancy(df):
    if df.shape[1] > 50:
        return [DiagnosticResult("feature_redundancy", None, None, "moderate", policy_required=True)]
    return []


def detect_multimodality(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        hist, _ = np.histogram(df[c].dropna(), bins=30)
        peaks, _ = find_peaks(hist)
        if len(peaks) > 2:
            res.append(DiagnosticResult("multimodality", c, len(peaks), "moderate", policy_required=True))
    return res


def detect_heteroscedasticity(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        parts = np.array_split(df[c].dropna(), 4)
        vars_ = [np.var(p) for p in parts if len(p) > 0]
        if len(vars_) > 1 and max(vars_) / (min(vars_) + 1e-6) > 5:
            res.append(DiagnosticResult("heteroscedasticity", c, None, "moderate", policy_required=True))
    return res


def detect_high_feature_to_sample_ratio(df):
    r = df.shape[1] / max(df.shape[0], 1)
    if r > 0.5:
        return [DiagnosticResult("high_feature_to_sample_ratio", None, r, "high", policy_required=True)]
    return []


def detect_target_separability(df, target=None):
    if not target or target not in df.columns:
        return []
    y = df[target]
    res = []
    for c in df.select_dtypes(include=np.number).drop(columns=[target], errors="ignore"):
        try:
            mi = mutual_info_classif(df[[c]].fillna(0), y)[0]
            if mi > 0.5:
                res.append(DiagnosticResult("target_separability", c, float(mi), "moderate", policy_required=True))
        except:
            pass
    return res


def detect_seasonality_presence(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        s = df[c].dropna()
        if len(s) < 50:
            continue
        ac = np.correlate(s - s.mean(), s - s.mean(), mode="full")
        ac = ac[len(ac)//2:]
        peaks, _ = find_peaks(ac)
        if len(peaks) > 3:
            res.append(DiagnosticResult("seasonality_presence", c, len(peaks), "moderate", policy_required=True))
    return res


def detect_time_frequency_irregularity(df):
    res = []
    for c in df.columns:
        ts = pd.to_datetime(df[c], errors="coerce").dropna()
        if len(ts) > 20:
            d = ts.sort_values().diff().dt.total_seconds().dropna()
            if len(d) > 0 and d.std() > d.mean():
                res.append(DiagnosticResult("time_frequency_irregularity", c, None, "moderate", policy_required=True))
    return res


# =====================================================
# INFO
# =====================================================

def detect_text_columns(df):
    return [DiagnosticResult("text_column", c, None, "info") for c in df.select_dtypes(include="object")]


def detect_boolean_columns(df):
    res = []
    for c in df.columns:
        if df[c].dropna().isin([0, 1, True, False]).all():
            res.append(DiagnosticResult("boolean_column", c, None, "info"))
    return res


def detect_ordinal_feature(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        u = np.sort(df[c].dropna().unique())
        if len(u) < 20 and np.all(np.diff(u) >= 0):
            res.append(DiagnosticResult("ordinal_feature", c, None, "info"))
    return res


def detect_small_sample(df):
    if len(df) < 500:
        return [DiagnosticResult("small_sample", None, len(df), "info")]
    return []


def detect_sampling_bias(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        q = pd.qcut(df[c], 4, duplicates="drop")
        if q.value_counts(normalize=True).max() > 0.7:
            res.append(DiagnosticResult("sampling_bias", c, None, "info"))
    return res

def detect_timestamp_columns(df):
    results = []

    for col in df.columns:

        # ---------- skip numeric ----------
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        # ---------- sample values ----------
        sample = df[col].dropna().astype(str).head(50)

        if len(sample) == 0:
            continue

        # ---------- heuristic check ----------
        # look for date-like patterns
        contains_date_pattern = sample.str.contains(
            r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}",
            regex=True
        ).mean()

        if contains_date_pattern < 0.3:
            continue

        # ---------- now safe to parse ----------
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")

        if parsed.notna().mean() > 0.8:
            results.append(
                DiagnosticResult("timestamp_column", col, None, "info")
            )

    return results


def detect_block_missingness(df):
    res = []
    for c in df.columns:
        mask = df[c].isna().astype(int)
        runs = mask.groupby((mask != mask.shift()).cumsum()).sum()
        if len(runs) and runs.max() > 10:
            res.append(DiagnosticResult("block_missingness", c, None, "info"))
    return res


def detect_missing_pattern(df):
    if df.isna().mean().mean() > 0.1:
        return [DiagnosticResult("missing_pattern", None, None, "info")]
    return []


def detect_feature_noise(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        if df[c].std() < 1e-3:
            res.append(DiagnosticResult("feature_noise", c, None, "info"))
    return res


def detect_distribution_shift(df):
    res = []
    mid = len(df)//2
    for c in df.select_dtypes(include=np.number):
        a = df[c].iloc[:mid].dropna()
        b = df[c].iloc[mid:].dropna()
        if len(a) > 10 and len(b) > 10 and ks_2samp(a, b).pvalue < 0.01: #type: ignore
            res.append(DiagnosticResult("distribution_shift", c, None, "info"))
    return res


def detect_feature_scaling_need(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        if df[c].std() > 100:
            res.append(DiagnosticResult("scaling_needed", c, None, "info"))
    return res


def detect_nonlinearity_score(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        x = df[c].dropna().values.reshape(-1,1)
        if len(x) < 20:
            continue
        y = np.arange(len(x))
        lin = LinearRegression().fit(x,y).score(x,y)
        poly = LinearRegression().fit(
            PolynomialFeatures(2).fit_transform(x), y
        ).score(PolynomialFeatures(2).fit_transform(x),y)
        if poly - lin > 0.2:
            res.append(DiagnosticResult("nonlinearity", c, None, "info"))
    return res


def detect_interaction_strength(df):
    num = df.select_dtypes(include=np.number)
    corr = num.corr().abs()
    if (corr > 0.7).sum().sum() > len(corr):
        return [DiagnosticResult("interaction_strength", None, None, "info")]
    return []


def detect_variance_instability(df):
    res = []
    for c in df.select_dtypes(include=np.number):
        r = df[c].rolling(50).var().dropna()
        if len(r) and r.max() > 5 * r.mean():
            res.append(DiagnosticResult("variance_instability", c, None, "info"))
    return res


# =====================================================
# REGISTRIES
# =====================================================

AUTO_FIX_DETECTORS = [
    detect_duplicate_columns,
    detect_duplicate_rows,
    # detect_text_missing,
    detect_near_constant,
    detect_extreme_kurtosis,
    # detect_heavy_tail,
    # detect_zero_inflation,
    detect_impossible_values,
    # detect_encoding_inconsistency,
    # detect_rare_categories,
    # detect_high_cardinality,
    # detect_outlier_clusters,
]

POLICY_DETECTORS = [
    detect_class_imbalance,
    # detect_multicollinearity,
    # detect_correlation_groups,
    # detect_feature_redundancy,
    # detect_multimodality,
    # detect_heteroscedasticity,
    detect_high_feature_to_sample_ratio,
    # detect_target_separability,
    # detect_seasonality_presence,
    # detect_time_frequency_irregularity,
]

INFO_DETECTORS = [
    detect_text_columns,
    detect_boolean_columns,
    detect_ordinal_feature,
    detect_small_sample,
    # detect_sampling_bias,
    detect_timestamp_columns,
    # detect_block_missingness,
    # detect_missing_pattern,
    # detect_feature_noise,
    # detect_distribution_shift,
    # detect_feature_scaling_need,
    # detect_nonlinearity_score,
    # detect_interaction_strength,
    # detect_variance_instability,
]
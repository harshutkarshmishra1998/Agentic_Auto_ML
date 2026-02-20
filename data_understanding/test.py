import os
import json
import pandas as pd
import numpy as np
from scipy.stats import skew
from groq import Groq
import api_keys


# =====================================================
# LOAD DATA
# =====================================================
def load_data(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError("Unsupported file format")


# =====================================================
# GLOBAL CONFIG
# =====================================================
AMBIGUITY_THRESHOLD = 0.75
ID_UNIQUE_THRESHOLD = 0.98


# =====================================================
# BASIC STAT HELPERS
# =====================================================
def safe_skew(series):
    s = series.dropna()
    return skew(s) if len(s) > 5 else 0


def is_binary(series):
    vals = set(series.dropna().unique())
    return vals.issubset({0, 1, True, False})


def is_numeric_categorical(series, n_rows):
    nunique = series.nunique()
    ratio = nunique / max(n_rows, 1)
    return ratio < 0.05 and nunique < 50


def evenly_spaced(series):
    vals = sorted(series.dropna().unique())
    if len(vals) < 3:
        return True
    diffs = np.diff(vals)
    return np.allclose(diffs, diffs[0])


# =====================================================
# SEMANTIC TYPE INFERENCE (DETERMINISTIC CORE)
# =====================================================
def infer_semantic_type(series, name, n_rows):
    name = name.lower()
    nunique = series.nunique(dropna=True)
    unique_ratio = nunique / max(n_rows, 1)

    # datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "temporal", 1.0

    # binary
    if is_binary(series):
        return "binary", 1.0

    # identifier
    if unique_ratio > ID_UNIQUE_THRESHOLD:
        return "identifier", 0.95

    # numeric-coded categorical
    categorical_keywords = ["code","id","state","type","plan","flag","status","area"]
    if any(k in name for k in categorical_keywords):
        return "categorical nominal", 0.8

    if pd.api.types.is_integer_dtype(series) and is_numeric_categorical(series, n_rows):
        if not evenly_spaced(series):
            return "categorical nominal", 0.75

    # text
    if series.dtype == "object":
        avg_len = series.dropna().astype(str).str.len().mean()
        if avg_len > 40:
            return "free text", 0.9
        return "categorical nominal", 0.8

    # numeric real
    if pd.api.types.is_numeric_dtype(series):
        return "numeric continuous", 0.7

    return "unknown", 0.3


# =====================================================
# UNIT DETECTION (IMPROVED)
# =====================================================
def detect_unit(name):
    name = name.lower()

    currency = ["price","amount","income","salary","cost","charge"]
    time = ["time","duration","seconds","minutes","hours"]
    geo = ["lat","lon","longitude","latitude"]
    count = ["count","calls","num"]

    if any(k in name for k in currency):
        return "currency"
    if any(k in name for k in time):
        return "time"
    if any(k in name for k in geo):
        return "geospatial"
    if any(k in name for k in count):
        return "count"

    return "unknown"


# =====================================================
# TRANSFORM HINT (SAFE)
# =====================================================
def transform_hint(series):
    if not pd.api.types.is_numeric_dtype(series):
        return "none"

    s = series.dropna()
    if len(s) < 10:
        return "none"

    if (s <= 0).any():
        return "none"

    if abs(safe_skew(s)) > 1:
        return "log_transform_candidate"

    return "none"


# =====================================================
# DATA QUALITY (FIXED)
# =====================================================
def data_quality(series, df):
    issues = []

    if series.isna().mean() > 0.4:
        issues.append("high_missing")

    if pd.api.types.is_numeric_dtype(series):
        if np.isinf(series).any():
            issues.append("infinite_values")

    return issues


# =====================================================
# TARGET SCORING (PROPER)
# =====================================================
def detect_target(df):
    scores = {}

    for col in df.columns:
        s = df[col]
        score = 0

        if s.nunique() <= 2:
            score += 3

        if any(k in col.lower() for k in ["target","label","outcome","churn"]):
            score += 5

        if s.nunique() < len(s) * 0.1:
            score += 1

        scores[col] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked if r[1] > 0]


# =====================================================
# LEAKAGE DETECTION (IMPROVED)
# =====================================================
def detect_leakage(df, target):
    if target not in df.columns:
        return {}

    y = df[target]
    leakage = {}

    for col in df.columns:
        if col == target:
            continue

        s = df[col]

        if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(y):
            corr = abs(s.corr(y))
            if corr > 0.9:
                leakage[col] = {"type": "high_target_correlation", "score": float(corr)}

        if any(k in col.lower() for k in ["final","approved","result","after"]):
            leakage[col] = {"type": "post_event_feature"}

    return leakage


# =====================================================
# LLM DISAMBIGUATION (ONLY IF NEEDED)
# =====================================================
def llm_resolve_column(name, stats):
    client = Groq()

    prompt = f"""
Classify semantic type of column.

Choose one:
numeric continuous
categorical nominal
categorical ordinal
binary
identifier

COLUMN:
{name}

STATS:
{json.dumps(stats)}

Return JSON {{type, reason}} only.
"""

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(res.choices[0].message.content)


# =====================================================
# COLUMN ANALYSIS
# =====================================================
def analyze_column(df, col):
    s = df[col]
    n = len(df)

    sem, confidence = infer_semantic_type(s, col, n)

    stats = {
        "unique": int(s.nunique()),
        "unique_ratio": float(s.nunique()/n),
        "dtype": str(s.dtype)
    }

    if confidence < AMBIGUITY_THRESHOLD:
        try:
            llm = llm_resolve_column(col, stats)
            sem = llm["type"]
        except:
            pass

    return {
        "semantic_type": sem,
        "unit": detect_unit(col),
        "transform_hint": transform_hint(s),
        "data_quality": data_quality(s, df)
    }


# =====================================================
# BUILD TABLE
# =====================================================
def build_column_table(df):
    rows = []

    for col in df.columns:
        analysis = analyze_column(df, col)
        rows.append({"column": col, **analysis})

    return pd.DataFrame(rows)


# =====================================================
# PIPELINE
# =====================================================
def stage1(path):
    df = load_data(path)

    column_table = build_column_table(df)

    targets = detect_target(df)
    primary_target = targets[0] if targets else None

    leakage = detect_leakage(df, primary_target)

    return column_table, {
        "target_candidates": targets,
        "primary_target": primary_target,
        "leakage": leakage
    }


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    file_path = "data_understanding/data.csv"

    table, meta = stage1(file_path)

    table.to_csv("column_analysis.csv", index=False)

    with open("stage1_profile.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Stage-1 analysis complete")
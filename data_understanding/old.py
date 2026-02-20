import os
import json
import pandas as pd
import numpy as np
from scipy.stats import skew
from groq import Groq


# =====================================================
# LOAD DATA
# =====================================================
def load_data(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format")


# =====================================================
# TYPE & DISTRIBUTION
# =====================================================
def semantic_type(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return "temporal"
    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric discrete" if series.nunique() <= 10 else "numeric continuous"
    if series.dtype == "object":
        avg_len = series.dropna().astype(str).str.len().mean()
        return "free text" if avg_len > 40 else "categorical nominal"
    return "unknown"


def distribution_shape(series):
    if not pd.api.types.is_numeric_dtype(series):
        return "N/A"
    s = series.dropna()
    if len(s) < 5:
        return "unknown"
    sk = skew(s)
    if abs(sk) < 0.5:
        return "symmetric"
    return "right skewed" if sk > 0 else "left skewed"


def outlier_flag(series):
    if not pd.api.types.is_numeric_dtype(series):
        return "no"
    s = series.dropna()
    if len(s) < 5:
        return "no"
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    return "yes" if outliers > 0 else "no"


# =====================================================
# STRUCTURE
# =====================================================
def cardinality_level(unique_ratio):
    if unique_ratio > 0.9:
        return "very high"
    if unique_ratio > 0.3:
        return "high"
    if unique_ratio > 0.05:
        return "medium"
    return "low"


def role_guess(unique_ratio):
    return "ID" if unique_ratio > 0.98 else "feature"


def encoding_needed(sem_type):
    if sem_type in ["categorical nominal", "binary"]:
        return "encode"
    if sem_type == "free text":
        return "NLP"
    if sem_type == "temporal":
        return "feature extract"
    if sem_type == "numeric continuous":
        return "scale"
    return "no"


# =====================================================
# MISSING
# =====================================================
def missing_pattern(series, df):
    if series.isna().sum() == 0:
        return "none"
    indicator = series.isna().astype(int)
    for col in df.select_dtypes(include=np.number).columns:
        if col == series.name:
            continue
        corr = abs(indicator.corr(df[col].fillna(0)))
        if corr > 0.3:
            return "MAR"
    return "random"


# =====================================================
# EXTRA ANALYSIS
# =====================================================
def category_imbalance(series):
    if not (series.dtype == "object" or series.nunique() < 20):
        return "N/A"
    counts = series.value_counts(normalize=True)
    if len(counts) == 0:
        return "N/A"
    if counts.iloc[0] > 0.75:
        return "high"
    if counts.iloc[0] > 0.5:
        return "moderate"
    return "low"


def text_complexity(series):
    if series.dtype != "object":
        return "none"
    avg_len = series.dropna().astype(str).str.len().mean()
    if avg_len > 100:
        return "long text"
    if avg_len > 30:
        return "medium text"
    return "short text"


def correlation_strength(col, df):
    if not pd.api.types.is_numeric_dtype(df[col]):
        return "unknown"
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return "none"
    corr = numeric.corr()[col].drop(col).abs().max()
    if pd.isna(corr):
        return "none"
    if corr > 0.8:
        return "strong"
    if corr > 0.4:
        return "moderate"
    return "low"


def data_quality_flags(series):
    issues = []
    if series.isna().mean() > 0.4:
        issues.append("high missing")
    if pd.api.types.is_numeric_dtype(series):
        if (series < 0).sum() > 0:
            issues.append("negative values present")
    if series.duplicated().sum() > 0 and series.nunique() < len(series):
        issues.append("duplicates present")
    return ", ".join(issues) if issues else "none"


# =====================================================
# UNIT DETECTION
# =====================================================
def detect_unit(series, name):
    n = name.lower()
    if any(k in n for k in ["price", "amount", "income", "salary", "cost"]):
        return "currency"
    if any(k in n for k in ["lat", "lon"]):
        return "geospatial"
    if any(k in n for k in ["time", "duration", "seconds"]):
        return "time"
    if "%" in str(series.head(20).tolist()):
        return "percentage"
    return "unknown"


# =====================================================
# TRANSFORM + MODELING
# =====================================================
def transform_hint(series):
    if not pd.api.types.is_numeric_dtype(series):
        return "none"
    s = series.dropna()
    if len(s) < 5:
        return "none"
    if abs(skew(s)) > 1:
        return "log_transform_candidate"
    return "none"


def modeling_hint(series):
    if pd.api.types.is_numeric_dtype(series):
        return "scaling_required"
    if series.dtype == "object":
        return "encoding_required"
    return "none"


# =====================================================
# TARGET DETECTION
# =====================================================
def detect_target(df):
    candidates = []
    for col in df.columns:
        s = df[col]
        if s.nunique() <= 2:
            candidates.append(col)
        if any(k in col.lower() for k in ["target", "label", "outcome"]):
            candidates.append(col)
    return list(set(candidates))


# =====================================================
# LEAKAGE DETECTION
# =====================================================
def detect_leakage(df, target):
    leakage = {}
    if target not in df.columns:
        return leakage

    y = df[target]

    for col in df.columns:
        if col == target:
            continue
        s = df[col]

        if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(y):
            corr = abs(s.corr(y))
            if corr > 0.9:
                leakage[col] = {
                    "risk": "high",
                    "type": "target_correlation",
                    "score": float(corr)
                }

        if any(k in col.lower() for k in ["final", "approved", "result", "after"]):
            leakage[col] = {
                "risk": "possible",
                "type": "post_event_feature"
            }

    return leakage


# =====================================================
# COLUMN INSPECTION TABLE
# =====================================================
def build_column_table(df):
    rows = []

    for col in df.columns:
        s = df[col]
        unique_ratio = s.nunique(dropna=True) / max(len(s), 1)
        sem = semantic_type(s)

        rows.append({
            "Column Name": col,
            "Technical Type": str(s.dtype),
            "Semantic Type": sem,
            "Role": role_guess(unique_ratio),
            "Cardinality Level": cardinality_level(unique_ratio),
            "Missing %": round(s.isna().mean() * 100, 2),
            "Missing Pattern": missing_pattern(s, df),
            "Distribution Shape": distribution_shape(s),
            "Outliers Present": outlier_flag(s),
            "Unique Ratio": round(unique_ratio, 4),
            "Encoding Needed": encoding_needed(sem),
            "Time Dependent": "yes" if sem == "temporal" else "no",
            "Unit / Scale": detect_unit(s, col),
            "Text Complexity": text_complexity(s),
            "Category Imbalance": category_imbalance(s),
            "Correlated With Others": correlation_strength(col, df),
            "Transform Hint": transform_hint(s),
            "Modeling Hint": modeling_hint(s),
            "Data Quality Issues": data_quality_flags(s)
        })

    return pd.DataFrame(rows)


# =====================================================
# LLM SEMANTIC REASONING
# =====================================================
def llm_reasoning(column_table):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    prompt = f"""
You are an expert data analyst.

Interpret dataset meaning using column summary.

Return JSON:
- dataset_type
- row_representation
- important_features
- recommended_target
- ml_task
- modeling_risks

COLUMN TABLE:
{column_table.to_json(orient="records")}
"""

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return json.loads(res.choices[0].message.content)
    except:
        return {"raw": res.choices[0].message.content}


# =====================================================
# FULL PIPELINE
# =====================================================
def stage1(path):
    df = load_data(path)

    column_table = build_column_table(df)
    targets = detect_target(df)
    leakage = detect_leakage(df, targets[0]) if targets else {}
    llm = llm_reasoning(column_table)

    return column_table, {
        "target_candidates": targets,
        "leakage_analysis": leakage,
        "llm_semantic": llm
    }


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    file_path = "your_dataset.csv"

    table, meta = stage1(file_path)

    table.to_csv("column_inspection_table.csv", index=False)

    with open("stage1_full_profile.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved column_inspection_table.csv")
    print("Saved stage1_full_profile.json")
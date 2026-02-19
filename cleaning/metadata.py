import pandas as pd
import numpy as np


# --------------------------------------------------
# TARGET DETECTION
# --------------------------------------------------

COMMON_TARGET_NAMES = {
    "target", "label", "y", "class", "outcome",
    "response", "churn", "price", "salary"
}


def _infer_target_column(df: pd.DataFrame, user_target=None):
    if user_target and user_target in df.columns:
        return user_target

    for col in df.columns:
        if col.lower() in COMMON_TARGET_NAMES:
            return col

    return None


def _infer_target_type(series: pd.Series):
    if series.dtype == "O" or series.nunique() <= 20:
        if series.nunique() == 2:
            return "binary_classification"
        return "multiclass_classification"
    return "regression"


# --------------------------------------------------
# FEATURE ROLE DETECTION
# --------------------------------------------------

def _is_encoded_categorical(series: pd.Series, n_rows: int):
    """
    Numeric column that behaves like category.
    """
    unique_count = series.nunique(dropna=True)
    ratio = unique_count / n_rows

    is_integer_like = pd.api.types.is_integer_dtype(series) or (
        series.dropna().apply(float.is_integer).all()
        if pd.api.types.is_numeric_dtype(series) else False
    )

    return (
        pd.api.types.is_numeric_dtype(series)
        and is_integer_like
        and unique_count <= 50
        and ratio < 0.05
    )


def _detect_datetime_columns(df: pd.DataFrame):
    datetime_cols = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue

        if df[col].dtype == "O":
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    datetime_cols.append(col)
            except Exception:
                pass

    return datetime_cols


def _detect_id_columns(df: pd.DataFrame, n_rows: int):
    id_cols = []
    for col in df.columns:
        if df[col].nunique(dropna=False) == n_rows:
            id_cols.append(col)
    return id_cols


def _detect_constant_columns(df: pd.DataFrame):
    return [c for c in df.columns if df[c].nunique(dropna=False) <= 1]


# --------------------------------------------------
# STATISTICAL CHARACTERISTICS
# --------------------------------------------------

def _class_imbalance(series: pd.Series):
    dist = series.value_counts(normalize=True)
    return dist.to_dict()


def _correlation_strength(df: pd.DataFrame):
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2:
        return None
    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return float(upper.max().max())


def _dataset_complexity(df: pd.DataFrame):
    """
    Simple interpretable complexity score (0–1 scale approx).
    """
    n_rows, n_cols = df.shape
    missing_ratio = df.isna().mean().mean()

    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] > 1:
        corr = numeric.corr().abs()
        redundancy = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        ).mean().mean()
    else:
        redundancy = 0

    dimensionality = n_cols / max(n_rows, 1)

    score = (
        0.4 * missing_ratio +
        0.3 * (redundancy if not np.isnan(redundancy) else 0) +
        0.3 * dimensionality
    )

    return float(np.clip(score, 0, 1))


def _leakage_candidates(df: pd.DataFrame, target_col: str | None):
    if target_col is None:
        return []

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return []

    num = df.select_dtypes(include=np.number)
    if target_col not in num.columns:
        return []

    corr = num.corr()[target_col].abs()
    return [
        col for col, val in corr.items()
        if col != target_col and val > 0.95
    ]


# --------------------------------------------------
# MAIN METADATA GENERATOR
# --------------------------------------------------

def generate_metadata(df: pd.DataFrame, target=None):

    n_rows = len(df)

    # ----- target -----
    target_col = _infer_target_column(df, target)

    if target_col:
        learning_type = "supervised"
        target_type = _infer_target_type(df[target_col])
    else:
        learning_type = "unsupervised"
        target_type = None

    # ----- feature roles -----
    datetime_cols = _detect_datetime_columns(df)
    id_cols = _detect_id_columns(df, n_rows)
    constant_cols = _detect_constant_columns(df)

    numeric_cols = []
    categorical_cols = []

    for col in df.columns:

        if col == target_col:
            continue

        if col in datetime_cols:
            continue

        if col in id_cols:
            continue

        if col in constant_cols:
            continue

        if df[col].dtype == "O":
            categorical_cols.append(col)
        elif _is_encoded_categorical(df[col], n_rows):
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # ----- statistics -----
    class_dist = (
        _class_imbalance(df[target_col])
        if target_col and target_type != "regression"
        else None
    )

    metadata = {
        "n_rows": n_rows,
        "n_features": df.shape[1],

        "learning_type": learning_type,
        "target_column": target_col,
        "target_type": target_type,

        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "id_columns": id_cols,
        "constant_columns": constant_cols,

        "feature_columns": [
            c for c in df.columns if c != target_col
        ],

        "class_distribution": class_dist,
        "max_feature_correlation": _correlation_strength(df),
        "dataset_complexity_score": _dataset_complexity(df),
        "leakage_candidates": _leakage_candidates(df, target_col)
    }

    return metadata

# --------------------------------------------------
# PUBLIC HELPER FOR OTHER MODULES
# --------------------------------------------------

def infer_feature_roles(df: pd.DataFrame, target=None):
    """
    Returns feature role classification used everywhere.
    """
    n_rows = len(df)

    target_col = _infer_target_column(df, target)
    datetime_cols = _detect_datetime_columns(df)
    id_cols = _detect_id_columns(df, n_rows)
    constant_cols = _detect_constant_columns(df)

    numeric_cols = []
    categorical_cols = []

    for col in df.columns:

        if col == target_col:
            continue
        if col in datetime_cols:
            continue
        if col in id_cols:
            continue
        if col in constant_cols:
            continue

        if df[col].dtype == "O":
            categorical_cols.append(col)
        elif _is_encoded_categorical(df[col], n_rows):
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "id": id_cols,
        "constant": constant_cols,
        "target": target_col
    }
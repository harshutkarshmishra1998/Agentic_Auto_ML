# import pandas as pd
# import numpy as np


# def generate_metadata(df: pd.DataFrame):
#     return {
#         "rows": len(df),
#         "features": df.shape[1],
#         "numeric_features": df.select_dtypes(include=np.number).shape[1],
#         "categorical_features": df.select_dtypes(exclude=np.number).shape[1]
#     }

import pandas as pd
import numpy as np


COMMON_TARGET_NAMES = {
    "target", "label", "y", "class", "outcome",
    "response", "churn", "price", "salary"
}


def _infer_target_column(df: pd.DataFrame, user_target=None):

    # ---------- explicit user target ----------
    if user_target and user_target in df.columns:
        return user_target

    # ---------- heuristic detection ----------
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


def generate_metadata(df: pd.DataFrame, target=None):

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    target_col = _infer_target_column(df, target)

    if target_col is None:
        learning_type = "unsupervised"
        target_type = None
    else:
        learning_type = "supervised"
        target_type = _infer_target_type(df[target_col])

    return {
        "n_rows": len(df),
        "n_features": df.shape[1],

        "learning_type": learning_type,
        "target_column": target_col,
        "target_type": target_type,

        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,

        "feature_columns": [c for c in df.columns if c != target_col],

        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols)
    }
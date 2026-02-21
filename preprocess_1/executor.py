import numpy as np
import pandas as pd
from .plan_schema import PreprocessPlan


def _drop_columns(df, cols):
    return df.drop(columns=cols, errors="ignore")


def _impute_numeric_median(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    return df


def _impute_categorical_mode(df, cols):
    for c in cols:
        if c in df.columns:
            if df[c].dropna().empty:
                continue
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df


def _log_transform(df, cols):
    for c in cols:
        if c in df.columns:
            min_val = df[c].min()
            shift = 1 - min_val if min_val <= 0 else 0
            df[c] = np.log(df[c] + shift)
    return df


STEP_EXECUTORS = {
    "drop_columns": _drop_columns,
    "impute_numeric_median": _impute_numeric_median,
    "impute_categorical_mode": _impute_categorical_mode,
    "log_transform": _log_transform,
}


def execute_plan(plan: PreprocessPlan):
    df = pd.read_csv(plan.dataset_path)

    for step in plan.steps:
        fn = STEP_EXECUTORS.get(step.step_type)
        if fn is None:
            step.status = "skipped"
            continue

        df = fn(df, step.columns)
        step.status = "executed"

    return df, plan

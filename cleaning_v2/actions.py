import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

ACTION_REGISTRY = {}


# =====================================================
# REGISTRY
# =====================================================

def register(name):
    def decorator(func):
        ACTION_REGISTRY[name] = func
        return func
    return decorator


def run_action(df: pd.DataFrame, action: str, column=None):
    if action not in ACTION_REGISTRY:
        raise ValueError(f"Unknown action: {action}")
    return ACTION_REGISTRY[action](df, column)


# =====================================================
# STRUCTURE
# =====================================================

@register("drop_feature")
def drop_feature(df, column):
    if column in df.columns:
        df = df.drop(columns=[column]).copy()
    return df


@register("drop_duplicate_rows")
def drop_duplicate_rows(df, column=None):
    return df.drop_duplicates().copy()


# =====================================================
# MISSING
# =====================================================

@register("impute_numeric_median")
def impute_numeric_median(df, column):
    if column in df.columns:
        df = df.copy()
        df.loc[:, column] = df[column].fillna(df[column].median())
    return df


@register("impute_text_missing")
def impute_text_missing(df, column):
    # if column in df.columns:
    #     df = df.copy()
    #     df.loc[:, column] = df[column].fillna("")
    return df


# =====================================================
# DISTRIBUTION
# =====================================================

@register("apply_power_transform")
def power_transform(df, column):

    if column not in df.columns:
        return df

    df = df.copy()

    pt = PowerTransformer()

    vals = df[[column]].replace([np.inf, -np.inf], np.nan)
    vals = vals.fillna(vals.median())

    transformed = pt.fit_transform(vals)

    df.loc[:, column] = transformed.ravel()

    return df


@register("clip_outliers")
def clip_outliers(df, column, z=5):

    if column not in df.columns:
        return df

    df = df.copy()

    mean = df[column].mean()
    std = df[column].std()

    df.loc[:, column] = df[column].clip(mean - z * std, mean + z * std)

    return df


# =====================================================
# CATEGORY
# =====================================================

@register("normalize_encoding")
def normalize_encoding(df, column):

    # if column not in df.columns:
    #     return df

    # series = df[column]

    # # skip long text (NLP column)
    # sample = series.dropna().astype(str).head(100)
    # if len(sample) > 0 and sample.str.len().mean() > 50:
    #     return df

    # df = df.copy()

    # df.loc[:, column] = (
    #     series.astype(str)
    #     .str.strip()
    #     .str.lower()
    # )

    return df


# @register("group_rare_categories")
# def group_rare(df, column, threshold=0.01):

#     if column not in df.columns:
#         return df

#     df = df.copy()

#     freq = df[column].value_counts(normalize=True)
#     rare = freq[freq < threshold].index

#     df.loc[:, column] = df[column].replace(rare, "other")

#     return df

# @register("group_rare_categories")
# def group_rare_categories(df, column, threshold=0.01):

#     if column not in df.columns:
#         return df

#     series = df[column]

#     # ---------- skip tiny categorical ----------
#     nunique = series.nunique(dropna=True)
#     if nunique < 10:
#         return df

#     # ---------- compute normalized freq ----------
#     freq = series.value_counts(normalize=True)

#     rare_values = freq[freq < threshold].index

#     if len(rare_values) == 0:
#         return df

#     # ---------- vectorized mask (FAST) ----------
#     mask = series.isin(rare_values)

#     if not mask.any():
#         return df

#     df = df.copy()
#     df.loc[mask, column] = "other"

#     return df


# @register("reduce_cardinality")
# def reduce_cardinality(df, column, max_categories=100):

#     if column not in df.columns:
#         return df

#     df = df.copy()

#     top = df[column].value_counts().nlargest(max_categories).index
#     df.loc[:, column] = df[column].where(df[column].isin(top), "other")

#     return df


# =====================================================
# VALUE FIX
# =====================================================

@register("clip_zero")
def clip_zero(df, column):
    if column in df.columns:
        df = df.copy()
        df.loc[:, column] = df[column].clip(lower=0)
    return df
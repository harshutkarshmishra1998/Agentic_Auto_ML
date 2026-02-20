import pandas as pd
import numpy as np
from itertools import combinations


# --------------------------------------------------
# SAFE NUMERIC MATRIX
# --------------------------------------------------
def numeric_nonconstant(df):
    num = df.select_dtypes(include="number")
    return num.loc[:, num.nunique(dropna=True) > 1]


# --------------------------------------------------
# CORRELATION PAIRS (store column names)
# --------------------------------------------------
def correlation_pairs(df, min_abs_corr=0.0):
    """
    Returns list of pairwise correlations.
    Only includes correlations > min_abs_corr.
    """

    num = numeric_nonconstant(df)

    if num.shape[1] < 2:
        return []

    corr = num.corr()

    pairs = []

    for c1, c2 in combinations(corr.columns, 2):
        val = corr.loc[c1, c2]

        if pd.isna(val):
            continue

        if abs(val) > min_abs_corr:
            pairs.append({
                "col_1": c1,
                "col_2": c2,
                "correlation": float(val)
            })

    return pairs


# --------------------------------------------------
# STRONG REDUNDANCY DETECTOR
# --------------------------------------------------
def redundant_features(df, threshold=0.95):
    """
    Features that are almost duplicates (|corr| >= threshold)
    """

    pairs = correlation_pairs(df, min_abs_corr=threshold)

    redundant = []

    for p in pairs:
        redundant.append({
            "feature_a": p["col_1"],
            "feature_b": p["col_2"],
            "correlation": abs(p["correlation"]),
            "relationship": "highly_correlated"
        })

    return redundant


# --------------------------------------------------
# DERIVED LINEAR FORMULA DETECTOR
# --------------------------------------------------
def derived_linear_relationships(df, threshold=0.999):
    """
    Detect deterministic linear relationships.
    Example: charge = minutes * rate
    """

    num = numeric_nonconstant(df)

    if num.shape[1] < 2:
        return []

    corr = num.corr().abs()

    derived = []

    for c1, c2 in combinations(corr.columns, 2):

        val = corr.loc[c1, c2]

        if val >= threshold:
            derived.append({
                "base_feature": c1,
                "derived_feature": c2,
                "correlation": float(val),
                "relationship": "deterministic_linear"
            })

    return derived


# --------------------------------------------------
# FEATURE DEPENDENCY GRAPH
# --------------------------------------------------
def feature_dependency_graph(df, threshold=0.7):
    """
    Graph representation of feature relationships.
    Node -> connected features
    """

    pairs = correlation_pairs(df, min_abs_corr=threshold)

    graph = {}

    for p in pairs:
        a = p["col_1"]
        b = p["col_2"]

        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)

    return graph


# --------------------------------------------------
# AUTO FEATURE PRUNING PLANNER
# --------------------------------------------------
def pruning_plan(df, redundancy_threshold=0.95):
    """
    Recommend which features to drop.
    Strategy:
        drop one feature from each highly correlated pair
        prefer keeping lower missing or higher variance
    """

    num = numeric_nonconstant(df)
    redundant = redundant_features(df, redundancy_threshold)

    drop = set()

    for r in redundant:

        a = r["feature_a"]
        b = r["feature_b"]

        if a in drop or b in drop:
            continue

        var_a = num[a].var()
        var_b = num[b].var()

        # drop lower variance feature
        if var_a < var_b:
            drop.add(a)
        else:
            drop.add(b)

    return list(drop)
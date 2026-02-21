import pandas as pd
from pathlib import Path
from .plan_schema import PreprocessPlan, Step


def _safe_colnames(df, names):
    return [c for c in names if c in df.columns]


def build_plan(dataset_path: str, column_profiles: list) -> PreprocessPlan:
    dataset_path = str(Path(dataset_path))
    dataset_name = Path(dataset_path).stem

    df = pd.read_csv(dataset_path)

    plan = PreprocessPlan(
        dataset_path=dataset_path,
        dataset_name=dataset_name
    )

    # ---------------------------------
    # 1. drop identifiers
    # ---------------------------------
    identifier_cols = [
        c["column_name"]
        for c in column_profiles
        if c.get("semantic_type") == "identifier"
    ]

    identifier_cols = _safe_colnames(df, identifier_cols)

    if identifier_cols:
        plan.add_step(Step(
            step_type="drop_columns",
            columns=identifier_cols,
            reason="identifier columns"
        ))

    # ---------------------------------
    # 2. missing value handling
    # ---------------------------------
    numeric_missing = []
    categorical_missing = []

    for c in column_profiles:
        if c.get("missing_pct", 0) > 0:
            if c.get("technical_type", "").startswith(("int", "float")):
                numeric_missing.append(c["column_name"])
            else:
                categorical_missing.append(c["column_name"])

    numeric_missing = _safe_colnames(df, numeric_missing)
    categorical_missing = _safe_colnames(df, categorical_missing)

    if numeric_missing:
        plan.add_step(Step(
            step_type="impute_numeric_median",
            columns=numeric_missing,
            reason="structural missing handling"
        ))

    if categorical_missing:
        plan.add_step(Step(
            step_type="impute_categorical_mode",
            columns=categorical_missing,
            reason="structural missing handling"
        ))

    # ---------------------------------
    # 3. skew stabilization
    # ---------------------------------
    log_candidates = [
        c["column_name"]
        for c in column_profiles
        if c.get("transform_hint") == "log_candidate"
    ]

    log_candidates = _safe_colnames(df, log_candidates)

    if log_candidates:
        plan.add_step(Step(
            step_type="log_transform",
            columns=log_candidates,
            reason="skew stabilization"
        ))

    # ---------------------------------
    # 4. deferred operations
    # ---------------------------------
    # plan.deferred_model_dependent.extend([
    #     "scaling",
    #     "encoding_strategy_selection",
    #     "feature_selection",
    #     "decorrelation",
    #     "representation_learning"
    # ])

        # ---------------------------------
    # 4. column-level deferred operations
    # ---------------------------------
    for c in column_profiles:

        col = c["column_name"]

        # encoding needed
        if c.get("encoding_required"):
            plan.add_deferred(
                column=col,
                strategy="encoding",
                reason=c.get("semantic_type", "categorical")
            )

        # scaling needed
        if c.get("modeling_hint") == "scaling":
            plan.add_deferred(
                column=col,
                strategy="scaling",
                reason="numeric_continuous"
            )

        # NLP pipeline later
        if c.get("modeling_hint") == "nlp":
            plan.add_deferred(
                column=col,
                strategy="text_vectorization",
                reason="text_freeform"
            )

        # high correlation (model dependent decision)
        if c.get("correlation_strength", 0) and c["correlation_strength"] > 0.8:
            plan.add_deferred(
                column=col,
                strategy="correlation_resolution",
                reason="highly_correlated"
            )

    return plan

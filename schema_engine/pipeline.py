from pathlib import Path

from .loader import load_table
from .profiler import profile_dataframe
from .deterministic import deterministic_role, Role
from .ambiguity import is_ambiguous
from .llm_resolver import resolve_with_llm
from .exporter import export_schema_result, export_user_inputs


# --------------------------------------------------
# VALIDATE USER INPUT
# --------------------------------------------------
def _validate_user_inputs(df, categorical_columns, target_column):

    dataset_columns = set(df.columns)

    if categorical_columns is not None:
        missing = set(categorical_columns) - dataset_columns
        if missing:
            raise ValueError(
                f"Categorical columns not found in dataset: {missing}"
            )

    if target_column is not None and target_column not in dataset_columns:
        raise ValueError(
            f"Target column not found in dataset: {target_column}"
        )


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def run_schema_inference(
    data_path,
    categorical_columns=None,
    target_column=None,
):
    """
    data_path : dataset file
    categorical_columns : list[str] | None
    target_column : str | None
    """

    df = load_table(data_path)

    if categorical_columns is None:
        categorical_columns = []

    _validate_user_inputs(df, categorical_columns, target_column)

    profiles = profile_dataframe(df)

    results = {}

    for col, prof in profiles.items():

        # ----------------------------
        # USER TARGET
        # ----------------------------
        if col == target_column:
            results[col] = {
                "role": Role.TARGET,
                "confidence": 1.0,
                "sample": prof.sample_values,
            }
            continue

        # ----------------------------
        # USER CATEGORICAL
        # ----------------------------
        if col in categorical_columns:
            results[col] = {
                "role": Role.CATEGORICAL_NOMINAL,
                "confidence": 0.99,
                "sample": prof.sample_values,
            }
            continue

        # ----------------------------
        # AUTO INFERENCE
        # ----------------------------
        det_role, det_conf = deterministic_role(prof)

        llm_role = llm_conf = None
        if is_ambiguous(det_role, det_conf):
            llm_role, llm_conf = resolve_with_llm(col, prof, det_role)

        final_role = det_role
        final_conf = det_conf

        if llm_role and llm_conf and llm_conf > det_conf:
            final_role = llm_role
            final_conf = llm_conf

        results[col] = {
            "role": final_role,
            "confidence": float(final_conf),
            "sample": prof.sample_values,
        }

    final_output = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "target": target_column,
        "columns": results,
    }

    # ----------------------------
    # EXPORT (APPEND MODE)
    # ----------------------------
    export_schema_result(data_path, final_output)

    export_user_inputs(data_path, categorical_columns, target_column)

    return final_output
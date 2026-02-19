from pathlib import Path
import json
import pandas as pd

from .diagnostics import pre_clean_checks, post_clean_checks
from .cleaning_engine import execute_cleaning
from .metadata import generate_metadata
from .fingerprint import dataset_hash
from .audit import record

# NEW
from .unified_diagnostics import run_unified_diagnostics


DATA_DIR = Path("data")


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


# --------------------------------------------------
# SAVE CLEANED DATASET
# --------------------------------------------------

def _save_cleaned_dataset(
    df: pd.DataFrame,
    dataset_id: str,
    original_name: str
) -> Path:

    DATA_DIR.mkdir(exist_ok=True)

    output_path = DATA_DIR / f"{original_name}_cleaned_{dataset_id}.csv"
    df.to_csv(output_path, index=False)

    return output_path


# --------------------------------------------------
# SAVE METADATA JSONL
# --------------------------------------------------

def _save_metadata_jsonl(
    metadata: dict,
    dataset_id: str,
    original_name: str
) -> Path:

    DATA_DIR.mkdir(exist_ok=True)

    file_path = DATA_DIR / f"{original_name}_metdata.jsonl"

    record_data = {
        "dataset_id": dataset_id,
        **metadata
    }

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record_data) + "\n")

    return file_path


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run_cleaning_pipeline(
    file_path: str,
    target_column: str | None = None
):
    """
    FULL INDUSTRIAL CLEANING PIPELINE

    Steps:
    load
    fingerprint
    unified diagnostics
    auto fixes
    standard cleaning diagnostics
    save cleaned dataset
    post-clean validation
    metadata generation
    save metadata
    """

    path = Path(file_path)

    # ==================================================
    # LOAD DATASET
    # ==================================================

    df = _load_dataset(path)

    dataset_id = dataset_hash(df)
    record(dataset_id, "dataset_loaded", {"file": path.name})


    # ==================================================
    # UNIFIED DIAGNOSTICS (NEW CORE ENGINE)
    # ==================================================

    diag = run_unified_diagnostics(df, target_column)

    auto_fix_results = diag["auto_fixable"]
    policy_results = diag["policy_required"]
    info_results = diag["informational"]

    record(dataset_id, "unified_diagnostics_completed", {
        "auto_fixable": len(auto_fix_results),
        "policy_required": len(policy_results),
        "informational": len(info_results)
    })


    # ==================================================
    # CONVERT AUTO FIX RESULTS → CLEANING ACTIONS
    # ==================================================

    auto_fix_actions = []

    for r in auto_fix_results:
        if r.recommended_action:
            auto_fix_actions.append({
                "column": r.column,
                "action": r.recommended_action
            })


    # ==================================================
    # EXECUTE AUTO FIX CLEANING
    # ==================================================

    if auto_fix_actions:
        df = execute_cleaning(df, dataset_id, auto_fix_actions)
        record(dataset_id, "auto_fix_cleaning_executed", {
            "actions": len(auto_fix_actions)
        })


    # ==================================================
    # STANDARD CLEANING DIAGNOSTICS (existing logic)
    # ==================================================

    pre_report = pre_clean_checks(df, target_column)
    record(dataset_id, "pre_clean_completed", {"steps": len(pre_report)})

    df = execute_cleaning(df, dataset_id, pre_report)


    # ==================================================
    # SAVE CLEANED DATASET
    # ==================================================

    cleaned_path = _save_cleaned_dataset(df, dataset_id, path.stem)
    record(dataset_id, "cleaned_dataset_saved", {"path": str(cleaned_path)})


    # ==================================================
    # POST CLEAN VALIDATION
    # ==================================================

    post_report = post_clean_checks(df)
    record(dataset_id, "post_clean_completed", post_report)


    # ==================================================
    # METADATA GENERATION
    # ==================================================

    metadata = generate_metadata(df, target_column)

    # ADD DIAGNOSTIC INTELLIGENCE (NEW)
    metadata["auto_fixes_applied"] = [r.__dict__ for r in auto_fix_results]
    metadata["policy_decisions_required"] = [r.__dict__ for r in policy_results]
    metadata["informational_diagnostics"] = [r.__dict__ for r in info_results]

    record(dataset_id, "metadata_generated", {
        "policy_flags": len(policy_results),
        "info_flags": len(info_results)
    })


    # ==================================================
    # SAVE METADATA JSONL
    # ==================================================

    metadata_log_path = _save_metadata_jsonl(
        metadata,
        dataset_id,
        path.stem
    )

    record(dataset_id, "metadata_saved", {
        "path": str(metadata_log_path)
    })


    # ==================================================
    # RETURN RESULTS
    # ==================================================

    return {
        "dataset_id": dataset_id,
        "cleaned_file_path": str(cleaned_path),
        "metadata_file_path": str(metadata_log_path),
        "df": df,
        "post_clean_report": post_report,
        "metadata": metadata
    }
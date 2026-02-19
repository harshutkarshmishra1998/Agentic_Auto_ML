from pathlib import Path
import json
import pandas as pd

from .diagnostics import pre_clean_checks, post_clean_checks
from .cleaning_engine import execute_cleaning
from .metadata import generate_metadata
from .fingerprint import dataset_hash
from .audit import record


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
    Full cleaning + metadata pipeline

    Steps:
    load → fingerprint → pre-clean → clean →
    save cleaned → post-clean → metadata → save metadata
    """

    path = Path(file_path)

    # ---------- LOAD ----------
    df = _load_dataset(path)

    dataset_id = dataset_hash(df)
    record(dataset_id, "dataset_loaded", {"file": path.name})

    # ---------- PRE CLEAN ----------
    pre_report = pre_clean_checks(df)
    record(dataset_id, "pre_clean_completed", {"steps": len(pre_report)})

    # ---------- CLEAN ----------
    df = execute_cleaning(df, dataset_id, pre_report)

    # ---------- SAVE CLEANED ----------
    cleaned_path = _save_cleaned_dataset(df, dataset_id, path.stem)
    record(dataset_id, "cleaned_dataset_saved", {"path": str(cleaned_path)})

    # ---------- POST CLEAN ----------
    post_report = post_clean_checks(df)
    record(dataset_id, "post_clean_completed", post_report)

    # ---------- METADATA ----------
    metadata = generate_metadata(df, target_column)
    record(dataset_id, "metadata_generated", metadata)

    # ---------- SAVE METADATA ----------
    metadata_log_path = _save_metadata_jsonl(
        metadata,
        dataset_id,
        path.stem
    )

    record(dataset_id, "metadata_saved", {
        "path": str(metadata_log_path)
    })

    # ---------- RETURN ----------
    return {
        "dataset_id": dataset_id,
        "cleaned_file_path": str(cleaned_path),
        "metadata_file_path": str(metadata_log_path),
        "df": df,
        "pre_clean_report": pre_report,
        "post_clean_report": post_report,
        "metadata": metadata
    }
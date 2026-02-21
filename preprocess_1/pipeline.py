import json
import uuid
from pathlib import Path

from .planner import build_plan
from .executor import execute_plan
from .logger import append_log


COLUMN_INSPECTION_PATH = Path("data/column_inspection.jsonl")
OUTPUT_LOG_PATH = Path("data/preprocesses_1.jsonl")
DATA_DIR = Path("data")


def _unique_suffix():
    """Short unique id for file versioning"""
    return uuid.uuid4().hex[:8]


# -------------------------------------------------
# main exposed function
# -------------------------------------------------
def run_preprocess_1(last_n: int):
    """
    Runs model-independent preprocessing for last n inspection entries.

    All outputs stored ONLY in data/
    Each output gets unique suffix to prevent overwrite.
    Output file path stored in preprocesses_1.jsonl
    """

    if not COLUMN_INSPECTION_PATH.exists():
        raise FileNotFoundError("data/column_inspection.jsonl not found")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(COLUMN_INSPECTION_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    records = records[-last_n:]

    results = []

    for rec in records:
        dataset_path = rec["dataset_file_path"]
        column_profiles = rec["column_profiles"]

        # -----------------------------
        # PLAN
        # -----------------------------
        plan = build_plan(dataset_path, column_profiles)

        # -----------------------------
        # EXECUTE
        # -----------------------------
        df, plan = execute_plan(plan)

        dataset_name = Path(dataset_path).stem
        uid = _unique_suffix()

        # -----------------------------
        # SAVE (unique version)
        # -----------------------------
        output_csv = DATA_DIR / f"{dataset_name}_preprocessed_1_{uid}.csv"
        df.to_csv(output_csv, index=False)

        # -----------------------------
        # LOG (includes output file info)
        # -----------------------------
        append_log(plan, output_csv, OUTPUT_LOG_PATH)

        results.append({
            "dataset": dataset_name,
            "output_csv": str(output_csv)
        })

    return results
from pathlib import Path
import json
from datetime import datetime


def export_schema_result(
    dataset_path,
    schema_result,
):
    """
    Append one JSON line per dataset classification.

    Output location:
        project_root/data/data_classification.jsonl
    """

    dataset_path = Path(dataset_path).resolve()

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = data_dir / "data_classification.jsonl"

    # ------------------------------
    # build feature mapping
    # ------------------------------
    feature_mapping = {}

    for col, info in schema_result["columns"].items():
        feature_mapping[col] = {
            "role": info["role"],
            "confidence": info["confidence"],
        }

    # ------------------------------
    # single dataset record
    # ------------------------------
    record = {
        "dataset_file_path": str(dataset_path),
        "dataset_file_name": dataset_path.name,
        "n_rows": schema_result["n_rows"],
        "n_columns": schema_result["n_columns"],
        "target_column": schema_result["target"],
        "feature_mapping": feature_mapping,
    }

    # ------------------------------
    # append JSONL
    # ------------------------------
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_user_inputs(
    data_path,
    categorical_columns=None,
    target_column=None,
):
    """
    Append user schema inputs to project_root/data/user_input.jsonl
    """

    # ---------------- normalize inputs ----------------
    data_path = Path(data_path).resolve()

    if categorical_columns is None:
        categorical_columns = []

    if target_column is None:
        target_column = []
    elif isinstance(target_column, str):
        target_column = [target_column]

    # ---------------- detect project root ----------------
    # assumes script is somewhere inside project
    project_root = Path(__file__).resolve().parents[1]

    output_dir = project_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "user_input.jsonl"

    # ---------------- build record ----------------
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_name": data_path.name,
        "dataset_path": str(data_path),
        "categorical_columns": list(categorical_columns),
        "target_columns": list(target_column),
    }

    # ---------------- append JSONL ----------------
    with output_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":")))
        f.write("\n")

    print(f"User input appended → {output_file}")
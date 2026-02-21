import json
from pathlib import Path

COLUMN_INSPECTION_PATH = Path("data/column_inspection.jsonl")
PREPROCESS_LOG_PATH = Path("data/preprocesses_1.jsonl")
CLASSIFICATION_PATH = Path("data/data_classification.jsonl")


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def extract_preprocessing(pre_record):
    return {
        "steps": pre_record.get("steps", []),
        "deferred": pre_record.get("deferred", []),
        "output_file_path": pre_record.get("output_file_path")
    }


def load_datasets(last_n):

    preprocess = read_jsonl(PREPROCESS_LOG_PATH)[-last_n:]
    columns = read_jsonl(COLUMN_INSPECTION_PATH)
    classes = read_jsonl(CLASSIFICATION_PATH)

    datasets = []

    for pre in preprocess:

        path = pre["dataset_path"]

        col = next((r for r in columns if r["dataset_file_path"] == path), None)
        cls = next((r for r in classes if r["dataset_file_path"] == path), None)

        if not col or not cls:
            continue

        datasets.append({
            "dataset_name": pre["dataset"],
            "raw_dataset_path": path,
            "preprocessed_file_path": pre["output_file_path"],
            "preprocessing": extract_preprocessing(pre),
            "column_profiles": col["column_profiles"],
            "target_column": cls.get("target_column"),
            "n_rows": cls["n_rows"],
            "n_columns": cls["n_columns"],
            "target_missing_ratio": cls.get("target_missing_ratio", 0),
            "overall_missing_ratio": cls.get("overall_missing_ratio", 0)
        })

    return datasets
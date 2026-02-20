import json
from pathlib import Path


def get_semantic_mapping(dataset_path):
    """
    Find semantic classification for dataset from data_classification.jsonl
    """

    dataset_path = str(Path(dataset_path).resolve())

    project_root = Path(__file__).resolve().parents[1]
    log_path = project_root / "data" / "data_classification.jsonl"

    if not log_path.exists():
        raise FileNotFoundError("data_classification.jsonl not found")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["dataset_file_path"] == dataset_path:
                return rec["feature_mapping"], rec["target_column"]

    raise ValueError("Dataset not found in data_classification.jsonl")
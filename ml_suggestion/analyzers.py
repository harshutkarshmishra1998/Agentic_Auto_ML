import json
from pathlib import Path
from typing import List
from .schemas import DatasetFingerprint
from .config import HIGH_CORRELATION_THRESHOLD


# ---------- METADATA LOADER ----------

def load_latest_metadata_entries(path: Path, last_n: int) -> List[dict]:
    with open(path, "r") as f:
        lines = f.readlines()

    selected = lines[-last_n:]
    return [json.loads(line) for line in selected]


# ---------- DATASET RESOLVER ----------

def resolve_cleaned_dataset_path(data_dir: Path, dataset_id: str) -> Path:
    extensions = ["csv", "xlsx", "xls"]

    for ext in extensions:
        matches = list(data_dir.glob(f"*{dataset_id}.{ext}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Cleaned dataset not found for id={dataset_id}")


# ---------- FINGERPRINT ----------

# def build_dataset_fingerprint(meta: dict) -> DatasetFingerprint:
#     return DatasetFingerprint(
#         dataset_id=meta["dataset_id"],
#         n_rows=meta.get("n_rows", 0),
#         n_features=meta.get("n_features", 0),
#         numeric_ratio=meta.get("numeric_ratio", 0.0),
#         categorical_ratio=meta.get("categorical_ratio", 0.0),
#         missing_ratio=meta.get("missing_ratio", 0.0),
#         target_type=meta.get("target_type", "unknown"),
#         class_balance=meta.get("class_balance"),
#         feature_correlation=meta.get("feature_correlation", 0.0),
#         complexity_score=meta.get("complexity_score", 0.0),
#     )

def build_dataset_fingerprint(meta: dict) -> DatasetFingerprint:

    n_rows = meta.get("n_rows", 0)
    n_features = meta.get("n_features", 0)

    numeric_ratio = (
        meta.get("numeric_ratio")
        or meta.get("num_numeric_features_ratio")
        or meta.get("numeric_feature_ratio")
        or 0.0
    )

    categorical_ratio = (
        meta.get("categorical_ratio")
        or meta.get("num_categorical_features_ratio")
        or meta.get("categorical_feature_ratio")
        or 0.0
    )

    # ✅ FALLBACK STRUCTURE INFERENCE
    if numeric_ratio == 0 and categorical_ratio == 0 and n_features > 0:
        numeric_ratio = 0.5
        categorical_ratio = 0.5

    missing_ratio = (
        meta.get("missing_ratio")
        or meta.get("missing_percentage")
        or 0.0
    )

    target_type = (
        meta.get("target_type")
        or meta.get("problem_type")
        or meta.get("task_type")
        or meta.get("learning_type")
        or "unknown"
    )

    class_balance = (
        meta.get("class_balance")
        or meta.get("minority_class_ratio")
        or None
    )

    feature_correlation = (
        meta.get("feature_correlation")
        or meta.get("avg_feature_correlation")
        or 0.0
    )

    complexity_score = (
        meta.get("complexity_score")
        or meta.get("dataset_complexity")
        or 0.0
    )

    return DatasetFingerprint(
        dataset_id=meta["dataset_id"],
        n_rows=n_rows,
        n_features=n_features,
        numeric_ratio=float(numeric_ratio),
        categorical_ratio=float(categorical_ratio),
        missing_ratio=float(missing_ratio),
        target_type=str(target_type),
        class_balance=class_balance,
        feature_correlation=float(feature_correlation),
        complexity_score=float(complexity_score),
    )


# ---------- TASK INFERENCE ----------

# def infer_task_type(fp: DatasetFingerprint) -> str:
#     if fp.target_type in ["binary", "multiclass"]:
#         return "classification"
#     if fp.target_type == "continuous":
#         return "regression"
#     if fp.target_type == "none":
#         return "clustering"
#     return "unknown"

def infer_task_type(fp: DatasetFingerprint) -> str:
    t = fp.target_type.lower()

    if "class" in t or "binary" in t or "multiclass" in t:
        return "classification"

    if "regress" in t or "continuous" in t or "numeric" in t:
        return "regression"

    if "cluster" in t or "unsupervised" in t or "none" in t:
        return "clustering"

    # SAFE fallback
    return "classification"


# ---------- RISK DETECTION ----------

def detect_statistical_risks(fp: DatasetFingerprint):
    risks = []

    if fp.feature_correlation > HIGH_CORRELATION_THRESHOLD:
        risks.append("high_multicollinearity")

    if fp.class_balance is not None and fp.class_balance < 0.2:
        risks.append("severe_class_imbalance")

    if fp.n_features > fp.n_rows:
        risks.append("high_dimensionality")

    return risks
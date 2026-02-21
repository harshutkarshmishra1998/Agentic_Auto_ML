from typing import Optional, List, Dict


def detect_problem_type(target_column: Optional[str], column_profiles: List[Dict]):

    if target_column is None:
        return "unsupervised", 1.0

    for c in column_profiles:
        if c["column_name"] == target_column:

            semantic = c.get("semantic_type")

            if semantic and semantic.startswith("categorical"):
                return "classification", 0.95

            if semantic == "numeric_continuous":
                return "regression", 0.95

    return "unknown", 0.3


def detect_semi_supervised(dataset):

    if not dataset.get("target_column"):
        return False, 0.0

    if dataset.get("target_missing_ratio", 0) > 0.4:
        return True, 0.9

    return False, 0.0
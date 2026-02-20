from pathlib import Path
from typing import List

from .analyzers import (
    load_latest_metadata_entries,
    build_dataset_fingerprint,
    infer_task_type,
    detect_statistical_risks,
    resolve_cleaned_dataset_path,
)
from .model_registry import get_eligible_models
from .scoring import compute_model_scores, apply_risk_penalties, normalize_scores
from .llm_advisor import maybe_refine_with_llm
from .formatter import build_ranking_output

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("data")
METADATA_FILE = PROJECT_ROOT / "data/data_metdata.jsonl"


def run_ml_suggestion(last_n: int = 1, use_llm: bool = False):
    """
    Main execution engine.
    Reads last N metadata entries and produces ranking per dataset.
    """

    metadata_entries = load_latest_metadata_entries(METADATA_FILE, last_n)

    results = []

    for meta in metadata_entries:
        fingerprint = build_dataset_fingerprint(meta)

        cleaned_path = resolve_cleaned_dataset_path(
            DATA_DIR, fingerprint.dataset_id
        )

        task_type = infer_task_type(fingerprint)
        risks = detect_statistical_risks(fingerprint)

        candidate_models = get_eligible_models(fingerprint, task_type)

        raw_scores = compute_model_scores(fingerprint, candidate_models)
        penalized_scores = apply_risk_penalties(raw_scores, risks)
        normalized_scores = normalize_scores(penalized_scores)

        final_scores = maybe_refine_with_llm(
            fingerprint=fingerprint,
            task_type=task_type,
            scores=normalized_scores,
            use_llm=use_llm,
        )

        result = build_ranking_output(
            fingerprint=fingerprint,
            task_type=task_type,
            scores=final_scores,
            risks=risks,
            cleaned_dataset_path=str(cleaned_path),
        )

        results.append(result)

        # print("FINGERPRINT:", fingerprint)
        # print("RAW SCORES:", normalized_scores)

    return result
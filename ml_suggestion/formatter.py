from .schemas import RankingResult
from dataclasses import asdict
import json
from pathlib import Path

OUTPUT_FILE = Path("data/ml_suggestions.jsonl")


def append_ml_suggestion(result):
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def build_ranking_output(
    fingerprint,
    task_type,
    scores,
    risks,
    cleaned_dataset_path,
):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    result = RankingResult(
        dataset_id=fingerprint.dataset_id,
        task_type=task_type,
        cleaned_dataset_path=cleaned_dataset_path,
        model_rankings=[{"model": m, "score": s} for m, s in ranked],
        risks=risks,
        confidence=max(scores.values()) if scores else 0.0,
    )

    # ✅ append to jsonl
    append_ml_suggestion(result)

    return result
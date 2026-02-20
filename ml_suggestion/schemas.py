from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DatasetFingerprint:
    dataset_id: str
    n_rows: int
    n_features: int
    numeric_ratio: float
    categorical_ratio: float
    missing_ratio: float
    target_type: str
    class_balance: Optional[float]
    feature_correlation: float
    complexity_score: float


@dataclass
class RankingResult:
    dataset_id: str
    task_type: str
    cleaned_dataset_path: str
    model_rankings: List[Dict]
    risks: List[str]
    confidence: float
# from typing import Dict
# from .schemas import DatasetFingerprint
# from .config import RISK_PENALTIES


# def compute_model_scores(fp: DatasetFingerprint, models: Dict):
#     scores = {}

#     for name, profile in models.items():
#         score = 0.5

#         if profile["nonlinear"] and fp.complexity_score > 0.3:
#             score += 0.2

#         if profile["categorical"] and fp.categorical_ratio > 0.4:
#             score += 0.15

#         scores[name] = score

#     return scores


# def apply_risk_penalties(scores: Dict, risks):
#     adjusted = scores.copy()

#     for risk in risks:
#         penalty = RISK_PENALTIES.get(risk, 1.0)
#         for m in adjusted:
#             adjusted[m] *= penalty

#     return adjusted


# def normalize_scores(scores: Dict):
#     total = sum(scores.values())
#     if total == 0:
#         return scores
#     return {k: v / total for k, v in scores.items()}

from typing import Dict
from .schemas import DatasetFingerprint
from .config import RISK_PENALTIES


# -----------------------------------------------------
# MAIN SUITABILITY SCORING
# -----------------------------------------------------

def compute_model_scores(fp: DatasetFingerprint, models: Dict):

    scores = {}

    for name, profile in models.items():

        score = 0.5  # baseline

        # ---------------------------------
        # Nonlinearity handling
        # ---------------------------------
        if fp.complexity_score > 0.4 and profile["nonlinear"]:
            score += 0.15

        if fp.complexity_score < 0.2 and profile.get("interpretable"):
            score += 0.05

        # ---------------------------------
        # Categorical data handling
        # ---------------------------------
        if fp.categorical_ratio > 0.4 and profile["categorical"]:
            score += 0.1

        # ---------------------------------
        # High dimensionality
        # ---------------------------------
        if fp.n_features > 100 and profile.get("high_dim"):
            score += 0.1

        # ---------------------------------
        # Dataset size vs training cost
        # ---------------------------------
        if fp.n_rows > 100_000 and profile.get("fast_train"):
            score += 0.1

        # ---------------------------------
        # Noise robustness
        # ---------------------------------
        if fp.missing_ratio > 0.1 and profile.get("robust_noise"):
            score += 0.1

        # ---------------------------------
        # Class imbalance
        # ---------------------------------
        if fp.class_balance is not None and fp.class_balance < 0.2:
            if profile.get("imbalance_ok"):
                score += 0.1

        scores[name] = score

    return scores


# -----------------------------------------------------
# RISK PENALTIES
# -----------------------------------------------------

def apply_risk_penalties(scores: Dict, risks):

    adjusted = scores.copy()

    for risk in risks:
        penalty = RISK_PENALTIES.get(risk, 1.0)
        for m in adjusted:
            adjusted[m] *= penalty

    return adjusted


# -----------------------------------------------------
# NORMALIZE
# -----------------------------------------------------

def normalize_scores(scores: Dict):

    total = sum(scores.values())
    if total == 0:
        return scores

    return {k: v / total for k, v in scores.items()}
# from typing import Dict
# from .schemas import DatasetFingerprint


# MODEL_PROFILES = {
#     "xgboost": dict(nonlinear=True, categorical=True, small_data=True),
#     "lightgbm": dict(nonlinear=True, categorical=True, small_data=True),
#     "random_forest": dict(nonlinear=True, categorical=True, small_data=True),
#     "logistic_regression": dict(nonlinear=False, categorical=False, small_data=True),
#     "svm_rbf": dict(nonlinear=True, categorical=False, small_data=True),
#     "neural_network": dict(nonlinear=True, categorical=False, small_data=False),
# }


# def get_eligible_models(fingerprint: DatasetFingerprint, task_type: str) -> Dict:
#     eligible = {}

#     for name, profile in MODEL_PROFILES.items():
#         if fingerprint.n_rows < 20000 and not profile["small_data"]:
#             continue
#         eligible[name] = profile

#     return eligible

from typing import Dict
from .schemas import DatasetFingerprint


# -----------------------------------------------------
# MODEL CAPABILITY REGISTRY
# -----------------------------------------------------
# nonlinear      → captures complex interactions
# categorical    → handles categorical data well
# small_data     → stable with small datasets
# interpretable  → human explainable
# fast_train     → computationally efficient
# robust_noise   → stable with noisy data
# high_dim       → works with many features
# imbalance_ok   → handles class imbalance well
#


MODEL_PROFILES = {

    # -----------------------------
    # Gradient Boosting Family
    # -----------------------------
    "xgboost": dict(nonlinear=True, categorical=True, small_data=True, interpretable=False, fast_train=True, robust_noise=True, high_dim=True, imbalance_ok=True),
    "lightgbm": dict(nonlinear=True, categorical=True, small_data=True, interpretable=False, fast_train=True, robust_noise=True, high_dim=True, imbalance_ok=True),
    "catboost": dict(nonlinear=True, categorical=True, small_data=True, interpretable=False, fast_train=True, robust_noise=True, high_dim=True, imbalance_ok=True),
    "gradient_boosting": dict(nonlinear=True, categorical=False, small_data=True, interpretable=False, fast_train=False, robust_noise=True, high_dim=False, imbalance_ok=True),

    # -----------------------------
    # Bagging / Tree Ensembles
    # -----------------------------
    "random_forest": dict(nonlinear=True, categorical=True, small_data=True, interpretable=False, fast_train=True, robust_noise=True, high_dim=True, imbalance_ok=True),
    "extra_trees": dict(nonlinear=True, categorical=True, small_data=True, interpretable=False, fast_train=True, robust_noise=True, high_dim=True, imbalance_ok=True),
    "decision_tree": dict(nonlinear=True, categorical=True, small_data=True, interpretable=True, fast_train=True, robust_noise=False, high_dim=False, imbalance_ok=False),

    # -----------------------------
    # Linear Models
    # -----------------------------
    "logistic_regression": dict(nonlinear=False, categorical=False, small_data=True, interpretable=True, fast_train=True, robust_noise=False, high_dim=True, imbalance_ok=True),
    "ridge_classifier": dict(nonlinear=False, categorical=False, small_data=True, interpretable=True, fast_train=True, robust_noise=False, high_dim=True, imbalance_ok=True),
    "linear_svm": dict(nonlinear=False, categorical=False, small_data=True, interpretable=False, fast_train=True, robust_noise=False, high_dim=True, imbalance_ok=True),

    # -----------------------------
    # Kernel Methods
    # -----------------------------
    "svm_rbf": dict(nonlinear=True, categorical=False, small_data=True, interpretable=False, fast_train=False, robust_noise=False, high_dim=False, imbalance_ok=True),
    "svm_poly": dict(nonlinear=True, categorical=False, small_data=True, interpretable=False, fast_train=False, robust_noise=False, high_dim=False, imbalance_ok=True),

    # -----------------------------
    # Instance Based
    # -----------------------------
    "knn": dict(nonlinear=True, categorical=False, small_data=True, interpretable=False, fast_train=True, robust_noise=False, high_dim=False, imbalance_ok=False),

    # -----------------------------
    # Probabilistic
    # -----------------------------
    "gaussian_nb": dict(nonlinear=False, categorical=False, small_data=True, interpretable=True, fast_train=True, robust_noise=False, high_dim=True, imbalance_ok=True),
    "bernoulli_nb": dict(nonlinear=False, categorical=True, small_data=True, interpretable=True, fast_train=True, robust_noise=False, high_dim=True, imbalance_ok=True),

    # -----------------------------
    # Neural
    # -----------------------------
    "mlp": dict(nonlinear=True, categorical=False, small_data=False, interpretable=False, fast_train=False, robust_noise=True, high_dim=True, imbalance_ok=True),
}


# -----------------------------------------------------
# ELIGIBILITY FILTER
# -----------------------------------------------------

def get_eligible_models(fingerprint: DatasetFingerprint, task_type: str) -> Dict:
    eligible = {}

    for name, profile in MODEL_PROFILES.items():

        # small dataset filter
        if fingerprint.n_rows < 20000 and not profile["small_data"]:
            continue

        # high dimensional data filter
        if fingerprint.n_features > 200 and not profile["high_dim"]:
            continue

        # heavy categorical dataset filter
        if fingerprint.categorical_ratio > 0.6 and not profile["categorical"]:
            continue

        eligible[name] = profile

    return eligible
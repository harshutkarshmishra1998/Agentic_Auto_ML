# from .problem_type_detector import detect_problem_type, detect_semi_supervised
# from .data_characteristics import compute_characteristics
# from .model_rules import rank_models, FUSION_WEIGHTS
# from .llm_reasoner import llm_model_selection
# from .utils import file_sha256, now_iso, new_experiment_id, SELECTOR_VERSION


# def build_llm_summary(dataset):

#     cp = dataset["column_profiles"]

#     return {
#         "dataset_name": dataset["dataset_name"],
#         "n_rows": dataset["n_rows"],
#         "n_features": dataset["n_columns"],
#         "target": dataset["target_column"],
#         "numeric_features": sum(1 for c in cp if "numeric" in c.get("semantic_type", "")),
#         "categorical_features": sum(1 for c in cp if "categorical" in c.get("semantic_type", "")),
#         "text_features": sum(1 for c in cp if c.get("semantic_type") == "text_freeform")
#     }


# def resolve_problem(rule_type, rule_conf, llm_type, llm_conf, semi_flag):

#     if rule_type == "unsupervised":
#         return "unsupervised"

#     if semi_flag:
#         return "semi_supervised"

#     if llm_conf > rule_conf + 0.25:
#         return llm_type

#     return rule_type


# def reconcile_preprocessing(llm_result, preprocessing):

#     suggested = llm_result.get("model_dependent_preprocessing", [])
#     applied = [s.get("type") for s in preprocessing.get("steps", [])]

#     missing = [s for s in suggested if s not in applied]

#     return {
#         "llm_suggested": suggested,
#         "missing_from_pipeline": missing
#     }

# def canonical_problem_type(problem_type: str | None):

#     if not problem_type:
#         return "unknown"

#     p = problem_type.lower()

#     if p in ("binary_classification", "multiclass_classification", "classification"):
#         return "classification"

#     if p in ("clustering", "unsupervised"):
#         return "unsupervised"

#     return p

# def build_model_plan(dataset):

#     experiment_id = new_experiment_id()

#     char = compute_characteristics(dataset)

#     rule_type, rule_conf = detect_problem_type(
#         dataset["target_column"],
#         dataset["column_profiles"]
#     )

#     semi_flag, _ = detect_semi_supervised(dataset)

#     llm_result = llm_model_selection(build_llm_summary(dataset))

#     llm_type = llm_result.get("problem_type", "unknown")
#     llm_conf = llm_result.get("problem_confidence", 0.5)

#     # final_problem = resolve_problem(rule_type, rule_conf, llm_type, llm_conf, semi_flag)
#     final_problem_raw = resolve_problem(rule_type, rule_conf, llm_type, llm_conf, semi_flag)
#     final_problem = canonical_problem_type(final_problem_raw)

#     ranking = rank_models(final_problem, char, llm_result)

#     primary_model = ranking[0]["model"] if ranking else None
#     top_k = [m["model"] for m in ranking[:3]]

#     preprocessing_recon = reconcile_preprocessing(llm_result, dataset["preprocessing"])

#     return {
#         "experiment_id": experiment_id,
#         "timestamp": now_iso(),
#         "model_selector_version": SELECTOR_VERSION,

#         "dataset": {
#             "name": dataset["dataset_name"],
#             "raw_path": dataset["raw_dataset_path"],
#             "preprocessed_path": dataset["preprocessed_file_path"],
#             "raw_hash": file_sha256(dataset["raw_dataset_path"]),
#             "preprocessed_hash": file_sha256(dataset["preprocessed_file_path"])
#         },

#         "schema": {
#             "target_column": dataset["target_column"],
#             "feature_counts": char["feature_type_counts"]
#         },

#         "preprocessing": dataset["preprocessing"],
#         "preprocessing_reconciliation": preprocessing_recon,

#         "problem_detection": {
#             "final_type": final_problem,
#             "rule_confidence": rule_conf,
#             "llm_confidence": llm_conf
#         },

#         "data_characteristics": char,

#         "model_selection": {
#             "primary_model": primary_model,
#             "top_k": top_k,
#             "ranked_models": ranking,
#             "fusion_weights": FUSION_WEIGHTS
#         },

#         "llm_analysis": llm_result
#     }

from .problem_type_detector import detect_problem_type, detect_semi_supervised
from .data_characteristics import compute_characteristics
from .model_rules import rank_models, FUSION_WEIGHTS
from .llm_reasoner import llm_model_selection
from .utils import file_sha256, now_iso, new_experiment_id, SELECTOR_VERSION


# =====================================================
# PROBLEM ONTOLOGY (EMBEDDED — NO EXTRA FILE)
# =====================================================

CANONICAL_PROBLEM_TYPES = {
    "classification",
    "regression",
    "unsupervised",
    "semi_supervised"
}

PROBLEM_TYPE_ALIASES = {

    # classification variants
    "binary_classification": "classification",
    "multiclass_classification": "classification",
    "multi_class_classification": "classification",
    "classification": "classification",

    # clustering / unsupervised variants
    "clustering": "unsupervised",
    "cluster_analysis": "unsupervised",
    "unsupervised_learning": "unsupervised",
    "unsupervised": "unsupervised",

    # regression variants
    "regression": "regression",

    # semi supervised
    "semi_supervised_learning": "semi_supervised",
    "semi_supervised": "semi_supervised"
}

PROBLEM_DEFAULT_METRICS = {
    "classification": ["accuracy", "f1", "roc_auc"],
    "regression": ["rmse", "mae", "r2"],
    "unsupervised": ["silhouette_score", "davies_bouldin"],
    "semi_supervised": ["accuracy", "f1"]
}

PROBLEM_TRAINING_MODE = {
    "classification": "supervised",
    "regression": "supervised",
    "unsupervised": "unsupervised",
    "semi_supervised": "semi_supervised"
}


def canonical_problem_type(problem_type: str | None) -> str:
    """
    Normalize any external problem description to canonical type.
    """

    if not problem_type:
        return "unknown"

    p = problem_type.lower().strip()

    if p in PROBLEM_TYPE_ALIASES:
        return PROBLEM_TYPE_ALIASES[p]

    if p in CANONICAL_PROBLEM_TYPES:
        return p

    return "unknown"


def validate_problem_configuration(problem_type: str, target_column):
    """
    Safety validation for logical consistency.
    """

    if problem_type == "unsupervised" and target_column:
        return "warning_target_present_for_unsupervised"

    if problem_type in ("classification", "regression") and not target_column:
        return "error_missing_target_for_supervised"

    return "ok"


def get_problem_metadata(problem_type: str):
    """
    Returns standardized execution metadata.
    """

    return {
        "canonical_type": problem_type,
        "training_mode": PROBLEM_TRAINING_MODE.get(problem_type),
        "default_metrics": PROBLEM_DEFAULT_METRICS.get(problem_type, [])
    }


# =====================================================
# LLM SUMMARY BUILDER
# =====================================================

def build_llm_summary(dataset):

    cp = dataset["column_profiles"]

    return {
        "dataset_name": dataset["dataset_name"],
        "n_rows": dataset["n_rows"],
        "n_features": dataset["n_columns"],
        "target": dataset["target_column"],
        "numeric_features": sum(1 for c in cp if "numeric" in c.get("semantic_type", "")),
        "categorical_features": sum(1 for c in cp if "categorical" in c.get("semantic_type", "")),
        "text_features": sum(1 for c in cp if c.get("semantic_type") == "text_freeform")
    }


# =====================================================
# ARBITRATION
# =====================================================

def resolve_problem(rule_type, rule_conf, llm_type, llm_conf, semi_flag):

    if rule_type == "unsupervised":
        return "unsupervised"

    if semi_flag:
        return "semi_supervised"

    if llm_conf > rule_conf + 0.25:
        return llm_type

    return rule_type


# =====================================================
# PREPROCESS RECONCILIATION
# =====================================================

def reconcile_preprocessing(llm_result, preprocessing):

    suggested = llm_result.get("model_dependent_preprocessing", {})
    applied = [s.get("type") for s in preprocessing.get("steps", [])]

    missing = []
    for model_name in suggested.keys():
        if model_name not in applied:
            missing.append(model_name)

    return {
        "llm_suggested": suggested,
        "missing_from_pipeline": missing
    }


# =====================================================
# MAIN MODEL PLAN BUILDER
# =====================================================

def build_model_plan(dataset):

    experiment_id = new_experiment_id()
    char = compute_characteristics(dataset)

    # ---------------- rule inference ----------------
    rule_type, rule_conf = detect_problem_type(
        dataset["target_column"],
        dataset["column_profiles"]
    )

    # ---------------- semi supervised ----------------
    semi_flag, _ = detect_semi_supervised(dataset)

    # ---------------- llm inference ----------------
    llm_result = llm_model_selection(build_llm_summary(dataset))
    llm_type = llm_result.get("problem_type", "unknown")
    llm_conf = llm_result.get("problem_confidence", 0.5)

    # ---------------- arbitration ----------------
    final_problem_raw = resolve_problem(
        rule_type, rule_conf,
        llm_type, llm_conf,
        semi_flag
    )

    # ---------------- ONTOLOGY NORMALIZATION ----------------
    final_problem = canonical_problem_type(final_problem_raw)

    # ---------------- VALIDATION ----------------
    configuration_status = validate_problem_configuration(
        final_problem,
        dataset["target_column"]
    )

    # ---------------- METADATA ----------------
    problem_meta = get_problem_metadata(final_problem)

    # ---------------- MODEL RANKING ----------------
    ranking = rank_models(final_problem, char, llm_result)

    primary_model = ranking[0]["model"] if ranking else None
    top_k = [m["model"] for m in ranking[:3]]

    preprocessing_recon = reconcile_preprocessing(
        llm_result,
        dataset["preprocessing"]
    )

    # =====================================================
    # FINAL EXPERIMENT MANIFEST
    # =====================================================

    return {

        "experiment_id": experiment_id,
        "timestamp": now_iso(),
        "model_selector_version": SELECTOR_VERSION,

        "dataset": {
            "name": dataset["dataset_name"],
            "raw_path": dataset["raw_dataset_path"],
            "preprocessed_path": dataset["preprocessed_file_path"],
            "raw_hash": file_sha256(dataset["raw_dataset_path"]),
            "preprocessed_hash": file_sha256(dataset["preprocessed_file_path"])
        },

        "schema": {
            "target_column": dataset["target_column"],
            "feature_counts": char["feature_type_counts"]
        },

        "problem_definition": {
            "canonical_type": final_problem,
            "training_mode": problem_meta["training_mode"],
            "default_metrics": problem_meta["default_metrics"],
            "configuration_status": configuration_status,
            "rule_confidence": rule_conf,
            "llm_confidence": llm_conf
        },

        "preprocessing": dataset["preprocessing"],
        "preprocessing_reconciliation": preprocessing_recon,

        "data_characteristics": char,

        "model_selection": {
            "primary_model": primary_model,
            "top_k": top_k,
            "ranked_models": ranking,
            "fusion_weights": FUSION_WEIGHTS
        },

        "llm_analysis": llm_result
    }
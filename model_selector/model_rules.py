# MODEL_REGISTRY = {
#     "random_forest": dict(classification=True, regression=True, clustering=False, nonlinear=True),
#     "xgboost": dict(classification=True, regression=True, clustering=False, nonlinear=True),
#     "lightgbm": dict(classification=True, regression=True, clustering=False, nonlinear=True),
#     "logistic_regression": dict(classification=True, regression=False, clustering=False, nonlinear=False),
#     "linear_regression": dict(classification=False, regression=True, clustering=False, nonlinear=False),
#     "kmeans": dict(classification=False, regression=False, clustering=True, nonlinear=False),
#     "dbscan": dict(classification=False, regression=False, clustering=True, nonlinear=True),
# }

# FUSION_WEIGHTS = {
#     "data": 0.6,
#     "llm": 0.4
# }


# def allowed_models(problem_type):

#     out = []

#     for m, caps in MODEL_REGISTRY.items():

#         if problem_type == "classification" and caps["classification"]:
#             out.append(m)

#         elif problem_type == "regression" and caps["regression"]:
#             out.append(m)

#         elif problem_type == "unsupervised" and caps["clustering"]:
#             out.append(m)

#         elif problem_type == "semi_supervised" and (caps["classification"] or caps["clustering"]):
#             out.append(m)

#     return out


# def data_score(model, char):

#     s = 0

#     if char["correlation_density"] > 0.5 and MODEL_REGISTRY[model]["nonlinear"]:
#         s += 1

#     if char["dimensionality_ratio"] > 0.3 and MODEL_REGISTRY[model]["nonlinear"]:
#         s += 1

#     if char["missing_ratio"] > 0.1:
#         s += 0.5

#     return s


# def llm_score(model, llm_ranked):

#     if model not in llm_ranked:
#         return 0

#     return 1 - (llm_ranked.index(model) / len(llm_ranked))


# def rank_models(problem_type, characteristics, llm_result):

#     allowed = allowed_models(problem_type)
#     llm_ranked = llm_result.get("recommended_models", [])

#     scored = []

#     for m in allowed:

#         d = data_score(m, characteristics)
#         l = llm_score(m, llm_ranked)

#         final = FUSION_WEIGHTS["data"] * d + FUSION_WEIGHTS["llm"] * l

#         scored.append({
#             "model": m,
#             "scores": {
#                 "data_fit": d,
#                 "llm_support": l
#             },
#             "final_score": final
#         })

#     scored.sort(key=lambda x: x["final_score"], reverse=True)

#     for i, m in enumerate(scored):
#         m["rank"] = i + 1

#     return scored

import re

MODEL_REGISTRY = {
    "random_forest": dict(classification=True, regression=True, clustering=False, nonlinear=True),
    "xgboost": dict(classification=True, regression=True, clustering=False, nonlinear=True),
    "lightgbm": dict(classification=True, regression=True, clustering=False, nonlinear=True),
    "logistic_regression": dict(classification=True, regression=False, clustering=False, nonlinear=False),
    "linear_regression": dict(classification=False, regression=True, clustering=False, nonlinear=False),
    "kmeans": dict(classification=False, regression=False, clustering=True, nonlinear=False),
    "dbscan": dict(classification=False, regression=False, clustering=True, nonlinear=True),
}

# --------------------------------------------------
# MODEL NAME NORMALIZATION
# --------------------------------------------------

MODEL_NAME_ALIASES = {

    # sklearn / LLM classifier names
    "randomforestclassifier": "random_forest",
    "random forest": "random_forest",
    "random forest classifier": "random_forest",

    "xgboostclassifier": "xgboost",
    "xgboost classifier": "xgboost",

    "lightgbmclassifier": "lightgbm",
    "lightgbm classifier": "lightgbm",

    "logisticregression": "logistic_regression",
    "logistic regression": "logistic_regression",

    "linearregression": "linear_regression",
    "linear regression": "linear_regression",

    # clustering naming
    "k-means": "kmeans",
    "kmeans": "kmeans",
    "k means": "kmeans",

    "dbscan": "dbscan",
    "hierarchical clustering": "dbscan",  # closest available
}


def canonical_model_name(name: str):
    """
    Normalize any external model naming to registry key.
    """

    if not name:
        return None

    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]+", "", name)  # remove punctuation
    name = name.strip()

    if name in MODEL_NAME_ALIASES:
        return MODEL_NAME_ALIASES[name]

    # already canonical?
    if name in MODEL_REGISTRY:
        return name

    return None


def normalize_llm_models(llm_models):
    """
    Convert LLM model list to registry canonical names.
    Remove unknown models safely.
    """

    out = []

    for m in llm_models:
        canon = canonical_model_name(m)
        if canon:
            out.append(canon)

    return out


# --------------------------------------------------
# SCORING
# --------------------------------------------------

FUSION_WEIGHTS = {
    "data": 0.6,
    "llm": 0.4
}


def allowed_models(problem_type):

    out = []

    for m, caps in MODEL_REGISTRY.items():

        if problem_type == "classification" and caps["classification"]:
            out.append(m)

        elif problem_type == "regression" and caps["regression"]:
            out.append(m)

        elif problem_type == "unsupervised" and caps["clustering"]:
            out.append(m)

        elif problem_type == "semi_supervised" and (caps["classification"] or caps["clustering"]):
            out.append(m)

    return out


def data_score(model, char):

    s = 0

    if char["correlation_density"] > 0.5 and MODEL_REGISTRY[model]["nonlinear"]:
        s += 1

    if char["dimensionality_ratio"] > 0.3 and MODEL_REGISTRY[model]["nonlinear"]:
        s += 1

    if char["missing_ratio"] > 0.1:
        s += 0.5

    return s


def llm_score(model, normalized_llm_ranked):

    if model not in normalized_llm_ranked:
        return 0

    return 1 - (normalized_llm_ranked.index(model) / len(normalized_llm_ranked))


# --------------------------------------------------
# FINAL RANK
# --------------------------------------------------

def rank_models(problem_type, characteristics, llm_result):

    allowed = allowed_models(problem_type)

    llm_models = llm_result.get("recommended_models", [])
    llm_ranked = normalize_llm_models(llm_models)

    scored = []

    for m in allowed:

        d = data_score(m, characteristics)
        l = llm_score(m, llm_ranked)

        final = FUSION_WEIGHTS["data"] * d + FUSION_WEIGHTS["llm"] * l

        scored.append({
            "model": m,
            "scores": {
                "data_fit": d,
                "llm_support": l
            },
            "final_score": final
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)

    for i, m in enumerate(scored):
        m["rank"] = i + 1

    return scored
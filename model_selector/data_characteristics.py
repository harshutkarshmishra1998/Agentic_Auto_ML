def compute_characteristics(dataset):

    cp = dataset["column_profiles"]

    numeric = sum(1 for c in cp if "numeric" in c.get("semantic_type", ""))
    categorical = sum(1 for c in cp if "categorical" in c.get("semantic_type", ""))
    text = sum(1 for c in cp if c.get("semantic_type") == "text_freeform")

    strong_corr = sum(
        1 for c in cp if (c.get("correlation_strength") or 0) > 0.7
    )

    return {
        "n_rows": dataset["n_rows"],
        "n_features": dataset["n_columns"],
        "feature_type_counts": {
            "numeric": numeric,
            "categorical": categorical,
            "text": text
        },
        "correlation_density": strong_corr / max(1, len(cp)),
        "missing_ratio": dataset.get("overall_missing_ratio", 0),
        "dimensionality_ratio": dataset["n_columns"] / max(1, dataset["n_rows"])
    }
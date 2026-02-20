from .loader import load_dataset
from .semantic_reader import get_semantic_mapping
from .column_profiler import *
from .exporter import export_column_inspection
from .feature_relationships import (
    correlation_pairs,
    redundant_features,
    derived_linear_relationships,
    feature_dependency_graph,
    pruning_plan
)

def run_data_understanding(dataset_path):

    df = load_dataset(dataset_path)
    semantic_map, target = get_semantic_mapping(dataset_path)

    column_records = []

    # --------------------------------
    # FEATURE RELATIONSHIPS
    # --------------------------------
    corr_pairs = correlation_pairs(df, min_abs_corr=0.0)
    redundant = redundant_features(df)
    derived = derived_linear_relationships(df)
    dependency_graph = feature_dependency_graph(df)
    drop_recommendations = pruning_plan(df)

    for col in df.columns:

        s = df[col]
        unique_ratio = s.nunique(dropna=True) / max(len(s), 1)

        sem = semantic_map[col]["role"] if col in semantic_map else "unknown"

        column_records.append({
            "column_name": col,
            "technical_type": str(s.dtype),
            "semantic_type": sem,
            "role": "target" if col == target else "feature",
            "cardinality_level": cardinality_level(unique_ratio),
            "missing_pct": float(s.isna().mean()),
            "missing_pattern": missing_pattern(s, df),
            "distribution_shape": distribution_shape(s),
            "outliers_present": outliers_present(s),
            "unique_ratio": float(unique_ratio),
            "encoding_required": encoding_required(sem),
            "time_dependent": sem == "datetime",
            "unit_scale": None,
            "text_complexity": text_complexity(s),
            "category_imbalance": category_imbalance(s),
            "correlation_strength": correlation_strength(col, df),
            "transform_hint": transform_hint(s),
            "modeling_hint": modeling_hint(sem),
            "data_quality_flags": None,
            "is_constant": is_constant(s),
        })

    export_column_inspection(dataset_path, 
        {"column_profiles": column_records,
        "correlation_pairs": corr_pairs,
        "redundant_features": redundant,
        "derived_relationships": derived,
        "dependency_graph": dependency_graph,
        "drop_recommendations": drop_recommendations})

    # print(dataset_path, column_records)

    return column_records
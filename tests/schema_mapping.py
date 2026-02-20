# schema_mapping.py

import json
import pandas as pd
from pathlib import Path


def extract_schema(metadata_path: str, data_path: str):
    """
    Map metadata categorical flags to dataset feature columns.
    Target column is excluded from categorical mapping.
    """

    metadata_path = Path(metadata_path) #type: ignore
    data_path = Path(data_path) #type: ignore

    if not metadata_path.exists(): #type: ignore
        raise FileNotFoundError(metadata_path)

    if not data_path.exists(): #type: ignore
        raise FileNotFoundError(data_path)

    # ---------- load metadata ----------
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    categorical_flags = meta.get("categorical_features")
    target_column = meta.get("target")

    if categorical_flags is None:
        raise ValueError("metadata missing 'categorical_features'")

    # ---------- load dataset ----------
    df = pd.read_csv(data_path)

    # remove columns that are fully empty
    df = df.dropna(axis=1, how="all")

    # normalize column names
    df.columns = df.columns.str.strip()

    dataset_columns = list(df.columns)

    # ---------- build feature column list ----------
    if target_column and target_column in dataset_columns:
        feature_columns = [c for c in dataset_columns if c != target_column]
    else:
        feature_columns = dataset_columns

    # ---------- validate alignment ----------
    if len(categorical_flags) != len(feature_columns):
        raise ValueError(
            "Metadata categorical flag count does not match feature columns.\n"
            f"flags = {len(categorical_flags)}\n"
            f"features (excluding target) = {len(feature_columns)}\n"
            f"target = {target_column}"
        )

    # ---------- map ----------
    categorical_columns = [
        col for col, is_cat in zip(feature_columns, categorical_flags) if is_cat
    ]

    return categorical_columns, target_column


if __name__ == "__main__":
    cats, target = extract_schema("uploaded_files/churn/metadata.json", "uploaded_files/churn/data.csv")
    print("CATEGORICAL_COLUMNS = ", cats)
    if target:
        # print("TARGET_COLUMN = \"", target, "\"")
        print(f'TARGET_COLUMN = "{target}"')
    else:
        print("TARGET_COLUMN = null")
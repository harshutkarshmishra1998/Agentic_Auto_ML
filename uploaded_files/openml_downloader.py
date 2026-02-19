"""
OpenML Dataset Downloader
Phase 0 of Agentic AutoML System

Downloads datasets from OpenML based on problem type
and stores them in local project data directory.

Execute the file:
python -m uploaded_files.openml_downloader --problem classification
python -m uploaded_files.openml_downloader --problem regression
python -m uploaded_files.openml_downloader --problem clustering

Features:
- Skip already downloaded datasets
- Continue searching until limit reached
- Dataset size filtering
- Largest datasets first (configurable)
- Saves CSV + metadata
"""

# from pathlib import Path
# import openml
# import pandas as pd


# # -----------------------------------------------------
# # Configuration
# # -----------------------------------------------------

# DATA_ROOT = Path("uploaded_files/")
# DATA_ROOT.mkdir(parents=True, exist_ok=True)


# # Map your ML testing needs to OpenML task types
# PROBLEM_TYPE_MAP = {
#     "classification": "Supervised Classification",
#     "regression": "Supervised Regression",
#     "clustering": "Unsupervised",
# }


# # -----------------------------------------------------
# # Core Functions
# # -----------------------------------------------------

# def search_datasets(
#     problem_type: str,
#     min_instances: int = 5000,
#     max_instances: int = 50000,
#     max_features: int = 100
# ) -> pd.DataFrame:
#     """
#     Search OpenML datasets filtered by problem type and size.
#     """

#     if problem_type not in PROBLEM_TYPE_MAP:
#         raise ValueError(f"Unsupported problem type: {problem_type}")

#     print(f"Searching datasets for problem type: {problem_type}")

#     df = openml.datasets.list_datasets(output_format="dataframe")

#     # basic filtering
#     df = df[
#         (df["NumberOfInstances"] >= min_instances) &
#         (df["NumberOfInstances"] <= max_instances) &
#         (df["NumberOfFeatures"] <= max_features)
#     ]

#     return df.sort_values("NumberOfInstances")


# # -----------------------------------------------------

# def download_dataset(dataset_id: int) -> tuple[pd.DataFrame, pd.Series, dict]:
#     """
#     Download dataset and return X, y, metadata.
#     """

#     dataset = openml.datasets.get_dataset(dataset_id)

#     X, y, categorical, feature_names = dataset.get_data(
#         dataset_format="dataframe",
#         target=dataset.default_target_attribute
#     )

#     metadata = {
#         "dataset_id": dataset_id,
#         "name": dataset.name,
#         "target": dataset.default_target_attribute,
#         "n_rows": len(X),
#         "n_features": X.shape[1], #type: ignore
#         "categorical_features": categorical,
#     }

#     return X, y, metadata #type: ignore


# # -----------------------------------------------------

# def save_dataset(
#     X: pd.DataFrame,
#     y: pd.Series,
#     metadata: dict
# ) -> Path:
#     """
#     Save dataset locally as CSV + metadata JSON.
#     """

#     dataset_name = metadata["name"].replace(" ", "_").lower()
#     dataset_dir = DATA_ROOT / dataset_name
#     dataset_dir.mkdir(parents=True, exist_ok=True)

#     data_path = dataset_dir / "data.csv"
#     meta_path = dataset_dir / "metadata.json"

#     df = X.copy()
#     df[metadata["target"]] = y
#     df.to_csv(data_path, index=False)

#     pd.Series(metadata).to_json(meta_path, indent=2)

#     print(f"Saved dataset → {dataset_dir}")

#     return dataset_dir


# # -----------------------------------------------------

# def fetch_datasets_for_testing(
#     problem_type: str,
#     limit: int = 3
# ) -> list[Path]:
#     """
#     Main function to download multiple datasets for testing.
#     """

#     candidates = search_datasets(problem_type)

#     if len(candidates) == 0:
#         print("No datasets found.")
#         return []

#     selected = candidates.head(limit)
#     saved_paths = []

#     for did in selected["did"]:
#         try:
#             print(f"Downloading dataset ID {did}")
#             X, y, meta = download_dataset(int(did))
#             path = save_dataset(X, y, meta)
#             saved_paths.append(path)

#         except Exception as e:
#             print(f"Failed dataset {did}: {e}")

#     return saved_paths


# # -----------------------------------------------------
# # CLI Entry
# # -----------------------------------------------------

# if __name__ == "__main__":

#     import argparse

#     parser = argparse.ArgumentParser(description="OpenML Dataset Downloader")
#     parser.add_argument(
#         "--problem",
#         type=str,
#         required=True,
#         choices=["classification", "regression", "clustering"],
#         help="Type of ML problem to fetch datasets for"
#     )
#     parser.add_argument(
#         "--limit",
#         type=int,
#         default=3,
#         help="Number of datasets to download"
#     )

#     args = parser.parse_args()

#     paths = fetch_datasets_for_testing(
#         problem_type=args.problem,
#         limit=args.limit
#     )

#     print("\nDownloaded datasets:")
#     for p in paths:
#         print(p)

from pathlib import Path
import openml
import pandas as pd
import argparse


# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

DATA_ROOT = Path("uploaded_files/")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

PROBLEM_TYPE_MAP = {
    "classification": "Supervised Classification",
    "regression": "Supervised Regression",
    "clustering": "Unsupervised",
}


# -----------------------------------------------------
# Dataset Search
# -----------------------------------------------------

def search_datasets(
    problem_type: str,
    min_instances: int = 5000,
    max_instances: int = 50000,
    max_features: int = 200,
    largest_first: bool = True
) -> pd.DataFrame:
    """
    Search OpenML datasets filtered by size and features.
    """

    if problem_type not in PROBLEM_TYPE_MAP:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    print(f"Searching datasets for problem type: {problem_type}")

    df = openml.datasets.list_datasets(output_format="dataframe")

    df = df[
        (df["NumberOfInstances"] >= min_instances) &
        (df["NumberOfInstances"] <= max_instances) &
        (df["NumberOfFeatures"] <= max_features)
    ]

    df = df.sort_values(
        "NumberOfInstances",
        ascending=not largest_first
    )

    print(f"Found {len(df)} candidate datasets")
    return df


# -----------------------------------------------------
# Download Single Dataset
# -----------------------------------------------------

def download_dataset(dataset_id: int):
    """
    Download dataset and return X, y, metadata.
    """

    dataset = openml.datasets.get_dataset(dataset_id)

    X, y, categorical, feature_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )

    metadata = {
        "dataset_id": dataset_id,
        "name": dataset.name,
        "target": dataset.default_target_attribute,
        "n_rows": len(X),
        "n_features": X.shape[1],
        "categorical_features": categorical,
    }

    return X, y, metadata


# -----------------------------------------------------
# Save Dataset
# -----------------------------------------------------

def save_dataset(X: pd.DataFrame, y: pd.Series, metadata: dict) -> Path:

    dataset_name = metadata["name"].replace(" ", "_").lower()
    dataset_dir = DATA_ROOT / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    data_path = dataset_dir / "data.csv"
    meta_path = dataset_dir / "metadata.json"

    df = X.copy()
    df[metadata["target"]] = y

    df.to_csv(data_path, index=False)
    pd.Series(metadata).to_json(meta_path, indent=2)

    print(f"Saved dataset → {dataset_dir}")
    return dataset_dir


# -----------------------------------------------------
# Main Fetch Logic
# -----------------------------------------------------

def fetch_datasets_for_testing(problem_type: str, limit: int = 3):

    candidates = search_datasets(problem_type)

    if len(candidates) == 0:
        print("No datasets found.")
        return []

    saved_paths = []

    for did in candidates["did"]:

        if len(saved_paths) >= limit:
            break

        try:
            print(f"\nChecking dataset ID {did}")

            dataset = openml.datasets.get_dataset(int(did))
            dataset_name = dataset.name.replace(" ", "_").lower()
            dataset_dir = DATA_ROOT / dataset_name

            # -------------------------------------------------
            # SKIP IF ALREADY DOWNLOADED
            # -------------------------------------------------
            if dataset_dir.exists():
                print(f"Skipping '{dataset_name}' (already exists)")
                continue

            print(f"Downloading '{dataset_name}'")

            X, y, meta = download_dataset(int(did))
            path = save_dataset(X, y, meta)
            saved_paths.append(path)

        except Exception as e:
            print(f"Failed dataset {did}: {e}")

    return saved_paths


# -----------------------------------------------------
# CLI Entry
# -----------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="OpenML Dataset Downloader")

    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        choices=["classification", "regression", "clustering"],
        help="Type of ML problem to fetch datasets for"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of datasets to download"
    )

    args = parser.parse_args()

    paths = fetch_datasets_for_testing(
        problem_type=args.problem,
        limit=args.limit
    )

    print("\nDownloaded datasets:")
    for p in paths:
        print(p)
from typing import TypedDict
import pandas as pd


class AgentState(TypedDict, total=False):
    file_path: str
    target_column: str | None

    dataset_id: str

    df: pd.DataFrame
    cleaned_file_path: str
    metadata_file_path: str

    post_clean_report: dict
    metadata: dict
from typing import TypedDict
import pandas as pd


class AgentState(TypedDict, total=False):
    file_path: str
    target_column: str | None
    df: pd.DataFrame
    dataset_id: str
    pre_clean_report: list
    post_clean_report: dict
    metadata: dict
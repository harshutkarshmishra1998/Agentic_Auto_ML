from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict, total=False):
    """
    Global state shared across graph nodes.
    """

    # -------- user inputs --------
    data_path: str
    categorical_columns: Optional[List[str]]
    target_column: Optional[str]

    # -------- outputs --------
    schema_result: Optional[Dict[str, Any]]
    data_understanding_result: Optional[Dict[str, Any]]
from typing import Dict, Any

from agent_state import AgentState
from data_understanding.pipeline import run_data_understanding


def data_understanding_node(state: AgentState) -> AgentState:
    """
    LangGraph node wrapper for data understanding pipeline.

    Input:
        state["data_path"]

    Output:
        state["data_understanding_result"]
    """

    data_path = state.get("data_path")

    if not data_path:
        raise ValueError("data_path missing in AgentState")

    result: Dict[str, Any] = run_data_understanding(data_path) #type: ignore

    return {
        **state,
        "data_understanding_result": result #type: ignore
    }
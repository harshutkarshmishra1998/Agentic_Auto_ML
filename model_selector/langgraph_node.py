from typing import Dict, Any
from agent_state import AgentState

from model_selector.pipeline import run_model_selection


def model_selector_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node wrapper for model selection pipeline.
    """

    last_n = state.get("preprocess_last_n", 1)

    results = run_model_selection(last_n)

    return {
        "model_selection_result": results
    }
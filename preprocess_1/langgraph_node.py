from typing import Dict, Any
from agent_state import AgentState

from .pipeline import run_preprocess_1


def preprocess_1_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node for model-independent preprocessing.

    Runs preprocess_1 pipeline and updates state with results.
    """

    # how many datasets to process
    # default = 1 if not provided upstream
    last_n = state.get("preprocess_last_n", 1)

    results = run_preprocess_1(last_n)

    return {
        "preprocess_1_result": results
    }
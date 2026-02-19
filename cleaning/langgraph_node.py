from agent_state import AgentState
from .pipeline import run_cleaning_pipeline


def cleaning_node(state: AgentState) -> AgentState:

    result = run_cleaning_pipeline(state["file_path"], state.get("target_column")) #type: ignore

    state["df"] = result["df"]
    state["dataset_id"] = result["dataset_id"]
    state["pre_clean_report"] = result["pre_clean_report"]
    state["post_clean_report"] = result["post_clean_report"]
    state["metadata"] = result["metadata"]

    return state
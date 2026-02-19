from agent_state import AgentState
from .pipeline import run_cleaning_pipeline


def cleaning_node(state: AgentState) -> AgentState:
    """
    LangGraph node for full industrial cleaning pipeline.
    """

    result = run_cleaning_pipeline(
        state["file_path"], #type: ignore
        state.get("target_column")
    )

    state["dataset_id"] = result["dataset_id"]
    state["df"] = result["df"]
    state["cleaned_file_path"] = result["cleaned_file_path"]
    state["metadata_file_path"] = result["metadata_file_path"]
    state["post_clean_report"] = result["post_clean_report"]
    state["metadata"] = result["metadata"]

    return state
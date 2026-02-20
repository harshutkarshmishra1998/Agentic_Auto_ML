from schema_engine.pipeline import run_schema_inference
from agent_state import AgentState


def schema_inference_node(state: AgentState) -> AgentState:
    """
    LangGraph node that performs schema inference.
    """

    data_path = state.get("data_path")
    categorical_columns = state.get("categorical_columns")
    target_column = state.get("target_column")

    if not data_path:
        raise ValueError("data_path missing in agent state")

    result = run_schema_inference(
        data_path=data_path,
        categorical_columns=categorical_columns,
        target_column=target_column,
    )

    state["schema_result"] = result
    return state
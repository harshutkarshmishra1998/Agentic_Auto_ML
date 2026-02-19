from agent_state import AgentState
from cleaning.langgraph_node import cleaning_node


def run_agent(file_path: str):

    state: AgentState = {
        "file_path": file_path
    }

    state = cleaning_node(state)
    return state


if __name__ == "__main__":
    result = run_agent("data/sample.csv")
    print(result["metadata"]) #type: ignore
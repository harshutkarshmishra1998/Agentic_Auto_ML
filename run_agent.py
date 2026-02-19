from agent_state import AgentState
from cleaning.langgraph_node import cleaning_node


def run_agent(file_path: str, target_column: str | None = None):

    state: AgentState = {
        "file_path": file_path,
        "target_column": target_column
    }

    state = cleaning_node(state)
    return state


if __name__ == "__main__":

    result = run_agent("uploaded_files/jigsaw_unintended_bias100k/data.csv", "target")

    print("\nDATASET ID:")
    print(result["dataset_id"]) #type: ignore

    print("\nCLEANED FILE:")
    print(result["cleaned_file_path"]) #type: ignore

    print("\nMETADATA FILE:")
    print(result["metadata_file_path"]) #type: ignore

    print("\nPOST CLEAN REPORT:")
    print(result["post_clean_report"]) #type: ignore

    print("\nMETADATA SUMMARY:")
    print(result["metadata"].keys()) #type: ignore
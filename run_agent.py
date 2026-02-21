from langgraph.graph import StateGraph, END

from agent_state import AgentState
from schema_engine.langgraph_node import schema_inference_node
from data_understanding.langgraph_node import data_understanding_node
from preprocess_1.langgraph_node import preprocess_1_node
from model_selector.langgraph_node import model_selector_node

from tests.schema_mapping import extract_schema
from tests.json_printer import print_last_n_role_constants

METADATA_FILE = "uploaded_files/churn/metadata.json"
DATA_FILE = "uploaded_files/churn/data.csv"

cats, target = extract_schema(METADATA_FILE, DATA_FILE)

print("CATEGORICAL_COLUMNS = ", cats)
if target:
    print(f'TARGET_COLUMN = "{target}"')
else:
    print("TARGET_COLUMN = null")


# --------------------------------------------------
# BUILD GRAPH
# --------------------------------------------------
def build_graph():

    builder = StateGraph(AgentState)

    builder.add_node("schema_inference", schema_inference_node)
    builder.add_node("data_understanding", data_understanding_node)
    builder.add_node("preprocess_1", preprocess_1_node)
    builder.add_node("model_selector", model_selector_node)

    builder.set_entry_point("schema_inference")
    builder.add_edge("schema_inference", "data_understanding")
    builder.add_edge("data_understanding", "preprocess_1")
    builder.add_edge("preprocess_1", "model_selector")
    builder.add_edge("model_selector", END)

    return builder.compile()


# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":

    graph = build_graph()

    initial_state: AgentState = {
        "data_path": DATA_FILE,
        "categorical_columns": cats,
        "target_column": target,
        "preprocess_last_n": 1
    }

    result = graph.invoke(initial_state)

    print("\n=== FINAL STATE ===\n")
    print(result)

    print("\n=== LAST CLASSIFICATION ===\n")
    print_last_n_role_constants("data/data_classification.jsonl", n=1)
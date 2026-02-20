from schema_engine.pipeline import run_schema_inference
from tests.schema_mapping import extract_schema
from tests.json_printer import print_last_n_role_constants

# python -m tests.test_pipeline - For executing schema engine

METADATA_FILE = "uploaded_files/traffic_violations_100k/metadata.json"
DATA_FILE = "uploaded_files/traffic_violations_100k/data.csv"

cats, target = extract_schema(METADATA_FILE, DATA_FILE)

print("CATEGORICAL_COLUMNS = ", cats)
if target:
    print(f'TARGET_COLUMN = "{target}"')
else:
    print("TARGET_COLUMN = null")

schema = run_schema_inference(
    DATA_FILE,
    categorical_columns=cats,
    target_column=target
)

print_last_n_role_constants(
        "data/data_classification.jsonl",
        n=1
    )

print("TEST COMPLETED!")


# import pandas as pd

# df = pd.read_csv(DATA_FILE)

# print("Column count:", len(df.columns))
# for i, c in enumerate(df.columns):
#     print(i, repr(c))
import json
from pathlib import Path

OUTPUT_PATH = Path("data/model_selection.jsonl")


def append_record(record):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
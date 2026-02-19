import json
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("data/cleaning_audit.jsonl")


def record(dataset_id, step, details):
    LOG_PATH.parent.mkdir(exist_ok=True)

    entry = {
        "time": datetime.utcnow().isoformat(),
        "dataset_id": dataset_id,
        "step": step,
        "details": details
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
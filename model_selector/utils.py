import hashlib
import uuid
from datetime import datetime
from pathlib import Path

SELECTOR_VERSION = "2.0.0"


def file_sha256(path: str):
    p = Path(path)
    if not p.exists():
        return None

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso():
    return datetime.utcnow().isoformat()


def new_experiment_id():
    return str(uuid.uuid4())
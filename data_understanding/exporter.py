import json
from pathlib import Path


# def export_column_inspection(dataset_path, column_records):

#     project_root = Path(__file__).resolve().parents[1]
#     out_path = project_root / "data" / "column_inspection.jsonl"
#     out_path.parent.mkdir(exist_ok=True)

#     record = {
#         "dataset_file_path": str(Path(dataset_path).resolve()),
#         "dataset_file_name": Path(dataset_path).name,
#         "columns": column_records
#     }

#     with open(out_path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(record) + "\n")

# def export_column_inspection(dataset_path, payload):

#     project_root = Path(__file__).resolve().parents[1]
#     out_path = project_root / "data" / "column_inspection.jsonl"
#     out_path.parent.mkdir(exist_ok=True)

#     record = {
#         "dataset_file_path": str(Path(dataset_path).resolve()),
#         "dataset_file_name": Path(dataset_path).name,
#         **payload
#     }

#     with open(out_path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(record) + "\n")

import json
from pathlib import Path


def export_column_inspection(dataset_path, payload):

    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "data" / "column_inspection.jsonl"
    out_path.parent.mkdir(exist_ok=True)

    record = {
        "dataset_file_path": str(Path(dataset_path).resolve()),
        "dataset_file_name": Path(dataset_path).name,
        **payload
    }

    # -----------------------------
    # ensure newline separation
    # -----------------------------
    with open(out_path, "a+", encoding="utf-8") as f:

        # move cursor to end
        f.seek(0, 2)

        # if file not empty and last char not newline → add newline
        if f.tell() > 0:
            f.seek(f.tell() - 1)
            if f.read(1) != "\n":
                f.write("\n")

        # write JSON object + newline
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")
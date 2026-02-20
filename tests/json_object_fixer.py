import json
from pathlib import Path


def repair_jsonl_file(input_path: str, output_path: str):
    """
    Convert malformed JSON stream into valid JSONL.
    Handles:
        - multi-line objects
        - concatenated objects
        - pretty-printed JSON blocks
    """

    text = Path(input_path).read_text(encoding="utf-8")

    objects = []
    buffer = ""
    depth = 0
    in_string = False
    escape = False

    for ch in text:
        buffer += ch

        # string tracking
        if ch == '"' and not escape:
            in_string = not in_string

        if ch == "\\" and not escape:
            escape = True
            continue
        escape = False

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

            # full object closed
            if depth == 0:
                obj = buffer.strip()
                if obj:
                    try:
                        objects.append(json.loads(obj))
                    except json.JSONDecodeError as e:
                        print("Skipping invalid object:", e)
                buffer = ""

    if depth != 0:
        raise ValueError("Unbalanced JSON braces — file corrupted")

    # write clean JSONL
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj, separators=(",", ":")))
            f.write("\n")

    print(f"\nRepaired JSONL saved → {out}")
    print(f"Recovered objects → {len(objects)}")


# --------------------------------------------------
# run standalone
# --------------------------------------------------
if __name__ == "__main__":

    INPUT = "data/column_inspection.jsonl"
    OUTPUT = "data/column_inspection_clean.jsonl"

    repair_jsonl_file(INPUT, OUTPUT)
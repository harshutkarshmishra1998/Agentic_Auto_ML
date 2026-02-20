import json
from pathlib import Path
from collections import defaultdict


def _role_to_constant(role: str) -> str:
    """
    Convert role name to CONSTANT_NAME_COLUMNS format.
    Example:
        categorical_nominal -> CATEGORICAL_NOMINAL_COLUMNS
    """
    role = role.upper().replace(" ", "_")
    return f"{role}_COLUMNS"


def print_last_n_role_constants(jsonl_path: str, n: int):
    """
    Print column lists per role as Python constant assignments.
    """

    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if n <= 0:
        raise ValueError("n must be > 0")

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    start = max(0, len(lines) - n)

    for record_idx, line in enumerate(lines[start:], start=start + 1):
        obj = json.loads(line)

        feature_mapping = obj.get("feature_mapping", {})
        target_column = obj.get("target_column")

        role_groups = defaultdict(list)

        for col, meta in feature_mapping.items():
            role = meta.get("role", "unknown")
            role_groups[role].append(col)

        print("\n" + "=" * 70)
        print(f"RECORD #{record_idx}")
        print("=" * 70)

        # print roles as constant lists
        for role, columns in sorted(role_groups.items()):
            const_name = _role_to_constant(role)
            print(f"{const_name} = {columns}")

        # print target
        if target_column:
            print(f'TARGET_COLUMN = "{target_column}"')
        else:
            print("TARGET_COLUMN = None")


# --------------------------------------------------
# Example usage
# --------------------------------------------------
if __name__ == "__main__":
    print_last_n_role_constants(
        "data/data_classification.jsonl",
        n=1
    )
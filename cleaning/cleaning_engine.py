import pandas as pd
from .actions import run_action
from .audit import record


def execute_cleaning(df: pd.DataFrame, dataset_id: str, actions: list):

    for step in actions:

        column = step.get("column")
        action = step["action"]

        # ---------- COLUMN VALIDATION ----------
        if column is not None and column not in df.columns:
            record(dataset_id, "action_skipped", {
                "reason": "column_missing_after_previous_transform",
                "column": column,
                "action": action
            })
            continue

        # ---------- EXECUTE ----------
        try:
            df = run_action(df, action, column)

            record(dataset_id, "clean_action", {
                "column": column,
                "action": action
            })

        except Exception as e:
            record(dataset_id, "action_failed", {
                "column": column,
                "action": action,
                "error": str(e)
            })

    return df
from .data_loader import load_datasets
from .selector import build_model_plan
from .logger import append_record


def run_model_selection(last_n):

    datasets = load_datasets(last_n)
    results = []

    for ds in datasets:
        plan = build_model_plan(ds)
        append_record(plan)
        results.append(plan)

    return results
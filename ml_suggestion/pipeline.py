from .engine import run_ml_suggestion


def suggest_models(last_n: int = 1, use_llm: bool = False):
    """
    Public API for tests and upstream pipeline.

    Automatically:
    - reads metadata
    - selects last N datasets
    - resolves cleaned dataset
    - ranks models
    """
    return run_ml_suggestion(last_n=last_n, use_llm=use_llm)
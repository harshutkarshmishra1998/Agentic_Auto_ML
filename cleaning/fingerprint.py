import hashlib
import pandas as pd


def dataset_hash(df: pd.DataFrame) -> str:
    content = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(content.tobytes()).hexdigest()[:8] #type: ignore
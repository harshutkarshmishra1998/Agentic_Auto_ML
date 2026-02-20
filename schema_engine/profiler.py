import pandas as pd
import numpy as np
import re
from dataclasses import dataclass


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n: int
    n_unique: int
    unique_ratio: float
    missing_ratio: float
    is_numeric: bool
    is_integer_like: bool
    mean: float | None
    std: float | None
    min_val: object
    max_val: object
    sample_values: list
    parseable_datetime_ratio: float


DATE_REGEXES = [
    r"\d{4}-\d{2}-\d{2}",
    r"\d{2}/\d{2}/\d{4}",
    r"\d{2}-\d{2}-\d{4}",
]


def _datetime_parse_ratio(series: pd.Series):
    s = series.dropna().astype(str)
    if len(s) == 0:
        return 0.0

    hits = 0
    for val in s.head(500):
        if any(re.fullmatch(rgx, val) for rgx in DATE_REGEXES):
            hits += 1

    return hits / min(len(s),500)


def profile_dataframe(df: pd.DataFrame):
    profiles = {}

    for col in df.columns:
        s = df[col]
        n = len(s)
        n_unique = s.nunique(dropna=True)

        numeric = pd.api.types.is_numeric_dtype(s)
        integer_like = False
        mean = std = None

        if numeric:
            integer_like = np.allclose(s.dropna() % 1, 0)
            mean = float(s.mean()) if n_unique else None
            std = float(s.std()) if n_unique > 1 else None

        profiles[col] = ColumnProfile(
            name=col,
            dtype=str(s.dtype),
            n=n,
            n_unique=n_unique,
            unique_ratio=n_unique/n if n else 0,
            missing_ratio=s.isna().mean(),
            is_numeric=numeric,
            is_integer_like=bool(integer_like),
            mean=mean,
            std=std,
            min_val=s.min(),
            max_val=s.max(),
            sample_values=s.dropna().astype(str).unique()[:10].tolist(),
            parseable_datetime_ratio=_datetime_parse_ratio(s),
        )

    return profiles
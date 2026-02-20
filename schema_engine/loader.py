from __future__ import annotations
import json
import io
import zipfile
from pathlib import Path
import pandas as pd


class DataLoadError(Exception):
    pass


def _read_excel_any(path: Path):
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, engine="openpyxl")


def _read_csv_robust(path: Path):
    try:
        df = pd.read_csv(path)
        df = df.dropna(axis=1, how="all")
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        try:
            df = pd.read_csv(path, sep=None, engine="python")
            df = df.dropna(axis=1, how="all")
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            return pd.read_csv(path, encoding="latin1")


def _flatten_json_records(obj):
    return pd.json_normalize(obj, sep="__")


def _read_json_any(path: Path):
    try:
        return pd.read_json(path)
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()

        if content.startswith("{") or content.startswith("["):
            return _flatten_json_records(json.loads(content))

        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        return _flatten_json_records(rows)


def _read_zip_first_table(path: Path):
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.lower().endswith(("csv","xlsx","xls","json","jsonl")):
                data = z.read(name)
                buffer = io.BytesIO(data)

                if name.endswith("csv"):
                    return pd.read_csv(buffer)
                if name.endswith(("xlsx","xls")):
                    return pd.read_excel(buffer)
                if name.endswith(("json","jsonl")):
                    return _flatten_json_records(json.loads(data.decode()))

    raise DataLoadError("No readable table inside zip")


def load_table(path: str | Path):
    path = Path(path)

    if not path.exists():
        raise DataLoadError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = _read_csv_robust(path)
    elif suffix in {".xlsx",".xls"}:
        df = _read_excel_any(path)
    elif suffix in {".json",".jsonl"}:
        df = _read_json_any(path)
    elif suffix == ".zip":
        df = _read_zip_first_table(path)
    else:
        raise DataLoadError(f"Unsupported format: {suffix}")

    df.columns = [str(c).strip() for c in df.columns]
    return df
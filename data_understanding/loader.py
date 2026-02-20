import pandas as pd


def load_dataset(path):
    if path.endswith(".csv"):
        # return pd.read_csv(path)
        df = pd.read_csv(path)
        df = df.dropna(axis=1, how="all")
        df.columns = df.columns.str.strip()
        return df
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError("Unsupported dataset format")
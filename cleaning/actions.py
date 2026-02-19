import pandas as pd
from sklearn.preprocessing import PowerTransformer

ACTION_REGISTRY = {}


def register(name):
    def decorator(func):
        ACTION_REGISTRY[name] = func
        return func
    return decorator


def run_action(df: pd.DataFrame, action: str, column=None):
    if action not in ACTION_REGISTRY:
        raise ValueError(f"Unknown action: {action}")
    return ACTION_REGISTRY[action](df, column)


@register("drop_feature")
def drop_feature(df, column):
    return df.drop(columns=[column])


@register("impute_numeric_median")
def impute_numeric_median(df, column):
    df[column] = df[column].fillna(df[column].median())
    return df


@register("apply_power_transform")
def power_transform(df, column):
    pt = PowerTransformer()
    df[column] = pt.fit_transform(df[[column]])
    return df
from .profiler import ColumnProfile


class Role:
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    IDENTIFIER = "identifier"
    DATETIME = "datetime"
    TEXT = "text_freeform"
    UNKNOWN = "unknown"
    TARGET = "target"


def deterministic_role(profile: ColumnProfile):

    if profile.parseable_datetime_ratio > 0.9:
        return Role.DATETIME, 0.95

    if profile.unique_ratio > 0.98 and profile.n_unique > 20:
        return Role.IDENTIFIER, 0.9

    if profile.is_numeric:
        if profile.is_integer_like and profile.n_unique < 30:
            return Role.NUMERIC_DISCRETE, 0.8
        return Role.NUMERIC_CONTINUOUS, 0.85

    if profile.n_unique > 100 and profile.unique_ratio > 0.5:
        return Role.TEXT, 0.7

    if profile.n_unique <= 50:
        return Role.CATEGORICAL_NOMINAL, 0.7

    return Role.UNKNOWN, 0.3
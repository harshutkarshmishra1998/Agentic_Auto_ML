from .deterministic import Role

def is_ambiguous(role, confidence):
    if confidence < 0.8:
        return True
    if role in {Role.NUMERIC_DISCRETE, Role.CATEGORICAL_NOMINAL}:
        return True
    return False
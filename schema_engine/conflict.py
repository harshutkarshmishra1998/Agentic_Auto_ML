from .deterministic import Role

def resolve_final_role(
    column,
    det_role,
    det_conf,
    llm_role,
    llm_conf,
    user_declared,
    target_column,
):
    if target_column and column == target_column:
        return Role.TARGET, 1.0

    if column in user_declared:
        return user_declared[column], 0.95

    if llm_role and llm_conf and llm_conf > det_conf:
        return llm_role, llm_conf

    return det_role, det_conf
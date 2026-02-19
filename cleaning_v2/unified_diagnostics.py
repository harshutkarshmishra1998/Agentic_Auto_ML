from .detectors import AUTO_FIX_DETECTORS, POLICY_DETECTORS, INFO_DETECTORS


def _safe(detector, *args):
    try:
        return detector(*args)
    except Exception:
        return []


def run_unified_diagnostics(df, target=None):

    auto = []
    policy = []
    info = []

    for d in AUTO_FIX_DETECTORS:
        auto.extend(_safe(d, df))

    for d in POLICY_DETECTORS:
        if d.__code__.co_argcount == 2:
            policy.extend(_safe(d, df, target))
        else:
            policy.extend(_safe(d, df))

    for d in INFO_DETECTORS:
        info.extend(_safe(d, df))

    return {
        "auto_fixable": auto,
        "policy_required": policy,
        "informational": info
    }
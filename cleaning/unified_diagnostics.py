from .detectors import (
    AUTO_FIX_DETECTORS,
    POLICY_DETECTORS,
    INFO_DETECTORS,
)


def run_unified_diagnostics(df, target=None):

    auto_fix = []
    policy = []
    info = []

    # auto fix
    for detector in AUTO_FIX_DETECTORS:
        auto_fix.extend(detector(df))

    # policy required
    for detector in POLICY_DETECTORS:
        if detector.__code__.co_argcount == 2:
            policy.extend(detector(df, target))
        else:
            policy.extend(detector(df))

    # informational
    for detector in INFO_DETECTORS:
        info.extend(detector(df))

    return {
        "auto_fixable": auto_fix,
        "policy_required": policy,
        "informational": info,
    }
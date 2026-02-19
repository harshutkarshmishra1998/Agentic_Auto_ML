from dataclasses import dataclass
from typing import Optional


@dataclass
class DiagnosticResult:
    detector: str
    column: Optional[str]
    value: Optional[float]
    severity: str
    auto_fixable: bool = False
    policy_required: bool = False
    recommended_action: Optional[str] = None

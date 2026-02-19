from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DiagnosticResult:
    detector: str
    column: Optional[str]
    value: Any
    severity: str

    auto_fixable: bool = False
    policy_required: bool = False

    recommended_action: Optional[str] = None
    details: Optional[dict] = None
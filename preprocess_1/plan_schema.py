# from dataclasses import dataclass, field
# from typing import List, Dict, Any


# @dataclass
# class Step:
#     step_type: str
#     columns: List[str] = field(default_factory=list)
#     params: Dict[str, Any] = field(default_factory=dict)
#     reason: str = ""
#     status: str = "planned"  # planned / executed / skipped


# @dataclass
# class PreprocessPlan:
#     dataset_path: str
#     dataset_name: str
#     steps: List[Step] = field(default_factory=list)
#     deferred_model_dependent: List[str] = field(default_factory=list)

#     def add_step(self, step: Step):
#         self.steps.append(step)

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Step:
    step_type: str
    columns: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    status: str = "planned"


@dataclass
class DeferredAction:
    column: str
    strategy: str
    reason: str


@dataclass
class PreprocessPlan:
    dataset_path: str
    dataset_name: str
    steps: List[Step] = field(default_factory=list)

    # column-level deferred operations
    deferred: List[DeferredAction] = field(default_factory=list)

    def add_step(self, step: Step):
        self.steps.append(step)

    def add_deferred(self, column: str, strategy: str, reason: str):
        self.deferred.append(
            DeferredAction(column, strategy, reason)
        )
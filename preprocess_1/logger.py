# # import json
# # from pathlib import Path
# # from datetime import datetime


# # def append_log(plan, output_path: Path):
# #     record = {
# #         "timestamp": datetime.utcnow().isoformat(),
# #         "dataset": plan.dataset_name,
# #         "dataset_path": plan.dataset_path,
# #         "steps": [
# #             {
# #                 "type": s.step_type,
# #                 "columns": s.columns,
# #                 "reason": s.reason,
# #                 "status": s.status
# #             }
# #             for s in plan.steps
# #         ],
# #         "deferred": plan.deferred_model_dependent
# #     }

# #     output_path.parent.mkdir(parents=True, exist_ok=True)

# #     with open(output_path, "a", encoding="utf-8") as f:
# #         f.write(json.dumps(record) + "\n")

# import json
# from pathlib import Path
# from datetime import datetime


# def append_log(plan, output_csv_path: Path, log_path: Path):
#     record = {
#         "timestamp": datetime.utcnow().isoformat(),

#         # dataset info
#         "dataset": plan.dataset_name,
#         "dataset_path": plan.dataset_path,

#         # output artifact (NEW)
#         "output_file_name": output_csv_path.name,
#         "output_file_path": str(output_csv_path.resolve()),

#         # steps
#         "steps": [
#             {
#                 "type": s.step_type,
#                 "columns": s.columns,
#                 "reason": s.reason,
#                 "status": s.status
#             }
#             for s in plan.steps
#         ],

#         # deferred work
#         "deferred": plan.deferred_model_dependent
#     }

#     log_path.parent.mkdir(parents=True, exist_ok=True)

#     with open(log_path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(record) + "\n")

import json
from pathlib import Path
from datetime import datetime


def append_log(plan, output_csv_path: Path, log_path: Path):

    record = {
        "timestamp": datetime.utcnow().isoformat(),

        "dataset": plan.dataset_name,
        "dataset_path": plan.dataset_path,

        "output_file_name": output_csv_path.name,
        "output_file_path": str(output_csv_path.resolve()),

        "steps": [
            {
                "type": s.step_type,
                "columns": s.columns,
                "reason": s.reason,
                "status": s.status
            }
            for s in plan.steps
        ],

        # COLUMN LEVEL DEFERRED
        "deferred": [
            {
                "column": d.column,
                "strategy": d.strategy,
                "reason": d.reason
            }
            for d in plan.deferred
        ]
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
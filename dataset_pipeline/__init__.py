"""C2HLS-Trajectory dataset pipeline (Pillar 8, schema_version 2.0).

Companion to the existing schema 1.0 jsonl in
[csynth_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl] and
[results/references_philip/]. v2.0 adds:

- per-scope feedback records (per-loop II / Slack / Issue / Violation /
  scheduler-blame / typed bottlenecks) — the Pillar 1 signal,
- a `validation_status` field (validated / unscored / failed) so
  trajectories that ran without csim are visible rather than silently
  treated as "pass" (Pillar 9 csim-gating),
- explicit step-effectiveness annotation (`step_effect` ∈ improved |
  regressed | no_op | absorbed) — the agent-side hook for Pillar 7,
- Vitis version + FPGA target as first-class run fields (Pillar 6 prep).

Existing schema 1.0 records remain valid; the recorder emits v2.0 in
parallel without disturbing the old jsonl producers.
"""

from .schema import (
    SCHEMA_VERSION,
    TrajectoryRecord,
    record_to_dict,
)
from .recorder import (
    record_step_outcome,
    classify_step_effect,
    classify_validation_status,
)
from .merge import merge_with_references
from .replay import replay_existing_results
from .external_adapter import (
    DatasetReport,
    FileClassification,
    adapt_external_kernel,
    classify_source_file,
    render_survey_markdown,
    survey_dataset,
)

__all__ = [
    "SCHEMA_VERSION",
    "TrajectoryRecord",
    "record_to_dict",
    "record_step_outcome",
    "classify_step_effect",
    "classify_validation_status",
    "merge_with_references",
    "replay_existing_results",
    "DatasetReport",
    "FileClassification",
    "adapt_external_kernel",
    "classify_source_file",
    "render_survey_markdown",
    "survey_dataset",
]

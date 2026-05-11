"""Schema v2.0 for the C2HLS-Trajectory dataset.

Each record is one (kernel, version, fpga, step, candidate) point along an
optimization trajectory. The schema is intentionally additive over v1.0:
all v1 fields appear in the same shape, plus `feedback` (Pillar 1) and the
trajectory-aware fields (`step_effect`, `validation_status`, `parent_hash`,
`candidate_hash`, `relative_advantage`, `skill_hits`).

Records are produced by `recorder.record_step_outcome()` and consumed by
`merge.merge_with_references()` and `replay.replay_existing_results()`.

Why a dataclass, not a free dict: the schema is part of the paper's
public contribution, and we want one place that documents every field +
any constraints. `record_to_dict()` then strips Nones and writes a
deterministic key order.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "2.0"


# Allowed values are documented here so downstream readers can validate
# without having to re-read the spec each time.
VALIDATION_STATUSES = ("validated", "unscored", "failed", "skipped")
STEP_EFFECTS = ("improved", "regressed", "no_op", "absorbed",
                "translation_failed", "synth_failed", "csim_failed",
                "errored", "unknown")


@dataclass
class RunMeta:
    """The (toolchain, target) cell this record belongs to. Pillar 6 will
    expand the value space; Phase 1 hard-codes Vitis 2023.2 / U280."""
    target: str = "vitis.csynth"        # vitis.csynth | vitis.sw_emu | vitis.hw_emu
    vitis_version: str = ""
    device: str = ""
    flow_target: str = "vitis"
    clock_ns: Optional[float] = None
    runtime_seconds: Optional[float] = None


@dataclass
class ImplementationMeta:
    """Provenance of the HLS code under test."""
    origin: str = "c2hls_orchestrator"  # | rodinia_hls_benchmark
    origin_version: str = ""             # model id / "upstream"
    multistep: bool = True
    step: str = ""                        # "tiling" | "pipeline" | …
    step_effect: str = "unknown"          # see STEP_EFFECTS
    validation_status: str = "unscored"   # see VALIDATION_STATUSES
    parent_hash: Optional[str] = None     # hash of the parent step's report
    candidate_hash: Optional[str] = None  # hash of THIS step's report
    relative_advantage: Optional[float] = None  # group-relative score (Pillar 3)
    skill_hits: List[str] = field(default_factory=list)  # skill ids applied
    rationale: str = ""                   # one-line LLM-emitted reason
    generated_at: str = ""                # ISO8601 timestamp


@dataclass
class HLSSynthPayload:
    """Top-level numerics that mirror the existing v1.0 schema."""
    status: str = "pass"                 # pass | fail | timeout | unscored
    latency_cycles: Optional[int] = None
    latency_ns: Optional[float] = None
    interval: Optional[int] = None
    bram: Optional[int] = None
    dsp: Optional[int] = None
    ff: Optional[int] = None
    lut: Optional[int] = None
    uram: Optional[int] = None
    fmax_mhz: Optional[float] = None
    estimated_clock_period_ns: Optional[float] = None
    requested_clock_period_ns: Optional[float] = None
    slack_ns: Optional[float] = None


@dataclass
class RTLSimPayload:
    """hw_emu / cosim payload — same shape as the existing reference jsonls."""
    status: str = "pass"  # pass | fail | timeout | not_run
    kernel_runtime_cycles: Optional[int] = None
    kernel_runtime_us: Optional[float] = None
    kernel_clock_freq_mhz: Optional[float] = None
    error: Optional[str] = None


@dataclass
class CsimPayload:
    """csim outcome (functional correctness gate; Pillar 9 csim-gating)."""
    available: bool = False
    ran: bool = False
    passed: bool = False
    error: Optional[str] = None


@dataclass
class TrajectoryRecord:
    schema_version: str = SCHEMA_VERSION
    report_type: str = "hls_synth"  # hls_synth | rtl_sim | sw_emu | hw_emu | csim
    run: RunMeta = field(default_factory=RunMeta)
    problem: Dict[str, Any] = field(default_factory=dict)  # {suite, group_path}
    implementation: ImplementationMeta = field(default_factory=ImplementationMeta)
    variant: Dict[str, Any] = field(default_factory=dict)  # {index, name}

    # Payload union — exactly one of these is populated, depending on
    # report_type. Keeping all three on the record makes downstream filtering
    # ergonomic ("rows where rtl_sim.status == pass") without having to
    # repeatedly re-shape.
    hls_synth: Optional[HLSSynthPayload] = None
    rtl_sim: Optional[RTLSimPayload] = None
    csim: Optional[CsimPayload] = None

    # Pillar 1 feedback (per-scope records, scheduler-blame, typed
    # bottlenecks, summary). Verbatim shape from hls_feedback.build_feedback.
    feedback: Optional[Dict[str, Any]] = None


# Field order for output JSON — keep it stable so diffs across runs of the
# pipeline are minimal.
_FIELD_ORDER = (
    "schema_version", "report_type", "run", "problem",
    "implementation", "variant",
    "hls_synth", "rtl_sim", "csim",
    "feedback",
)


def record_to_dict(record: TrajectoryRecord, *, drop_none: bool = True) -> Dict[str, Any]:
    """Convert a TrajectoryRecord to a dict suitable for `json.dumps()` with
    deterministic key order. By default, top-level keys whose value is None
    are removed (matches the v1.0 conventions)."""
    raw = asdict(record)
    out: Dict[str, Any] = {}
    for k in _FIELD_ORDER:
        v = raw.get(k)
        if drop_none and v is None:
            continue
        out[k] = v
    # Trim empty optional sub-payloads.
    for k in ("hls_synth", "rtl_sim", "csim"):
        v = out.get(k)
        if drop_none and isinstance(v, dict) and all(
            sub_v is None or sub_v == [] or sub_v is False for sub_v in v.values()
        ):
            del out[k]
    # implementation.skill_hits empty → drop only that field, not the parent.
    impl = out.get("implementation")
    if isinstance(impl, dict) and impl.get("skill_hits") == []:
        impl.pop("skill_hits", None)
    return out

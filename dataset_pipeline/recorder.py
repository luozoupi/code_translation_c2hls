"""Convert a c2hls multistep `step_result` (or a baseline_report) into one
or more v2.0 TrajectoryRecord dicts.

Three concerns it owns:
1. **csim-gating** (Pillar 9 MVP item 4): if a step's csim is unavailable
   or did not run, mark `validation_status = "unscored"` rather than
   silently inheriting the synth-only `pass`.
2. **Step-effectiveness annotation** (Pillar 7 / 9): classify the step's
   effect as `improved` | `regressed` | `no_op` | `synth_failed` | etc.
   so trajectory consumers can filter without re-deriving from numbers.
3. **Hashing**: stable `parent_hash` / `candidate_hash` from a canonical
   subset of the synth report so downstream cache lookups work across
   reruns.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .schema import (
    CsimPayload,
    HLSSynthPayload,
    ImplementationMeta,
    RTLSimPayload,
    RunMeta,
    TrajectoryRecord,
    VALIDATION_STATUSES,
    STEP_EFFECTS,
    record_to_dict,
)


# Tuple of fields that drive step-effect classification. Same set as the
# Pillar 9 no-op detector so the two stay in sync.
_NUMERIC_HASH_FIELDS = (
    "latency_cycles", "interval", "bram", "dsp", "ff", "lut", "fmax_mhz",
)


# === Hashing ===============================================================


def report_hash(report: Optional[Dict[str, Any]]) -> Optional[str]:
    """Stable hash over the synthesis-driving numerics. Two runs producing
    identical numbers share a hash so the dataset can deduplicate
    inadvertently re-synthesized candidates."""
    if not report:
        return None
    payload = {k: report.get(k) for k in _NUMERIC_HASH_FIELDS}
    if all(v is None for v in payload.values()):
        return None
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# === Classification helpers ================================================


def classify_validation_status(
    csim: Optional[Dict[str, Any]],
    synth_status: str = "pass",
) -> str:
    """Pillar 9 csim-gating: a step is 'validated' only when csim ran and
    passed; otherwise it's 'unscored' (or 'failed' if synth itself failed)."""
    if synth_status not in {"pass", "ok", "success"}:
        return "failed"
    if not csim:
        return "unscored"
    if not csim.get("available", csim.get("ran", False)):
        return "unscored"
    if not csim.get("ran", False):
        return "unscored"
    if csim.get("passed", csim.get("status") == "pass"):
        return "validated"
    return "failed"


def classify_step_effect(
    new_report: Optional[Dict[str, Any]],
    prev_report: Optional[Dict[str, Any]],
    *,
    success: bool = True,
    csim_passed: Optional[bool] = None,
    error: Optional[str] = None,
    epsilon: float = 0.005,
) -> str:
    """Bucket the step's outcome so downstream filtering is one column away.

    - `synth_failed`     : success=False and the underlying synth never
                            produced a report (or error names a synth crash)
    - `csim_failed`      : synth passed, csim ran and failed
    - `translation_failed`: error string contains 'translation' / 'no code'
    - `no_op`            : new_report numerics tuple == prev_report tuple
    - `regressed`        : net latency_ns or 3+ resources grew >10%
    - `improved`         : net latency_ns shrank by > epsilon
    - `absorbed`         : neither improved nor regressed; resources stayed
                            within +/- (1 + 2*epsilon) — Vitis already did
                            the optimization (Pillar 7 'Avoid' candidate)
    """
    if not success and error:
        e = error.lower()
        if "translation" in e or "no code" in e or "translator" in e:
            return "translation_failed"
        return "synth_failed"
    if csim_passed is False:
        return "csim_failed"
    if not new_report:
        return "synth_failed"
    if prev_report is None:
        # First step has no parent to compare to; we still want a non-noisy
        # bucket. Treat as 'improved' if it produced a populated report.
        return "improved"
    new_tuple = tuple(new_report.get(k) for k in _NUMERIC_HASH_FIELDS)
    prev_tuple = tuple(prev_report.get(k) for k in _NUMERIC_HASH_FIELDS)
    populated = sum(1 for v in new_tuple if v is not None)
    if new_tuple == prev_tuple and populated >= 3:
        return "no_op"

    new_lat = _as_float(new_report.get("latency_ns"))
    prev_lat = _as_float(prev_report.get("latency_ns"))
    if new_lat is not None and prev_lat is not None and prev_lat > 0:
        ratio = new_lat / prev_lat
        if ratio > 1.10:
            return "regressed"
        if ratio < (1.0 - epsilon):
            return "improved"

    grown = 0
    for k in ("lut", "ff", "bram", "dsp"):
        new_v = _as_float(new_report.get(k))
        prev_v = _as_float(prev_report.get(k))
        if new_v is not None and prev_v is not None and prev_v > 0:
            r = new_v / prev_v
            if r > 1.10:
                grown += 1
    if grown >= 3:
        return "regressed"

    return "absorbed"


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# === Recorder ==============================================================


def _populate_synth_payload(report: Optional[Dict[str, Any]],
                            success: bool) -> HLSSynthPayload:
    if not report:
        return HLSSynthPayload(status="fail" if not success else "pass")
    p = HLSSynthPayload(
        status="pass" if success else "fail",
        latency_cycles=_as_int(report.get("latency_cycles")),
        latency_ns=_as_float(report.get("latency_ns")),
        interval=_as_int(report.get("interval")),
        bram=_as_int(report.get("bram")),
        dsp=_as_int(report.get("dsp")),
        ff=_as_int(report.get("ff")),
        lut=_as_int(report.get("lut")),
        uram=_as_int(report.get("uram")),
        fmax_mhz=_as_float(report.get("fmax_mhz")),
        estimated_clock_period_ns=_as_float(report.get("estimated_clock_period_ns")),
        requested_clock_period_ns=_as_float(report.get("requested_clock_period_ns")),
        slack_ns=_as_float(report.get("slack_ns")),
    )
    return p


def _as_int(value: Any) -> Optional[int]:
    f = _as_float(value)
    return int(f) if f is not None else None


def record_step_outcome(
    *,
    step_result: Dict[str, Any],
    suite: str,
    group_path: List[str],
    variant_index: int,
    variant_name: str,
    run_meta: RunMeta,
    parent_report: Optional[Dict[str, Any]] = None,
    origin: str = "c2hls_orchestrator",
    origin_version: str = "",
    multistep: bool = True,
    rationale: str = "",
    skill_hits: Optional[List[str]] = None,
    relative_advantage: Optional[float] = None,
    rtl_sim_payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convert one multistep step_result into a list of v2.0 records.

    Emits up to two records per step:
    - the `hls_synth` record (always)
    - an `rtl_sim` record when `rtl_sim_payload` is provided (hw_emu /
      sw_emu / cosim outcome from the bench harness)

    Returns a list of dicts ready to write to jsonl.
    """
    success = bool(step_result.get("success"))
    new_report = step_result.get("report") or step_result.get("rejected_report")
    csim = step_result.get("csim")
    csim_passed = None
    if csim is not None:
        csim_passed = bool(
            csim.get("passed", csim.get("status") in {"pass", "ok"})
        )

    validation_status = classify_validation_status(
        csim,
        synth_status="pass" if success else "fail",
    )

    step_effect = classify_step_effect(
        new_report,
        parent_report,
        success=success,
        csim_passed=csim_passed,
        error=step_result.get("error"),
    )
    # If the step_result already carries explicit no-op markers from
    # Pillar 9, prefer them.
    if step_result.get("no_op_reasons") or step_result.get("error") == "no_op_persisted":
        step_effect = "no_op"

    impl = ImplementationMeta(
        origin=origin,
        origin_version=origin_version,
        multistep=multistep,
        step=variant_name,
        step_effect=step_effect,
        validation_status=validation_status,
        parent_hash=report_hash(parent_report),
        candidate_hash=report_hash(new_report),
        relative_advantage=relative_advantage,
        skill_hits=list(skill_hits or []),
        rationale=rationale,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    feedback = None
    if isinstance(new_report, dict):
        feedback = new_report.get("feedback")

    records: List[Dict[str, Any]] = []

    synth_record = TrajectoryRecord(
        report_type="hls_synth",
        run=run_meta,
        problem={"suite": suite, "group_path": list(group_path)},
        implementation=impl,
        variant={"index": variant_index, "name": variant_name},
        hls_synth=_populate_synth_payload(new_report, success),
        feedback=feedback,
    )
    if csim is not None:
        synth_record.csim = CsimPayload(
            available=bool(csim.get("available", csim.get("ran", False))),
            ran=bool(csim.get("ran", False)),
            passed=bool(csim.get("passed", csim.get("status") in {"pass", "ok"})),
            error=csim.get("error"),
        )
    records.append(record_to_dict(synth_record))

    if rtl_sim_payload:
        rtl_record = TrajectoryRecord(
            report_type="rtl_sim",
            run=RunMeta(
                target=rtl_sim_payload.get("target", "vitis.hw_emu"),
                vitis_version=run_meta.vitis_version,
                device=rtl_sim_payload.get("device", run_meta.device),
                flow_target=run_meta.flow_target,
                clock_ns=run_meta.clock_ns,
            ),
            problem={"suite": suite, "group_path": list(group_path)},
            implementation=impl,
            variant={"index": variant_index, "name": variant_name},
            rtl_sim=RTLSimPayload(
                status=rtl_sim_payload.get("status", "fail"),
                kernel_runtime_cycles=_as_int(rtl_sim_payload.get("kernel_runtime_cycles")),
                kernel_runtime_us=_as_float(rtl_sim_payload.get("kernel_runtime_us")),
                kernel_clock_freq_mhz=_as_float(rtl_sim_payload.get("kernel_clock_freq_mhz")),
                error=rtl_sim_payload.get("error"),
            ),
            feedback=None,  # rtl_sim records intentionally omit per-scope
                            # feedback (it belongs on the source synth row).
        )
        records.append(record_to_dict(rtl_record))

    return records

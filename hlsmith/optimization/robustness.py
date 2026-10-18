"""Pillar 9 (full) + Pillar 7 robustness primitives.

Phase 1 already shipped the MVP slice (no-op detector, csim-gating,
xrt.ini auto-inject). Phase 2 completes the remaining items:

- ``trajectory_collapse_check`` — abort when 3 consecutive steps are
  no-ops (Pillar 9 item 2).
- ``throughput_regression_check`` — flag a step when interval >
  latency or interval grew >5% vs parent (Pillar 9 item 3).
- ``translation_failure_retry_payload`` — build the feedback prompt for
  the translator when it returned no code (Pillar 9 item 6).
- ``step_effect_rollback_decision`` — given the per-step Δscore,
  decide whether to keep or roll back (Pillar 9 item 7).

Plus Pillar 7:

- ``mark_absorbed_skills`` — given the trajectory's per-step effects,
  promote skills tagged with the corresponding bottleneck kinds into
  the skill library's Avoid band when 'absorbed' is observed N times.

All functions are pure / side-effect-free except `mark_absorbed_skills`,
which mutates the skill library it is given.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Tunables. Conservative thresholds — easy to lower later once we have
# trajectory data.
TRAJECTORY_COLLAPSE_THRESHOLD = 3   # consecutive no-ops to abort
INTERVAL_REGRESSION_RATIO = 1.05    # interval growth that flags a regression
ABSORBED_OBSERVATIONS_TO_AVOID = 2  # times a skill must be 'absorbed' before
                                     # we move it into the Avoid band


# ---- Pillar 9 item 2: trajectory-collapse guard --------------------------


@dataclass
class CollapseDecision:
    should_abort: bool
    consecutive_no_ops: int
    reason: str


def trajectory_collapse_check(step_effects: Sequence[str],
                              *, threshold: int = TRAJECTORY_COLLAPSE_THRESHOLD) -> CollapseDecision:
    """Inspect the *trailing* run of ``step_effects`` and decide whether
    to abort the trajectory. Only a contiguous tail of 'no_op' /
    'absorbed' counts; any 'improved' / 'regressed' / 'failed' resets
    the count.
    """
    n = 0
    for effect in reversed(step_effects):
        if effect in {"no_op", "absorbed"}:
            n += 1
        else:
            break
    abort = n >= threshold
    return CollapseDecision(
        should_abort=abort,
        consecutive_no_ops=n,
        reason=(
            f"{n} consecutive no-op/absorbed steps reached threshold {threshold}"
            if abort else
            f"{n} trailing no-op/absorbed steps (threshold {threshold} not yet reached)"
        ),
    )


# ---- Pillar 9 item 3: hidden-throughput-regression gate -----------------


@dataclass
class ThroughputCheck:
    flagged: bool
    reasons: List[str]


def throughput_regression_check(new_report: Optional[Dict[str, Any]],
                                 prev_report: Optional[Dict[str, Any]],
                                 *, ratio_limit: float = INTERVAL_REGRESSION_RATIO) -> ThroughputCheck:
    """Two complementary checks:

    1. ``interval > latency`` on the same scope = the kernel can't
       start a new transaction until later than the previous one
       finishes (the pathfinder doublebuffer failure mode from the
       smoke test).
    2. interval grew by > ``ratio_limit`` vs parent = throughput
       regressed even if latency dropped.

    Either firing returns a ThroughputCheck with reasons; both empty
    means the step's throughput is OK."""
    reasons: List[str] = []
    if not new_report:
        return ThroughputCheck(False, reasons)

    new_lat = _coerce_int(new_report.get("latency_cycles"))
    new_ii = _coerce_int(new_report.get("interval"))
    if (new_lat is not None and new_ii is not None and new_lat > 0
            and new_ii > new_lat * 1.05 and new_ii > 16):
        # Vitis reports interval = latency + 1 for non-dataflow kernels in
        # steady state — that's the +1 artifact, not a regression. We only
        # flag when the ratio is meaningfully above 1 (≥1.05x) AND the
        # absolute value is large enough to matter (>16 cycles), so trivial
        # control-only kernels don't false-positive.
        reasons.append(
            f"interval ({new_ii}) significantly exceeds latency ({new_lat}; "
            f"ratio {new_ii / new_lat:.2f}x) — throughput regression hidden "
            f"inside an apparent latency improvement"
        )

    if prev_report:
        prev_ii = _coerce_int(prev_report.get("interval"))
        if (new_ii is not None and prev_ii is not None
                and prev_ii > 0 and new_ii / prev_ii > ratio_limit):
            ratio = new_ii / prev_ii
            reasons.append(
                f"interval grew {ratio:.2f}x vs parent ({prev_ii} → {new_ii}); "
                f"limit {ratio_limit:.2f}x"
            )

    return ThroughputCheck(flagged=bool(reasons), reasons=reasons)


# ---- Pillar 9 item 6: translation-failure retry payload -----------------


def translation_failure_retry_payload(
    *,
    translator_log: str,
    benchmark_name: str,
    benchmark_context: str = "",
) -> str:
    """Build the prompt fragment delivered to the translator when its
    previous attempt produced no usable HLS code. Tells the model
    exactly what was wrong and what shape we expect.

    Caller responsibility: the multistep orchestrator should use this
    payload as `additional_guidance` on a single retry, then surface
    `translation_failed` if the retry also fails."""
    excerpt = translator_log.strip()
    if len(excerpt) > 1500:
        excerpt = excerpt[:1500] + "\n...[truncated]"
    return (
        f"Your previous attempt at translating `{benchmark_name}` produced "
        f"no usable HLS code (the response did not contain a parseable C/C++ "
        f"code fence, or the code did not define an `extern \"C\" workload(...)` "
        f"top function).\n\n"
        f"Translator log excerpt (verbatim):\n```\n{excerpt}\n```\n\n"
        f"On this retry: emit a single ```cpp …``` fenced code block "
        f"containing the full HLS kernel. Define exactly one top function "
        f"matching the benchmark's expected signature. Do not include any "
        f"prose outside the code fence."
        + (f"\n\nBenchmark context:\n{benchmark_context}" if benchmark_context else "")
    )


# ---- Pillar 9 item 7: step-effectiveness rollback decision -------------


@dataclass
class RollbackDecision:
    should_rollback: bool
    reason: str


def step_effect_rollback_decision(
    *,
    delta_score: Optional[float],
    step_effect: str,
    epsilon: float = 0.005,
) -> RollbackDecision:
    """Rule:

    - ``synth_failed`` / ``csim_failed`` / ``regressed`` / ``no_op``
      → roll back.
    - ``absorbed`` → keep, but mark the skill that produced it as a
      candidate for the Avoid band (Pillar 7).
    - ``improved`` → keep.
    - ``unknown`` with no Δscore data → keep (conservative).
    """
    bad = {"synth_failed", "csim_failed", "regressed", "no_op",
           "translation_failed"}
    if step_effect in bad:
        return RollbackDecision(True, f"step_effect={step_effect}")
    if step_effect == "improved":
        return RollbackDecision(False, "step_effect=improved")
    if step_effect == "absorbed":
        return RollbackDecision(False, "step_effect=absorbed (kept; mark as Avoid candidate)")
    if delta_score is not None and delta_score < -epsilon:
        return RollbackDecision(True, f"Δscore={delta_score:.4f} < -ε({epsilon})")
    return RollbackDecision(False, "no rollback signal")


# ---- Pillar 7: absorbed-by-Vitis → Avoid band -------------------------


def mark_absorbed_skills(
    skill_library,        # type: skill_library.SkillLibrary  (forward-ref)
    *,
    observations: Iterable[Dict[str, Any]],
    threshold: int = ABSORBED_OBSERVATIONS_TO_AVOID,
) -> List[str]:
    """Inspect a stream of observations
    ``[{skill_id, step_effect, vitis_version, ...}, …]`` and move any
    skill that has been observed as ``absorbed`` ≥ ``threshold`` times
    into the library's Avoid band.

    Returns the list of skill ids that were demoted on this call.
    """
    counts: Dict[str, int] = {}
    for obs in observations:
        if obs.get("step_effect") == "absorbed":
            sid = obs.get("skill_id")
            if sid:
                counts[sid] = counts.get(sid, 0) + 1

    demoted: List[str] = []
    for sid, n in counts.items():
        if n >= threshold:
            sk = skill_library.mark_avoid(sid, reason=f"absorbed-x{n}")
            if sk is not None:
                demoted.append(sid)
    if demoted:
        logging.info("Pillar 7: demoted %d skill(s) into Avoid band: %s",
                     len(demoted), demoted)
    return demoted


# ---- Helpers ----------------------------------------------------------


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

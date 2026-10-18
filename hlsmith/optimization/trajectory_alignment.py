"""Phase 3 trajectory-aware step evaluation.

The Phase 1 regression guard reverts a step when its synthesis numbers
look worse than the previous step. That's the right call when the step
is supposed to *improve* PPA in isolation — but in the canonical
Rodinia-HLS multistep progression, **some steps are enablers that
regress PPA on their own and only pay off when later steps land**.

The clearest live evidence is in [philip's knn reference jsonl](csynth_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl):

    baseline    1 048 818 cycles
    tiling      4 276 372 cycles  ← +4.08x latency, BUT
    pipeline    4 276 372 cycles  ← byte-identical (no-op)
    unroll      4 044 880 cycles  ← +5% improvement
    doublebuffer  1 740 530 cycles ← below baseline at last
    coalescing      262 480 cycles ← 4x below baseline (the win)

A pure improvement-per-step reverter would have killed `tiling` after
step 1. The reference kept it because tiling is the structural
prerequisite for the dataflow + double-buffer + memory-coalescing wins
later. Our Phase 1 revert mechanism would have done the same thing on
the gen side.

This module adds two pieces:

1. **GT-aware regression tolerance** — if the agent's step regressed
   PPA but the corresponding GT step also regressed at this position
   in the canonical trajectory, the regression is *expected*. Don't
   revert.
2. **Step-effect taxonomy expansion** — new labels
   ``enabling_regress`` (regression that matches GT shape) and
   ``improvement_realized`` (improvement that follows enabling
   regressions) so trajectory consumers can distinguish "bad
   regression" from "expected enabler" downstream.

Used by `c2hls.run_optimization_step` to consult before reverting, and
by `dataset_pipeline.recorder.classify_step_effect` to label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


_RATIO_TOLERANCE = 0.20   # gen ratio within ±20% of GT ratio is "consistent"
_GT_REGRESSED_MIN = 1.10  # GT had to regress >=10% itself for tolerance to apply


@dataclass
class TrajectoryAlignmentDecision:
    consistent_with_gt: bool
    reason: str
    gt_latency_ratio: Optional[float] = None
    gen_latency_ratio: Optional[float] = None
    gt_resource_ratios: Optional[Dict[str, float]] = None


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _ratio(new_v: Any, parent_v: Any) -> Optional[float]:
    n = _to_float(new_v)
    p = _to_float(parent_v)
    if n is None or p is None or p <= 0:
        return None
    return n / p


def is_consistent_with_gt_trajectory(
    *,
    gen_report: Dict[str, Any],
    parent_gen_report: Optional[Dict[str, Any]],
    gt_report: Optional[Dict[str, Any]],
    parent_gt_report: Optional[Dict[str, Any]],
    ratio_tolerance: float = _RATIO_TOLERANCE,
) -> TrajectoryAlignmentDecision:
    """Decide whether a regression in ``gen_report`` (vs
    ``parent_gen_report``) is *consistent with* the corresponding step
    in the GT reference (``gt_report`` vs ``parent_gt_report``).

    "Consistent" means: the GT also regressed by a similar magnitude at
    this position, so the regression is structural / expected and the
    step should be kept rather than reverted.

    Returns ``TrajectoryAlignmentDecision`` with the per-axis ratios so
    callers can render diagnostic prompts.
    """
    gen_lat_ratio = _ratio(
        (gen_report or {}).get("latency_cycles") or (gen_report or {}).get("latency_ns"),
        (parent_gen_report or {}).get("latency_cycles") or (parent_gen_report or {}).get("latency_ns"),
    )
    gt_lat_ratio = _ratio(
        (gt_report or {}).get("latency_cycles") or (gt_report or {}).get("latency_ns"),
        (parent_gt_report or {}).get("latency_cycles") or (parent_gt_report or {}).get("latency_ns"),
    )

    if gt_lat_ratio is None:
        return TrajectoryAlignmentDecision(
            consistent_with_gt=False,
            reason="no GT trajectory data to compare against",
            gen_latency_ratio=gen_lat_ratio,
        )

    # Only tolerate gen regression if GT itself regressed by ≥ _GT_REGRESSED_MIN.
    # If GT improved at this step, gen has no excuse to regress.
    if gt_lat_ratio < _GT_REGRESSED_MIN:
        return TrajectoryAlignmentDecision(
            consistent_with_gt=False,
            reason=(
                f"GT latency at this step ratio={gt_lat_ratio:.3f} did not "
                f"regress; gen ratio={gen_lat_ratio} should also not regress"
            ),
            gen_latency_ratio=gen_lat_ratio,
            gt_latency_ratio=gt_lat_ratio,
        )

    # GT did regress here. If the gen ratio is within ±tolerance of the GT
    # ratio (multiplicative), call it consistent.
    if gen_lat_ratio is not None:
        # Ratio of ratios: how does gen compare to GT?
        rr = gen_lat_ratio / gt_lat_ratio
        if (1 - ratio_tolerance) <= rr <= (1 + ratio_tolerance):
            return TrajectoryAlignmentDecision(
                consistent_with_gt=True,
                reason=(
                    f"gen latency ratio {gen_lat_ratio:.3f} ≈ GT ratio "
                    f"{gt_lat_ratio:.3f} (within ±{ratio_tolerance:.0%}); "
                    f"this step is a structural enabler (matches GT shape)"
                ),
                gen_latency_ratio=gen_lat_ratio,
                gt_latency_ratio=gt_lat_ratio,
            )

    # Gen regressed differently from GT — looks like a real bad step.
    return TrajectoryAlignmentDecision(
        consistent_with_gt=False,
        reason=(
            f"gen latency ratio {gen_lat_ratio} departs from GT ratio "
            f"{gt_lat_ratio:.3f} by more than ±{ratio_tolerance:.0%}; "
            f"genuine regression"
        ),
        gen_latency_ratio=gen_lat_ratio,
        gt_latency_ratio=gt_lat_ratio,
    )


def render_alignment_for_history(decision: TrajectoryAlignmentDecision,
                                  step_name: str) -> str:
    """One-line history record for a trajectory-alignment decision."""
    icon = "✓" if decision.consistent_with_gt else "✗"
    parts = [
        f"[{step_name}] alignment={icon}: {decision.reason}",
    ]
    if decision.gen_latency_ratio is not None:
        parts.append(f"gen_lat_ratio={decision.gen_latency_ratio:.3f}")
    if decision.gt_latency_ratio is not None:
        parts.append(f"gt_lat_ratio={decision.gt_latency_ratio:.3f}")
    return " ".join(parts)


def render_alignment_for_prompt(decision: TrajectoryAlignmentDecision,
                                 step_name: str) -> str:
    """If the gen step regressed but the GT also regressed at this
    position, deliver a context block to the LLM explaining that this
    is *expected* so it doesn't try to "fix" what isn't broken.
    Returns "" when no message is needed."""
    if not decision.consistent_with_gt:
        return ""
    return (
        f"NOTE on the `{step_name}` step: this step is known to be a "
        f"*structural enabler* — the canonical reference trajectory also "
        f"regresses here (latency ratio {decision.gt_latency_ratio:.2f}x "
        f"vs the previous step in the gold reference). The PPA cost will "
        f"only pay off when later steps (typically doublebuffer / "
        f"coalescing) land. Stay close to the gold reference's pragma "
        f"set; do NOT add aggressive optimizations to mask the regression."
    )


def classify_step_effect_with_alignment(
    base_effect: str,
    alignment: Optional[TrajectoryAlignmentDecision] = None,
    *,
    realized_after_enabling: bool = False,
) -> str:
    """Map the existing 5-effect taxonomy through the alignment signal.

    - ``regressed`` + ``consistent_with_gt=True`` → ``enabling_regress``
    - ``improved`` + ``realized_after_enabling=True`` → ``improvement_realized``
    - everything else → unchanged base effect

    The ``realized_after_enabling`` flag is computed by the caller from
    the trajectory shape (one or more recent steps were
    ``enabling_regress``). We don't compute it here so this function
    stays pure.
    """
    if alignment is not None and alignment.consistent_with_gt and base_effect == "regressed":
        return "enabling_regress"
    if base_effect == "improved" and realized_after_enabling:
        return "improvement_realized"
    return base_effect

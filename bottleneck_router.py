"""Bottleneck-driven action routing (Pillar 5).

Replaces the fixed `tiling → pipeline → unroll → doublebuffer →
coalescing` order with a routing table that consults the Pillar 1
feedback (typed bottleneck records) and the Pillar 3 skill library to
pick what to apply next. This is the "human-engineer cognitive
alignment" piece of the plan: profile first, then fix the dominant
bottleneck.

Two modes:

    static   (default backward-compatible behavior):
             return DEFAULT_OPT_STEPS in order.
    dynamic  (Pillar 5):
             pick the highest-severity unsolved bottleneck, query the
             skill library for the best matching skill, return the
             corresponding step name (or skill template if no fixed
             step matches).

If the library has no relevant skill, fall back to the static order so
the multistep flow always makes progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from skill_library import (
    Skill,
    SkillLibrary,
    TIER_AVOID,
    render_skill_set_for_prompt,
)


# Map skill `tags`/id → c2hls step name. The router uses this to convert
# a chosen skill back into one of the existing OPTIMIZATION_PROMPTS keys
# so the multistep orchestrator can run it without changes. When no
# mapping exists, the router falls back to the skill's template/strategy
# delivered as a generic-step prompt.
_SKILL_TO_STEP: Dict[str, str] = {
    "prompt-tiling": "tiling",
    "prompt-pipeline": "pipeline",
    "prompt-unroll": "unroll",
    "prompt-doublebuffer": "doublebuffer",
    "prompt-coalescing": "coalescing",
}

# Direct bottleneck-kind → preferred step. Used when the skill library
# is empty or filtered out everything.
_BOTTLENECK_FALLBACK: Dict[str, str] = {
    "non_pipelined_hot_loop": "pipeline",
    "ii_target_miss": "unroll",
    "interval_exceeds_latency": "doublebuffer",
    "port_conflict": "coalescing",
    "loop_carried_dep": "pipeline",
    "pipeline_blocked": "pipeline",
    "dataflow_blocked": "doublebuffer",
    "timing_violation": "pipeline",
}


@dataclass
class RoutingDecision:
    step_name: str
    reason: str
    bottleneck_kind: Optional[str]
    skill_id: Optional[str]
    confidence: Optional[str]
    fallback: bool = False  # True if we couldn't find a matching skill


def select_next_step(
    *,
    feedback: Optional[Dict[str, Any]],
    library: Optional[SkillLibrary],
    completed_steps: Sequence[str] = (),
    available_steps: Optional[Sequence[str]] = None,
    vitis_version: Optional[str] = None,
    fpga: Optional[str] = None,
    static_order: Sequence[str] = ("tiling", "pipeline", "unroll",
                                    "doublebuffer", "coalescing"),
) -> RoutingDecision:
    """Pick the next step to run.

    Resolution order:
    1. If feedback has bottlenecks, pick the highest-severity one and
       query the skill library for matching, version-applicable skills.
    2. If a candidate skill maps to an OPTIMIZATION_PROMPTS step that
       hasn't been tried yet, return it.
    3. Otherwise fall back to static_order, skipping completed_steps.
    """
    completed = set(completed_steps or ())
    candidates = list(available_steps or static_order)
    candidates = [s for s in candidates if s not in completed]

    bottlenecks = []
    if feedback:
        bottlenecks = list(feedback.get("bottlenecks") or [])

    # Severity-ranked iteration.
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    bottlenecks.sort(key=lambda b: severity_rank.get(b.get("severity"), 3))

    for bn in bottlenecks:
        kind = bn.get("kind")
        if not kind:
            continue
        # Skill-library route.
        if library is not None:
            for sk in library.query(
                bottleneck_kind=kind,
                vitis_version=vitis_version,
                fpga=fpga,
                include_avoid=False,
            ):
                step = _SKILL_TO_STEP.get(sk.id)
                if step and step in candidates:
                    return RoutingDecision(
                        step_name=step,
                        reason=(
                            f"matched bottleneck '{kind}' → skill '{sk.id}' "
                            f"(confidence={sk.confidence}, adv={sk.mean_advantage:.3f})"
                        ),
                        bottleneck_kind=kind,
                        skill_id=sk.id,
                        confidence=sk.confidence,
                        fallback=False,
                    )
        # Direct fallback by bottleneck kind.
        step = _BOTTLENECK_FALLBACK.get(kind)
        if step and step in candidates:
            return RoutingDecision(
                step_name=step,
                reason=(
                    f"matched bottleneck '{kind}' via direct fallback "
                    f"(no skill in library matched)"
                ),
                bottleneck_kind=kind,
                skill_id=None,
                confidence=None,
                fallback=True,
            )

    # No bottleneck routing succeeded — fall back to static order.
    if not candidates:
        return RoutingDecision(
            step_name="",
            reason="no remaining steps in static_order; trajectory complete",
            bottleneck_kind=None, skill_id=None, confidence=None, fallback=True,
        )
    return RoutingDecision(
        step_name=candidates[0],
        reason="no actionable bottleneck found; advancing static order",
        bottleneck_kind=None,
        skill_id=None,
        confidence=None,
        fallback=True,
    )


def plan_dynamic_trajectory(
    *,
    initial_feedback: Optional[Dict[str, Any]],
    library: Optional[SkillLibrary],
    available_steps: Sequence[str],
    vitis_version: Optional[str] = None,
    fpga: Optional[str] = None,
    max_steps: int = 5,
) -> List[RoutingDecision]:
    """Offline planner: given the *initial* feedback, project an entire
    trajectory greedily. Used when the orchestrator wants to print the
    plan before running it. Real online use should call
    `select_next_step()` after every step."""
    decisions: List[RoutingDecision] = []
    completed: List[str] = []
    feedback = initial_feedback
    for _ in range(max_steps):
        d = select_next_step(
            feedback=feedback,
            library=library,
            completed_steps=completed,
            available_steps=available_steps,
            vitis_version=vitis_version,
            fpga=fpga,
        )
        if not d.step_name:
            break
        decisions.append(d)
        completed.append(d.step_name)
        # We don't have post-step feedback at planning time, so the loop
        # converges via static_order once it runs out of skill matches.
    return decisions


def render_decision_for_prompt(decision: RoutingDecision,
                               library: Optional[SkillLibrary] = None) -> str:
    lines = [f"Routing decision: step={decision.step_name}",
             f"  reason: {decision.reason}"]
    if decision.bottleneck_kind:
        lines.append(f"  bottleneck: {decision.bottleneck_kind}")
    if decision.skill_id and library is not None:
        sk = library.get(decision.skill_id)
        if sk is not None:
            lines.append("  skill context:")
            lines.append("    " + render_skill_set_for_prompt([sk]).replace("\n", "\n    "))
    return "\n".join(lines)

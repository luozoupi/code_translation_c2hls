"""Versioned setup registry and deterministic HLS tournament selection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


REGISTRY_SCHEMA_VERSION = "c2hls.setup-registry.v2"
TOURNAMENT_SCHEMA_VERSION = "c2hls.setup-tournament.v1"
LEGACY_VERSION = "legacy_v1"
CORRECTED_VERSION = "corrected_v2"

LEGACY_SKILL_SCOPES = (
    "skillless",
    "matched",
    "smart_best_fit",
    "smart_exhaustive",
    "all_positive",
)
CORRECTED_SKILL_SCOPES = (
    "skillless",
    "matched_positive",
    "smart_best_fit_v2",
    "smart_exhaustive_v2",
    "all_positive_preconditions",
)
STRATEGIES = ("flash", "multistep")
U280_CAPACITY = {
    "bram": 4032,
    "dsp": 9024,
    "ff": 2_607_360,
    "lut": 1_303_680,
    "uram": 960,
}


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class SetupSpec:
    setup_id: str
    behavior_version: str
    strategy: str
    skill_scope: str
    prompt_mode: str
    router_version: int
    candidate_policy: str

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": REGISTRY_SCHEMA_VERSION,
                **asdict(self),
            }
        )

    def to_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "setup_fingerprint": self.fingerprint,
        }


def setup_registry(version: str = CORRECTED_VERSION) -> list[SetupSpec]:
    if version == LEGACY_VERSION:
        scopes = LEGACY_SKILL_SCOPES
    elif version == CORRECTED_VERSION:
        scopes = CORRECTED_SKILL_SCOPES
    else:
        raise ValueError(f"unknown setup registry version: {version!r}")

    records: list[SetupSpec] = []
    for strategy in STRATEGIES:
        for scope in scopes:
            corrected = version == CORRECTED_VERSION
            prompt_mode = (
                "positive_with_preconditions"
                if corrected and scope != "skillless"
                else "literal_empty"
                if scope == "skillless"
                else "action_only"
                if scope in {"smart_best_fit", "smart_exhaustive", "all_positive"}
                else "default"
            )
            records.append(
                SetupSpec(
                    setup_id=f"{version}:{strategy}:{scope}",
                    behavior_version=version,
                    strategy=strategy,
                    skill_scope=scope,
                    prompt_mode=prompt_mode,
                    router_version=(
                        2
                        if corrected and scope.startswith("smart_")
                        else 1
                        if scope.startswith("smart_")
                        else 0
                    ),
                    candidate_policy=(
                        "separate_skill_directed_candidates"
                        if scope == "smart_exhaustive_v2"
                        else "single_completion"
                    ),
                )
            )
    return records


def registry_by_id(version: str = CORRECTED_VERSION) -> dict[str, SetupSpec]:
    return {setup.setup_id: setup for setup in setup_registry(version)}


def resolve_policy_setups(
    *,
    policy: str,
    predicted_setup_ids: Sequence[str] = (),
    version: str = CORRECTED_VERSION,
    prediction_metadata: dict[str, Any] | None = None,
) -> list[SetupSpec]:
    """Resolve exhaustive, advisory, or learned setup evaluation."""

    registry = registry_by_id(version)
    normalized = policy.strip().lower().replace("-", "_")
    if normalized == "exhaustive":
        return list(registry.values())
    ranked = [
        registry[setup_id]
        for setup_id in predicted_setup_ids
        if setup_id in registry
    ]
    ranked = list({setup.setup_id: setup for setup in ranked}.values())
    if normalized == "advisory":
        return ranked
    if normalized not in {"learned_top_k", "adaptive_diverse_top_k"}:
        raise ValueError(f"unknown tournament policy: {policy!r}")
    baseline_id = f"{version}:multistep:skillless"
    selected = [registry[baseline_id]]
    alternatives = [
        setup for setup in ranked if setup.setup_id != baseline_id
    ]
    if normalized == "learned_top_k":
        selected.extend(alternatives)
        selected = list(
            {setup.setup_id: setup for setup in selected}.values()
        )
        return selected[:3]

    budget = adaptive_candidate_budget(prediction_metadata)
    selected.extend(alternatives[:2])
    remaining = [
        setup
        for setup in alternatives[2:]
        if setup.setup_id not in {item.setup_id for item in selected}
    ]
    while len(selected) < budget and remaining:
        strategies = {item.strategy for item in selected}
        scopes = {item.skill_scope for item in selected}
        ranked_remaining = {item.setup_id: i for i, item in enumerate(remaining)}
        next_setup = min(
            remaining,
            key=lambda item: (
                -(item.skill_scope not in scopes),
                -(item.strategy not in strategies),
                ranked_remaining[item.setup_id],
                item.fingerprint,
            ),
        )
        selected.append(next_setup)
        remaining.remove(next_setup)
    if len(selected) < budget:
        selected.extend(
            setup
            for setup in registry.values()
            if setup.setup_id not in {item.setup_id for item in selected}
        )
    return selected[:budget]


def adaptive_candidate_budget(
    prediction_metadata: dict[str, Any] | None,
) -> int:
    """Return the conservative 3/5/8 candidate budget for a prediction."""

    metadata = prediction_metadata or {}
    explicit = metadata.get("recommended_candidate_budget")
    if explicit is None:
        explicit = metadata.get("candidate_budget")
    if explicit is not None:
        try:
            parsed = int(explicit)
        except (TypeError, ValueError):
            parsed = 8
        return min(8, max(3, parsed))

    disagreement = metadata.get("committee_disagreement")
    ood_score = metadata.get("ood_score")
    try:
        disagreement_value = float(disagreement)
    except (TypeError, ValueError):
        disagreement_value = 1.0
    try:
        ood_value = float(ood_score)
    except (TypeError, ValueError):
        ood_value = 1.0
    if disagreement_value >= 0.67 or ood_value >= 0.75:
        return 8
    if disagreement_value > 0.0 or ood_value >= 0.4:
        return 5
    return 3


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def candidate_feasibility(candidate: dict[str, Any]) -> dict[str, Any]:
    report = (
        candidate.get("final_report")
        or candidate.get("synth_report")
        or candidate.get("report")
        or {}
    )
    csim = candidate.get("csim") or {}
    cycles = _positive_int(
        report.get("latency_cycles_worst")
        or report.get("latency_cycles")
    )
    csim_passed = (
        isinstance(csim, dict)
        and csim.get("ran") is True
        and csim.get("passed") is True
    )
    csynth_passed = bool(report) and cycles is not None

    estimated = report.get("estimated_clock_period_ns")
    requested = report.get("requested_clock_period_ns")
    slack = report.get("slack_ns")
    timing_met = None
    if isinstance(slack, (int, float)) and not isinstance(slack, bool):
        timing_met = float(slack) >= 0.0
    elif all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (estimated, requested)
    ):
        timing_met = float(estimated) <= float(requested)

    resources: dict[str, int | None] = {}
    resource_fit = True
    for name, capacity in U280_CAPACITY.items():
        value = report.get(name)
        parsed = (
            int(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )
        resources[name] = parsed
        if parsed is None or parsed < 0 or parsed > capacity:
            resource_fit = False

    reasons: list[str] = []
    if not csim_passed:
        reasons.append("csim_not_passed")
    if not csynth_passed:
        reasons.append("csynth_missing_exact_cycles")
    if timing_met is not True:
        reasons.append("timing_not_met_or_unavailable")
    if not resource_fit:
        reasons.append("resource_fit_failed_or_unavailable")
    return {
        "feasible": not reasons,
        "reasons": reasons,
        "csim_passed": csim_passed,
        "csynth_passed": csynth_passed,
        "timing_met": timing_met,
        "resource_fit": resource_fit,
        "latency_cycles": cycles,
        "resources": resources,
    }


def _candidate_setup_fingerprint(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("setup_fingerprint")
        or (candidate.get("setup") or {}).get("setup_fingerprint")
        or ""
    )


def _code_hash(candidate: dict[str, Any]) -> str:
    existing = candidate.get("code_sha256") or candidate.get(
        "selected_code_sha256"
    )
    if existing:
        return str(existing)
    code = str(candidate.get("hls_code") or candidate.get("code") or "")
    return hashlib.sha256(code.encode("utf-8")).hexdigest() if code else ""


def _candidate_setup_id(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("setup_id")
        or (candidate.get("setup") or {}).get("setup_id")
        or ""
    )


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _winner_trajectory(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory = []
    retained_best = None
    for item in candidate.get("best_so_far_history") or []:
        if not isinstance(item, dict):
            continue
        report = item.get("report") or {}
        cycles = _positive_int(
            report.get("latency_cycles_worst")
            or report.get("latency_cycles")
        )
        if cycles is None:
            continue
        previous_best = retained_best
        improved = retained_best is None or cycles < retained_best
        if improved:
            retained_best = cycles
        trajectory.append(
            {
                "step_name": str(item.get("step_name") or ""),
                "candidate_latency_cycles": cycles,
                "candidate_improved_best": improved,
                "retained_best_latency_cycles": retained_best,
                "speedup_over_previous_retained_best": (
                    previous_best / retained_best
                    if previous_best is not None and improved
                    else None
                ),
            }
        )
    return trajectory


def _winner_skill_evidence(candidate: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "catalog_skill_ids",
        "routed_skill_ids",
        "rendered_skill_ids",
        "declared_applied_skill_ids",
        "verified_applied_skill_ids",
        "synthesized_candidate_skill_ids",
    )
    aggregate: dict[str, list[str]] = {field: [] for field in fields}
    steps = []
    source_steps = (
        candidate.get("generated_step_history")
        or candidate.get("optimization_history")
        or candidate.get("steps")
        or []
    )
    for item in source_steps:
        if not isinstance(item, dict):
            continue
        prompt = item.get("skill_prompt")
        if not isinstance(prompt, dict) or not prompt:
            continue
        routing = item.get("routing_decision") or {}
        step = {
            "step_name": str(item.get("step_name") or ""),
            "routing_reason": (
                str(routing.get("reason") or "")
                if isinstance(routing, dict)
                else ""
            ),
        }
        for field in fields:
            values = _ordered_unique(prompt.get(field) or [])
            step[field] = values
            aggregate[field].extend(values)
        steps.append(step)
    for field in fields:
        aggregate[field] = _ordered_unique(aggregate[field])
        aggregate[field.replace("_ids", "_count")] = len(aggregate[field])
    return {**aggregate, "steps": steps}


def _mode_fit_evidence(
    *,
    winner: dict[str, Any],
    feasible: list[dict[str, Any]],
    runner_up: dict[str, Any] | None,
) -> dict[str, Any]:
    setup_id = _candidate_setup_id(winner)
    setup = winner.get("setup") or {}
    winner_cycles = winner["tournament_feasibility"]["latency_cycles"]
    skillless = [
        candidate
        for candidate in feasible
        if _candidate_setup_id(candidate).endswith(":skillless")
    ]
    best_skillless = (
        min(
            skillless,
            key=lambda candidate: (
                candidate["tournament_feasibility"]["latency_cycles"],
                candidate["setup_fingerprint"],
            ),
        )
        if skillless
        else None
    )
    best_skillless_cycles = (
        best_skillless["tournament_feasibility"]["latency_cycles"]
        if best_skillless is not None
        else None
    )
    baseline_report = winner.get("baseline_report") or {}
    baseline_cycles = _positive_int(
        baseline_report.get("latency_cycles_worst")
        or baseline_report.get("latency_cycles")
    )
    bottlenecks = _ordered_unique(
        item.get("kind")
        for item in (
            (baseline_report.get("feedback") or {}).get("bottlenecks")
            or []
        )
        if isinstance(item, dict)
    )
    final_report = (
        winner.get("final_report")
        or winner.get("synth_report")
        or winner.get("report")
        or {}
    )
    skill_evidence = _winner_skill_evidence(winner)
    observations = [
        (
            f"The selected setup used {setup.get('strategy') or 'unknown'} "
            f"strategy with {setup.get('skill_scope') or 'unknown'} routing."
        )
    ]
    if baseline_cycles:
        observations.append(
            f"Its retained candidate reduced the frozen Phase-B latency from "
            f"{baseline_cycles} to {winner_cycles} cycles "
            f"({baseline_cycles / winner_cycles:.4f}x)."
        )
    if best_skillless_cycles:
        observations.append(
            f"It was {best_skillless_cycles / winner_cycles:.4f}x faster "
            "than the best feasible skillless setup in this tournament."
        )
    if bottlenecks:
        observations.append(
            "The Phase-B report exposed these typed bottlenecks: "
            + ", ".join(bottlenecks)
            + "."
        )
    if skill_evidence["verified_applied_skill_ids"]:
        observations.append(
            "Code-difference telemetry verified application of: "
            + ", ".join(skill_evidence["verified_applied_skill_ids"])
            + "."
        )
    elif skill_evidence["rendered_skill_ids"]:
        observations.append(
            "Skills were rendered, but none were verified by the current "
            "code-difference detector."
        )
    else:
        observations.append("The selected setup rendered no skills.")
    return {
        "schema_version": "c2hls.mode-fit-evidence.v1",
        "winner_setup": {
            "setup_id": setup_id,
            "strategy": setup.get("strategy"),
            "skill_scope": setup.get("skill_scope"),
            "prompt_mode": setup.get("prompt_mode"),
            "candidate_policy": setup.get("candidate_policy"),
        },
        "phase_b": {
            "latency_cycles": baseline_cycles,
            "bottleneck_kinds": bottlenecks,
        },
        "winner": {
            "latency_cycles": winner_cycles,
            "estimated_clock_period_ns": final_report.get(
                "estimated_clock_period_ns"
            ),
            "requested_clock_period_ns": final_report.get(
                "requested_clock_period_ns"
            ),
            "slack_ns": final_report.get("slack_ns"),
            "resources": {
                name: final_report.get(name) for name in U280_CAPACITY
            },
        },
        "best_skillless": {
            "setup_id": (
                _candidate_setup_id(best_skillless)
                if best_skillless is not None
                else None
            ),
            "latency_cycles": best_skillless_cycles,
            "winner_speedup": (
                best_skillless_cycles / winner_cycles
                if best_skillless_cycles
                else None
            ),
        },
        "runner_up": {
            "setup_id": (
                _candidate_setup_id(runner_up)
                if runner_up is not None
                else None
            ),
            "latency_cycles": (
                runner_up["tournament_feasibility"]["latency_cycles"]
                if runner_up is not None
                else None
            ),
        },
        "step_measurements": _winner_trajectory(winner),
        "skill_evidence": skill_evidence,
        "observations": observations,
        "interpretation_limit": (
            "These are within-run measured associations. They explain the "
            "selection evidence but do not establish that the setup or any "
            "individual skill caused the latency reduction."
        ),
    }


def select_tournament_winner(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Select the deterministic minimum-cycle feasible candidate."""

    measurements: list[dict[str, Any]] = []
    for index, raw in enumerate(candidates):
        candidate = dict(raw)
        feasibility = candidate_feasibility(candidate)
        measurements.append(
            {
                **candidate,
                "candidate_index": candidate.get("candidate_index", index),
                "setup_fingerprint": _candidate_setup_fingerprint(candidate),
                "code_sha256": _code_hash(candidate),
                "tournament_feasibility": feasibility,
            }
        )
    feasible = [
        candidate
        for candidate in measurements
        if candidate["tournament_feasibility"]["feasible"]
    ]
    if not feasible:
        return {
            "schema_version": TOURNAMENT_SCHEMA_VERSION,
            "success": False,
            "winner": None,
            "candidate_measurements": measurements,
            "winner_explanation": {
                "summary": "No candidate passed CSim, CSynth, timing, and resource-fit gates.",
                "feasible_candidate_count": 0,
            },
        }

    winner = min(
        feasible,
        key=lambda candidate: (
            candidate["tournament_feasibility"]["latency_cycles"],
            candidate["setup_fingerprint"],
            candidate["code_sha256"],
            int(candidate["candidate_index"]),
        ),
    )
    runner_ups = sorted(
        (candidate for candidate in feasible if candidate is not winner),
        key=lambda candidate: (
            candidate["tournament_feasibility"]["latency_cycles"],
            candidate["setup_fingerprint"],
            candidate["code_sha256"],
            int(candidate["candidate_index"]),
        ),
    )
    runner_up = runner_ups[0] if runner_ups else None
    next_cycles = (
        runner_up["tournament_feasibility"]["latency_cycles"]
        if runner_up is not None
        else None
    )
    winner_cycles = winner["tournament_feasibility"]["latency_cycles"]
    margin = (
        (next_cycles / winner_cycles) if next_cycles is not None else None
    )
    setup_id = _candidate_setup_id(winner)
    mode_fit_evidence = _mode_fit_evidence(
        winner=winner,
        feasible=feasible,
        runner_up=runner_up,
    )
    explanation = {
        "schema_version": "c2hls.tournament-winner-explanation.v2",
        "selection_rule": (
            "minimum exact CSynth cycles among candidates passing executed "
            "CSim, CSynth, target timing, and U280 resource-fit gates; "
            "ties use setup fingerprint, code hash, then candidate index"
        ),
        "winner_setup_id": setup_id,
        "winner_setup_fingerprint": winner["setup_fingerprint"],
        "winner_latency_cycles": winner_cycles,
        "runner_up_latency_cycles": next_cycles,
        "speedup_over_runner_up": margin,
        "feasible_candidate_count": len(feasible),
        "evaluated_candidate_count": len(measurements),
        "mode_fit_evidence": mode_fit_evidence,
        "summary": (
            f"{setup_id or winner['setup_fingerprint']} won with "
            f"{winner_cycles} cycles after passing all feasibility gates"
            + (
                f", {margin:.4f}x faster than the next feasible candidate."
                if margin is not None and math.isfinite(margin)
                else "."
            )
        ),
    }
    return {
        "schema_version": TOURNAMENT_SCHEMA_VERSION,
        "success": True,
        "winner": winner,
        "candidate_measurements": measurements,
        "winner_explanation": explanation,
    }

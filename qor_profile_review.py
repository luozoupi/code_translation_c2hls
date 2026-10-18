"""Reference-blind, evidence-qualified skill and hardware-tradeoff reviews."""

from __future__ import annotations

import hashlib
import math
import os
import re

from qor_design_space import extract_qor_metrics
from skill_usage import _skill_family


REVIEW_VERSION = "profile_skill_tradeoff_v1"
REVIEW_ENV = "C2HLS_QOR_PROFILE_REVIEW"
RESOURCES = ("dsp", "bram", "lut", "ff", "uram")


def enabled():
    return os.getenv(REVIEW_ENV, "0").lower() in {"1", "true", "yes", "on"}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def source_text(code):
    return re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                  lambda m: re.sub(r"[^\n]", " ", m.group()), code or "", flags=re.S)


def source_evidence(code, family):
    text = source_text(code)
    markers = {
        "pipeline": r"#\s*pragma\s+HLS\s+PIPELINE\b[^\n]*",
        "unroll": r"#\s*pragma\s+HLS\s+UNROLL\b[^\n]*",
        "partition": r"#\s*pragma\s+HLS\s+ARRAY_(?:PARTITION|RESHAPE)\b[^\n]*",
        "dataflow": r"#\s*pragma\s+HLS\s+(?:DATAFLOW|STREAM)\b[^\n]*",
        "dependence": r"#\s*pragma\s+HLS\s+DEPENDENCE\b[^\n]*",
        "coalescing": r"max_widen_bitwidth\s*=\s*\w+|ap_u?int\s*<\s*\d+\s*>",
        "multibank": r"bundle\s*=\s*\w+",
        "tiling": r"[^\n;]*\b(?:tile\w*|block\w*)\b[^\n;]*",
        "reduction": r"[^\n;]*\b(?:partial|lane)_?sum\w*\b[^\n;]*",
    }
    pattern = markers.get(family)
    if not pattern:
        return []
    return sorted(set(re.sub(r"\s+", " ", m.group().strip()).lower()
                      for m in re.finditer(pattern, text, re.I)))


def bottlenecks(report):
    records = ((report or {}).get("feedback") or {}).get("bottlenecks") or []
    return [{"kind": str(r.get("kind") or ""),
             "scope": str(r.get("scope_id") or r.get("scope") or "unknown"),
             "evidence": str(r.get("evidence") or r.get("message") or "")[:600]}
            for r in records if isinstance(r, dict)]


def resource_guidance(report, capacity):
    usage = []
    for key in RESOURCES:
        used, cap = number((report or {}).get(key)), number(capacity.get(key))
        if used is not None and cap and cap > 0:
            usage.append(f"{key.upper()} {used:g}/{cap:g} ({100 * used / cap:.2f}% used)")
    return (
        "HARDWARE/PERFORMANCE DESIGN REVIEW:\n"
        "Optimize the latency/resource tradeoff, not minimum hardware usage. Device capacity is a "
        "physical feasibility ceiling, not the previous kernel's allocation budget. A faster design "
        "may legitimately use substantially more DSP, BRAM, LUT, FF or URAM. Do not preserve a "
        "serial low-utilization architecture merely to keep its resource counts small.\n"
        "Use measured hot-loop, dependency and memory-port evidence to choose between a modest "
        "design and a wider parallel design. Match compute lanes with memory banks/bandwidth and "
        "reduction structure; scale tiles/buffers together with their bounds and boundary guards. "
        "Choose one concrete candidate for this call and state the expected bottleneck effect and "
        "hardware cost in a short code comment. More resources are justified only by measured "
        "benefit; do not target an arbitrary utilization percentage. Preserve true dependencies, "
        "CSim correctness, target timing, and device fit.\n"
        "Selection stays latency-first among feasible designs; retain alternative Pareto tradeoffs "
        "in the experiment record. Missing resource/profile evidence means unknown, not zero.\n"
        "Measured current utilization: " + ("; ".join(usage) or "unavailable")
    )


def review_candidate(*, parent_code, candidate_code, parent_report, candidate_report,
                     skill_meta, library, feasibility, capacity, synth_success):
    before, after = extract_qor_metrics(parent_report), extract_qor_metrics(candidate_report)
    parent_cycles, candidate_cycles = before.get("latency_cycles_worst"), after.get("latency_cycles_worst")
    valid = bool(synth_success and feasibility.get("feasible"))
    comparable = valid and parent_cycles is not None and parent_cycles > 0 and candidate_cycles is not None and candidate_cycles > 0
    effect = ("improved" if candidate_cycles < parent_cycles else "regressed" if candidate_cycles > parent_cycles else "unchanged") if comparable else "unmeasured_or_invalid"
    deltas = {key: after[key] - before[key] for key in (*RESOURCES, "interval", "achieved_pipeline_ii_max", "estimated_clock_period_ns")
              if number(after.get(key)) is not None and number(before.get(key)) is not None}
    resource_known = all(number(before.get(k)) is not None and number(after.get(k)) is not None for k in RESOURCES)
    dominates = bool(comparable and resource_known and candidate_cycles <= parent_cycles
                     and all(deltas[k] <= 0 for k in RESOURCES)
                     and (candidate_cycles < parent_cycles or any(deltas[k] < 0 for k in RESOURCES)))
    dominated = bool(comparable and resource_known and candidate_cycles >= parent_cycles
                    and all(deltas[k] >= 0 for k in RESOURCES)
                    and (candidate_cycles > parent_cycles or any(deltas[k] > 0 for k in RESOURCES)))
    tradeoff = "candidate_dominates_parent" if dominates else "parent_dominates_candidate" if dominated else "tradeoff_or_equal" if comparable and resource_known else "insufficient_evidence"
    prior_bns, current_bns = bottlenecks(parent_report), bottlenecks(candidate_report)
    rendered = list(dict.fromkeys(skill_meta.get("rendered_skill_ids") or []))
    declared = set(skill_meta.get("declared_applied_skill_ids") or [])
    reviews = []
    # Only exposed skills are inspected. In skillless mode the library is never queried.
    for skill_id in rendered:
        skill = library.get(skill_id) if library is not None else None
        family = _skill_family(skill) if skill is not None else "unknown"
        prior_marks, current_marks = source_evidence(parent_code, family), source_evidence(candidate_code, family)
        source_status = "new_or_changed_family_evidence" if current_marks and current_marks != prior_marks else "preexisting_family_evidence" if current_marks else "no_static_family_evidence"
        targets = set(getattr(skill, "bottleneck_kinds", []) or [])
        relevant = [bn for bn in prior_bns if bn["kind"] in targets]
        current_keys = {(bn["kind"], bn["scope"]) for bn in current_bns}
        persistent = [bn for bn in relevant if (bn["kind"], bn["scope"]) in current_keys]
        profile_effect = "target_bottleneck_still_reported" if persistent else "target_no_longer_reported_not_proven_resolved" if relevant and current_bns else "insufficient_profile_evidence"
        reviews.append({
            "skill_id": skill_id, "declared": skill_id in declared, "family": family,
            "source_status": source_status, "source_evidence": current_marks[:6],
            "recipe_correctness": "not_proven_by_static_markers",
            "applicability_evidence": relevant[:4], "profile_effect": profile_effect,
            "candidate_correctness_passed": bool(feasibility.get("correctness_ok")),
            "candidate_latency_effect": effect,
            "causal_attribution": "not_isolated_per_skill",
        })
    unsupported = [r["skill_id"] for r in reviews if r["declared"] and r["source_status"] != "new_or_changed_family_evidence"]
    available = {key: {"used": number(candidate_report.get(key)), "capacity": number(capacity.get(key)),
                       "utilization": number(candidate_report.get(key)) / number(capacity[key])}
                 for key in RESOURCES if number(candidate_report.get(key)) is not None and number(capacity.get(key)) and number(capacity[key]) > 0}
    retry = bool(valid and effect in {"unchanged", "regressed"} and not dominates
                 and (unsupported or current_bns or dominated))
    guidance = [f"QoR REVIEW ({REVIEW_VERSION}): candidate latency effect={effect}; resource relation={tradeoff}.",
                f"Measured cycles: {parent_cycles} -> {candidate_cycles}. Measured metric deltas: {deltas}."]
    if unsupported:
        guidance.append("Declared skills without new source evidence: " + ", ".join(unsupported) + ". Inspect the actual loop/array implementation; an added pragma or declaration does not prove the recipe was implemented correctly.")
    for bn in current_bns[:4]:
        guidance.append(f"Remaining profile evidence [{bn['kind']}] {bn['scope']}: {bn['evidence']}")
    if retry:
        guidance.append("Revise the design to address the measured bottleneck; increasing parallelism/banking is allowed within physical capacity. Match coupled controls and measure the result rather than repeating ineffective directives.")
    guidance.append("CSim validates only exercised inputs. Skill-level causality and full semantic correctness require more than static markers or aggregate cycle changes.")
    return {
        "schema_version": "c2hls.qor-profile-review.v1", "review_version": REVIEW_VERSION,
        "reference_blind": True, "reviewer_kind": "deterministic_evidence_review",
        "parent_code_sha256": hashlib.sha256((parent_code or "").encode()).hexdigest(),
        "candidate_code_sha256": hashlib.sha256((candidate_code or "").encode()).hexdigest(),
        "feasible": valid, "latency_effect": effect,
        "parent_cycles": parent_cycles, "candidate_cycles": candidate_cycles,
        "metric_deltas": deltas, "resource_utilization": available,
        "pareto_relation": tradeoff, "skill_reviews": reviews,
        "unsupported_declarations": unsupported,
        "rendered_skill_count": len(rendered), "profile_bottlenecks": current_bns[:8],
        "retry_recommended": retry, "guidance": "\n".join(guidance),
    }

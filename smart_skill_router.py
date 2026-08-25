"""Deterministic, auditable skill routing for HLS optimization prompts.

The router has two search widths:

* ``smart_best_fit`` scores a focused pool supported by the current
  bottleneck, optimization step, or an explicitly requested skill.
* ``smart_exhaustive`` scores every applicable positive skill.

Neither mode forces a result.  A skill must have task evidence and clear the
configured score threshold; otherwise the returned selection is empty.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from skill_library import Skill, SkillLibrary, TIER_AVOID


SMART_SKILL_SCOPES = {
    "smart_best_fit",
    "smart-best-fit",
    "best_fit",
    "best-fit",
    "smart_exhaustive",
    "smart-exhaustive",
    "exhaustive",
    "smart_best_fit_v2",
    "smart-best-fit-v2",
    "best_fit_v2",
    "best-fit-v2",
    "smart_exhaustive_v2",
    "smart-exhaustive-v2",
    "exhaustive_v2",
    "exhaustive-v2",
}

SKILLLESS_SCOPES = {
    "skillless",
    "no_skills",
    "no-skills",
    "none",
}

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}")
_STOPWORDS = {
    "all",
    "any",
    "array",
    "arrays",
    "as",
    "at",
    "about",
    "after",
    "also",
    "and",
    "are",
    "be",
    "before",
    "being",
    "bit",
    "by",
    "can",
    "code",
    "complete",
    "current",
    "each",
    "for",
    "from",
    "has",
    "have",
    "high",
    "hls",
    "if",
    "in",
    "into",
    "is",
    "it",
    "kernel",
    "more",
    "must",
    "no",
    "not",
    "of",
    "on",
    "only",
    "or",
    "other",
    "report",
    "return",
    "should",
    "that",
    "the",
    "their",
    "then",
    "this",
    "to",
    "through",
    "using",
    "vitis",
    "warning",
    "when",
    "where",
    "which",
    "while",
    "with",
}
_CONFIDENCE_PRIOR = {
    "high": 0.75,
    "medium": 0.35,
    "low": 0.0,
}
_MEMORY_BOTTLENECKS = {
    "axi_burst_failed",
    "axi_dependency",
    "axi_port_contention",
    "bank_imbalance",
    "global_memory_latency",
    "global_memory_reuse",
    "memory_bandwidth",
    "memory_bandwidth_improved_but_latency_still_high",
    "non_contiguous_access",
    "poor_locality",
    "repeated_global_access",
    "single_bundle_bottleneck",
    "underutilized_wide_memory",
}
_PORT_BOTTLENECKS = {
    "local_memory_port_limited",
    "memory_port_limited",
    "port_conflict",
    "unroll_blocked_by_memory_ports",
}
_PIPELINE_BOTTLENECKS = {
    "ii_target_miss",
    "interval_exceeds_latency",
    "non_pipelined_hot_loop",
    "pipeline_blocked",
    "pipeline_body_latency",
    "resource_limited_ii",
}
_RECURRENCE_BOTTLENECKS = {
    "dynamic_programming_dependency",
    "loop_carried_dep",
    "recurrence_limited_ii",
    "reduction_loop",
    "true_loop_carried_dep",
}
_DATAFLOW_BOTTLENECKS = {
    "dataflow_blocked",
    "function_stage_serialization",
    "latency_high_after_tiling",
    "load_compute_serialize",
    "store_compute_serialize",
}


@dataclass(frozen=True)
class SmartSkillRoute:
    """Selected skill objects plus JSON-serializable routing evidence."""

    selected: list[Skill]
    audit: dict[str, Any]


def normalize_skill_scope(scope: Optional[str]) -> str:
    value = (scope or "").strip().lower()
    if value in {"smart-best-fit-v2", "best_fit_v2", "best-fit-v2"}:
        return "smart_best_fit_v2"
    if value in {
        "smart-exhaustive-v2",
        "exhaustive_v2",
        "exhaustive-v2",
    }:
        return "smart_exhaustive_v2"
    if value in {"smart-best-fit", "best_fit", "best-fit"}:
        return "smart_best_fit"
    if value in {"smart-exhaustive", "exhaustive"}:
        return "smart_exhaustive"
    if value in SKILLLESS_SCOPES:
        return "skillless"
    return value or "matched"


def render_empty_skill_source(source: str, *, reason: str) -> str:
    """Make a zero-skill condition explicit in the model-visible prompt."""

    return (
        f"SKILL SOURCE: {source}\n"
        "AVAILABLE SKILLS: []\n"
        f"SELECTION RESULT: {reason}\n"
        "Use the task code and synthesis feedback as the optimization context."
    )


def _tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, default=str)
    return {
        token.lower()
        for token in _TOKEN_RE.findall(value)
        if token.lower() not in _STOPWORDS
    }


def _skill_signature_tokens(skill: Skill) -> set[str]:
    return _tokens(
        " ".join(
            [
                skill.id,
                skill.kind,
                " ".join(skill.bottleneck_kinds),
                " ".join(skill.tags),
            ]
        )
    )


def _applicable(
    skill: Skill,
    *,
    vitis_version: Optional[str],
    fpga: Optional[str],
) -> bool:
    if skill.confidence == TIER_AVOID:
        return False
    if vitis_version and skill.applicable_versions:
        if vitis_version not in skill.applicable_versions:
            return False
    if fpga and skill.applicable_fpgas:
        fpga_l = fpga.lower()
        if not any(
            fpga_l == item.lower()
            or fpga_l.startswith(item.lower())
            or item.lower().startswith(fpga_l)
            for item in skill.applicable_fpgas
        ):
            return False
    return True


def _bottleneck_kinds(report: Optional[dict[str, Any]]) -> list[str]:
    feedback = (report or {}).get("feedback") or {}
    records = feedback.get("bottlenecks") or []
    out: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        kind = str(record.get("kind") or "").strip()
        if kind and kind not in out:
            out.append(kind)
    return out


def _task_features(
    current_code: str,
    bottlenecks: list[str],
) -> set[str]:
    code = current_code or ""
    code_l = code.lower()
    features: set[str] = set()
    if len(re.findall(r"\bfor\s*\(", code)) >= 2:
        features.add("nested_or_multiphase_loops")
    if re.search(r"\[[^\]]+\]\s*\[[^\]]+\]", code):
        features.add("multidimensional_array")
    if "+=" in code or re.search(r"\b(sum|acc|accum)\w*\b", code_l):
        features.add("reduction")
    if "m_axi" in code_l:
        features.add("axi_memory")
    if code_l.count("m_axi") >= 2:
        features.add("multiple_axi_arrays")
    if "hls::stream" in code_l or "#pragma hls dataflow" in code_l:
        features.add("stream_or_dataflow")
    if re.search(
        r"\[[^\]]*[a-zA-Z_]\w*\s*[+-]\s*(?:\d+|[a-zA-Z_]\w*)[^\]]*\]",
        code,
    ):
        features.add("neighbor_indexing")
    compile_time_bounds = set(
        re.findall(
            r"\b(?:const|constexpr)\s+\w+(?:\s+\w+)*\s+([a-zA-Z_]\w*)\s*=",
            code,
        )
    )
    loop_bounds = set(
        re.findall(
            r"\bfor\s*\([^;]+;[^;]+<\s*([a-zA-Z_]\w*)\s*;",
            code,
        )
    )
    if any(
        bound not in compile_time_bounds and not bound.isupper()
        for bound in loop_bounds
    ):
        features.add("runtime_loop_bound")

    kinds = set(bottlenecks)
    if kinds & _MEMORY_BOTTLENECKS:
        features.add("memory_bottleneck")
    if kinds & _PORT_BOTTLENECKS:
        features.add("memory_port_bottleneck")
    if kinds & _PIPELINE_BOTTLENECKS:
        features.add("pipeline_bottleneck")
    if kinds & _RECURRENCE_BOTTLENECKS:
        features.add("recurrence_bottleneck")
    if kinds & _DATAFLOW_BOTTLENECKS:
        features.add("dataflow_bottleneck")
    if kinds & {"repeated_neighbor_access", "stencil_dependency"}:
        features.add("stencil_bottleneck")
    if "false_dependence" in kinds:
        features.add("false_dependence_reported")
    return features


def _specialty_compatible(
    skill: Skill,
    *,
    task_features: set[str],
) -> tuple[bool, Optional[str]]:
    signature = _skill_signature_tokens(skill)
    skill_id = skill.id.lower()

    if signature & {"stencil", "halo", "in-place"}:
        if not (
            task_features
            & {"neighbor_indexing", "stencil_bottleneck"}
        ):
            return False, "stencil_evidence_missing"
    if "false-dependence" in signature or "false_dependence" in signature:
        if "false_dependence_reported" not in task_features:
            return False, "false_dependence_evidence_missing"
    if (
        signature
        & {"dynamic-programming", "recurrence", "shift-register"}
        and "recurrence_bottleneck" not in task_features
        and not (
            "reduction" in task_features
            and "reduction" in signature
        )
    ):
        return False, "recurrence_evidence_missing"
    if signature & {"512-bit", "wide-bus", "coalescing"}:
        if "memory_bottleneck" not in task_features:
            return False, "memory_bandwidth_evidence_missing"
    if signature & {"multiddr", "memory-bank", "traffic-balance"}:
        if not {
            "memory_bottleneck",
            "multiple_axi_arrays",
        } <= task_features:
            return False, "multibank_evidence_missing"
    if "loop-tripcount" in signature or "loop_tripcount" in signature:
        if "runtime_loop_bound" not in task_features:
            return False, "runtime_bound_evidence_missing"
    if (
        signature & {"doublebuffer", "ping-pong-buffer"}
        and not (
            task_features
            & {"dataflow_bottleneck", "stream_or_dataflow"}
        )
    ):
        return False, "overlap_evidence_missing"
    if "partition-lane-buffers" in skill_id and not (
        task_features
        & {"memory_bottleneck", "memory_port_bottleneck"}
    ):
        return False, "lane_buffer_evidence_missing"
    return True, None


def _skill_family(skill: Skill) -> str:
    signature = _skill_signature_tokens(skill)
    if signature & {"stencil", "halo", "in-place"}:
        return "stencil"
    if "false-dependence" in signature or "false_dependence" in signature:
        return "dependence"
    if signature & {"512-bit", "wide-bus", "coalescing"}:
        return "coalescing"
    if signature & {"doublebuffer", "ping-pong-buffer", "dataflow"}:
        return "dataflow"
    if signature & {"reduction", "partial-sums", "tree-reduction"}:
        return "reduction"
    if signature & {"array-partition", "banking", "port-conflict"}:
        return "partition"
    if signature & {"tiling", "locality", "reuse"}:
        return "tiling"
    if signature & {"unroll", "parallelization", "processing-elements"}:
        return "unroll"
    if signature & {"pipeline", "ii", "hot-loop"}:
        return "pipeline"
    if signature & {"multiddr", "memory-bank", "traffic-balance"}:
        return "multibank"
    return skill.kind or "other"


def _focused_candidates(
    skills: Iterable[Skill],
    *,
    bottlenecks: list[str],
    step_tokens: set[str],
    code_tokens: set[str],
    requested_skill_id: Optional[str],
) -> list[Skill]:
    focused: list[Skill] = []
    for skill in skills:
        signature = _tokens(
            " ".join(
                [
                    skill.id,
                    skill.kind,
                    " ".join(skill.tags),
                    " ".join(skill.bottleneck_kinds),
                ]
            )
        )
        if (
            skill.id == requested_skill_id
            or bool(set(skill.bottleneck_kinds) & set(bottlenecks))
            or bool(signature & step_tokens)
            or len(signature & code_tokens) >= 2
        ):
            focused.append(skill)
    return focused


def _score_skill(
    skill: Skill,
    *,
    bottlenecks: list[str],
    step_tokens: set[str],
    code_tokens: set[str],
    task_features: set[str],
    requested_skill_id: Optional[str],
    strict_evidence: bool = False,
) -> tuple[float, bool, dict[str, Any]]:
    score = 0.0
    reasons: list[str] = []
    hard_evidence = False
    independent_evidence = False
    specialty_ok, specialty_reason = _specialty_compatible(
        skill,
        task_features=task_features,
    )

    if requested_skill_id and skill.id == requested_skill_id:
        score += 20.0
        hard_evidence = True
        reasons.append("explicit_request:+20.00")

    bottleneck_hits = [
        kind for kind in bottlenecks if kind in skill.bottleneck_kinds
    ]
    if bottleneck_hits:
        rank_weights = (10.0, 6.0, 4.0, 3.0, 2.0)
        bonus = sum(
            rank_weights[index] if index < len(rank_weights) else 1.0
            for index, kind in enumerate(bottlenecks)
            if kind in skill.bottleneck_kinds
        )
        bonus = min(22.0, bonus)
        score += bonus
        hard_evidence = True
        independent_evidence = True
        reasons.append(
            f"bottleneck={','.join(bottleneck_hits)}:+{bonus:.2f}"
        )

    signature_tokens = _skill_signature_tokens(skill)
    step_hits = sorted(step_tokens & signature_tokens)
    if step_hits:
        bonus = min(8.0, 4.0 * len(step_hits))
        score += bonus
        hard_evidence = True
        reasons.append(f"step={','.join(step_hits)}:+{bonus:.2f}")

    context_hits = sorted(code_tokens & signature_tokens)
    if len(context_hits) >= 2:
        bonus = min(4.0, 0.7 * len(context_hits))
        score += bonus
        hard_evidence = True
        independent_evidence = True
        reasons.append(
            f"context={','.join(context_hits[:10])}:+{bonus:.2f}"
        )

    family = _skill_family(skill)
    feature_bonus = 0.0
    if family == "pipeline" and "pipeline_bottleneck" in task_features:
        feature_bonus += 4.0
    if family == "reduction" and "reduction" in task_features:
        feature_bonus += 5.0
    if family == "tiling" and task_features & {
        "multidimensional_array",
        "nested_or_multiphase_loops",
    }:
        feature_bonus += 3.0
    if family == "partition" and "memory_port_bottleneck" in task_features:
        feature_bonus += 5.0
    elif family == "partition" and "multidimensional_array" in task_features:
        feature_bonus += 1.5
    if family in {"coalescing", "multibank"} and (
        "memory_bottleneck" in task_features
    ):
        feature_bonus += 5.0
    if family == "dataflow" and "dataflow_bottleneck" in task_features:
        feature_bonus += 5.0
    if family == "unroll" and "nested_or_multiphase_loops" in task_features:
        feature_bonus += 1.5
    if feature_bonus:
        score += feature_bonus
        hard_evidence = True
        independent_evidence = True
        reasons.append(f"task_features:+{feature_bonus:.2f}")

    prior = _CONFIDENCE_PRIOR.get(skill.confidence, 0.0)
    if skill.mean_advantage > 0:
        prior += min(0.75, float(skill.mean_advantage))
    if prior:
        score += prior
        reasons.append(f"prior:+{prior:.2f}")

    eligible_evidence = (
        (independent_evidence if strict_evidence else hard_evidence)
        and specialty_ok
    )
    if not specialty_ok:
        reasons.append(f"specialty_mismatch:{specialty_reason}")
    return score, eligible_evidence, {
        "skill_id": skill.id,
        "family": family,
        "score": round(score, 4),
        "hard_evidence": eligible_evidence,
        "independent_code_or_report_evidence": independent_evidence,
        "specialty_compatible": specialty_ok,
        "specialty_reason": specialty_reason,
        "reasons": reasons,
    }


def route_smart_skills(
    library: SkillLibrary,
    *,
    scope: str,
    step_name: str,
    current_code: str,
    synth_report: Optional[dict[str, Any]],
    vitis_version: Optional[str] = None,
    fpga: Optional[str] = None,
    requested_skill_id: Optional[str] = None,
    max_skills: Optional[int] = None,
    min_score: Optional[float] = None,
) -> SmartSkillRoute:
    """Route zero or more positive skills with deterministic score evidence."""

    normalized_scope = normalize_skill_scope(scope)
    if normalized_scope not in {
        "smart_best_fit",
        "smart_exhaustive",
        "smart_best_fit_v2",
        "smart_exhaustive_v2",
    }:
        raise ValueError(f"unsupported smart skill scope: {scope!r}")

    if max_skills is None:
        max_skills = int(os.getenv("C2HLS_SKILL_ROUTER_MAX_SKILLS", "3"))
    if min_score is None:
        min_score = float(os.getenv("C2HLS_SKILL_ROUTER_MIN_SCORE", "4.0"))
    max_skills = max(0, max_skills)

    bottlenecks = _bottleneck_kinds(synth_report)
    step_tokens = _tokens(step_name.replace("_", " ").replace("-", " "))
    code_tokens = _tokens(current_code)
    task_features = _task_features(current_code, bottlenecks)
    applicable = [
        skill
        for skill in library.all()
        if _applicable(skill, vitis_version=vitis_version, fpga=fpga)
    ]
    focused_scope = normalized_scope in {
        "smart_best_fit",
        "smart_best_fit_v2",
    }
    strict_evidence = normalized_scope.endswith("_v2")
    if focused_scope:
        candidates = _focused_candidates(
            applicable,
            bottlenecks=bottlenecks,
            step_tokens=step_tokens,
            code_tokens=code_tokens,
            requested_skill_id=requested_skill_id,
        )
    else:
        candidates = applicable

    scored: list[tuple[float, Skill, dict[str, Any]]] = []
    for skill in candidates:
        score, hard_evidence, detail = _score_skill(
            skill,
            bottlenecks=bottlenecks,
            step_tokens=step_tokens,
            code_tokens=code_tokens,
            task_features=task_features,
            requested_skill_id=requested_skill_id,
            strict_evidence=strict_evidence,
        )
        detail["eligible"] = bool(hard_evidence and score >= min_score)
        scored.append((score, skill, detail))
    scored.sort(key=lambda item: (-item[0], item[1].id))

    selected: list[Skill] = []
    selected_families: set[str] = set()
    diversify_families = normalized_scope != "smart_exhaustive_v2"
    for _score, skill, detail in scored:
        family = str(detail["family"])
        if (
            not detail["eligible"]
            or (diversify_families and family in selected_families)
        ):
            continue
        selected.append(skill)
        selected_families.add(family)
        if len(selected) >= max_skills:
            break
    selected_ids = [skill.id for skill in selected]
    audit = {
        "router": normalized_scope,
        "search_policy": (
            "focused_candidate_pool"
            if focused_scope
            else "all_applicable_positive_skills"
        ),
        "router_version": 2 if strict_evidence else 1,
        "strict_evidence": strict_evidence,
        "family_diversity_constraint": diversify_families,
        "catalog_positive_applicable_count": len(applicable),
        "candidate_count": len(candidates),
        "bottleneck_kinds": bottlenecks,
        "task_features": sorted(task_features),
        "step_name": step_name,
        "requested_skill_id": requested_skill_id,
        "min_score": min_score,
        "max_skills": max_skills,
        "selected_skill_ids": selected_ids,
        "selected_skill_count": len(selected_ids),
        "no_match": not selected_ids,
        "scores": [detail for _, _, detail in scored],
    }
    return SmartSkillRoute(selected=selected, audit=audit)

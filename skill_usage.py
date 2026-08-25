"""Auditable skill exposure and best-effort semantic-application telemetry."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from skill_library import Skill, SkillLibrary


_DECLARATION_RE = re.compile(
    r"^\s*SKILL_USAGE\s*:\s*(\[[^\r\n]*\])\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_skill_usage_declaration(
    response: str,
    *,
    rendered_skill_ids: Iterable[str],
) -> dict[str, Any]:
    """Parse an optional ``SKILL_USAGE: [...]`` line before the code block."""

    rendered = [str(value) for value in rendered_skill_ids if value]
    match = _DECLARATION_RE.search(response or "")
    if not match:
        return {
            "declared_applied_skill_ids": [],
            "declared_applied_skill_count": 0,
            "skill_declaration_status": "not_declared",
            "undeclared_or_unrendered_skill_ids": [],
        }
    try:
        raw = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {
            "declared_applied_skill_ids": [],
            "declared_applied_skill_count": 0,
            "skill_declaration_status": "malformed",
            "undeclared_or_unrendered_skill_ids": [],
        }
    if not isinstance(raw, list) or not all(
        isinstance(value, str) for value in raw
    ):
        return {
            "declared_applied_skill_ids": [],
            "declared_applied_skill_count": 0,
            "skill_declaration_status": "malformed",
            "undeclared_or_unrendered_skill_ids": [],
        }
    declared = list(dict.fromkeys(value for value in raw if value))
    rendered_set = set(rendered)
    unsupported = [
        value for value in declared if value not in rendered_set
    ]
    accepted = [value for value in declared if value in rendered_set]
    return {
        "declared_applied_skill_ids": accepted,
        "declared_applied_skill_count": len(accepted),
        "skill_declaration_status": (
            "contains_unrendered_ids" if unsupported else "declared"
        ),
        "undeclared_or_unrendered_skill_ids": unsupported,
    }


def _skill_family(skill: Skill) -> str:
    text = " ".join(
        [skill.id, skill.kind, *skill.tags, *skill.bottleneck_kinds]
    ).lower()
    ordered = (
        ("stencil", ("stencil", "halo", "in-place")),
        ("coalescing", ("coalescing", "wide-bus", "512-bit")),
        ("dataflow", ("doublebuffer", "double-buffer", "dataflow", "ping-pong")),
        ("dependence", ("false-dependence", "dependence")),
        ("reduction", ("reduction", "partial-sums", "tree-reduction")),
        ("partition", ("array-partition", "banking", "port-conflict")),
        ("tiling", ("tiling", "locality", "reuse")),
        ("unroll", ("unroll", "parallelization", "processing-elements")),
        ("pipeline", ("pipeline", "hot-loop", " ii ")),
        ("multibank", ("multiddr", "memory-bank", "traffic-balance")),
    )
    for family, markers in ordered:
        if any(marker in text for marker in markers):
            return family
    return "other"


def transformation_families(code: str) -> set[str]:
    """Return conservative transformation-family markers present in code."""

    lowered = (code or "").lower()
    families: set[str] = set()
    if "max_widen_bitwidth" in lowered or re.search(
        r"ap_u?int\s*<\s*512", lowered
    ):
        families.add("coalescing")
    if "#pragma hls dataflow" in lowered or (
        "ping" in lowered and "pong" in lowered
    ):
        families.add("dataflow")
    if "#pragma hls pipeline" in lowered:
        families.add("pipeline")
    if re.search(r"\btile\w*\b", lowered):
        families.add("tiling")
    if "#pragma hls unroll" in lowered:
        families.add("unroll")
    if "#pragma hls array_partition" in lowered:
        families.add("partition")
    if "#pragma hls dependence" in lowered:
        families.add("dependence")
    if re.search(r"\b(partial|lane)_?sum", lowered):
        families.add("reduction")
    if len(set(re.findall(r"bundle\s*=\s*([a-zA-Z_]\w*)", lowered))) >= 2:
        families.add("multibank")
    return families


def verify_declared_skill_usage(
    *,
    parent_code: str,
    candidate_code: str,
    declared_skill_ids: Iterable[str],
    library: SkillLibrary | None,
) -> dict[str, Any]:
    """Verify only declarations with a newly observable family marker.

    This is deliberately conservative. Skills without a reliable static marker
    remain ``unverified`` rather than being credited from prompt exposure.
    """

    declared = [str(value) for value in declared_skill_ids if value]
    before = transformation_families(parent_code)
    after = transformation_families(candidate_code)
    introduced = after - before
    verified: list[str] = []
    unverified: list[str] = []
    family_by_id: dict[str, str] = {}
    for skill_id in declared:
        skill = library.get(skill_id) if library is not None else None
        if skill is None:
            unverified.append(skill_id)
            continue
        family = _skill_family(skill)
        family_by_id[skill_id] = family
        if family in introduced:
            verified.append(skill_id)
        else:
            unverified.append(skill_id)
    return {
        "verified_applied_skill_ids": verified,
        "verified_applied_skill_count": len(verified),
        "unverified_declared_skill_ids": unverified,
        "verification_method": "new_family_marker_v1",
        "parent_transformation_families": sorted(before),
        "candidate_transformation_families": sorted(after),
        "introduced_transformation_families": sorted(introduced),
        "declared_skill_families": family_by_id,
    }


def usage_declaration_instruction(rendered_skill_ids: Iterable[str]) -> str:
    rendered = [str(value) for value in rendered_skill_ids if value]
    return (
        "Before the C++ code block, emit exactly one machine-readable line "
        "listing only the rendered skills you intentionally used. Use an empty "
        "list when none were used:\n"
        "SKILL_USAGE: []\n"
        f"Allowed IDs: {json.dumps(rendered)}\n"
        "Replace the empty list with the applicable IDs; do not list mere "
        "prompt exposure."
    )

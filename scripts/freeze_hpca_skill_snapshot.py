#!/usr/bin/env python3
"""Freeze reviewed HLSFactory skill evidence into an immutable snapshot.

This utility is intentionally a *freezer*, not an experiment runner.  It reads
an explicit review manifest, validates the referenced result artifacts, and
copies only the named skill definitions into a content-addressed directory.
It never opens the mutable source library for writing.

The input contract and intended HPCA workflow are documented in
``docs/hpca2027_frozen_skills.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from skill_library import (  # noqa: E402  (repo import after path setup)
    SCHEMA_VERSION as SKILL_SCHEMA_VERSION,
    TIER_AVOID,
    TIER_HIGH,
    TIER_LOW,
    TIER_MEDIUM,
    _coerce_skill_entry,
)


INPUT_SCHEMA = "hpca2027.validated-hlsfactory-skills.v1"
SNAPSHOT_SCHEMA = "hpca2027.frozen-skill-snapshot.v1"
CONTENT_ADDRESS_PREFIX = "sha256-"

# These are the reference-isolated primary kernels in the checked-in HPCA matrix.  The
# built-in deny-list is deliberate: an input manifest cannot redefine the
# evaluation set to make a Rodinia trajectory look like development data.
RODINIA_EVALUATION_KERNELS = frozenset(
    {
        "streamcluster",
        "hotspot",
        "kmeans",
        "knn",
        "lavamd",
        "lud",
        "nw",
        "pathfinder",
        "srad",
    }
)

_HEX_DIGITS = frozenset("0123456789abcdef")
_ALLOWED_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "source_suite",
        "benchmark_role",
        "skill_source",
        "evaluation_kernels",
        "trajectories",
        "notes",
    }
)


class SnapshotValidationError(ValueError):
    """The proposed snapshot lacks required evidence or violates isolation."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in _HEX_DIGITS for char in text):
        raise SnapshotValidationError(f"{field} must be a lowercase SHA-256 hex digest")
    return text


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SnapshotValidationError(f"{field} must be a JSON object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise SnapshotValidationError(f"{field} must be a JSON array")
    return value


def _normalize_kernel(value: Any) -> str:
    text = "".join(char for char in str(value or "").lower() if char.isalnum())
    for prefix in ("hlsfactory", "rodiniahls", "rodinia"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text


def _is_hlsfactory(value: Any) -> bool:
    return "".join(char for char in str(value or "").lower() if char.isalnum()) == "hlsfactory"


def _resolve_input_path(manifest_path: Path, reference: Any, field: str) -> tuple[Path, str]:
    reference_text = str(reference or "").strip()
    if not reference_text:
        raise SnapshotValidationError(f"{field} is required")
    reference_path = Path(reference_text)
    if reference_path.is_absolute():
        raise SnapshotValidationError(
            f"{field} must be relative to the review manifest (absolute paths are not portable)"
        )
    resolved = (manifest_path.parent / reference_path).resolve()
    if not resolved.is_file():
        raise SnapshotValidationError(f"{field} does not name a regular file: {reference_text}")
    # Preserve the user-supplied portable reference in error messages only.
    # Snapshot metadata records hashes, not workstation paths.
    return resolved, reference_path.as_posix()


def _load_json_object(path: Path, field: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SnapshotValidationError(f"cannot read {field}: {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotValidationError(f"{field} is not valid UTF-8 JSON: {exc}") from exc
    return _require_mapping(payload, field), raw


def _check_declared_bytes_hash(raw: bytes, expected: Any, field: str) -> str:
    expected_hash = _require_sha256(expected, f"{field}.sha256")
    observed = _sha256_bytes(raw)
    if observed != expected_hash:
        raise SnapshotValidationError(
            f"{field} hash mismatch: manifest={expected_hash}, observed={observed}"
        )
    return observed


def _artifact_source_values(payload: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for container in (
        payload,
        payload.get("meta"),
        payload.get("run"),
        (payload.get("run") or {}).get("benchmark")
        if isinstance(payload.get("run"), Mapping)
        else None,
    ):
        if not isinstance(container, Mapping):
            continue
        for key in ("source_suite", "source_repo", "suite"):
            if container.get(key) not in (None, ""):
                values.append(str(container[key]))
    return values


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise SnapshotValidationError(f"{field} must be a positive integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotValidationError(f"{field} must be a positive integer") from exc
    if converted <= 0:
        raise SnapshotValidationError(f"{field} must be a positive integer")
    return converted


def _positive_finite(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise SnapshotValidationError(f"{field} must be a positive finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotValidationError(f"{field} must be a positive finite number") from exc
    if not math.isfinite(converted) or converted <= 0:
        raise SnapshotValidationError(f"{field} must be a positive finite number")
    return converted


def _report_scalar_latency(
    report_value: Any,
    field: str,
) -> dict[str, Any]:
    """Return the controller's cycle-first scalar latency for one report.

    The ordering intentionally mirrors ``C2HLSOrchestrator._best_so_far_score``:
    worst-case cycles, cycles, worst-case nanoseconds, then nanoseconds.  The
    freezer never trusts a trajectory's precomputed ``vs_previous`` or skill
    update.  It derives the advantage from the two pinned reports instead.
    """

    report = _require_mapping(report_value, field)
    for key, unit in (
        ("latency_cycles_worst", "cycles"),
        ("latency_cycles", "cycles"),
        ("latency_ns_worst", "nanoseconds"),
        ("latency_ns", "nanoseconds"),
    ):
        if report.get(key) is None:
            continue
        return {
            "value": _positive_finite(report.get(key), f"{field}.{key}"),
            "field": key,
            "unit": unit,
        }
    raise SnapshotValidationError(
        f"{field} lacks a positive core scalar latency "
        "(latency_cycles_worst/latency_cycles/latency_ns_worst/latency_ns)"
    )


def _validate_csim_golden(
    csim_value: Any,
    *,
    golden_output_hash: str,
    field: str,
) -> None:
    """Require an executed CSim whose *own* independent-golden check passed."""

    csim = _require_mapping(csim_value, field)
    if csim.get("ran") is not True:
        raise SnapshotValidationError(f"{field}.ran must be true")
    if csim.get("passed") is not True or csim.get("success") is not True:
        raise SnapshotValidationError(f"{field} must be an executed passing CSim")
    correctness = _require_mapping(csim.get("correctness"), f"{field}.correctness")
    if correctness.get("passed") is not True:
        raise SnapshotValidationError(f"{field}.correctness.passed must be true")
    if correctness.get("correctness_status") != "passed":
        raise SnapshotValidationError(
            f"{field}.correctness.correctness_status must be 'passed'"
        )
    if csim.get("golden_output_sha256") != golden_output_hash:
        raise SnapshotValidationError(
            f"{field}.golden_output_sha256 does not match the trajectory's "
            "independent golden output"
        )


def _step_feasibility(step: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    # Current controller results use ``feasibility``.  Accept the longer name
    # only for a pinned transitional result, while still requiring the proof to
    # live on this step (never the top-level selected winner).
    value = step.get("feasibility")
    name = "feasibility"
    if value is None and step.get("candidate_feasibility") is not None:
        value = step.get("candidate_feasibility")
        name = "candidate_feasibility"
    feasibility = _require_mapping(value, f"{field}.{name}")
    if feasibility.get("feasible") is not True:
        raise SnapshotValidationError(f"{field}.{name}.feasible must be true")
    return feasibility


def _validated_step_skill_evidence(
    payload: Mapping[str, Any],
    *,
    skill_id: str,
    golden_output_hash: str,
    field: str,
) -> dict[str, Any]:
    """Prove one reviewed skill was routed, injected, and accepted together.

    A final passing candidate cannot mask a failed or unused skill application:
    routing, prompt injection, CSim/golden correctness, feasibility, synthesis
    report, and success must all occur on the same step.  Earlier accepted
    steps update the comparison parent; failed/reverted steps do not.
    """

    steps = _require_list(payload.get("steps"), f"{field}.steps")
    if not steps:
        raise SnapshotValidationError(f"{field}.steps cannot be empty")

    baseline_report = _require_mapping(
        payload.get("baseline_report"), f"{field}.baseline_report"
    )
    previous_latency = _report_scalar_latency(
        baseline_report, f"{field}.baseline_report"
    )
    _validate_csim_golden(
        payload.get("baseline_csim"),
        golden_output_hash=golden_output_hash,
        field=f"{field}.baseline_csim",
    )

    routed_indices: list[int] = []
    injected_indices: list[int] = []
    routed_and_injected_indices: list[int] = []
    unsuccessful_joint_indices: list[int] = []
    eligible: list[dict[str, Any]] = []

    for step_index, step_value in enumerate(steps):
        step_field = f"{field}.steps[{step_index}]"
        step = _require_mapping(step_value, step_field)
        routing = step.get("routing_decision")
        routing = routing if isinstance(routing, Mapping) else {}
        prompt = step.get("skill_prompt")
        prompt = prompt if isinstance(prompt, Mapping) else {}
        routed = str(routing.get("skill_id") or "").strip() == skill_id
        injected_ids_value = prompt.get("injected_skill_ids")
        injected_ids = (
            [str(value).strip() for value in injected_ids_value]
            if isinstance(injected_ids_value, list)
            else []
        )
        injected = prompt.get("injected") is True and skill_id in injected_ids
        if routed:
            routed_indices.append(step_index)
        if injected:
            injected_indices.append(step_index)
        if routed and injected:
            routed_and_injected_indices.append(step_index)

        accepted = step.get("success") is True
        if not accepted:
            if routed and injected:
                unsuccessful_joint_indices.append(step_index)
            continue
        if step.get("reverted_to_prev") is True or step.get("budget_exhausted") is True:
            raise SnapshotValidationError(
                f"{step_field} claims success but is reverted or budget-exhausted"
            )

        # Every claimed accepted step must be independently auditable so it can
        # safely become the comparison parent for a later reviewed skill.
        _validate_csim_golden(
            step.get("csim"),
            golden_output_hash=golden_output_hash,
            field=f"{step_field}.csim",
        )
        _step_feasibility(step, step_field)
        report = _require_mapping(step.get("report"), f"{step_field}.report")
        current_latency = _report_scalar_latency(report, f"{step_field}.report")
        if current_latency["unit"] != previous_latency["unit"]:
            raise SnapshotValidationError(
                f"{step_field}.report scalar latency unit {current_latency['unit']!r} "
                f"does not match previous accepted report unit {previous_latency['unit']!r}"
            )

        if routed and injected:
            requested = str(prompt.get("requested_skill_id") or "").strip()
            if requested and requested != skill_id:
                raise SnapshotValidationError(
                    f"{step_field}.skill_prompt.requested_skill_id conflicts with "
                    f"the routed skill {skill_id!r}"
                )
            previous_value = float(previous_latency["value"])
            current_value = float(current_latency["value"])
            eligible.append(
                {
                    "step_index": step_index,
                    "step_name": str(step.get("step_name") or ""),
                    "previous_latency": previous_value,
                    "previous_latency_field": previous_latency["field"],
                    "current_latency": current_value,
                    "current_latency_field": current_latency["field"],
                    "latency_unit": current_latency["unit"],
                    "relative_advantage": (previous_value - current_value)
                    / previous_value,
                }
            )

        previous_latency = current_latency

    if not eligible:
        if routed_and_injected_indices:
            raise SnapshotValidationError(
                f"{field} routes and injects skill {skill_id!r} only in "
                f"unsuccessful steps {unsuccessful_joint_indices or routed_and_injected_indices}; "
                "a final/top-level pass cannot validate that application"
            )
        if routed_indices and not injected_indices:
            raise SnapshotValidationError(
                f"{field} routes skill {skill_id!r} at steps {routed_indices} but never "
                "injects it into the same step prompt"
            )
        if injected_indices and not routed_indices:
            raise SnapshotValidationError(
                f"{field} injects skill {skill_id!r} at steps {injected_indices} but "
                "never routes it"
            )
        raise SnapshotValidationError(
            f"{field} contains no successful step that both routes and injects "
            f"skill {skill_id!r}"
        )
    if len(eligible) != 1:
        raise SnapshotValidationError(
            f"{field} has {len(eligible)} successful routed-and-injected applications "
            f"of skill {skill_id!r}; use one unambiguous trajectory per validation"
        )
    return eligible[0]


def _validate_result_evidence(
    payload: Mapping[str, Any],
    *,
    kernel: str,
    field: str,
) -> dict[str, Any]:
    artifact_benchmark = str(payload.get("benchmark") or "")
    if _normalize_kernel(artifact_benchmark) != _normalize_kernel(kernel):
        raise SnapshotValidationError(
            f"{field}.benchmark does not match manifest kernel {kernel!r}"
        )

    source_values = _artifact_source_values(payload)
    has_hlsfactory_prefix = artifact_benchmark.lower().startswith("hlsfactory_")
    if source_values and any(not _is_hlsfactory(value) for value in source_values):
        raise SnapshotValidationError(f"{field} contains a non-HLSFactory source declaration")
    if not has_hlsfactory_prefix and not any(_is_hlsfactory(value) for value in source_values):
        raise SnapshotValidationError(
            f"{field} must identify itself as an HLSFactory artifact"
        )

    if payload.get("correctness_status") != "passed":
        raise SnapshotValidationError(f"{field}.correctness_status must be 'passed'")

    golden = _require_mapping(payload.get("independent_golden"), f"{field}.independent_golden")
    if golden.get("required") is not True:
        raise SnapshotValidationError(f"{field}.independent_golden.required must be true")
    if golden.get("status") != "passed":
        raise SnapshotValidationError(f"{field}.independent_golden.status must be 'passed'")
    if golden.get("source") != "pragma_stripped_plain_c_and_public_testbench":
        raise SnapshotValidationError(
            f"{field}.independent_golden.source is not the independent CPU oracle"
        )
    golden_output_hash = _require_sha256(
        golden.get("output_sha256"), f"{field}.independent_golden.output_sha256"
    )
    golden_specs_hash = _require_sha256(
        golden.get("specs_sha256"), f"{field}.independent_golden.specs_sha256"
    )
    output_count = _positive_int(
        golden.get("output_count"), f"{field}.independent_golden.output_count"
    )
    value_count = _positive_int(
        golden.get("value_count"), f"{field}.independent_golden.value_count"
    )

    _validate_csim_golden(
        payload.get("csim"),
        golden_output_hash=golden_output_hash,
        field=f"{field}.csim",
    )

    synthesis = _require_mapping(
        payload.get("synthesis_evaluations"), f"{field}.synthesis_evaluations"
    )
    synthesis_count = _positive_int(
        synthesis.get("count"), f"{field}.synthesis_evaluations.count"
    )
    synthesis_events = _require_list(
        synthesis.get("events"), f"{field}.synthesis_evaluations.events"
    )
    if not any(
        isinstance(event, Mapping)
        and event.get("synthesis_ran") is True
        and event.get("success") is True
        for event in synthesis_events
    ):
        raise SnapshotValidationError(
            f"{field} lacks an executed successful synthesis event"
        )

    feasibility = _require_mapping(
        payload.get("candidate_feasibility"), f"{field}.candidate_feasibility"
    )
    if feasibility.get("feasible") is not True:
        raise SnapshotValidationError(f"{field}.candidate_feasibility.feasible must be true")

    report = payload.get("synth_report") or payload.get("final_report")
    report = _require_mapping(report, f"{field}.synth_report/final_report")
    latency_value = report.get("latency_cycles_worst")
    if latency_value is None:
        latency_value = report.get("latency_cycles")
    latency_cycles = _positive_finite(latency_value, f"{field}.report.latency_cycles")

    return {
        "golden_output_sha256": golden_output_hash,
        "golden_specs_sha256": golden_specs_hash,
        "output_count": output_count,
        "value_count": value_count,
        "synthesis_evaluation_count": synthesis_count,
        "latency_cycles": latency_cycles,
    }


def _load_skill_definitions(
    source_payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if str(source_payload.get("schema") or "") != SKILL_SCHEMA_VERSION:
        raise SnapshotValidationError(
            f"skill source schema must be {SKILL_SCHEMA_VERSION!r}"
        )
    entries = _require_list(source_payload.get("skills"), "skill_source.skills")
    definitions: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        skill = _coerce_skill_entry(entry)
        if skill is None:
            raise SnapshotValidationError(f"skill_source.skills[{index}] is malformed")
        if skill.id in definitions:
            raise SnapshotValidationError(f"duplicate skill id in source: {skill.id}")
        definitions[skill.id] = asdict(skill)
    return definitions


def _validated_skill_records(value: Any, field: str) -> list[dict[str, Any]]:
    records = _require_list(value, field)
    if not records:
        raise SnapshotValidationError(f"{field} cannot be empty")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(records):
        item = _require_mapping(item, f"{field}[{index}]")
        unexpected = set(item) - {"id", "relative_advantage"}
        if unexpected:
            raise SnapshotValidationError(
                f"{field}[{index}] has unsupported fields: {sorted(unexpected)}"
            )
        skill_id = str(item.get("id") or "").strip()
        if not skill_id:
            raise SnapshotValidationError(f"{field}[{index}].id is required")
        if skill_id in seen:
            raise SnapshotValidationError(f"{field} repeats skill id {skill_id!r}")
        seen.add(skill_id)
        advantage = item.get("relative_advantage")
        if advantage is not None:
            if isinstance(advantage, bool):
                raise SnapshotValidationError(
                    f"{field}[{index}].relative_advantage must be finite"
                )
            try:
                advantage = float(advantage)
            except (TypeError, ValueError) as exc:
                raise SnapshotValidationError(
                    f"{field}[{index}].relative_advantage must be finite"
                ) from exc
            if not math.isfinite(advantage):
                raise SnapshotValidationError(
                    f"{field}[{index}].relative_advantage must be finite"
                )
        output.append({"id": skill_id, "declared_relative_advantage": advantage})
    return output


def _source_skill_bundle_provenance(
    source_path: Path,
    *,
    source_hash: str,
) -> dict[str, Any]:
    """Verify and fingerprint a sibling frozen-snapshot bundle when present."""

    parent = source_path.parent
    manifest_path = parent / "snapshot_manifest.json"
    sums_path = parent / "SHA256SUMS"
    indicators = [path for path in (manifest_path, sums_path) if path.exists()]
    if not indicators:
        return {"present": False}
    if source_path.name != "skills.json":
        raise SnapshotValidationError(
            "a source snapshot bundle must be referenced through its skills.json"
        )
    # Do not silently consume one file from a partial/tampered bundle.
    verified = verify_snapshot(parent)
    descriptor = _require_mapping(
        verified.get("content_descriptor"), "source snapshot content_descriptor"
    )
    bundled_skills = _require_mapping(
        descriptor.get("skills"), "source snapshot content_descriptor.skills"
    )
    if bundled_skills.get("sha256") != source_hash:
        raise SnapshotValidationError(
            "source snapshot manifest does not fingerprint the referenced skills.json"
        )
    return {
        "present": True,
        "schema_version": verified.get("schema_version"),
        "content_id": verified.get("content_id"),
        "content_address": verified.get("content_address"),
        "snapshot_manifest_sha256": _sha256_file(manifest_path),
        "sha256sums_sha256": _sha256_file(sums_path),
    }


def _evidence_confidence(source_confidence: str, count: int, advantages: list[float]) -> str:
    """Derive routing confidence from accepted evidence, not mutable statistics."""

    if source_confidence == TIER_AVOID:
        # An avoid recipe is negative knowledge, not a successfully applied
        # optimization.  It cannot enter this positive-skill snapshot.
        raise SnapshotValidationError("avoid-tier skills cannot be frozen as validated successes")
    mean = sum(advantages) / len(advantages) if advantages else 0.0
    if count >= 3 and mean >= 0.0:
        return TIER_HIGH
    if advantages and mean < -0.5:
        return TIER_LOW
    return TIER_MEDIUM


def _build_snapshot(
    manifest_path: Path,
) -> tuple[dict[str, Any], bytes, bytes, str]:
    manifest_path = manifest_path.resolve()
    manifest, manifest_raw = _load_json_object(manifest_path, "review manifest")
    unexpected = set(manifest) - _ALLOWED_TOP_LEVEL
    if unexpected:
        raise SnapshotValidationError(
            f"review manifest has unsupported fields: {sorted(unexpected)}"
        )
    if manifest.get("schema_version") != INPUT_SCHEMA:
        raise SnapshotValidationError(f"schema_version must be {INPUT_SCHEMA!r}")
    if not _is_hlsfactory(manifest.get("source_suite")):
        raise SnapshotValidationError("source_suite must be exactly HLSFactory")
    if manifest.get("benchmark_role") != "development":
        raise SnapshotValidationError("benchmark_role must be 'development'")

    evaluation_names = set(RODINIA_EVALUATION_KERNELS)
    for value in _require_list(manifest.get("evaluation_kernels", []), "evaluation_kernels"):
        normalized = _normalize_kernel(value)
        if not normalized:
            raise SnapshotValidationError("evaluation_kernels cannot contain an empty name")
        evaluation_names.add(normalized)

    skill_source = _require_mapping(manifest.get("skill_source"), "skill_source")
    if set(skill_source) != {"path", "sha256"}:
        raise SnapshotValidationError("skill_source must contain exactly path and sha256")
    skill_source_path, _ = _resolve_input_path(
        manifest_path, skill_source.get("path"), "skill_source.path"
    )
    source_payload, source_raw = _load_json_object(skill_source_path, "skill source")
    skill_source_hash = _check_declared_bytes_hash(
        source_raw, skill_source.get("sha256"), "skill_source"
    )
    source_skill_bundle = _source_skill_bundle_provenance(
        skill_source_path,
        source_hash=skill_source_hash,
    )
    skill_definitions = _load_skill_definitions(source_payload)

    trajectories = _require_list(manifest.get("trajectories"), "trajectories")
    if not trajectories:
        raise SnapshotValidationError("trajectories cannot be empty")

    evidence: list[dict[str, Any]] = []
    validations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectory_hashes: set[str] = set()
    for index, item in enumerate(trajectories):
        field = f"trajectories[{index}]"
        item = _require_mapping(item, field)
        required_keys = {
            "path",
            "sha256",
            "kernel",
            "source_suite",
            "benchmark_role",
            "validated_skills",
        }
        if set(item) != required_keys:
            raise SnapshotValidationError(
                f"{field} must contain exactly {sorted(required_keys)}"
            )
        if not _is_hlsfactory(item.get("source_suite")):
            raise SnapshotValidationError(f"{field}.source_suite must be HLSFactory")
        if item.get("benchmark_role") != "development":
            raise SnapshotValidationError(f"{field}.benchmark_role must be 'development'")
        kernel = str(item.get("kernel") or "").strip()
        normalized_kernel = _normalize_kernel(kernel)
        if not normalized_kernel:
            raise SnapshotValidationError(f"{field}.kernel is required")
        if normalized_kernel in evaluation_names:
            raise SnapshotValidationError(
                f"{field}.kernel {kernel!r} is a primary evaluation kernel"
            )

        trajectory_path, _ = _resolve_input_path(
            manifest_path, item.get("path"), f"{field}.path"
        )
        trajectory_payload, trajectory_raw = _load_json_object(trajectory_path, field)
        trajectory_hash = _check_declared_bytes_hash(
            trajectory_raw, item.get("sha256"), field
        )
        if trajectory_hash in trajectory_hashes:
            raise SnapshotValidationError(f"{field} duplicates a trajectory artifact")
        trajectory_hashes.add(trajectory_hash)
        proof = _validate_result_evidence(
            trajectory_payload, kernel=kernel, field=field
        )
        skill_records = _validated_skill_records(
            item.get("validated_skills"), f"{field}.validated_skills"
        )
        for record in skill_records:
            skill_id = record["id"]
            if skill_id not in skill_definitions:
                raise SnapshotValidationError(
                    f"{field} validates unknown skill id {skill_id!r}"
                )
            step_proof = _validated_step_skill_evidence(
                trajectory_payload,
                skill_id=skill_id,
                golden_output_hash=proof["golden_output_sha256"],
                field=field,
            )
            declared_advantage = record["declared_relative_advantage"]
            derived_advantage = float(step_proof["relative_advantage"])
            if declared_advantage is not None and not math.isclose(
                float(declared_advantage),
                derived_advantage,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise SnapshotValidationError(
                    f"{field}.validated_skills declares relative_advantage "
                    f"{declared_advantage!r} for {skill_id!r}, but the previous "
                    f"accepted report and validated step derive {derived_advantage!r}"
                )
            validations[skill_id].append(
                {
                    "kernel": normalized_kernel,
                    "trajectory_sha256": trajectory_hash,
                    "relative_advantage": derived_advantage,
                    "step_evidence": step_proof,
                }
            )

            # Snapshot evidence carries only derived values.  A reviewer may
            # state an expected value as a cross-check, but cannot author it.
            record["relative_advantage"] = derived_advantage
            record.pop("declared_relative_advantage", None)

        evidence.append(
            {
                "kernel": normalized_kernel,
                "trajectory_sha256": trajectory_hash,
                "validated_skills": skill_records,
                **proof,
            }
        )

    if not validations:
        raise SnapshotValidationError("no skills were validated")

    frozen_skills: list[dict[str, Any]] = []
    skill_evidence: list[dict[str, Any]] = []
    for skill_id in sorted(validations):
        observations = sorted(
            validations[skill_id],
            key=lambda item: (item["kernel"], item["trajectory_sha256"]),
        )
        advantages = [
            item["relative_advantage"]
            for item in observations
            if item["relative_advantage"] is not None
        ]
        skill = dict(skill_definitions[skill_id])
        skill["occurrences"] = len(observations)
        skill["sec_pass"] = len(observations)
        skill["mean_advantage"] = (
            sum(advantages) / len(advantages) if advantages else 0.0
        )
        skill["last_used_at"] = None
        skill["confidence"] = _evidence_confidence(
            str(skill.get("confidence") or ""), len(observations), advantages
        )
        frozen_skills.append(skill)
        skill_evidence.append(
            {
                "id": skill_id,
                "validated_trajectory_count": len(observations),
                "relative_advantage_observation_count": len(advantages),
                "source_definition_sha256": _sha256_bytes(
                    _canonical_json_bytes(skill_definitions[skill_id])
                ),
                "observations": observations,
            }
        )

    skill_payload = {
        "schema": SKILL_SCHEMA_VERSION,
        "skills": frozen_skills,
    }
    skills_bytes = _pretty_json_bytes(skill_payload)
    skills_hash = _sha256_bytes(skills_bytes)
    evidence.sort(key=lambda item: (item["kernel"], item["trajectory_sha256"]))
    descriptor = {
        "schema_version": SNAPSHOT_SCHEMA,
        "input_manifest_sha256": _sha256_bytes(manifest_raw),
        "source_skill_library_sha256": skill_source_hash,
        "source_skill_bundle": source_skill_bundle,
        "source_suite": "HLSFactory",
        "benchmark_role": "development",
        "isolation": {
            "rodinia_evaluation_kernels_rejected": sorted(RODINIA_EVALUATION_KERNELS),
            "additional_evaluation_kernels_rejected": sorted(
                evaluation_names - set(RODINIA_EVALUATION_KERNELS)
            ),
            "independent_golden_required": True,
            "successful_synthesis_required": True,
            "online_mutation_allowed": False,
        },
        "skills": {
            "path": "skills.json",
            "schema": SKILL_SCHEMA_VERSION,
            "sha256": skills_hash,
            "count": len(frozen_skills),
            "ids": [skill["id"] for skill in frozen_skills],
        },
        "skill_evidence": skill_evidence,
        "trajectory_evidence": evidence,
    }
    content_id = _sha256_bytes(_canonical_json_bytes(descriptor))
    snapshot_manifest = {
        "schema_version": SNAPSHOT_SCHEMA,
        "content_address": f"sha256:{content_id}",
        "content_id": content_id,
        "content_descriptor": descriptor,
    }
    snapshot_manifest_bytes = _pretty_json_bytes(snapshot_manifest)
    return snapshot_manifest, skills_bytes, snapshot_manifest_bytes, content_id


def _expected_bundle(
    skills_bytes: bytes, snapshot_manifest_bytes: bytes
) -> dict[str, bytes]:
    manifest_hash = _sha256_bytes(snapshot_manifest_bytes)
    skills_hash = _sha256_bytes(skills_bytes)
    sums = (
        f"{skills_hash}  skills.json\n"
        f"{manifest_hash}  snapshot_manifest.json\n"
    ).encode("ascii")
    return {
        "skills.json": skills_bytes,
        "snapshot_manifest.json": snapshot_manifest_bytes,
        "SHA256SUMS": sums,
    }


def _verify_expected_directory(path: Path, expected: Mapping[str, bytes]) -> None:
    if not path.is_dir():
        raise SnapshotValidationError(f"existing snapshot path is not a directory: {path}")
    names = {child.name for child in path.iterdir()}
    if names != set(expected):
        raise SnapshotValidationError(
            f"existing snapshot has unexpected contents: expected {sorted(expected)}, got {sorted(names)}"
        )
    mismatches = [
        name for name, content in expected.items() if (path / name).read_bytes() != content
    ]
    if mismatches:
        raise SnapshotValidationError(
            "content-addressed snapshot exists with hash-mismatched files: "
            + ", ".join(sorted(mismatches))
        )


def freeze_snapshot(manifest_path: Path | str, output_root: Path | str) -> dict[str, Any]:
    """Validate ``manifest_path`` and atomically materialize its snapshot.

    Repeating a freeze is idempotent only when the existing addressed bundle
    is byte-for-byte identical.  No existing file is ever overwritten.
    """

    manifest_path = Path(manifest_path)
    output_root = Path(output_root).resolve()
    snapshot_manifest, skills_bytes, manifest_bytes, content_id = _build_snapshot(
        manifest_path
    )
    expected = _expected_bundle(skills_bytes, manifest_bytes)
    target = output_root / f"{CONTENT_ADDRESS_PREFIX}{content_id}"
    if target.exists():
        _verify_expected_directory(target, expected)
        return {**snapshot_manifest, "snapshot_path": str(target), "created": False}

    output_root.mkdir(parents=True, exist_ok=True)
    temp_path = Path(tempfile.mkdtemp(prefix=".hpca-skill-freeze-", dir=output_root))
    try:
        for name, content in expected.items():
            destination = temp_path / name
            destination.write_bytes(content)
            destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            temp_path.rename(target)
        except FileExistsError:
            # Another freezer won the race.  Validate it instead of replacing
            # any bytes.
            _verify_expected_directory(target, expected)
        else:
            temp_path = target
        return {**snapshot_manifest, "snapshot_path": str(target), "created": True}
    finally:
        if temp_path.exists() and temp_path != target:
            shutil.rmtree(temp_path)


def verify_snapshot(snapshot_path: Path | str) -> dict[str, Any]:
    """Verify the internal hashes and content address of a frozen bundle."""

    snapshot_path = Path(snapshot_path).resolve()
    expected_names = {"snapshot_manifest.json", "skills.json", "SHA256SUMS"}
    if not snapshot_path.is_dir():
        raise SnapshotValidationError(f"snapshot path is not a directory: {snapshot_path}")
    observed_names = {entry.name for entry in snapshot_path.iterdir()}
    if observed_names != expected_names:
        raise SnapshotValidationError(
            "snapshot has unexpected contents: "
            f"expected {sorted(expected_names)}, got {sorted(observed_names)}"
        )
    manifest_path = snapshot_path / "snapshot_manifest.json"
    skills_path = snapshot_path / "skills.json"
    sums_path = snapshot_path / "SHA256SUMS"
    for path in (manifest_path, skills_path, sums_path):
        if not path.is_file():
            raise SnapshotValidationError(f"snapshot is missing {path.name}")

    manifest, manifest_bytes = _load_json_object(manifest_path, "snapshot_manifest.json")
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA:
        raise SnapshotValidationError("snapshot manifest schema mismatch")
    descriptor = _require_mapping(manifest.get("content_descriptor"), "content_descriptor")
    content_id = _sha256_bytes(_canonical_json_bytes(descriptor))
    if manifest.get("content_id") != content_id:
        raise SnapshotValidationError("snapshot content_id does not match descriptor")
    if manifest.get("content_address") != f"sha256:{content_id}":
        raise SnapshotValidationError("snapshot content_address does not match descriptor")
    if snapshot_path.name != f"{CONTENT_ADDRESS_PREFIX}{content_id}":
        raise SnapshotValidationError("snapshot directory name does not match content address")

    skills_hash = _sha256_file(skills_path)
    descriptor_skills = _require_mapping(descriptor.get("skills"), "content_descriptor.skills")
    if descriptor_skills.get("sha256") != skills_hash:
        raise SnapshotValidationError("skills.json hash does not match snapshot manifest")
    expected_sums = _expected_bundle(skills_path.read_bytes(), manifest_bytes)["SHA256SUMS"]
    if sums_path.read_bytes() != expected_sums:
        raise SnapshotValidationError("SHA256SUMS mismatch")
    return dict(manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="validate evidence and create a snapshot")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="verify an existing snapshot")
    verify.add_argument("--snapshot", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "freeze":
            result = freeze_snapshot(args.manifest, args.output_root)
        else:
            result = verify_snapshot(args.snapshot)
    except SnapshotValidationError as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

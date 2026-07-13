#!/usr/bin/env python3
"""Normalize an explicit, hash-pinned HPCA freeze index to result schema v2.

This tool performs no directory discovery and never chooses a best run.  The
freeze index names exactly one source artifact and JSON pointer for every
kernel/method/seed row and for every baseline/expert record.  Source bytes are
verified before parsing.  Candidate events are derived only from telemetry
written by the corresponding runner; unavailable counters are hard errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.generate_hpca_paper_artifacts import (  # noqa: E402
    PASS,
    RUN_ID_RE,
    SCHEMA_VERSION,
    ManifestError as ResultManifestError,
    _validate_record,
)
from evaluation_repro import (  # noqa: E402
    FINGERPRINT_SCHEMA,
    PAPER_PROFILE,
    REFERENCE_BLIND_OVERRIDES,
    effective_llm_call_issues,
    fingerprint_completeness,
)


INDEX_SCHEMA = "c2hls.hpca-freeze-index.v1"
NORMALIZER_SCHEMA = "c2hls.hpca-freeze-normalizer.v1"
BASELINE_RESULT_SCHEMA = "c2hls.paper-baseline.v1"
REFERENCE_AUDIT_SCHEMA = "c2hls.reference-isolation-audit.v1"
BASELINE_METHODS = {"one_shot_best_of_five", "pragma_only"}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
AGENTIC_METHOD_CONTRACTS = {
    "flash_c2hls": {"strategy": "flash", "skill_mode": "skill_off"},
    "dynamic_no_skills": {"strategy": "dynamic", "skill_mode": "skill_off"},
    "dynamic_frozen_skills": {"strategy": "dynamic", "skill_mode": "skill_on"},
}
RESOURCE_KEYS = ("bram", "dsp", "ff", "lut", "uram")
# XCU280 physical capacities in the units emitted by Vitis CSynth.  An
# explicit target.resource_capacities mapping may be supplied by a freeze
# index; for the paper's fixed U280 target it must agree with this part table.
U280_RESOURCE_CAPACITIES = {
    "bram": 4032,
    "dsp": 9024,
    "ff": 2_607_360,
    "lut": 1_303_680,
    "uram": 960,
}


class FreezeNormalizationError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        location: str = "",
        missing_fields: Sequence[str] = (),
        producer_functions: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.location = location
        self.missing_fields = list(missing_fields)
        self.producer_functions = list(producer_functions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": "freeze_normalization_refused",
            "code": self.code,
            "message": str(self),
            "location": self.location or None,
            "missing_fields": self.missing_fields,
            "producer_functions": self.producer_functions,
        }


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise FreezeNormalizationError(
                "duplicate_json_key", f"duplicate JSON key {key!r}"
            )
        output[key] = value
    return output


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise FreezeNormalizationError(
            "artifact_unreadable", f"cannot read {path}: {exc}", location=str(path)
        ) from exc


def _parse_json(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=_reject_duplicates)
    except FreezeNormalizationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreezeNormalizationError(
            "json_unreadable", f"cannot read JSON {path}: {exc}", location=str(path)
        ) from exc
    if not isinstance(value, dict):
        raise FreezeNormalizationError(
            "json_root_not_object", f"JSON root must be an object: {path}"
        )
    return value


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = _read_bytes(path)
    return _parse_json(raw, path), hashlib.sha256(raw).hexdigest()


def _sha256_json(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FreezeNormalizationError(
            "fingerprint_payload_invalid",
            f"run fingerprint payload is not canonical JSON: {exc}",
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _report_sha256(value: Any, location: str, *, required: bool) -> str | None:
    """Hash a Vitis report exactly as the agentic producer does."""
    if not isinstance(value, dict) or not value:
        if required:
            raise FreezeNormalizationError(
                "synthesis_evidence_missing",
                f"{location} must be a non-empty report object",
                location=location,
            )
        return None
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FreezeNormalizationError(
            "synthesis_report_not_canonical",
            f"{location} cannot be hashed as canonical JSON: {exc}",
            location=location,
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _required_sha256(value: Any, location: str) -> str:
    digest = _required_string(value, location).lower()
    if SHA256_RE.fullmatch(digest) is None:
        raise FreezeNormalizationError(
            "sha256_invalid",
            f"{location} must be a SHA-256 digest",
            location=location,
        )
    return digest


def _required_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FreezeNormalizationError(
            "type_error", f"{location} must be an object", location=location
        )
    return value


def _required_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise FreezeNormalizationError(
            "type_error", f"{location} must be an array", location=location
        )
    return value


def _required_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise FreezeNormalizationError(
            "type_error", f"{location} must be a non-empty string", location=location
        )
    return value


def _required_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise FreezeNormalizationError(
            "type_error",
            f"{location} must be an integer >= {minimum}",
            location=location,
        )
    return value


def _exact_positive_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreezeNormalizationError(
            "measured_integer_missing",
            f"{location} must be a recorded positive integer",
            location=location,
        )
    if isinstance(value, int):
        if value <= 0:
            raise FreezeNormalizationError(
                "measured_integer_invalid",
                f"{location} must be an exact positive integer; rounding is forbidden",
                location=location,
            )
        return value
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        raise FreezeNormalizationError(
            "measured_integer_invalid",
            f"{location} must be an exact positive integer; rounding is forbidden",
            location=location,
        )
    return int(numeric)


def _exact_nonnegative_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreezeNormalizationError(
            "measured_resource_missing",
            f"{location} must be a recorded non-negative integer",
            location=location,
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise FreezeNormalizationError(
            "measured_resource_invalid",
            f"{location} must be an exact non-negative integer",
            location=location,
        )
    return int(numeric)


def _positive_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreezeNormalizationError(
            "measured_fmax_missing",
            f"{location} must be a recorded positive finite number",
            location=location,
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise FreezeNormalizationError(
            "measured_fmax_invalid",
            f"{location} must be a recorded positive finite number",
            location=location,
        )
    return numeric


def _resource_capacities_for_target(
    target: Mapping[str, Any], location: str = "target"
) -> tuple[dict[str, int], str]:
    part = _required_string(target.get("part"), f"{location}.part").lower()
    explicit = target.get("resource_capacities")
    known = U280_RESOURCE_CAPACITIES if part.startswith("xcu280") else None
    if explicit is None:
        if known is None:
            raise FreezeNormalizationError(
                "resource_capacities_missing",
                f"{location}.resource_capacities is required for unrecognized part {part!r}",
                location=location,
            )
        return dict(known), "xcu280_part_table"
    mapping = _required_mapping(explicit, f"{location}.resource_capacities")
    if set(mapping) != set(RESOURCE_KEYS):
        raise FreezeNormalizationError(
            "resource_capacities_invalid",
            f"{location}.resource_capacities must contain exactly {list(RESOURCE_KEYS)}",
            location=f"{location}.resource_capacities",
        )
    capacities = {
        key: _exact_positive_int(
            mapping.get(key), f"{location}.resource_capacities.{key}"
        )
        for key in RESOURCE_KEYS
    }
    if known is not None and capacities != known:
        raise FreezeNormalizationError(
            "resource_capacities_target_mismatch",
            f"{location}.resource_capacities disagrees with the XCU280 part table",
            location=f"{location}.resource_capacities",
        )
    return capacities, "target.resource_capacities"


def _normalize_synthesis_metrics(
    report: Any,
    capacities: Mapping[str, int],
    location: str,
    *,
    required: bool,
) -> dict[str, Any] | None:
    if not required:
        return None
    measured = _required_mapping(report, location)
    resources: dict[str, dict[str, int | float]] = {}
    for key in RESOURCE_KEYS:
        used = _exact_nonnegative_int(measured.get(key), f"{location}.{key}")
        capacity = _exact_positive_int(
            capacities.get(key), f"{location}.device_capacity.{key}"
        )
        resources[key] = {
            "used": used,
            "capacity": capacity,
            "utilization": used / capacity,
        }
    return {
        "source": "vitis_csynth_report",
        "report_sha256": _report_sha256(report, location, required=True),
        "fmax_mhz": _positive_number(measured.get("fmax_mhz"), f"{location}.fmax_mhz"),
        "resources": resources,
    }


def _json_pointer(document: Any, pointer: str, location: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise FreezeNormalizationError(
            "invalid_json_pointer", f"{location} must be empty or start with /"
        )
    current = document
    for raw_token in pointer[1:].split("/"):
        if re.search(r"~(?![01])", raw_token):
            raise FreezeNormalizationError(
                "invalid_json_pointer",
                f"{location} contains an invalid RFC 6901 escape",
                location=location,
            )
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise FreezeNormalizationError(
                    "json_pointer_missing",
                    f"{location} does not resolve; missing object key {token!r}",
                    location=location,
                )
            current = current[token]
        elif isinstance(current, list):
            try:
                index = int(token)
            except ValueError as exc:
                raise FreezeNormalizationError(
                    "json_pointer_invalid_index",
                    f"{location} uses non-integer array index {token!r}",
                ) from exc
            if index < 0 or index >= len(current):
                raise FreezeNormalizationError(
                    "json_pointer_missing",
                    f"{location} array index {index} is out of range",
                )
            current = current[index]
        else:
            raise FreezeNormalizationError(
                "json_pointer_scalar_traversal",
                f"{location} traverses through a scalar",
            )
    return current


def _load_attested_source(
    index_path: Path, source_spec: Mapping[str, Any], location: str
) -> tuple[dict[str, Any], Any, str, str]:
    artifact = _required_mapping(source_spec.get("artifact"), f"{location}.artifact")
    raw_path = _required_string(artifact.get("path"), f"{location}.artifact.path")
    expected_hash = _required_string(
        artifact.get("sha256"), f"{location}.artifact.sha256"
    ).lower()
    if SHA256_RE.fullmatch(expected_hash) is None:
        raise FreezeNormalizationError(
            "invalid_sha256",
            f"{location}.artifact.sha256 must contain 64 hexadecimal characters",
        )
    path = Path(raw_path)
    if not path.is_absolute():
        path = index_path.parent / path
    path = path.resolve()
    raw = _read_bytes(path)
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != expected_hash:
        raise FreezeNormalizationError(
            "artifact_hash_mismatch",
            f"hash mismatch for {location}: expected {expected_hash}, got {actual_hash}",
            location=location,
        )
    root = _parse_json(raw, path)
    pointer = source_spec.get("json_pointer")
    if not isinstance(pointer, str):
        raise FreezeNormalizationError(
            "json_pointer_missing",
            f"{location}.json_pointer must be a string (empty selects the document root)",
        )
    selected = _json_pointer(root, pointer, f"{location}.json_pointer")
    return root, selected, actual_hash, pointer


def _validate_isolation_evidence(
    index_path: Path,
    row: Mapping[str, Any],
    root: Mapping[str, Any],
    location: str,
) -> tuple[bool, list[dict[str, Any]]]:
    transcript_spec = _required_mapping(row.get("transcript"), f"{location}.transcript")
    transcript_root, transcript_selected, transcript_digest, transcript_pointer = (
        _load_attested_source(index_path, transcript_spec, f"{location}.transcript")
    )
    if transcript_selected is not transcript_root:
        raise FreezeNormalizationError(
            "transcript_pointer_must_be_root",
            f"{location}.transcript.json_pointer must be empty",
        )
    if not isinstance(transcript_root.get("messages"), list):
        raise FreezeNormalizationError(
            "transcript_messages_missing",
            f"{location} transcript root must contain a messages array",
        )

    audit_spec = _required_mapping(
        row.get("reference_isolation_audit"),
        f"{location}.reference_isolation_audit",
    )
    _, audit_selected, audit_digest, audit_pointer = _load_attested_source(
        index_path, audit_spec, f"{location}.reference_isolation_audit"
    )
    audit = _required_mapping(
        audit_selected, f"{location}.reference_isolation_audit selected value"
    )
    embedded = _required_mapping(
        root.get("reference_isolation_audit"),
        f"{location}.root.reference_isolation_audit",
    )
    if audit != embedded:
        raise FreezeNormalizationError(
            "reference_audit_copy_mismatch",
            f"{location} attested and embedded reference-isolation audits disagree",
        )
    missing: list[str] = []
    if audit.get("transcript_sha256") is None:
        missing.append("reference_isolation_audit.transcript_sha256")
    if missing:
        raise FreezeNormalizationError(
            "reference_audit_binding_missing",
            f"{location} audit is not cryptographically bound to its transcript",
            location=location,
            missing_fields=missing,
            producer_functions=[
                "reference_isolation.audit_history_file (hash the exact transcript bytes before parsing)",
                "paper_baselines.finalize_baseline_result / run_agentic_sweep.main (persist the bound audit)",
            ],
        )
    if audit.get("schema_version") != REFERENCE_AUDIT_SCHEMA:
        raise FreezeNormalizationError(
            "reference_audit_schema_mismatch",
            f"{location} audit schema must be {REFERENCE_AUDIT_SCHEMA}",
        )
    if str(audit.get("transcript_sha256")).lower() != transcript_digest:
        raise FreezeNormalizationError(
            "reference_audit_transcript_mismatch",
            f"{location} audit transcript hash does not match the pinned transcript",
        )
    findings = _required_list(
        audit.get("findings"), f"{location}.reference_isolation_audit.findings"
    )
    finding_count = _required_int(
        audit.get("finding_count"),
        f"{location}.reference_isolation_audit.finding_count",
    )
    if finding_count != len(findings):
        raise FreezeNormalizationError(
            "reference_audit_count_mismatch",
            f"{location} reference audit finding count disagrees with its findings",
        )
    _required_mapping(
        audit.get("finding_counts"),
        f"{location}.reference_isolation_audit.finding_counts",
    )
    passed = audit.get("passed")
    if not isinstance(passed, bool):
        raise FreezeNormalizationError(
            "reference_audit_status_missing",
            f"{location}.reference_isolation_audit.passed must be boolean",
        )
    has_error = bool(str(audit.get("error") or "").strip())
    expected_passed = finding_count == 0 and not has_error
    if passed != expected_passed:
        raise FreezeNormalizationError(
            "reference_audit_status_inconsistent",
            f"{location} reference audit pass flag disagrees with findings/error",
        )
    provenance = [
        {
            "role": "transcript",
            "source_sha256": transcript_digest,
            "json_pointer": transcript_pointer,
        },
        {
            "role": "reference_isolation_audit",
            "source_sha256": audit_digest,
            "json_pointer": audit_pointer,
        },
    ]
    return passed, provenance


def _validate_content_manifest(value: Any, location: str) -> dict[str, Any]:
    manifest = _required_mapping(value, location)
    files = _required_list(manifest.get("files"), f"{location}.files")
    if manifest.get("file_count") != len(files):
        raise FreezeNormalizationError(
            "fingerprint_manifest_count_mismatch",
            f"{location}.file_count disagrees with its files",
        )
    expected = manifest.get("sha256")
    if not isinstance(expected, str) or expected.lower() != _sha256_json(files):
        raise FreezeNormalizationError(
            "fingerprint_manifest_digest_mismatch",
            f"{location}.sha256 does not attest its files",
        )
    return manifest


def _fingerprint(root: Mapping[str, Any], location: str) -> dict[str, Any]:
    run = root.get("run") if isinstance(root.get("run"), dict) else {}
    top_level = root.get("run_fingerprint")
    nested = run.get("run_fingerprint")
    if top_level is not None and nested is not None and top_level != nested:
        raise FreezeNormalizationError(
            "fingerprint_copies_disagree",
            f"{location} top-level and run-nested fingerprints disagree",
            location=location,
        )
    fingerprint = top_level or nested
    fingerprint = _required_mapping(fingerprint, f"{location}.run_fingerprint")
    if fingerprint.get("schema_version") != FINGERPRINT_SCHEMA:
        raise FreezeNormalizationError(
            "fingerprint_schema_mismatch",
            f"{location} fingerprint schema must be {FINGERPRINT_SCHEMA}",
        )
    digest = fingerprint.get("sha256")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise FreezeNormalizationError(
            "fingerprint_missing", f"{location} lacks a complete run fingerprint"
        )
    payload = fingerprint.get("payload")
    if not isinstance(payload, dict) or _sha256_json(payload) != digest.lower():
        raise FreezeNormalizationError(
            "fingerprint_digest_mismatch",
            f"{location} run fingerprint digest does not attest its payload",
            location=location,
        )
    if payload.get("profile") != PAPER_PROFILE:
        raise FreezeNormalizationError(
            "fingerprint_profile_mismatch",
            f"{location} fingerprint profile must be {PAPER_PROFILE}",
        )
    implementation = _required_mapping(
        payload.get("implementation"),
        f"{location}.fingerprint.payload.implementation",
    )
    if not isinstance(implementation.get("git_head"), str) or not implementation.get(
        "git_head"
    ):
        raise FreezeNormalizationError(
            "implementation_identity_missing",
            f"{location} fingerprint lacks implementation.git_head",
        )
    _validate_content_manifest(
        implementation.get("sources"),
        f"{location}.fingerprint.payload.implementation.sources",
    )
    _validate_content_manifest(
        payload.get("prompts"), f"{location}.fingerprint.payload.prompts"
    )
    benchmark = _required_mapping(
        payload.get("benchmark"), f"{location}.fingerprint.payload.benchmark"
    )
    _validate_content_manifest(
        benchmark.get("inputs"),
        f"{location}.fingerprint.payload.benchmark.inputs",
    )
    reference_isolation = _required_mapping(
        payload.get("reference_isolation"),
        f"{location}.fingerprint.payload.reference_isolation",
    )
    if reference_isolation != REFERENCE_BLIND_OVERRIDES:
        raise FreezeNormalizationError(
            "reference_blind_fingerprint_mismatch",
            f"{location} fingerprint does not contain the exact paper isolation overrides",
        )
    computed_completeness = fingerprint_completeness(fingerprint)
    if computed_completeness.get("complete") is not True:
        raise FreezeNormalizationError(
            "fingerprint_incomplete",
            f"{location} fingerprint completeness check failed: {computed_completeness.get('issues')}",
            location=location,
        )
    call_issues = effective_llm_call_issues(root, fingerprint)
    if call_issues:
        raise FreezeNormalizationError(
            "effective_llm_call_mismatch",
            f"{location} actual LLM call telemetry disagrees with the fingerprint: {call_issues}",
            location=f"{location}.llm_usage.events",
        )
    reproducibility = run.get("reproducibility")
    if (
        not isinstance(reproducibility, dict)
        or reproducibility.get("complete") is not True
    ):
        raise FreezeNormalizationError(
            "fingerprint_incomplete",
            f"{location}.run.reproducibility.complete must be true",
            location=location,
        )
    if run.get("reference_blind") is not True:
        raise FreezeNormalizationError(
            "not_reference_blind", f"{location}.run.reference_blind must be true"
        )
    return fingerprint


def _validate_cohort(
    fingerprint: Mapping[str, Any], cohort: Mapping[str, Any], location: str
) -> None:
    payload = _required_mapping(
        fingerprint.get("payload"), f"{location}.fingerprint.payload"
    )
    hash_contracts = {
        "implementation_sha256": payload.get("implementation"),
        "prompts_sha256": payload.get("prompts"),
        "reference_isolation_sha256": payload.get("reference_isolation"),
    }
    for field, value in hash_contracts.items():
        expected = _required_string(cohort.get(field), f"cohort.{field}").lower()
        if SHA256_RE.fullmatch(expected) is None:
            raise FreezeNormalizationError(
                "cohort_digest_invalid", f"cohort.{field} must be a SHA-256 digest"
            )
        actual = _sha256_json(value)
        if actual != expected:
            raise FreezeNormalizationError(
                "cohort_identity_mismatch",
                f"{location} {field}={actual} != cohort value {expected}",
            )
    expected_decoding = _required_mapping(cohort.get("decoding"), "cohort.decoding")
    decoding = _required_mapping(
        payload.get("decoding"), f"{location}.fingerprint.payload.decoding"
    )
    for field in ("temperature", "top_p", "max_completion_tokens"):
        if decoding.get(field) != expected_decoding.get(field):
            raise FreezeNormalizationError(
                "cohort_identity_mismatch",
                f"{location} decoding.{field} does not match the freeze cohort",
            )
    expected_budgets = _required_mapping(cohort.get("budgets"), "cohort.budgets")
    budgets = _required_mapping(
        payload.get("budgets"), f"{location}.fingerprint.payload.budgets"
    )
    for field in ("candidate_budget", "llm_candidate_budget"):
        if str(budgets.get(field)) != str(expected_budgets.get(field)):
            raise FreezeNormalizationError(
                "cohort_identity_mismatch",
                f"{location} budgets.{field} does not match the freeze cohort",
            )
    expected_toolchain = _required_mapping(cohort.get("toolchain"), "cohort.toolchain")
    toolchain = _required_mapping(
        payload.get("toolchain"), f"{location}.fingerprint.payload.toolchain"
    )
    for field, expected in expected_toolchain.items():
        if toolchain.get(field) != expected:
            raise FreezeNormalizationError(
                "cohort_identity_mismatch",
                f"{location} toolchain.{field} does not match the freeze cohort",
            )


def _verify_identity(
    root: Mapping[str, Any],
    *,
    kernel: str,
    seed: str,
    target: Mapping[str, Any],
    method_spec: Mapping[str, Any],
    cohort: Mapping[str, Any],
    location: str,
) -> dict[str, Any]:
    fingerprint = _fingerprint(root, location)
    _validate_cohort(fingerprint, cohort, location)
    payload = _required_mapping(
        fingerprint.get("payload"), f"{location}.fingerprint.payload"
    )
    benchmark = _required_mapping(
        payload.get("benchmark"), f"{location}.fingerprint.payload.benchmark"
    )
    if str(benchmark.get("name")) != kernel:
        raise FreezeNormalizationError(
            "benchmark_identity_mismatch",
            f"{location} fingerprint benchmark {benchmark.get('name')!r} != {kernel!r}",
        )
    decoding = _required_mapping(
        payload.get("decoding"), f"{location}.fingerprint.payload.decoding"
    )
    if str(decoding.get("seed")) != seed:
        raise FreezeNormalizationError(
            "seed_identity_mismatch",
            f"{location} fingerprint seed {decoding.get('seed')!r} != {seed!r}",
        )
    model_contract = _required_mapping(
        method_spec.get("model"), f"{location}.method.model"
    )
    model = _required_mapping(
        payload.get("model"), f"{location}.fingerprint.payload.model"
    )
    if model.get("id") != model_contract.get("id"):
        raise FreezeNormalizationError(
            "model_identity_mismatch",
            f"{location} fingerprint model {model.get('id')!r} != {model_contract.get('id')!r}",
        )
    revision = _required_mapping(
        model.get("revision"), f"{location}.fingerprint.payload.model.revision"
    )
    if revision.get("resolved") is not True or revision.get(
        "value"
    ) != model_contract.get("revision"):
        raise FreezeNormalizationError(
            "model_revision_mismatch",
            f"{location} fingerprint model revision does not match the method contract",
        )
    toolchain = _required_mapping(
        payload.get("toolchain"), f"{location}.fingerprint.payload.toolchain"
    )
    for field in ("vitis_version", "part"):
        if str(toolchain.get(field)) != str(target.get(field)):
            raise FreezeNormalizationError(
                "target_identity_mismatch",
                f"{location} toolchain {field}={toolchain.get(field)!r} != {target.get(field)!r}",
            )
    try:
        source_clock = float(toolchain.get("clock_ns"))
        target_clock = float(target.get("clock_ns"))
    except (TypeError, ValueError) as exc:
        raise FreezeNormalizationError(
            "target_identity_missing", f"{location} lacks numeric clock_ns"
        ) from exc
    if source_clock != target_clock:
        raise FreezeNormalizationError(
            "target_identity_mismatch",
            f"{location} clock_ns={source_clock} != {target_clock}",
        )
    return fingerprint


def _summary_status(summary: Any, *, synthesis: bool = False) -> str:
    if not isinstance(summary, dict):
        return "not_run"
    raw_status = str(summary.get("status") or "").strip().lower()
    error = str(summary.get("error") or summary.get("skip_reason") or "").lower()
    if "timeout" in raw_status or "timed out" in error:
        return "timeout"
    if summary.get("ran") is False or raw_status in {
        "not_run",
        "not_supported",
        "skipped",
    }:
        return "not_run"
    if raw_status in {"tool_error", "error"}:
        return "tool_failure"
    pass_signal = (
        summary.get("passed") is True
        or summary.get("success") is True
        or raw_status in {"pass", "passed", "success"}
    )
    if pass_signal:
        return PASS if summary.get("ran") is True else "not_run"
    if synthesis and summary.get("ran") is True and summary.get("success") is False:
        return "failed"
    if summary.get("passed") is False or raw_status in {"fail", "failed", "mismatch"}:
        return "failed"
    return "tool_failure" if error else "not_run"


def _typed_status(value: Any, location: str) -> str:
    normalized = str(value or "").strip().lower()
    normalized = {
        "pass": PASS,
        "passed": PASS,
        "success": PASS,
        "fail": "failed",
        "tool_error": "tool_failure",
        "error": "tool_failure",
    }.get(normalized, normalized)
    if normalized not in {PASS, "failed", "not_run", "tool_failure", "timeout"}:
        raise FreezeNormalizationError(
            "typed_status_missing",
            f"{location} has unsupported status {value!r}",
            location=location,
        )
    return normalized


def _executed_cycles(cosim: Any, location: str, alternate: Any = None) -> int | None:
    if not isinstance(cosim, dict):
        if alternate is not None:
            raise FreezeNormalizationError(
                "inferred_cycle_count_forbidden",
                f"{location} supplies cycles without an executed cosim record",
            )
        return None
    policy = (
        cosim.get("cosim_policy") if isinstance(cosim.get("cosim_policy"), dict) else {}
    )
    if (
        policy.get("classification") == "predicted_timeout"
        or cosim.get("skip_reason") == "predicted_longer_than_gold"
    ):
        if alternate is not None:
            raise FreezeNormalizationError(
                "predicted_cycle_count_forbidden",
                f"{location} cannot use a predicted cosim skip as measurement",
            )
        return None
    status = _summary_status(cosim)
    raw_cycles = cosim.get("kernel_runtime_cycles")
    if raw_cycles is None:
        raw_cycles = cosim.get("cycles")
    if status != PASS:
        if raw_cycles is not None or alternate is not None:
            raise FreezeNormalizationError(
                "inferred_cycle_count_forbidden",
                f"{location} reports cycles without a passing executed cosim",
            )
        return None
    if cosim.get("ran") is not True:
        raise FreezeNormalizationError(
            "cosim_execution_missing",
            f"{location}.ran must be true for measured cycles",
        )
    cycles = _exact_positive_int(raw_cycles, f"{location}.kernel_runtime_cycles")
    if (
        alternate is not None
        and _exact_positive_int(alternate, f"{location}.alternate_cycles") != cycles
    ):
        raise FreezeNormalizationError(
            "cycle_count_mismatch", f"{location} cycle fields disagree"
        )
    return cycles


def _latency_from_report(report: Any, location: str, *, required: bool) -> int | None:
    if not isinstance(report, dict):
        if required:
            raise FreezeNormalizationError(
                "csynth_latency_missing", f"{location} lacks a Vitis synthesis report"
            )
        return None
    value = report.get("latency_cycles_worst")
    field = "latency_cycles_worst"
    if value is None:
        value = report.get("latency_cycles")
        field = "latency_cycles"
    if value is None:
        if required:
            raise FreezeNormalizationError(
                "csynth_latency_missing",
                f"{location} lacks latency_cycles_worst/latency_cycles; latency_ns is not converted",
            )
        return None
    return _exact_positive_int(value, f"{location}.{field}")


def _failure_class(
    *,
    correctness: str,
    synthesis: str,
    resource_fit: bool | None,
    timing_met: bool | None,
    cosim: str = "not_run",
    rejection_reason: str = "",
    isolation_passed: bool = True,
) -> str | None:
    if not isolation_passed:
        return "reference_isolation_failure"
    reason = rejection_reason.lower()
    if correctness in {"failed", "timeout"}:
        return "wrong_output" if correctness == "failed" else "tool_failure"
    if correctness == "tool_failure":
        return "tool_failure"
    if correctness == "not_run":
        if "missing" in reason or "code" in reason or "pragma" in reason:
            return "malformed_output"
        return "compile_or_interface_failure"
    if synthesis == "timeout":
        return "synthesis_timeout"
    if synthesis in {"failed", "tool_failure"}:
        return "tool_failure"
    if synthesis == "not_run":
        return (
            "candidate_budget_exhausted"
            if "budget" in reason
            else "compile_or_interface_failure"
        )
    if resource_fit is False:
        return "infeasible_resources"
    if timing_met is False:
        return "timing_failure"
    if cosim == "timeout":
        return "cosim_timeout"
    if cosim in {"failed", "tool_failure"}:
        return "cosim_failure" if cosim == "failed" else "tool_failure"
    if cosim == "not_run":
        return "missing_executed_cosim"
    return None


def _baseline_candidate_events(
    root: Mapping[str, Any], run_id: str, location: str
) -> list[dict[str, Any]]:
    candidates = _required_list(root.get("candidates"), f"{location}.candidates")
    llm_usage = _required_mapping(root.get("llm_usage"), f"{location}.llm_usage")
    llm_events = _required_list(llm_usage.get("events"), f"{location}.llm_usage.events")
    synthesis_summary = _required_mapping(
        root.get("synthesis_evaluations"), f"{location}.synthesis_evaluations"
    )
    synthesis_events = _required_list(
        synthesis_summary.get("events"), f"{location}.synthesis_evaluations.events"
    )
    missing: list[str] = []
    if root.get("total_synthesis_calls") is None:
        missing.append("total_synthesis_calls")
    if root.get("selected_winner_cosim_count") is None:
        missing.append("selected_winner_cosim_count")
    for index, candidate in enumerate(candidates):
        if (
            not isinstance(candidate, dict)
            or candidate.get("cumulative_elapsed_seconds") is None
        ):
            missing.append(f"candidates[{index}].cumulative_elapsed_seconds")
    if missing:
        raise FreezeNormalizationError(
            "paper_baseline_candidate_telemetry_incomplete",
            "run_paper_baseline result cannot faithfully produce schema-v2 candidate events",
            location=location,
            missing_fields=missing,
            producer_functions=[
                "paper_baselines.PaperBaselineEngine._evaluate (record candidate completion time)",
                "paper_baselines.PaperBaselineEngine.run (record placeholder completion times and total_synthesis_calls including selected cosim)",
            ],
        )
    llm_by_candidate: dict[int, dict[str, Any]] = {}
    for event_index, event_raw in enumerate(llm_events):
        event = _required_mapping(
            event_raw, f"{location}.llm_usage.events[{event_index}]"
        )
        candidate_index = _required_int(
            event.get("candidate_index"),
            f"{location}.llm_usage.events[{event_index}].candidate_index",
        )
        if candidate_index in llm_by_candidate:
            raise FreezeNormalizationError(
                "duplicate_llm_event",
                f"duplicate LLM event for candidate {candidate_index}",
            )
        llm_by_candidate[candidate_index] = event
    out_of_range_llm = sorted(
        candidate_index
        for candidate_index in llm_by_candidate
        if candidate_index >= len(candidates)
    )
    if out_of_range_llm:
        raise FreezeNormalizationError(
            "orphan_llm_event",
            f"{location} has LLM events outside its candidate array: {out_of_range_llm}",
        )
    synth_by_candidate: dict[int, dict[str, Any]] = {}
    for index, event in enumerate(synthesis_events):
        event = _required_mapping(
            event, f"{location}.synthesis_evaluations.events[{index}]"
        )
        candidate_index = _required_int(
            event.get("candidate_index"),
            f"{location}.synthesis_evaluations.events[{index}].candidate_index",
        )
        if candidate_index in synth_by_candidate:
            raise FreezeNormalizationError(
                "duplicate_synthesis_event",
                f"duplicate synthesis event for candidate {candidate_index}",
            )
        synth_by_candidate[candidate_index] = event
    out_of_range_synth = sorted(
        candidate_index
        for candidate_index in synth_by_candidate
        if candidate_index >= len(candidates)
    )
    if out_of_range_synth:
        raise FreezeNormalizationError(
            "orphan_synthesis_event",
            f"{location} has synthesis events outside its candidate array: {out_of_range_synth}",
        )

    selected_index = root.get("selected_candidate_index")
    cumulative_tokens = 0
    cumulative_syntheses = 0
    previous_elapsed = 0.0
    output: list[dict[str, Any]] = []
    for index, candidate_raw in enumerate(candidates):
        candidate = _required_mapping(candidate_raw, f"{location}.candidates[{index}]")
        source_candidate_index = _required_int(
            candidate.get("index"), f"{location}.candidates[{index}].index"
        )
        if source_candidate_index != index:
            raise FreezeNormalizationError(
                "candidate_index_mismatch",
                f"{location} candidate indices are not contiguous",
            )
        llm_event = llm_by_candidate.get(index)
        if llm_event is not None:
            if llm_event.get("usage_available") is not True:
                raise FreezeNormalizationError(
                    "token_telemetry_missing",
                    f"{location}.llm_usage event for candidate {index} lacks provider token usage",
                )
            tokens = _required_int(
                llm_event.get("total_tokens"),
                f"{location}.llm_usage event for candidate {index}.total_tokens",
            )
            cumulative_tokens += tokens
            candidate_response_hash = _required_string(
                candidate.get("response_sha256"),
                f"{location}.candidates[{index}].response_sha256",
            ).lower()
            llm_response_hash = _required_string(
                llm_event.get("response_sha256"),
                f"{location}.llm_usage event for candidate {index}.response_sha256",
            ).lower()
            if (
                SHA256_RE.fullmatch(candidate_response_hash) is None
                or candidate_response_hash != llm_response_hash
            ):
                raise FreezeNormalizationError(
                    "baseline_candidate_llm_join_mismatch",
                    f"{location} candidate {index} response hashes disagree",
                )
        elapsed = candidate.get("cumulative_elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
        ):
            raise FreezeNormalizationError(
                "candidate_elapsed_invalid",
                f"{location}.candidates[{index}].cumulative_elapsed_seconds is invalid",
            )
        elapsed = float(elapsed)
        if elapsed < previous_elapsed:
            raise FreezeNormalizationError(
                "candidate_elapsed_nonmonotonic",
                f"{location} candidate completion times are nonmonotonic",
            )
        csim = candidate.get("csim")
        correctness = _summary_status(csim)
        synthesis_summary_raw = candidate.get("synthesis")
        synthesis = _summary_status(synthesis_summary_raw, synthesis=True)
        synthesis_event = synth_by_candidate.get(index)
        synthesis_ran = (
            isinstance(synthesis_summary_raw, dict)
            and synthesis_summary_raw.get("ran") is True
        )
        if synthesis_ran:
            if synthesis_event is None:
                raise FreezeNormalizationError(
                    "synthesis_event_missing",
                    f"{location} candidate {index} ran synthesis without an event",
                )
            cumulative_syntheses += 1
            candidate_code_hash = _required_string(
                candidate.get("code_sha256"),
                f"{location}.candidates[{index}].code_sha256",
            ).lower()
            synthesis_code_hash = _required_string(
                synthesis_event.get("code_sha256"),
                f"{location}.synthesis event for candidate {index}.code_sha256",
            ).lower()
            if (
                SHA256_RE.fullmatch(candidate_code_hash) is None
                or candidate_code_hash != synthesis_code_hash
                or synthesis_event.get("success")
                is not synthesis_summary_raw.get("success")
            ):
                raise FreezeNormalizationError(
                    "baseline_candidate_synthesis_join_mismatch",
                    f"{location} candidate {index} synthesis hashes/statuses disagree",
                )
        elif synthesis_event is not None:
            raise FreezeNormalizationError(
                "orphan_synthesis_event",
                f"{location} candidate {index} has an orphan synthesis event",
            )
        feasibility = (
            candidate.get("feasibility")
            if isinstance(candidate.get("feasibility"), dict)
            else {}
        )
        resource_fit = (
            feasibility.get("resource_fit")
            if isinstance(feasibility.get("resource_fit"), bool)
            else None
        )
        timing_met = (
            feasibility.get("timing_met")
            if isinstance(feasibility.get("timing_met"), bool)
            else None
        )
        if synthesis == PASS and (resource_fit is None or timing_met is None):
            raise FreezeNormalizationError(
                "candidate_feasibility_telemetry_missing",
                f"{location}.candidates[{index}].feasibility must record resource_fit and timing_met after synthesis",
            )
        latency = _latency_from_report(
            candidate.get("report"),
            f"{location}.candidates[{index}].report",
            required=synthesis == PASS,
        )
        feasible = bool(
            correctness == PASS
            and synthesis == PASS
            and resource_fit is True
            and timing_met is True
            and latency is not None
        )
        rejection_reason = str(candidate.get("rejection_reason") or "")
        failure = (
            None
            if feasible
            else _failure_class(
                correctness=correctness,
                synthesis=synthesis,
                resource_fit=resource_fit,
                timing_met=timing_met,
                rejection_reason=rejection_reason,
            )
        )
        output.append(
            {
                "event_id": f"{run_id}.c{index + 1}",
                "candidate_index": index + 1,
                "code_sha256": _required_sha256(
                    candidate.get("code_sha256"),
                    f"{location}.candidates[{index}].code_sha256",
                ),
                "report_sha256": _report_sha256(
                    candidate.get("report"),
                    f"{location}.candidates[{index}].report",
                    required=synthesis == PASS,
                ),
                "cumulative_tokens": cumulative_tokens,
                "cumulative_llm_calls": sum(
                    1 for candidate_id in llm_by_candidate if candidate_id <= index
                ),
                "cumulative_synthesis_evaluations": cumulative_syntheses,
                "cumulative_elapsed_seconds": elapsed,
                "correctness_status": correctness,
                "synthesis_status": synthesis,
                "resource_fit": resource_fit,
                "timing_met": timing_met,
                "synthesized_latency_cycles": latency,
                "latency_source": (
                    "vitis_csynth_report" if latency is not None else "none"
                ),
                "failure_class": failure,
                "selected_for_executed_cosim": selected_index == index,
            }
        )
        previous_elapsed = elapsed
    if cumulative_tokens != _required_int(
        llm_usage.get("total_tokens"), f"{location}.llm_usage.total_tokens"
    ):
        raise FreezeNormalizationError(
            "token_total_mismatch",
            f"{location} LLM event token sum disagrees with total",
        )
    if len(llm_by_candidate) != _required_int(
        llm_usage.get("calls"), f"{location}.llm_usage.calls"
    ):
        raise FreezeNormalizationError(
            "llm_call_total_mismatch",
            f"{location} LLM event count disagrees with total",
        )
    if cumulative_syntheses != _required_int(
        synthesis_summary.get("count"), f"{location}.synthesis_evaluations.count"
    ):
        raise FreezeNormalizationError(
            "synthesis_total_mismatch",
            f"{location} synthesis event count disagrees with total",
        )
    return output


def _agentic_candidate_events(
    root: Mapping[str, Any], run_id: str, location: str
) -> list[dict[str, Any]]:
    summary = root.get("synthesis_evaluations")
    summary = summary if isinstance(summary, dict) else {}
    events = summary.get("events") if isinstance(summary.get("events"), list) else []
    required_event_fields = [
        "code_sha256",
        "report_sha256",
        "cumulative_tokens",
        "cumulative_llm_calls",
        "cumulative_synthesis_evaluations",
        "cumulative_elapsed_seconds",
        "correctness_status",
        "synthesis_status",
        "resource_fit",
        "timing_met",
        "synthesized_latency_cycles",
        "latency_source",
        "failure_class",
        "selected_for_executed_cosim",
    ]
    missing: list[str] = []
    if summary.get("complete_candidate_event_stream") is not True:
        missing.append("synthesis_evaluations.complete_candidate_event_stream")
    if root.get("total_synthesis_calls") is None:
        missing.append("total_synthesis_calls")
    llm_usage = root.get("llm_usage") if isinstance(root.get("llm_usage"), dict) else {}
    llm_events = (
        llm_usage.get("events") if isinstance(llm_usage.get("events"), list) else []
    )
    if not llm_events:
        missing.append("llm_usage.events[*]")
    elif any(
        not isinstance(event, dict) or "candidate_evaluation_index" not in event
        for event in llm_events
    ):
        missing.append("llm_usage.events[*].candidate_evaluation_index")
    if llm_usage.get("usage_missing_calls") is None:
        missing.append("llm_usage.usage_missing_calls")
    if not events:
        missing.append("synthesis_evaluations.events[*]")
    else:
        for field in required_event_fields:
            if any(field not in event for event in events if isinstance(event, dict)):
                missing.append(f"synthesis_evaluations.events[*].{field}")
    if missing:
        raise FreezeNormalizationError(
            "agentic_candidate_telemetry_incomplete",
            "run_agentic_sweep result cannot faithfully join controller candidates to schema-v2 events",
            location=location,
            missing_fields=missing,
            producer_functions=[
                "c2hls.C2HLSOrchestrator._record_llm_usage (add candidate-evaluation ID and cumulative token/call snapshot)",
                "c2hls.C2HLSOrchestrator._synth_and_test (record CSim status, report latency, fit/timing, failure and cumulative elapsed)",
                "c2hls.C2HLSOrchestrator._optimization_step_attempt_single (emit compile/CSim-rejected candidates into the unified stream)",
                "c2hls.C2HLSOrchestrator._run_selected_winner_cosim (mark the selected event and count total synthesis calls)",
            ],
        )

    if len(events) > 5:
        raise FreezeNormalizationError(
            "candidate_budget_exceeded",
            f"{location} unified candidate stream exceeds the five-candidate budget",
        )
    candidate_requests = _required_int(
        llm_usage.get("candidate_requests"),
        f"{location}.llm_usage.candidate_requests",
    )
    if candidate_requests != len(events) or candidate_requests > 5:
        raise FreezeNormalizationError(
            "candidate_budget_mismatch",
            f"{location} candidate-request count disagrees with the unified stream",
        )
    selected_cosim_count = _required_int(
        root.get("selected_winner_cosim_count"),
        f"{location}.selected_winner_cosim_count",
    )
    selection_synthesis_count = _required_int(
        summary.get("count"), f"{location}.synthesis_evaluations.count"
    )
    if (
        selected_cosim_count not in {0, 1}
        or root.get("total_synthesis_calls")
        != selection_synthesis_count + selected_cosim_count
    ):
        raise FreezeNormalizationError(
            "synthesis_attribution_mismatch",
            f"{location} total synthesis calls must equal selection syntheses plus selected cosim flow",
        )

    if (
        _required_int(
            llm_usage.get("usage_missing_calls"),
            f"{location}.llm_usage.usage_missing_calls",
        )
        != 0
    ):
        raise FreezeNormalizationError(
            "token_telemetry_missing",
            f"{location} has LLM calls without provider token usage",
        )
    if len(llm_events) != _required_int(
        llm_usage.get("calls"), f"{location}.llm_usage.calls"
    ):
        raise FreezeNormalizationError(
            "llm_call_total_mismatch",
            f"{location} LLM event count disagrees with total",
        )
    cumulative_tokens_by_candidate = [0] * len(events)
    cumulative_calls_by_candidate = [0] * len(events)
    running_tokens = 0
    running_calls = 0
    llm_by_candidate: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(events))
    }
    for event_index, raw_llm_event in enumerate(llm_events):
        llm_event = _required_mapping(
            raw_llm_event, f"{location}.llm_usage.events[{event_index}]"
        )
        if llm_event.get("usage_available") is not True:
            raise FreezeNormalizationError(
                "token_telemetry_missing",
                f"{location}.llm_usage.events[{event_index}] lacks provider usage",
            )
        candidate_index = _required_int(
            llm_event.get("candidate_evaluation_index"),
            f"{location}.llm_usage.events[{event_index}].candidate_evaluation_index",
        )
        if candidate_index >= len(events):
            raise FreezeNormalizationError(
                "orphan_llm_event",
                f"{location} LLM event {event_index} has no candidate event",
            )
        llm_by_candidate[candidate_index].append(llm_event)
    for candidate_index in range(len(events)):
        if not llm_by_candidate[candidate_index]:
            raise FreezeNormalizationError(
                "candidate_llm_join_incomplete",
                f"{location} candidate {candidate_index} has no attributed LLM call",
            )
        for llm_event in llm_by_candidate[candidate_index]:
            running_tokens += _required_int(
                llm_event.get("total_tokens"),
                f"{location}.llm_usage candidate {candidate_index}.total_tokens",
            )
            running_calls += 1
        cumulative_tokens_by_candidate[candidate_index] = running_tokens
        cumulative_calls_by_candidate[candidate_index] = running_calls
    if running_tokens != _required_int(
        llm_usage.get("total_tokens"), f"{location}.llm_usage.total_tokens"
    ):
        raise FreezeNormalizationError(
            "token_total_mismatch", f"{location} LLM token events disagree with total"
        )

    output: list[dict[str, Any]] = []
    for index, raw_event in enumerate(events):
        event = _required_mapping(
            raw_event, f"{location}.synthesis_evaluations.events[{index}]"
        )
        forbidden_fields = sorted(
            field
            for field in event
            if "predicted" in field.lower()
            or field.lower().startswith("estimated_latency")
            or field.lower().startswith("gold_relative")
        )
        if forbidden_fields:
            raise FreezeNormalizationError(
                "forbidden_candidate_telemetry",
                f"{location} candidate event contains predicted/oracle fields: {forbidden_fields}",
            )
        source_index = _required_int(
            event.get("candidate_evaluation_index"),
            f"{location}.synthesis_evaluations.events[{index}].candidate_evaluation_index",
        )
        if source_index != index:
            raise FreezeNormalizationError(
                "agentic_candidate_index_mismatch",
                f"{location} candidate_evaluation_index must be contiguous from zero",
            )
        if event.get("cumulative_tokens") != cumulative_tokens_by_candidate[index]:
            raise FreezeNormalizationError(
                "candidate_llm_join_mismatch",
                f"{location} candidate {index} cumulative token count disagrees with attributed LLM events",
            )
        if event.get("cumulative_llm_calls") != cumulative_calls_by_candidate[index]:
            raise FreezeNormalizationError(
                "candidate_llm_join_mismatch",
                f"{location} candidate {index} cumulative LLM-call count disagrees with attributed events",
            )
        normalized = {
            "event_id": f"{run_id}.c{index + 1}",
            "candidate_index": index + 1,
            **{field: event[field] for field in required_event_fields},
        }
        normalized["code_sha256"] = _required_sha256(
            normalized["code_sha256"],
            f"{location}.synthesis_evaluations.events[{index}].code_sha256",
        )
        report_digest = normalized["report_sha256"]
        if report_digest is not None:
            report_digest = _required_sha256(
                report_digest,
                f"{location}.synthesis_evaluations.events[{index}].report_sha256",
            )
        if normalized["synthesis_status"] == PASS and report_digest is None:
            raise FreezeNormalizationError(
                "candidate_report_hash_missing",
                f"{location} synthesized candidate {index} lacks report_sha256",
                location=f"{location}.synthesis_evaluations.events[{index}]",
            )
        normalized["report_sha256"] = report_digest
        output.append(normalized)
    return output


def _authenticate_selected_winner(
    root: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    synthesis_metrics: Mapping[str, Any] | None,
    location: str,
) -> tuple[str | None, str | None]:
    """Bind the normalized final report and RTL target to one candidate event."""
    selected = [event for event in events if event.get("selected_for_executed_cosim")]
    selected_count = _required_int(
        root.get("selected_winner_cosim_count"),
        f"{location}.selected_winner_cosim_count",
    )
    if selected_count not in {0, 1} or len(selected) != selected_count:
        raise FreezeNormalizationError(
            "selected_winner_binding_mismatch",
            f"{location} selected candidate count disagrees with the executed-cosim flow",
            location=location,
        )

    selected_hash_raw = root.get("selected_code_sha256")
    cosim_hash_raw = root.get("cosim_target_code_sha256")
    if selected_count == 0:
        if selected_hash_raw is not None or cosim_hash_raw is not None:
            raise FreezeNormalizationError(
                "selected_winner_binding_mismatch",
                f"{location} exposes winner hashes without a selected-winner cosim flow",
                location=location,
            )
        return None, None

    selected_hash = _required_sha256(
        selected_hash_raw, f"{location}.selected_code_sha256"
    )
    cosim_hash = _required_sha256(
        cosim_hash_raw, f"{location}.cosim_target_code_sha256"
    )
    event = selected[0]
    if selected_hash != cosim_hash or event.get("code_sha256") != selected_hash:
        raise FreezeNormalizationError(
            "selected_winner_code_hash_mismatch",
            f"{location} final selection, candidate event, and cosim target hashes disagree",
            location=location,
        )
    cosim = root.get("cosim")
    if isinstance(cosim, dict) and cosim.get("target_code_sha256") is not None:
        if _required_sha256(
            cosim.get("target_code_sha256"),
            f"{location}.cosim.target_code_sha256",
        ) != selected_hash:
            raise FreezeNormalizationError(
                "selected_winner_code_hash_mismatch",
                f"{location} cosim result targets a different candidate",
                location=f"{location}.cosim.target_code_sha256",
            )
    if synthesis_metrics is None:
        raise FreezeNormalizationError(
            "selected_winner_report_hash_mismatch",
            f"{location} selected winner lacks normalized synthesis metrics",
            location=location,
        )
    report_hash = synthesis_metrics.get("report_sha256")
    if event.get("report_sha256") != report_hash:
        raise FreezeNormalizationError(
            "selected_winner_report_hash_mismatch",
            f"{location} final report is not the selected candidate's synthesized report",
            location=location,
        )
    return selected_hash, cosim_hash


def _normalize_generated_record(
    root: Mapping[str, Any],
    row: Mapping[str, Any],
    method_spec: Mapping[str, Any],
    resource_capacities: Mapping[str, int],
    location: str,
    *,
    audit_passed: bool,
) -> dict[str, Any]:
    run_id = _required_string(row.get("run_id"), f"{location}.run_id")
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise FreezeNormalizationError(
            "invalid_run_id", f"{location}.run_id is not opaque"
        )
    runner = _required_string(row.get("runner"), f"{location}.runner")
    method_id = _required_string(row.get("method"), f"{location}.method")
    if method_id != method_spec.get("id"):
        raise FreezeNormalizationError(
            "method_identity_mismatch",
            f"{location} row method does not match its method contract",
        )
    if runner != method_spec.get("runner"):
        raise FreezeNormalizationError(
            "runner_identity_mismatch",
            f"{location} row runner does not match its method contract",
        )
    runner_method = _required_string(
        method_spec.get("runner_method"), f"{location}.method.runner_method"
    )
    fingerprint = root.get("run_fingerprint") or (root.get("run") or {}).get(
        "run_fingerprint"
    )
    payload = fingerprint.get("payload") if isinstance(fingerprint, dict) else {}
    skills = payload.get("skills") if isinstance(payload, dict) else {}
    skill_on = runner_method == "dynamic_frozen_skills"
    if (
        not isinstance(skills, dict)
        or skills.get("mode") != ("skill_on" if skill_on else "skill_off")
        or skills.get("prompt_injection") is not skill_on
        or skills.get("frozen") is not True
        or skills.get("persistence") is not False
        or skills.get("online_statistics") is not False
    ):
        raise FreezeNormalizationError(
            "skill_isolation_contract_mismatch",
            f"{location} fingerprint skill controls do not match {runner_method}",
        )
    if skill_on and (
        skills.get("source_mode") != "explicit_frozen_snapshot"
        or skills.get("file_count") != 1
        or not skills.get("expected_sha256")
        or skills.get("matches_expected") is not True
    ):
        raise FreezeNormalizationError(
            "skill_freeze_missing",
            f"{location} frozen-skill fingerprint lacks an exact matched snapshot",
        )
    if runner == "run_paper_baseline.py":
        if runner_method not in BASELINE_METHODS:
            raise FreezeNormalizationError(
                "unknown_baseline_method",
                f"{location} unsupported paper baseline method {runner_method!r}",
            )
        if root.get("schema_version") != BASELINE_RESULT_SCHEMA:
            raise FreezeNormalizationError(
                "runner_schema_mismatch",
                f"{location} expected {BASELINE_RESULT_SCHEMA}, got {root.get('schema_version')!r}",
            )
        if root.get("method") != runner_method:
            raise FreezeNormalizationError(
                "method_identity_mismatch",
                f"{location} source method {root.get('method')!r} != {runner_method!r}",
            )
        baseline_contract = (
            payload.get("paper_baseline") if isinstance(payload, dict) else {}
        )
        if (
            not isinstance(baseline_contract, dict)
            or baseline_contract.get("schema_version") != BASELINE_RESULT_SCHEMA
            or baseline_contract.get("method") != runner_method
            or baseline_contract.get("max_llm_candidates") != 5
            or baseline_contract.get("max_synthesis_evaluations") != 5
            or baseline_contract.get("correctness_order")
            != "csim_golden_before_synthesis"
            or baseline_contract.get("cosim_policy") != "selected_winner_only"
        ):
            raise FreezeNormalizationError(
                "method_identity_mismatch",
                f"{location} fingerprint paper-baseline contract does not match {runner_method}",
            )
        if (
            root.get("candidate_count") != len(root.get("candidates") or [])
            or root.get("candidate_count") != 5
        ):
            raise FreezeNormalizationError(
                "baseline_candidate_contract_mismatch",
                f"{location} baseline must expose all five candidate outcomes",
            )
        events = _baseline_candidate_events(root, run_id, location)
        selected_cosim_count = _required_int(
            root.get("selected_winner_cosim_count"),
            f"{location}.selected_winner_cosim_count",
        )
        if (
            selected_cosim_count not in {0, 1}
            or root.get("total_synthesis_calls")
            != root.get("synthesis_evaluations", {}).get("count") + selected_cosim_count
        ):
            raise FreezeNormalizationError(
                "synthesis_attribution_mismatch",
                f"{location} total synthesis calls must equal selection syntheses plus selected cosim flow",
            )
    elif runner == "run_agentic_sweep.py":
        contract = AGENTIC_METHOD_CONTRACTS.get(runner_method)
        if contract is None:
            raise FreezeNormalizationError(
                "unknown_agentic_method",
                f"{location} unsupported agentic method {runner_method!r}",
            )
        search = payload.get("search") if isinstance(payload, dict) else {}
        if (
            search.get("strategy") != contract["strategy"]
            or skills.get("mode") != contract["skill_mode"]
        ):
            raise FreezeNormalizationError(
                "method_identity_mismatch",
                f"{location} fingerprint strategy/skill mode does not match {runner_method}",
            )
        if (
            runner_method == "dynamic_frozen_skills"
            and skills.get("frozen") is not True
        ):
            raise FreezeNormalizationError(
                "skill_freeze_missing",
                f"{location} dynamic skill run is not fingerprinted frozen",
            )
        if runner_method == "dynamic_frozen_skills":
            integrity = root.get("skill_snapshot_integrity")
            if not isinstance(integrity, dict) or not isinstance(
                integrity.get("unchanged"), bool
            ):
                raise FreezeNormalizationError(
                    "skill_integrity_telemetry_missing",
                    f"{location}.skill_snapshot_integrity.unchanged must be recorded",
                )
        events = _agentic_candidate_events(root, run_id, location)
    else:
        raise FreezeNormalizationError(
            "unknown_runner",
            f"{location}.runner must be run_agentic_sweep.py or run_paper_baseline.py",
        )

    skill_integrity = root.get("skill_snapshot_integrity")
    skill_integrity_failed = bool(
        isinstance(skill_integrity, dict) and skill_integrity.get("unchanged") is False
    )
    evaluation = _required_mapping(
        root.get("evaluation_status"), f"{location}.evaluation_status"
    )
    llm_usage = _required_mapping(root.get("llm_usage"), f"{location}.llm_usage")
    if evaluation.get("schema_version") != "c2hls.evaluation-status.v1":
        raise FreezeNormalizationError(
            "status_schema_mismatch",
            f"{location}.evaluation_status has an unsupported schema",
        )
    correctness = _typed_status(
        evaluation.get("correctness_status"),
        f"{location}.evaluation_status.correctness_status",
    )
    synthesis = _typed_status(
        evaluation.get("synthesis_status"),
        f"{location}.evaluation_status.synthesis_status",
    )
    root_correctness = _typed_status(
        root.get("correctness_status"), f"{location}.correctness_status"
    )
    if root_correctness != correctness:
        raise FreezeNormalizationError(
            "typed_status_mismatch",
            f"{location} root and evaluation correctness statuses disagree",
        )
    final_report = root.get("final_report")
    if synthesis == PASS and not (
        isinstance(final_report, dict) and bool(final_report)
    ):
        raise FreezeNormalizationError(
            "synthesis_evidence_missing",
            f"{location} reports passing synthesis without final_report",
        )
    if synthesis != PASS and isinstance(final_report, dict) and final_report:
        raise FreezeNormalizationError(
            "typed_status_mismatch",
            f"{location} has final_report despite nonpassing synthesis status",
        )
    synthesis_metrics = _normalize_synthesis_metrics(
        final_report,
        resource_capacities,
        f"{location}.final_report",
        required=synthesis == PASS,
    )
    selected_code_sha256, cosim_target_code_sha256 = _authenticate_selected_winner(
        root, events, synthesis_metrics, location
    )
    cosim = root.get("cosim")
    cosim_status = _summary_status(cosim)
    typed_cosim_status = _typed_status(
        evaluation.get("cosim_execution_status"),
        f"{location}.evaluation_status.cosim_execution_status",
    )
    if root.get("executed_cosim_status") != evaluation.get("cosim_execution_status"):
        raise FreezeNormalizationError(
            "typed_status_mismatch",
            f"{location} executed-cosim status copies disagree",
        )
    if typed_cosim_status != cosim_status:
        raise FreezeNormalizationError(
            "typed_status_mismatch",
            f"{location} evaluation and cosim evidence statuses disagree",
        )
    cosim_ran = bool(isinstance(cosim, dict) and cosim.get("ran") is True)
    if evaluation.get("cosim_ran") is not cosim_ran:
        raise FreezeNormalizationError(
            "typed_status_mismatch",
            f"{location}.evaluation_status.cosim_ran disagrees with cosim evidence",
        )
    predicted_skip = evaluation.get("cosim_predicted_skip")
    if (
        not isinstance(predicted_skip, bool)
        or root.get("predicted_cosim_skip") is not predicted_skip
    ):
        raise FreezeNormalizationError(
            "typed_status_mismatch",
            f"{location} predicted-cosim status copies disagree",
        )
    if predicted_skip:
        raise FreezeNormalizationError(
            "predicted_cosim_forbidden",
            f"{location} reference-blind paper result contains a predicted cosim skip",
        )
    llm_events = _required_list(
        llm_usage.get("events"), f"{location}.llm_usage.events"
    )
    actual_provider_failure = any(
        isinstance(event, dict) and bool(event.get("error")) for event in llm_events
    )
    if not isinstance(evaluation.get("provider_failure"), bool) or evaluation.get(
        "provider_failure"
    ) is not actual_provider_failure:
        raise FreezeNormalizationError(
            "typed_status_mismatch",
            f"{location}.evaluation_status.provider_failure disagrees with attributed LLM-event errors",
        )
    if "provider_failure" in root and root.get(
        "provider_failure"
    ) is not actual_provider_failure:
        raise FreezeNormalizationError(
            "typed_status_mismatch",
            f"{location} root provider_failure copy disagrees with LLM-event errors",
        )
    expected_timeout = (
        correctness == "timeout"
        or synthesis == "timeout"
        or cosim_status == "timeout"
    )
    if evaluation.get("timeout") is not expected_timeout or root.get(
        "timeout_status"
    ) != ("timeout" if expected_timeout else "none"):
        raise FreezeNormalizationError(
            "typed_status_mismatch",
            f"{location} timeout status copies disagree",
        )
    expected_tool_failure = (
        correctness == "tool_failure"
        or synthesis == "tool_failure"
        or cosim_status == "tool_failure"
        or actual_provider_failure
    )
    if evaluation.get("tool_failure") is not expected_tool_failure or root.get(
        "tool_failure_status"
    ) != ("tool_failure" if expected_tool_failure else "none"):
        raise FreezeNormalizationError(
            "typed_status_mismatch",
            f"{location} tool-failure status copies disagree",
        )
    alternate_cycles = root.get("executed_cosim_cycles")
    cycles = _executed_cycles(cosim, f"{location}.cosim", alternate_cycles)
    feasibility = root.get("candidate_feasibility")
    feasibility = feasibility if isinstance(feasibility, dict) else {}
    resource_fit = (
        feasibility.get("resource_fit")
        if isinstance(feasibility.get("resource_fit"), bool)
        else None
    )
    timing_met = (
        feasibility.get("timing_met")
        if isinstance(feasibility.get("timing_met"), bool)
        else None
    )
    if synthesis == PASS and (resource_fit is None or timing_met is None):
        raise FreezeNormalizationError(
            "final_feasibility_telemetry_missing",
            f"{location}.candidate_feasibility must record resource_fit and timing_met",
        )
    if not audit_passed or skill_integrity_failed:
        cycles = None
        cosim_status = "not_run"
    solved = bool(
        audit_passed
        and not skill_integrity_failed
        and correctness == PASS
        and synthesis == PASS
        and resource_fit is True
        and timing_met is True
        and cosim_status == PASS
        and cycles is not None
    )
    if solved:
        failure = None
    elif not audit_passed:
        failure = "reference_isolation_failure"
    elif skill_integrity_failed:
        failure = "other"
    else:
        failure = _failure_class(
            correctness=correctness,
            synthesis=synthesis,
            resource_fit=resource_fit,
            timing_met=timing_met,
            cosim=cosim_status,
            isolation_passed=True,
        )
        if actual_provider_failure:
            failure = "tool_failure"
    raw_success = root.get("success")
    if not isinstance(raw_success, bool):
        raise FreezeNormalizationError(
            "terminal_status_missing", f"{location}.success must be boolean"
        )
    expected_raw_success = solved
    if raw_success != expected_raw_success:
        raise FreezeNormalizationError(
            "terminal_status_inconsistent",
            f"{location}.success={raw_success} disagrees with typed correctness/feasibility/cosim evidence",
        )
    run = _required_mapping(root.get("run"), f"{location}.run")
    if run.get("paper_method_wall_time_field") != "search_elapsed_seconds":
        raise FreezeNormalizationError(
            "wall_time_contract_missing",
            f"{location}.run.paper_method_wall_time_field must select search_elapsed_seconds",
        )
    search_elapsed = run.get("search_elapsed_seconds")
    if (
        isinstance(search_elapsed, bool)
        or not isinstance(search_elapsed, (int, float))
        or not math.isfinite(float(search_elapsed))
        or float(search_elapsed) < 0
    ):
        raise FreezeNormalizationError(
            "wall_time_telemetry_missing",
            f"{location}.run.search_elapsed_seconds must be non-negative and finite",
        )
    record = {
        "run_id": run_id,
        "terminal_status": "success" if solved else "failure",
        "correctness_status": correctness,
        "synthesis_status": synthesis,
        "resource_fit": resource_fit,
        "timing_met": timing_met,
        "cosim_status": cosim_status,
        "cycle_source": "executed_rtl_cosim" if cycles is not None else "none",
        "executed_cosim_cycles": cycles,
        "failure_class": failure,
        "synthesis_metrics": synthesis_metrics,
        "selected_code_sha256": selected_code_sha256,
        "cosim_target_code_sha256": cosim_target_code_sha256,
        "provider_failure": actual_provider_failure,
        "reference_isolation_status": PASS if audit_passed else "failed",
        "tokens": _required_int(
            llm_usage.get("total_tokens"), f"{location}.llm_usage.total_tokens"
        ),
        "llm_calls": _required_int(
            llm_usage.get("calls"), f"{location}.llm_usage.calls"
        ),
        "synthesis_calls": _required_int(
            root.get("total_synthesis_calls"), f"{location}.total_synthesis_calls"
        ),
        "selection_synthesis_evaluations": _required_int(
            (root.get("synthesis_evaluations") or {}).get("count"),
            f"{location}.synthesis_evaluations.count",
        ),
        "wall_time_seconds": float(search_elapsed),
        "candidates_evaluated": len(events),
        "candidate_events": events,
    }
    if failure == "other" and skill_integrity_failed:
        record["failure_detail"] = "skill_snapshot_integrity_failure"
    try:
        return _validate_record(record, location, generated=True)
    except ResultManifestError as exc:
        raise FreezeNormalizationError(
            "normalized_record_invalid", f"{location}: {exc}"
        ) from exc


def _normalize_frontier_record(
    root: Mapping[str, Any],
    entry_raw: Any,
    spec: Mapping[str, Any],
    resource_capacities: Mapping[str, int],
    location: str,
) -> dict[str, Any]:
    entry = _required_mapping(entry_raw, location)
    if spec.get("source_kind") != "reference_workflow_entry":
        raise FreezeNormalizationError(
            "frontier_source_kind_invalid",
            f"{location}.source_kind must be reference_workflow_entry",
        )
    run_id = _required_string(spec.get("run_id"), f"{location}.run_id")
    synthesis = _summary_status(entry.get("synthesis"), synthesis=True)
    correctness = _summary_status(entry.get("csim"))
    cosim_status = _summary_status(entry.get("cosim"))
    cycles = _executed_cycles(entry.get("cosim"), f"{location}.cosim")
    feasibility = (
        entry.get("feasibility") if isinstance(entry.get("feasibility"), dict) else {}
    )
    resource_fit = (
        feasibility.get("resource_fit")
        if isinstance(feasibility.get("resource_fit"), bool)
        else None
    )
    timing_met = (
        feasibility.get("timing_met")
        if isinstance(feasibility.get("timing_met"), bool)
        else None
    )
    synthesis_metrics = _normalize_synthesis_metrics(
        entry.get("report"),
        resource_capacities,
        f"{location}.report",
        required=synthesis == PASS,
    )
    solved = bool(
        entry.get("benchmark_ready") is True
        and correctness == PASS
        and synthesis == PASS
        and resource_fit is True
        and timing_met is True
        and cosim_status == PASS
        and cycles is not None
    )
    failure = (
        None
        if solved
        else _failure_class(
            correctness=correctness,
            synthesis=synthesis,
            resource_fit=resource_fit,
            timing_met=timing_met,
            cosim=cosim_status,
        )
    )
    if failure is None and not solved:
        failure = "invalid_reference"
    record = {
        "run_id": run_id,
        "terminal_status": "success" if solved else "failure",
        "correctness_status": correctness,
        "synthesis_status": synthesis,
        "resource_fit": resource_fit,
        "timing_met": timing_met,
        "cosim_status": cosim_status,
        "cycle_source": "executed_rtl_cosim" if cycles is not None else "none",
        "executed_cosim_cycles": cycles,
        "failure_class": failure,
        "synthesis_metrics": synthesis_metrics,
    }
    try:
        return _validate_record(record, location, generated=False)
    except ResultManifestError as exc:
        raise FreezeNormalizationError(
            "normalized_frontier_invalid", f"{location}: {exc}"
        ) from exc


def _authenticate_frontier_role(
    root: Mapping[str, Any], entry: Mapping[str, Any], role: str, location: str
) -> None:
    validation = _required_mapping(
        root.get("reference_validation"), f"{location}.root.reference_validation"
    )
    workflow = _required_list(
        validation.get("workflow"), f"{location}.root.reference_validation.workflow"
    )
    if not any(candidate is entry for candidate in workflow):
        raise FreezeNormalizationError(
            "frontier_pointer_not_workflow_entry",
            f"{location} pointer does not select an exact workflow entry",
        )
    if (
        validation.get("validation_scope") != "all"
        or validation.get("reference_source") != "local_vitis"
        or validation.get("skipped_candidates") not in ([], None)
        or validation.get("frontier_synthesis_csim_valid") is not True
        or validation.get("rtl_measurement_pair_valid") is not True
        or validation.get("benchmark_ready") is not True
    ):
        raise FreezeNormalizationError(
            "frontier_validation_contract_mismatch",
            f"{location} does not come from a complete local-Vitis frontier and RTL-pair preflight",
        )
    contract_audit = _required_mapping(
        entry.get("public_contract_audit"), f"{location}.public_contract_audit"
    )
    if (
        entry.get("reference_contract_status") != "passed"
        or contract_audit.get("passed") is not True
        or contract_audit.get("differences") not in ([], None)
    ):
        raise FreezeNormalizationError(
            "frontier_public_contract_failed",
            f"{location} reference variant did not pass the public interface contract audit",
        )
    if role == "baseline":
        baseline = _required_mapping(
            validation.get("baseline_reference"),
            f"{location}.root.reference_validation.baseline_reference",
        )
        if (
            entry.get("step_name") != "baseline"
            or validation.get("baseline_reference_cosim_measurement_valid") is not True
            or baseline.get("step_name") != "baseline"
            or baseline.get("file") != entry.get("file")
            or baseline.get("variant_name") != entry.get("variant_name")
            or baseline.get("synthesis") != entry.get("synthesis")
            or baseline.get("csim") != entry.get("csim")
            or baseline.get("cosim") != entry.get("cosim")
        ):
            raise FreezeNormalizationError(
                "baseline_frontier_identity_mismatch",
                f"{location} is not the designated measured plain-C baseline",
            )
        return

    if role != "expert":
        raise FreezeNormalizationError(
            "frontier_role_invalid", f"{location} has unsupported role {role!r}"
        )
    if (
        entry.get("selected") is not True
        or validation.get("selected_reference_cosim_measurement_valid") is not True
        or entry.get("file") != validation.get("selected_variant_file")
        or entry.get("variant_name") != validation.get("selected_variant_name")
        or entry.get("step_name") != validation.get("selected_variant_step")
    ):
        raise FreezeNormalizationError(
            "expert_frontier_identity_mismatch",
            f"{location} is not the selected measured expert frontier",
        )
    feasible: list[tuple[int, Mapping[str, Any]]] = []
    for candidate_index, candidate_raw in enumerate(workflow):
        if not isinstance(candidate_raw, dict):
            raise FreezeNormalizationError(
                "frontier_workflow_invalid",
                f"{location} workflow entry {candidate_index} is not an object",
            )
        if (
            candidate_raw.get("benchmark_ready") is not True
            or (candidate_raw.get("feasibility") or {}).get("feasible") is not True
        ):
            continue
        latency = _latency_from_report(
            candidate_raw.get("report"),
            f"{location}.workflow[{candidate_index}].report",
            required=True,
        )
        feasible.append((int(latency), candidate_raw))
    if not feasible:
        raise FreezeNormalizationError(
            "expert_frontier_empty", f"{location} workflow has no feasible reference"
        )
    expert_latency = _latency_from_report(
        entry.get("report"), f"{location}.report", required=True
    )
    if expert_latency != min(latency for latency, _ in feasible):
        raise FreezeNormalizationError(
            "expert_frontier_not_fastest",
            f"{location} selected expert is not minimum-latency correct and feasible",
        )


def normalize_freeze_index(index_path: Path) -> dict[str, Any]:
    index_path = index_path.resolve()
    index, index_digest = _load_json(index_path)
    if index.get("schema_version") != INDEX_SCHEMA:
        raise FreezeNormalizationError(
            "index_schema_unsupported",
            f"schema_version must be {INDEX_SCHEMA!r}",
        )
    target = _required_mapping(index.get("target"), "target")
    for field in ("vitis_version", "part"):
        _required_string(target.get(field), f"target.{field}")
    clock_value = target.get("clock_ns")
    if isinstance(clock_value, bool) or not isinstance(clock_value, (int, float, str)):
        raise FreezeNormalizationError(
            "target_clock_invalid", "target.clock_ns must be a positive finite number"
        )
    try:
        clock_number = float(clock_value)
    except ValueError as exc:
        raise FreezeNormalizationError(
            "target_clock_invalid", "target.clock_ns must be a positive finite number"
        ) from exc
    if not math.isfinite(clock_number) or clock_number <= 0:
        raise FreezeNormalizationError(
            "target_clock_invalid", "target.clock_ns must be a positive finite number"
        )
    resource_capacities, resource_capacity_source = _resource_capacities_for_target(
        target
    )

    cohort = _required_mapping(index.get("cohort"), "cohort")
    methods_raw = _required_list(index.get("methods"), "methods")
    methods: list[dict[str, str]] = []
    method_specs: dict[str, dict[str, Any]] = {}
    for method_index, raw in enumerate(methods_raw):
        item = _required_mapping(raw, f"methods[{method_index}]")
        method_id = _required_string(item.get("id"), f"methods[{method_index}].id")
        runner = _required_string(item.get("runner"), f"methods[{method_index}].runner")
        if runner not in {"run_paper_baseline.py", "run_agentic_sweep.py"}:
            raise FreezeNormalizationError(
                "unknown_runner", f"methods[{method_index}].runner is unsupported"
            )
        runner_method = _required_string(
            item.get("runner_method"), f"methods[{method_index}].runner_method"
        )
        model = _required_mapping(item.get("model"), f"methods[{method_index}].model")
        model_contract = {
            "id": _required_string(
                model.get("id"), f"methods[{method_index}].model.id"
            ),
            "revision": _required_string(
                model.get("revision"),
                f"methods[{method_index}].model.revision",
            ),
        }
        display_name = _required_string(
            item.get("display_name"), f"methods[{method_index}].display_name"
        )
        methods.append({"id": method_id, "display_name": display_name})
        method_specs[method_id] = {
            "id": method_id,
            "display_name": display_name,
            "runner": runner,
            "runner_method": runner_method,
            "model": model_contract,
        }
    method_ids = [item["id"] for item in methods]
    if not method_ids or len(set(method_ids)) != len(method_ids):
        raise FreezeNormalizationError(
            "method_ids_invalid", "method IDs must be non-empty and unique"
        )

    kernels = [
        _required_string(value, f"expected_kernels[{index_value}]")
        for index_value, value in enumerate(
            _required_list(index.get("expected_kernels"), "expected_kernels")
        )
    ]
    if not kernels or len(set(kernels)) != len(kernels):
        raise FreezeNormalizationError(
            "kernel_ids_invalid", "expected kernels must be non-empty and unique"
        )
    cells: list[tuple[str, str, int | str, str]] = []
    for cell_index, raw in enumerate(
        _required_list(index.get("expected_cells"), "expected_cells")
    ):
        item = _required_mapping(raw, f"expected_cells[{cell_index}]")
        kernel = _required_string(
            item.get("kernel"), f"expected_cells[{cell_index}].kernel"
        )
        seed_value = item.get("seed")
        if isinstance(seed_value, bool) or not isinstance(seed_value, (int, str)):
            raise FreezeNormalizationError(
                "seed_invalid", f"expected_cells[{cell_index}].seed is invalid"
            )
        method_id = _required_string(
            item.get("method"), f"expected_cells[{cell_index}].method"
        )
        cells.append((kernel, str(seed_value), seed_value, method_id))
    cell_keys = [(kernel, seed, method_id) for kernel, seed, _, method_id in cells]
    if (
        not cells
        or len(set(cell_keys)) != len(cell_keys)
        or any(kernel not in kernels for kernel, _, _, _ in cells)
        or any(method_id not in method_ids for _, _, _, method_id in cells)
    ):
        raise FreezeNormalizationError(
            "expected_cells_invalid",
            "expected cells must be unique and use expected kernels and methods",
        )
    units: list[tuple[str, str, int | str]] = []
    for kernel, seed, output_seed, _ in cells:
        if (kernel, seed) not in {(item[0], item[1]) for item in units}:
            units.append((kernel, seed, output_seed))

    rows_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row_index, raw in enumerate(
        _required_list(index.get("generated_rows"), "generated_rows")
    ):
        row = _required_mapping(raw, f"generated_rows[{row_index}]")
        seed_value = row.get("seed")
        key = (str(row.get("kernel")), str(seed_value), str(row.get("method")))
        if key in rows_by_key:
            raise FreezeNormalizationError(
                "duplicate_row", f"duplicate generated row {key}"
            )
        rows_by_key[key] = row
    expected_keys = cell_keys
    if set(rows_by_key) != set(expected_keys):
        missing = [key for key in expected_keys if key not in rows_by_key]
        extra = [key for key in rows_by_key if key not in expected_keys]
        raise FreezeNormalizationError(
            "row_coverage_mismatch",
            f"generated row coverage mismatch; missing={missing}, extra={extra}",
        )

    source_provenance: list[dict[str, Any]] = []
    normalized_units: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for kernel, seed, output_seed in units:
        normalized_results: dict[str, Any] = {}
        unit_method_ids = [
            method_id
            for method_id in method_ids
            if (kernel, seed, method_id) in set(cell_keys)
        ]
        for method_id in unit_method_ids:
            key = (kernel, seed, method_id)
            row = rows_by_key[key]
            location = f"generated_rows[{kernel},{seed},{method_id}]"
            root, selected, digest, pointer = _load_attested_source(
                index_path, row, location
            )
            if selected is not root:
                raise FreezeNormalizationError(
                    "generated_row_pointer_must_be_root",
                    f"{location}.json_pointer must be empty for runner result artifacts",
                )
            _verify_identity(
                root,
                kernel=kernel,
                seed=seed,
                target=target,
                method_spec=method_specs[method_id],
                cohort=cohort,
                location=location,
            )
            if str(root.get("benchmark")) != kernel:
                raise FreezeNormalizationError(
                    "benchmark_identity_mismatch",
                    f"{location}.benchmark does not match {kernel}",
                )
            audit_passed, isolation_sources = _validate_isolation_evidence(
                index_path, row, root, location
            )
            record = _normalize_generated_record(
                root,
                row,
                method_specs[method_id],
                resource_capacities,
                location,
                audit_passed=audit_passed,
            )
            if record["run_id"] in seen_run_ids:
                raise FreezeNormalizationError(
                    "duplicate_run_id", f"duplicate run ID {record['run_id']}"
                )
            seen_run_ids.add(record["run_id"])
            normalized_results[method_id] = record
            source_provenance.append(
                {
                    "run_id": record["run_id"],
                    "role": "generated",
                    "adapter": row["runner"],
                    "source_sha256": digest,
                    "json_pointer": pointer,
                }
            )
            for isolation_source in isolation_sources:
                source_provenance.append(
                    {
                        "run_id": record["run_id"],
                        "adapter": row["runner"],
                        **isolation_source,
                    }
                )
        normalized_units.append(
            {"kernel": kernel, "seed": output_seed, "results": normalized_results}
        )

    frontier_specs = _required_list(index.get("frontiers"), "frontiers")
    if len(frontier_specs) != len(kernels):
        raise FreezeNormalizationError(
            "frontier_coverage_mismatch", "frontiers must contain one row per kernel"
        )
    frontiers_by_kernel: dict[str, Mapping[str, Any]] = {}
    for item_index, raw in enumerate(frontier_specs):
        item = _required_mapping(raw, f"frontiers[{item_index}]")
        kernel = str(item.get("kernel"))
        if kernel in frontiers_by_kernel:
            raise FreezeNormalizationError(
                "duplicate_frontier", f"duplicate frontier row for {kernel}"
            )
        frontiers_by_kernel[kernel] = item
    if set(frontiers_by_kernel) != set(kernels):
        raise FreezeNormalizationError(
            "frontier_coverage_mismatch",
            "frontier kernels do not match expected kernels",
        )

    normalized_frontiers: list[dict[str, Any]] = []
    for kernel in kernels:
        item = frontiers_by_kernel[kernel]
        normalized_roles: dict[str, Any] = {"kernel": kernel}
        for role in ("baseline", "expert"):
            spec = _required_mapping(item.get(role), f"frontiers[{kernel}].{role}")
            location = f"frontiers[{kernel}].{role}"
            root, selected, digest, pointer = _load_attested_source(
                index_path, spec, location
            )
            # Reference workflow entries are produced inside a fingerprinted
            # runner result. They share target identity but are seed-independent.
            fingerprint = _fingerprint(root, location)
            _validate_cohort(fingerprint, cohort, location)
            payload = _required_mapping(
                fingerprint.get("payload"), f"{location}.fingerprint.payload"
            )
            toolchain = _required_mapping(
                payload.get("toolchain"), f"{location}.fingerprint.payload.toolchain"
            )
            if str((payload.get("benchmark") or {}).get("name")) != kernel:
                raise FreezeNormalizationError(
                    "benchmark_identity_mismatch", f"{location} benchmark mismatch"
                )
            try:
                source_clock = float(toolchain.get("clock_ns"))
                target_clock = float(target.get("clock_ns"))
            except (TypeError, ValueError) as exc:
                raise FreezeNormalizationError(
                    "target_identity_missing", f"{location} lacks numeric clock_ns"
                ) from exc
            if (
                str(toolchain.get("vitis_version")) != str(target.get("vitis_version"))
                or str(toolchain.get("part")) != str(target.get("part"))
                or source_clock != target_clock
            ):
                raise FreezeNormalizationError(
                    "target_identity_mismatch", f"{location} target mismatch"
                )
            selected_entry = _required_mapping(selected, location)
            _authenticate_frontier_role(root, selected_entry, role, location)
            record = _normalize_frontier_record(
                root, selected, spec, resource_capacities, location
            )
            if record["run_id"] in seen_run_ids:
                raise FreezeNormalizationError(
                    "duplicate_run_id", f"duplicate run ID {record['run_id']}"
                )
            seen_run_ids.add(record["run_id"])
            normalized_roles[role] = record
            source_provenance.append(
                {
                    "run_id": record["run_id"],
                    "role": role,
                    "adapter": "reference_workflow_entry",
                    "source_sha256": digest,
                    "json_pointer": pointer,
                }
            )
        normalized_frontiers.append(normalized_roles)

    return {
        "schema_version": SCHEMA_VERSION,
        "methods": methods,
        "expected_cells": [
            {"kernel": kernel, "seed": output_seed, "method": method_id}
            for kernel, _, output_seed, method_id in cells
        ],
        "baseline_expert": normalized_frontiers,
        "evaluation_units": normalized_units,
        "normalization_provenance": {
            "schema_version": NORMALIZER_SCHEMA,
            "freeze_index_sha256": index_digest,
            "target": dict(target),
            "device_resource_capacities": dict(resource_capacities),
            "resource_capacity_source": resource_capacity_source,
            "cohort": dict(cohort),
            "method_contracts": [method_specs[method_id] for method_id in method_ids],
            "sources": source_provenance,
        },
    }


def normalize_to_file(index_path: Path, output_path: Path) -> Path:
    normalized = normalize_freeze_index(index_path)
    encoded = (json.dumps(normalized, indent=2, sort_keys=False) + "\n").encode("utf-8")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.tmp.", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output_path)
        except FileExistsError:
            if output_path.read_bytes() == encoded:
                return output_path
            raise FreezeNormalizationError(
                "output_exists",
                f"refusing to overwrite non-identical normalized manifest {output_path}",
            )
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = normalize_to_file(args.freeze_index, args.output)
    except FreezeNormalizationError as exc:
        print(json.dumps(exc.as_dict(), indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

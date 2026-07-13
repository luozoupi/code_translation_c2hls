#!/usr/bin/env python3
"""Generate HPCA paper artifacts from a frozen, explicitly attested run set.

The generator intentionally has no knowledge of experiment output directories.
It accepts two small JSON manifests: a result manifest containing normalized run
records, and an evidence manifest that pins the result file and auxiliary audit
artifacts by SHA-256.  Missing records are errors, not implicit failures, and a
cycle count is publishable only when its source is executed RTL co-simulation.

The output is an immutable directory named by the frozen run-set hash.  This
keeps a paper build from observing a mixture of files from different run sets.
No output is written unless both manifests validate completely.
"""

from __future__ import annotations

import argparse
import csv
import datetime as datetime_module
import hashlib
import html
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 2
GENERATOR_VERSION = "2.1"
PASS = "passed"
RESOURCE_KEYS = ("bram", "dsp", "ff", "lut", "uram")
U280_RESOURCE_CAPACITIES = {
    "bram": 4032,
    "dsp": 9024,
    "ff": 2_607_360,
    "lut": 1_303_680,
    "uram": 960,
}
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
FAILURE_CLASSES = {
    "malformed_output",
    "wrong_output",
    "compile_or_interface_failure",
    "synthesis_timeout",
    "tool_failure",
    "infeasible_resources",
    "timing_failure",
    "cosim_failure",
    "cosim_timeout",
    "reference_isolation_failure",
    "missing_executed_cosim",
    "invalid_reference",
    "candidate_budget_exhausted",
    "other",
}
PAPER_PROFILE_TAUS = [1.0, 1.25, 2.0, 4.0, 10.0]
PAPER_PROFILE_TAU_MAX = 10.0
PAPER_BOOTSTRAP = {
    "confidence": 0.95,
    "replicates": 10_000,
    "seed": 2027,
}


class ManifestError(ValueError):
    """The supplied evidence is incomplete, inconsistent, or unauthenticated."""


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ManifestError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"manifest {path} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ManifestError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _require_sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ManifestError(f"{where} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ManifestError(f"{where} is not hexadecimal") from exc
    return value.lower()


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{where} must be an object")
    return value


def _require_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManifestError(f"{where} must be an array")
    return value


def _require_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{where} must be a non-empty string")
    return value


def _require_bool(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise ManifestError(f"{where} must be a boolean")
    return value


def _require_nonnegative_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{where} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ManifestError(f"{where} must be a non-negative finite number")
    return result


def _require_positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{where} must be a positive integer")
    return value


def _record_is_solved(record: Mapping[str, Any]) -> bool:
    cycles = record.get("executed_cosim_cycles")
    return (
        record.get("terminal_status") == "success"
        and record.get("correctness_status") == PASS
        and record.get("synthesis_status") == PASS
        and record.get("resource_fit") is True
        and record.get("timing_met") is True
        and record.get("cosim_status") == PASS
        and record.get("cycle_source") == "executed_rtl_cosim"
        and isinstance(cycles, int)
        and not isinstance(cycles, bool)
        and cycles > 0
    )


def _candidate_is_feasible(event: Mapping[str, Any]) -> bool:
    latency = event.get("synthesized_latency_cycles")
    return (
        event.get("correctness_status") == PASS
        and event.get("synthesis_status") == PASS
        and event.get("resource_fit") is True
        and event.get("timing_met") is True
        and event.get("latency_source") == "vitis_csynth_report"
        and isinstance(latency, int)
        and not isinstance(latency, bool)
        and latency > 0
    )


def _validate_synthesis_metrics(
    value: Any,
    where: str,
    *,
    required: bool,
    expected_capacities: Mapping[str, int] | None = None,
) -> dict[str, Any] | None:
    if not required:
        if value is not None:
            raise ManifestError(
                f"{where} must be null when synthesis did not pass"
            )
        return None
    metrics = _require_mapping(value, where)
    if metrics.get("source") != "vitis_csynth_report":
        raise ManifestError(f"{where}.source must be vitis_csynth_report")
    _require_sha256(metrics.get("report_sha256"), f"{where}.report_sha256")
    fmax = metrics.get("fmax_mhz")
    if isinstance(fmax, bool) or not isinstance(fmax, (int, float)):
        raise ManifestError(f"{where}.fmax_mhz must be a positive finite number")
    fmax_number = float(fmax)
    if not math.isfinite(fmax_number) or fmax_number <= 0:
        raise ManifestError(f"{where}.fmax_mhz must be a positive finite number")
    resources = _require_mapping(metrics.get("resources"), f"{where}.resources")
    if set(resources) != set(RESOURCE_KEYS):
        raise ManifestError(
            f"{where}.resources must contain exactly {list(RESOURCE_KEYS)}"
        )
    for key in RESOURCE_KEYS:
        resource = _require_mapping(resources[key], f"{where}.resources.{key}")
        used = resource.get("used")
        capacity = resource.get("capacity")
        utilization = resource.get("utilization")
        if (
            isinstance(used, bool)
            or not isinstance(used, int)
            or used < 0
        ):
            raise ManifestError(
                f"{where}.resources.{key}.used must be a non-negative integer"
            )
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or capacity <= 0
        ):
            raise ManifestError(
                f"{where}.resources.{key}.capacity must be a positive integer"
            )
        if expected_capacities is not None and capacity != expected_capacities[key]:
            raise ManifestError(
                f"{where}.resources.{key}.capacity disagrees with the normalized target capacity"
            )
        if (
            isinstance(utilization, bool)
            or not isinstance(utilization, (int, float))
            or not math.isfinite(float(utilization))
            or float(utilization) < 0
            or not math.isclose(
                float(utilization), used / capacity, rel_tol=1e-12, abs_tol=1e-15
            )
        ):
            raise ManifestError(
                f"{where}.resources.{key}.utilization must equal used/capacity"
            )
    return metrics


def _validate_candidate_events(record: dict[str, Any], where: str) -> None:
    events = _require_list(record.get("candidate_events"), f"{where}.candidate_events")
    expected_count = record["candidates_evaluated"]
    if expected_count <= 0 or len(events) != expected_count:
        raise ManifestError(
            f"{where}.candidate_events must contain exactly one event per evaluated candidate"
        )

    previous_tokens = 0
    previous_llm_calls = 0
    previous_syntheses = 0
    previous_elapsed = 0.0
    selected_count = 0
    selected_latency: int | None = None
    selected_code_sha256: str | None = None
    selected_report_sha256: str | None = None
    feasible_latencies: list[int] = []
    seen_event_ids: set[str] = set()
    for index, raw_event in enumerate(events, start=1):
        event_where = f"{where}.candidate_events[{index - 1}]"
        event = _require_mapping(raw_event, event_where)
        forbidden_fields = [
            field
            for field in event
            if "predicted" in field.lower()
            or field.lower().startswith("estimated_latency")
            or field.lower().startswith("gold_relative")
        ]
        if forbidden_fields:
            raise ManifestError(
                f"{event_where} contains forbidden predicted/oracle fields: {forbidden_fields}"
            )
        event_id = _require_string(event.get("event_id"), f"{event_where}.event_id")
        if RUN_ID_RE.fullmatch(event_id) is None or event_id in seen_event_ids:
            raise ManifestError(f"{event_where}.event_id must be a unique opaque identifier")
        seen_event_ids.add(event_id)
        if event.get("candidate_index") != index:
            raise ManifestError(f"{event_where}.candidate_index must be the contiguous value {index}")
        code_sha256 = _require_sha256(
            event.get("code_sha256"), f"{event_where}.code_sha256"
        )
        report_sha256_raw = event.get("report_sha256")
        report_sha256 = (
            None
            if report_sha256_raw is None
            else _require_sha256(report_sha256_raw, f"{event_where}.report_sha256")
        )

        counters: dict[str, int] = {}
        for field in (
            "cumulative_tokens",
            "cumulative_llm_calls",
            "cumulative_synthesis_evaluations",
        ):
            value = event.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ManifestError(f"{event_where}.{field} must be a non-negative integer")
            counters[field] = value
        elapsed = _require_nonnegative_number(
            event.get("cumulative_elapsed_seconds"),
            f"{event_where}.cumulative_elapsed_seconds",
        )
        if counters["cumulative_tokens"] < previous_tokens:
            raise ManifestError(f"{event_where}.cumulative_tokens must be monotonic")
        if counters["cumulative_llm_calls"] < previous_llm_calls:
            raise ManifestError(f"{event_where}.cumulative_llm_calls must be monotonic")
        synth_delta = counters["cumulative_synthesis_evaluations"] - previous_syntheses
        if synth_delta not in (0, 1):
            raise ManifestError(
                f"{event_where} must account for zero or one new selection synthesis"
            )
        if elapsed < previous_elapsed:
            raise ManifestError(f"{event_where}.cumulative_elapsed_seconds must be monotonic")

        for field, domain in {
            "correctness_status": {PASS, "failed", "not_run", "tool_failure", "timeout"},
            "synthesis_status": {PASS, "failed", "not_run", "tool_failure", "timeout"},
        }.items():
            if event.get(field) not in domain:
                raise ManifestError(f"{event_where}.{field} has unsupported value")
        for field in ("resource_fit", "timing_met"):
            if event.get(field) is not None and type(event.get(field)) is not bool:
                raise ManifestError(f"{event_where}.{field} must be boolean or null")

        latency = event.get("synthesized_latency_cycles")
        latency_source = event.get("latency_source")
        if latency is not None and (
            not isinstance(latency, int) or isinstance(latency, bool) or latency <= 0
        ):
            raise ManifestError(
                f"{event_where}.synthesized_latency_cycles must be a positive integer or null"
            )
        if latency_source not in {"vitis_csynth_report", "none"}:
            raise ManifestError(
                f"{event_where}.latency_source must be vitis_csynth_report or none; predicted data is forbidden"
            )
        if (latency is None) != (latency_source == "none"):
            raise ManifestError(f"{event_where} latency value/source are inconsistent")
        if latency is not None and event.get("synthesis_status") != PASS:
            raise ManifestError(f"{event_where} reports synthesis latency without passing synthesis")
        if event.get("synthesis_status") == PASS and report_sha256 is None:
            raise ManifestError(
                f"{event_where}.report_sha256 is required after passing synthesis"
            )
        if synth_delta == 0 and event.get("synthesis_status") != "not_run":
            raise ManifestError(f"{event_where} has synthesis evidence without a budget event")
        if synth_delta == 1 and event.get("synthesis_status") == "not_run":
            raise ManifestError(f"{event_where} consumes synthesis budget but marks synthesis not_run")
        if event.get("correctness_status") == PASS and synth_delta != 1:
            raise ManifestError(f"{event_where} passed CSim but was not synthesis-evaluated")
        if event.get("correctness_status") != PASS and (
            synth_delta != 0 or event.get("synthesis_status") != "not_run"
        ):
            raise ManifestError(
                f"{event_where} bypasses the correctness gate before synthesis"
            )

        feasible = _candidate_is_feasible(event)
        failure_class = event.get("failure_class")
        if feasible:
            feasible_latencies.append(int(event["synthesized_latency_cycles"]))
            if failure_class not in (None, "none"):
                raise ManifestError(f"{event_where} is feasible but has a failure class")
        else:
            if failure_class not in FAILURE_CLASSES:
                raise ManifestError(f"{event_where} must explicitly classify its failed outcome")
            if failure_class == "other":
                _require_string(event.get("failure_detail"), f"{event_where}.failure_detail")

        selected = _require_bool(
            event.get("selected_for_executed_cosim"),
            f"{event_where}.selected_for_executed_cosim",
        )
        selected_count += int(selected)
        if selected and not feasible:
            raise ManifestError(f"{event_where} selects an infeasible candidate for RTL co-simulation")
        if selected:
            selected_latency = int(event["synthesized_latency_cycles"])
            selected_code_sha256 = code_sha256
            selected_report_sha256 = report_sha256

        previous_tokens = counters["cumulative_tokens"]
        previous_llm_calls = counters["cumulative_llm_calls"]
        previous_syntheses = counters["cumulative_synthesis_evaluations"]
        previous_elapsed = elapsed

    if previous_tokens != record["tokens"]:
        raise ManifestError(f"{where} token total disagrees with its final candidate event")
    if previous_llm_calls != record["llm_calls"]:
        raise ManifestError(f"{where} LLM-call total disagrees with its final candidate event")
    if previous_syntheses != record["selection_synthesis_evaluations"]:
        raise ManifestError(f"{where} synthesis-evaluation total disagrees with its candidate events")
    if record["synthesis_calls"] < record["selection_synthesis_evaluations"]:
        raise ManifestError(f"{where} total synthesis calls cannot be below selection evaluations")
    if previous_elapsed > record["wall_time_seconds"] + 1e-9:
        raise ManifestError(f"{where} candidate elapsed time exceeds total wall time")
    if selected_count > 1:
        raise ManifestError(f"{where} selects more than one candidate for executed RTL co-simulation")
    if _record_is_solved(record) and selected_count != 1:
        raise ManifestError(f"{where} successful run must identify its one co-simulated winner")
    root_selected_hash = record.get("selected_code_sha256")
    root_cosim_hash = record.get("cosim_target_code_sha256")
    if selected_count == 0:
        if root_selected_hash is not None or root_cosim_hash is not None:
            raise ManifestError(f"{where} exposes winner hashes without a selected candidate")
    else:
        if (
            _require_sha256(root_selected_hash, f"{where}.selected_code_sha256")
            != selected_code_sha256
            or _require_sha256(root_cosim_hash, f"{where}.cosim_target_code_sha256")
            != selected_code_sha256
        ):
            raise ManifestError(
                f"{where} selected candidate and cosim target hashes disagree"
            )
        metrics = _require_mapping(record.get("synthesis_metrics"), f"{where}.synthesis_metrics")
        if metrics.get("report_sha256") != selected_report_sha256:
            raise ManifestError(
                f"{where} selected candidate report hash disagrees with final synthesis metrics"
            )
    if selected_latency is not None and selected_latency != min(feasible_latencies):
        raise ManifestError(
            f"{where} selected winner is not the minimum-latency feasible synthesized candidate"
        )


def _validate_record(
    record: Any,
    where: str,
    *,
    generated: bool,
    expected_capacities: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    rec = _require_mapping(record, where)
    run_id = _require_string(rec.get("run_id"), f"{where}.run_id")
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ManifestError(
            f"{where}.run_id must be an opaque identifier, not a path or free-form label"
        )
    if rec.get("terminal_status") not in {"success", "failure"}:
        raise ManifestError(f"{where}.terminal_status must be success or failure")
    status_domains = {
        "correctness_status": {PASS, "failed", "not_run", "tool_failure", "timeout"},
        "synthesis_status": {PASS, "failed", "not_run", "tool_failure", "timeout"},
        "cosim_status": {PASS, "failed", "not_run", "tool_failure", "timeout"},
    }
    for field, domain in status_domains.items():
        if rec.get(field) not in domain:
            raise ManifestError(f"{where}.{field} has unsupported value {rec.get(field)!r}")
    for field in ("resource_fit", "timing_met"):
        if rec.get(field) is not None and type(rec.get(field)) is not bool:
            raise ManifestError(f"{where}.{field} must be boolean or null")
    _validate_synthesis_metrics(
        rec.get("synthesis_metrics"),
        f"{where}.synthesis_metrics",
        required=rec.get("synthesis_status") == PASS,
        expected_capacities=expected_capacities,
    )
    if rec.get("cycle_source") not in {
        "executed_rtl_cosim",
        "predicted",
        "estimated",
        "none",
    }:
        raise ManifestError(f"{where}.cycle_source has unsupported value")

    cycles = rec.get("executed_cosim_cycles")
    if cycles is not None and (
        isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0
    ):
        raise ManifestError(f"{where}.executed_cosim_cycles must be a positive integer or null")
    if cycles is not None and (
        rec.get("cycle_source") != "executed_rtl_cosim"
        or rec.get("cosim_status") != PASS
    ):
        raise ManifestError(
            f"{where} reports cycles without a passing executed RTL co-simulation"
        )
    if rec.get("cycle_source") == "executed_rtl_cosim" and cycles is None:
        raise ManifestError(f"{where} labels cycles as executed but supplies no cycle count")

    solved = _record_is_solved(rec)
    if (rec.get("terminal_status") == "success") != solved:
        raise ManifestError(
            f"{where}.terminal_status disagrees with correctness/feasibility/cosim evidence"
        )
    failure_class = rec.get("failure_class")
    if solved:
        if failure_class not in (None, "none"):
            raise ManifestError(f"{where} is successful but has failure_class={failure_class!r}")
    else:
        if failure_class not in FAILURE_CLASSES:
            raise ManifestError(
                f"{where}.failure_class must explicitly classify the failed outcome"
            )
        if failure_class == "other":
            _require_string(rec.get("failure_detail"), f"{where}.failure_detail")

    if generated:
        _require_bool(rec.get("provider_failure"), f"{where}.provider_failure")
        isolation_status = rec.get("reference_isolation_status")
        if isolation_status not in {PASS, "failed"}:
            raise ManifestError(
                f"{where}.reference_isolation_status must be passed or failed"
            )
        if isolation_status == "failed" and (
            rec.get("terminal_status") != "failure"
            or rec.get("failure_class") != "reference_isolation_failure"
            or rec.get("executed_cosim_cycles") is not None
        ):
            raise ManifestError(
                f"{where} with failed reference isolation must be a non-measured reference_isolation_failure"
            )
        for field in (
            "tokens",
            "llm_calls",
            "synthesis_calls",
            "selection_synthesis_evaluations",
            "wall_time_seconds",
            "candidates_evaluated",
        ):
            _require_nonnegative_number(rec.get(field), f"{where}.{field}")
        for field in (
            "tokens", "llm_calls", "synthesis_calls", "selection_synthesis_evaluations",
            "candidates_evaluated",
        ):
            if not isinstance(rec.get(field), int) or isinstance(rec.get(field), bool):
                raise ManifestError(f"{where}.{field} must be an integer")
        _validate_candidate_events(rec, where)
    return rec


@dataclass(frozen=True)
class Method:
    method_id: str
    display_name: str


@dataclass
class ValidatedInput:
    evidence_path: Path
    result_path: Path
    evidence: dict[str, Any]
    results: dict[str, Any]
    run_set_hash: str
    evidence_hash: str
    methods: list[Method]
    expected_kernels: list[str]
    expected_units: list[tuple[str, str]]
    expected_cells: list[tuple[str, str, str]]
    headline_units: dict[str, tuple[str, str]]
    profile_units: list[tuple[str, str]]
    bootstrap_units: list[tuple[str, str]]
    frontiers: dict[str, dict[str, dict[str, Any]]]
    evaluations: dict[tuple[str, str], dict[str, dict[str, Any]]]
    artifacts: dict[str, dict[str, Any]]


def _unit_key(value: Mapping[str, Any], where: str) -> tuple[str, str]:
    kernel = _require_string(value.get("kernel"), f"{where}.kernel")
    if (
        "seed" not in value
        or isinstance(value["seed"], bool)
        or not isinstance(value["seed"], (int, str))
    ):
        raise ManifestError(f"{where}.seed must be an integer or string")
    seed = str(value["seed"])
    if not seed:
        raise ManifestError(f"{where}.seed must not be empty")
    return kernel, seed


def _cell_key(value: Mapping[str, Any], where: str) -> tuple[str, str, str]:
    kernel, seed = _unit_key(value, where)
    method = _require_string(value.get("method"), f"{where}.method")
    return kernel, seed, method


def _resolve_attested_file(base: Path, spec: Mapping[str, Any], where: str) -> Path:
    raw_path = _require_string(spec.get("path"), f"{where}.path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    expected = _require_sha256(spec.get("sha256"), f"{where}.sha256")
    actual = _sha256_file(path)
    if actual != expected:
        raise ManifestError(
            f"{where} hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return path


def load_and_validate(evidence_path: Path) -> ValidatedInput:
    evidence_path = evidence_path.resolve()
    evidence = _load_json(evidence_path)
    if evidence.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("unsupported evidence manifest schema_version")
    if evidence.get("frozen") is not True:
        raise ManifestError("evidence manifest must explicitly set frozen=true")
    freeze_timestamp = _require_string(
        evidence.get("evidence_freeze_timestamp"), "evidence_freeze_timestamp"
    )
    try:
        freeze_datetime = datetime_module.datetime.fromisoformat(
            freeze_timestamp.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ManifestError("evidence_freeze_timestamp must be ISO-8601/RFC-3339") from exc
    if freeze_datetime.tzinfo is None:
        raise ManifestError("evidence_freeze_timestamp must include a timezone")

    run_spec = _require_mapping(evidence.get("run_set"), "run_set")
    result_path = _resolve_attested_file(evidence_path.parent, run_spec, "run_set")
    run_set_hash = _sha256_file(result_path)
    results = _load_json(result_path)
    if results.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("unsupported result manifest schema_version")
    normalization = _require_mapping(
        results.get("normalization_provenance"),
        "results.normalization_provenance",
    )
    if normalization.get("schema_version") != "c2hls.hpca-freeze-normalizer.v1":
        raise ManifestError("result manifest lacks supported freeze-normalizer provenance")
    normalized_target = _require_mapping(
        normalization.get("target"), "results.normalization_provenance.target"
    )
    target_part = _require_string(
        normalized_target.get("part"),
        "results.normalization_provenance.target.part",
    )
    if not target_part.lower().startswith("xcu280"):
        raise ManifestError("paper result manifest target must be the preregistered XCU280")
    normalized_capacities = _require_mapping(
        normalization.get("device_resource_capacities"),
        "results.normalization_provenance.device_resource_capacities",
    )
    if normalized_capacities != U280_RESOURCE_CAPACITIES:
        raise ManifestError(
            "normalized device resource capacities disagree with the XCU280 part table"
        )
    if normalization.get("resource_capacity_source") not in {
        "xcu280_part_table",
        "target.resource_capacities",
    }:
        raise ManifestError("normalized resource capacity source is unsupported")

    expected_kernels_raw = _require_list(
        evidence.get("expected_kernels"), "expected_kernels"
    )
    expected_kernels = [
        _require_string(item, f"expected_kernels[{index}]")
        for index, item in enumerate(expected_kernels_raw)
    ]
    if not expected_kernels or len(set(expected_kernels)) != len(expected_kernels):
        raise ManifestError("expected_kernels must be a non-empty unique list")

    methods_raw = _require_list(results.get("methods"), "results.methods")
    methods: list[Method] = []
    for index, value in enumerate(methods_raw):
        spec = _require_mapping(value, f"results.methods[{index}]")
        methods.append(
            Method(
                _require_string(spec.get("id"), f"results.methods[{index}].id"),
                _require_string(
                    spec.get("display_name"), f"results.methods[{index}].display_name"
                ),
            )
        )
    method_ids = [method.method_id for method in methods]
    if not method_ids or len(set(method_ids)) != len(method_ids):
        raise ManifestError("results.methods IDs must be non-empty and unique")
    expected_methods = _require_list(evidence.get("expected_methods"), "expected_methods")
    if expected_methods != method_ids:
        raise ManifestError(
            "expected_methods must exactly match results.methods IDs and order"
        )

    expected_cells_raw = _require_list(
        evidence.get("expected_cells"), "expected_cells"
    )
    expected_cells = [
        _cell_key(
            _require_mapping(item, f"expected_cells[{index}]"),
            f"expected_cells[{index}]",
        )
        for index, item in enumerate(expected_cells_raw)
    ]
    if not expected_cells or len(set(expected_cells)) != len(expected_cells):
        raise ManifestError("expected_cells must be a non-empty unique list")
    if any(kernel not in expected_kernels for kernel, _, _ in expected_cells):
        raise ManifestError("every expected cell kernel must appear in expected_kernels")
    if any(method not in method_ids for _, _, method in expected_cells):
        raise ManifestError("every expected cell method must appear in expected_methods")
    result_cells_raw = _require_list(
        results.get("expected_cells"), "results.expected_cells"
    )
    result_cells = [
        _cell_key(
            _require_mapping(item, f"results.expected_cells[{index}]"),
            f"results.expected_cells[{index}]",
        )
        for index, item in enumerate(result_cells_raw)
    ]
    if result_cells != expected_cells:
        raise ManifestError(
            "results.expected_cells must exactly match evidence expected_cells and order"
        )
    expected_units: list[tuple[str, str]] = []
    for kernel, seed, _ in expected_cells:
        key = (kernel, seed)
        if key not in expected_units:
            expected_units.append(key)
    headline_units_raw = _require_list(evidence.get("headline_units"), "headline_units")
    headline_unit_list = [
        _unit_key(
            _require_mapping(item, f"headline_units[{index}]"),
            f"headline_units[{index}]",
        )
        for index, item in enumerate(headline_units_raw)
    ]
    if len(headline_unit_list) != len(expected_kernels):
        raise ManifestError("headline_units must select exactly one unit per expected kernel")
    if [kernel for kernel, _ in headline_unit_list] != expected_kernels:
        raise ManifestError("headline_units kernels must exactly match expected_kernels and order")
    if any(key not in expected_units for key in headline_unit_list):
        raise ManifestError("every headline unit must be a preregistered expected unit")
    headline_units = {kernel: key for kernel, key in zip(expected_kernels, headline_unit_list)}

    def selected_units(field: str) -> list[tuple[str, str]]:
        raw = _require_list(evidence.get(field), field)
        selected = [
            _unit_key(_require_mapping(item, f"{field}[{index}]"), f"{field}[{index}]")
            for index, item in enumerate(raw)
        ]
        if not selected or len(set(selected)) != len(selected):
            raise ManifestError(f"{field} must be a non-empty unique list")
        if any(key not in expected_units for key in selected):
            raise ManifestError(f"every {field} entry must be a preregistered expected unit")
        return selected

    profile_units = selected_units("profile_units")
    bootstrap_units = selected_units("bootstrap_units")

    run_ids: set[str] = set()
    frontiers: dict[str, dict[str, dict[str, Any]]] = {}
    frontiers_raw = _require_list(results.get("baseline_expert"), "results.baseline_expert")
    for index, value in enumerate(frontiers_raw):
        spec = _require_mapping(value, f"results.baseline_expert[{index}]")
        kernel = _require_string(spec.get("kernel"), f"results.baseline_expert[{index}].kernel")
        if kernel in frontiers:
            raise ManifestError(f"duplicate baseline/expert entry for {kernel}")
        baseline = _validate_record(
            spec.get("baseline"),
            f"results.baseline_expert[{index}].baseline",
            generated=False,
            expected_capacities=U280_RESOURCE_CAPACITIES,
        )
        expert = _validate_record(
            spec.get("expert"),
            f"results.baseline_expert[{index}].expert",
            generated=False,
            expected_capacities=U280_RESOURCE_CAPACITIES,
        )
        for record in (baseline, expert):
            if record["run_id"] in run_ids:
                raise ManifestError(f"duplicate run_id: {record['run_id']}")
            run_ids.add(record["run_id"])
        frontiers[kernel] = {"baseline": baseline, "expert": expert}
    if list(frontiers) != expected_kernels:
        raise ManifestError(
            "baseline_expert kernels must exactly match expected_kernels and order"
        )

    evaluations: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    units_raw = _require_list(results.get("evaluation_units"), "results.evaluation_units")
    for index, value in enumerate(units_raw):
        spec = _require_mapping(value, f"results.evaluation_units[{index}]")
        key = _unit_key(spec, f"results.evaluation_units[{index}]")
        if key in evaluations:
            raise ManifestError(f"duplicate evaluation unit: {key}")
        records_raw = _require_mapping(
            spec.get("results"), f"results.evaluation_units[{index}].results"
        )
        expected_unit_methods = [
            method_id
            for method_id in method_ids
            if (key[0], key[1], method_id) in set(expected_cells)
        ]
        if list(records_raw) != expected_unit_methods:
            raise ManifestError(
                f"unit {key} must contain exactly its explicitly expected methods in global method order"
            )
        records: dict[str, dict[str, Any]] = {}
        for method_id in expected_unit_methods:
            rec = _validate_record(
                records_raw[method_id],
                f"results.evaluation_units[{index}].results.{method_id}",
                generated=True,
                expected_capacities=U280_RESOURCE_CAPACITIES,
            )
            if rec["run_id"] in run_ids:
                raise ManifestError(f"duplicate run_id: {rec['run_id']}")
            run_ids.add(rec["run_id"])
            records[method_id] = rec
        evaluations[key] = records
    if list(evaluations) != expected_units:
        raise ManifestError(
            "evaluation_units must exactly match the unit order induced by expected_cells; missing runs are not inferred"
        )

    def budget_checkpoints(field: str) -> list[int]:
        raw = _require_list(evidence.get(field), field)
        values: list[int] = []
        for index, value in enumerate(raw):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ManifestError(f"{field}[{index}] must be a positive integer")
            values.append(value)
        if not values or values != sorted(set(values)):
            raise ManifestError(f"{field} must be a non-empty sorted unique list")
        return values

    synthesis_checkpoints = budget_checkpoints("budget_synthesis_checkpoints")
    if synthesis_checkpoints[-1] != 5:
        raise ManifestError("budget_synthesis_checkpoints must end at the preregistered limit of five")
    token_checkpoints = budget_checkpoints("budget_token_checkpoints")
    maximum_tokens = max(
        record["tokens"]
        for records in evaluations.values()
        for record in records.values()
    )
    if token_checkpoints[-1] < maximum_tokens:
        raise ManifestError(
            "budget_token_checkpoints must include a final point covering every complete trace"
        )

    claim_methods = _require_mapping(evidence.get("claim_methods"), "claim_methods")
    for name in ("primary", "one_shot", "dynamic_no_skill", "dynamic_frozen_skill"):
        method_id = _require_string(claim_methods.get(name), f"claim_methods.{name}")
        if method_id not in method_ids:
            raise ManifestError(f"claim_methods.{name} references unknown method {method_id}")
    primary_method = str(claim_methods["primary"])
    if any(primary_method not in evaluations[key] for key in headline_unit_list):
        raise ManifestError("every headline unit must contain the primary claim method")
    if any(set(evaluations[key]) != set(method_ids) for key in profile_units):
        raise ManifestError(
            "every profile unit must contain every expected method; sparse cells belong only in replication sets"
        )
    bootstrap_method_ids = {
        str(claim_methods["one_shot"]),
        str(claim_methods["dynamic_no_skill"]),
        str(claim_methods["dynamic_frozen_skill"]),
    }
    if any(
        not bootstrap_method_ids.issubset(evaluations[key])
        for key in bootstrap_units
    ):
        raise ManifestError(
            "every bootstrap unit must contain one-shot, dynamic-no-skill, and dynamic-frozen-skill cells"
        )

    taus = _require_list(evidence.get("profile_taus"), "profile_taus")
    tau_values = [_require_nonnegative_number(value, f"profile_taus[{i}]") for i, value in enumerate(taus)]
    if not tau_values or tau_values[0] < 1 or tau_values != sorted(set(tau_values)):
        raise ManifestError("profile_taus must be unique, sorted, and at least 1")
    tau_max = _require_nonnegative_number(evidence.get("profile_tau_max"), "profile_tau_max")
    if tau_max <= 1 or tau_max < tau_values[-1]:
        raise ManifestError("profile_tau_max must be greater than 1 and at least the largest profile tau")
    if tau_values != PAPER_PROFILE_TAUS or tau_max != PAPER_PROFILE_TAU_MAX:
        raise ManifestError(
            "performance-profile settings differ from the preregistered paper contract"
        )

    bootstrap = _require_mapping(evidence.get("bootstrap"), "bootstrap")
    confidence = _require_nonnegative_number(bootstrap.get("confidence"), "bootstrap.confidence")
    if not 0 < confidence < 1:
        raise ManifestError("bootstrap.confidence must be between 0 and 1")
    _require_positive_int(bootstrap.get("replicates"), "bootstrap.replicates")
    if not isinstance(bootstrap.get("seed"), int) or isinstance(bootstrap.get("seed"), bool):
        raise ManifestError("bootstrap.seed must be an integer")
    if dict(bootstrap) != PAPER_BOOTSTRAP:
        raise ManifestError(
            "bootstrap settings differ from the preregistered paper contract"
        )

    artifacts: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(_require_list(evidence.get("artifacts", []), "artifacts")):
        spec = _require_mapping(value, f"artifacts[{index}]")
        artifact_id = _require_string(spec.get("id"), f"artifacts[{index}].id")
        if artifact_id in artifacts:
            raise ManifestError(f"duplicate artifact ID {artifact_id}")
        path = _resolve_attested_file(evidence_path.parent, spec, f"artifacts[{index}]")
        artifacts[artifact_id] = {
            "sha256": _sha256_file(path),
        }

    return ValidatedInput(
        evidence_path=evidence_path,
        result_path=result_path,
        evidence=evidence,
        results=results,
        run_set_hash=run_set_hash,
        evidence_hash=_sha256_file(evidence_path),
        methods=methods,
        expected_kernels=expected_kernels,
        expected_units=expected_units,
        expected_cells=expected_cells,
        headline_units=headline_units,
        profile_units=profile_units,
        bootstrap_units=bootstrap_units,
        frontiers=frontiers,
        evaluations=evaluations,
        artifacts=artifacts,
    )


def _cycles(record: Mapping[str, Any]) -> int | None:
    return int(record["executed_cosim_cycles"]) if _record_is_solved(record) else None


def _valid_frontier_pair(frontier: Mapping[str, Mapping[str, Any]]) -> bool:
    baseline = _cycles(frontier["baseline"])
    expert = _cycles(frontier["expert"])
    return baseline is not None and expert is not None and baseline > expert > 0


def expert_recovery(baseline_cycles: int, generated_cycles: int, expert_cycles: int) -> float:
    """Return log(B/G)/log(B/E), rejecting a degenerate expert frontier."""
    if min(baseline_cycles, generated_cycles, expert_cycles) <= 0:
        raise ValueError("cycle counts must be positive")
    denominator = math.log(baseline_cycles / expert_cycles)
    if denominator <= 0:
        raise ValueError("expert cycles must be strictly lower than baseline cycles")
    return math.log(baseline_cycles / generated_cycles) / denominator


def _performance_ratios(
    data: ValidatedInput,
    units: Sequence[tuple[str, str]],
    method_ids: Sequence[str] | None = None,
) -> dict[str, list[float]]:
    selected_methods = (
        list(method_ids)
        if method_ids is not None
        else [method.method_id for method in data.methods]
    )
    ratios = {method_id: [] for method_id in selected_methods}
    for key in units:
        cycles = {
            method_id: _cycles(data.evaluations[key][method_id])
            for method_id in selected_methods
        }
        finite = [value for value in cycles.values() if value is not None]
        best = min(finite) if finite else None
        for method_id in selected_methods:
            value = cycles[method_id]
            ratios[method_id].append(
                math.inf if value is None or best is None else value / best
            )
    return ratios


def performance_profile(ratios: Sequence[float], taus: Sequence[float]) -> list[float]:
    if not ratios:
        raise ValueError("performance profile requires at least one evaluation unit")
    count = len(ratios)
    return [sum(math.isfinite(value) and value <= tau for value in ratios) / count for tau in taus]


def _profile_auc_scores(ratios: Sequence[float], tau_max: float) -> list[float]:
    denominator = tau_max - 1.0
    return [
        0.0
        if not math.isfinite(value) or value >= tau_max
        else (tau_max - value) / denominator
        for value in ratios
    ]


def _profile_dominates(candidate: Sequence[float], reference: Sequence[float]) -> bool:
    breakpoints = sorted(
        {1.0}
        | {value for value in candidate if math.isfinite(value)}
        | {value for value in reference if math.isfinite(value)}
    )
    cand_profile = performance_profile(candidate, breakpoints)
    ref_profile = performance_profile(reference, breakpoints)
    return all(a + 1e-12 >= b for a, b in zip(cand_profile, ref_profile)) and any(
        a > b + 1e-12 for a, b in zip(cand_profile, ref_profile)
    )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    index = (len(sorted_values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_bootstrap(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    """Paired percentile bootstrap for the mean candidate-reference delta."""
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("paired bootstrap requires equal, non-empty samples")
    deltas = [a - b for a, b in zip(candidate, reference)]
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        samples.append(statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": len(deltas),
        "estimate": statistics.fmean(deltas),
        "ci_low": _percentile(samples, alpha),
        "ci_high": _percentile(samples, 1.0 - alpha),
        "confidence": confidence,
        "replicates": replicates,
        "seed": seed,
    }


def _best_event_latency(
    record: Mapping[str, Any], budget_field: str, budget: int
) -> int | None:
    latencies = [
        int(event["synthesized_latency_cycles"])
        for event in record["candidate_events"]
        if event[budget_field] <= budget and _candidate_is_feasible(event)
    ]
    return min(latencies) if latencies else None


def _budget_curve_analysis(
    data: ValidatedInput, tau_max: float
) -> dict[str, Any]:
    unit_best: dict[tuple[str, str], int | None] = {}
    for key in data.profile_units:
        final_latencies = [
            int(event["synthesized_latency_cycles"])
            for method in data.methods
            for event in data.evaluations[key][method.method_id]["candidate_events"]
            if _candidate_is_feasible(event)
        ]
        unit_best[key] = min(final_latencies) if final_latencies else None

    specifications = [
        (
            "synthesis_evaluations",
            "cumulative_synthesis_evaluations",
            data.evidence["budget_synthesis_checkpoints"],
        ),
        ("tokens", "cumulative_tokens", data.evidence["budget_token_checkpoints"]),
    ]
    rows: list[dict[str, Any]] = []
    series: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for budget_type, event_field, checkpoints in specifications:
        series[budget_type] = {method.method_id: [] for method in data.methods}
        for budget in checkpoints:
            for method in data.methods:
                ratios: list[float] = []
                failures = 0
                for key in data.profile_units:
                    latency = _best_event_latency(
                        data.evaluations[key][method.method_id], event_field, int(budget)
                    )
                    reference = unit_best[key]
                    if latency is None or reference is None:
                        ratios.append(math.inf)
                        failures += 1
                    else:
                        ratios.append(latency / reference)
                auc = statistics.fmean(_profile_auc_scores(ratios, tau_max))
                solve_rate = 1.0 - failures / len(data.profile_units)
                row = {
                    "budget_type": budget_type,
                    "budget": int(budget),
                    "method": method.method_id,
                    "qor_profile_auc": auc,
                    "correct_solve_rate": solve_rate,
                    "failure_count": failures,
                    "unit_count": len(data.profile_units),
                }
                rows.append(row)
                series[budget_type][method.method_id].append(row)
    return {"rows": rows, "series": series}


def _attested_gate(data: ValidatedInput, gate_name: str) -> tuple[bool, str]:
    gates = _require_mapping(data.evidence.get("gate_evidence", {}), "gate_evidence")
    spec = gates.get(gate_name)
    if not isinstance(spec, dict):
        return False, "missing gate evidence"
    if spec.get("status") != PASS:
        return False, f"status={spec.get('status', 'missing')}"
    artifact_id = spec.get("artifact_id")
    if artifact_id is not None and artifact_id not in data.artifacts:
        return False, f"unattested artifact_id={artifact_id}"
    return True, "passed"


def _solve_rate(
    data: ValidatedInput, method_id: str, units: Sequence[tuple[str, str]]
) -> float:
    return sum(
        _record_is_solved(data.evaluations[key][method_id]) for key in units
    ) / len(units)


def _records_for_method(
    data: ValidatedInput, method_id: str
) -> list[dict[str, Any]]:
    return [
        data.evaluations[(kernel, seed)][method_id]
        for kernel, seed, expected_method in data.expected_cells
        if expected_method == method_id
    ]


def _resource_measurement_row(
    record: Mapping[str, Any],
    *,
    kernel: str,
    seed: str,
    method: str,
    role: str,
) -> dict[str, Any]:
    metrics = record.get("synthesis_metrics")
    available = record.get("synthesis_status") == PASS and isinstance(metrics, dict)
    row: dict[str, Any] = {
        "kernel": kernel,
        "seed": seed,
        "method": method,
        "role": role,
        "run_id": record["run_id"],
        "terminal_status": record["terminal_status"],
        "correctness_status": record["correctness_status"],
        "synthesis_status": record["synthesis_status"],
        "resource_fit": record["resource_fit"],
        "timing_met": record["timing_met"],
        "fmax_mhz": metrics["fmax_mhz"] if available else None,
        "failure_class": record.get("failure_class"),
        "provider_failure": record.get("provider_failure") if role == "generated" else None,
        "measurement_status": "available" if available else "not_available",
    }
    resources = metrics["resources"] if available else {}
    for key in RESOURCE_KEYS:
        resource = resources.get(key, {})
        row[f"{key}_used"] = resource.get("used")
        row[f"{key}_capacity"] = resource.get("capacity")
        row[f"{key}_utilization"] = resource.get("utilization")
    return row


def _resource_rows(data: ValidatedInput) -> list[dict[str, Any]]:
    rows = [
        _resource_measurement_row(
            data.evaluations[(kernel, seed)][method_id],
            kernel=kernel,
            seed=seed,
            method=method_id,
            role="generated",
        )
        for kernel, seed, method_id in data.expected_cells
    ]
    for kernel in data.expected_kernels:
        for role in ("baseline", "expert"):
            rows.append(
                _resource_measurement_row(
                    data.frontiers[kernel][role],
                    kernel=kernel,
                    seed="",
                    method=role,
                    role=role,
                )
            )
    return rows


def _resource_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    measured = [
        record["synthesis_metrics"]
        for record in records
        if record.get("synthesis_status") == PASS
    ]
    summary: dict[str, Any] = {
        "qor_records": len(measured),
        "mean_fmax_mhz": (
            statistics.fmean(float(item["fmax_mhz"]) for item in measured)
            if measured
            else None
        ),
    }
    for key in RESOURCE_KEYS:
        summary[f"mean_{key}_utilization"] = (
            statistics.fmean(
                float(item["resources"][key]["utilization"]) for item in measured
            )
            if measured
            else None
        )
    return summary


def _cost_complete(data: ValidatedInput) -> bool:
    fields = (
        "tokens", "llm_calls", "synthesis_calls", "selection_synthesis_evaluations",
        "wall_time_seconds", "candidates_evaluated",
    )
    return all(
        all(field in data.evaluations[(kernel, seed)][method_id] for field in fields)
        for kernel, seed, method_id in data.expected_cells
    )


def _matched_budget(data: ValidatedInput) -> tuple[bool, str]:
    gates = _require_mapping(data.evidence.get("gate_evidence", {}), "gate_evidence")
    spec = gates.get("matched_budget")
    if not isinstance(spec, dict) or spec.get("status") != PASS:
        return False, "missing or failed matched-budget attestation"
    try:
        candidate_limit = _require_positive_int(spec.get("candidate_limit"), "matched_budget.candidate_limit")
        synthesis_limit = _require_positive_int(spec.get("synthesis_limit"), "matched_budget.synthesis_limit")
    except ManifestError as exc:
        return False, str(exc)
    if candidate_limit != 5 or synthesis_limit != 5:
        return False, "HPCA matched budget requires maximum five candidates and five selection syntheses"
    for kernel, seed, method_id in data.expected_cells:
        rec = data.evaluations[(kernel, seed)][method_id]
        if rec["candidates_evaluated"] > candidate_limit:
            return False, f"{rec['run_id']} exceeds candidate limit"
        if rec["selection_synthesis_evaluations"] > synthesis_limit:
            return False, f"{rec['run_id']} exceeds selection synthesis limit"
    return True, "passed and independently checked against every run"


def compute_analysis(data: ValidatedInput) -> dict[str, Any]:
    methods = {method.method_id: method for method in data.methods}
    claim_methods = data.evidence["claim_methods"]
    primary = claim_methods["primary"]
    ratios = _performance_ratios(data, data.profile_units)
    bootstrap_method_ids = list(
        dict.fromkeys(
            [
                claim_methods["one_shot"],
                claim_methods["dynamic_no_skill"],
                claim_methods["dynamic_frozen_skill"],
            ]
        )
    )
    bootstrap_ratios = _performance_ratios(
        data, data.bootstrap_units, bootstrap_method_ids
    )
    taus = [float(value) for value in data.evidence["profile_taus"]]
    tau_max = float(data.evidence["profile_tau_max"])

    recovery_rows: list[dict[str, Any]] = []
    valid_pair_count = 0
    for kernel in data.expected_kernels:
        frontier = data.frontiers[kernel]
        pair_valid = _valid_frontier_pair(frontier)
        valid_pair_count += int(pair_valid)
        # Extra fixed seeds remain in profiles and bootstrap intervals; the
        # evidence manifest explicitly chooses the one headline unit per kernel.
        generated = data.evaluations[data.headline_units[kernel]][primary]
        baseline_cycles = _cycles(frontier["baseline"])
        expert_cycles = _cycles(frontier["expert"])
        generated_cycles = _cycles(generated)
        recovery = None
        if pair_valid and generated_cycles is not None:
            recovery = expert_recovery(baseline_cycles, generated_cycles, expert_cycles)  # type: ignore[arg-type]
        if not _record_is_solved(frontier["baseline"]):
            row_status = f"baseline:{frontier['baseline']['failure_class']}"
        elif not _record_is_solved(frontier["expert"]):
            row_status = f"expert:{frontier['expert']['failure_class']}"
        elif not pair_valid:
            row_status = "invalid_expert_frontier"
        elif generated_cycles is None:
            row_status = generated["failure_class"]
        else:
            row_status = "success"
        recovery_rows.append(
            {
                "kernel": kernel,
                "baseline_run_id": frontier["baseline"]["run_id"],
                "generated_run_id": generated["run_id"],
                "expert_run_id": frontier["expert"]["run_id"],
                "baseline_cycles": baseline_cycles,
                "generated_cycles": generated_cycles,
                "expert_cycles": expert_cycles,
                "recovery": recovery,
                "pair_valid": pair_valid,
                "status": row_status,
            }
        )

    bootstrap_spec = data.evidence["bootstrap"]
    no_skill = claim_methods["dynamic_no_skill"]
    with_skill = claim_methods["dynamic_frozen_skill"]
    no_skill_scores = _profile_auc_scores(bootstrap_ratios[no_skill], tau_max)
    with_skill_scores = _profile_auc_scores(bootstrap_ratios[with_skill], tau_max)
    skill_bootstrap = paired_bootstrap(
        with_skill_scores,
        no_skill_scores,
        replicates=int(bootstrap_spec["replicates"]),
        confidence=float(bootstrap_spec["confidence"]),
        seed=int(bootstrap_spec["seed"]),
    )
    no_skill_solve_rate = _solve_rate(data, no_skill, data.bootstrap_units)
    skill_solve_rate = _solve_rate(data, with_skill, data.bootstrap_units)
    one_shot = claim_methods["one_shot"]
    workflow_bootstrap = paired_bootstrap(
        _profile_auc_scores(ratios[no_skill], tau_max),
        _profile_auc_scores(ratios[one_shot], tau_max),
        replicates=int(bootstrap_spec["replicates"]),
        confidence=float(bootstrap_spec["confidence"]),
        seed=int(bootstrap_spec["seed"]),
    )
    skill_solve_bootstrap = paired_bootstrap(
        [float(_record_is_solved(data.evaluations[key][with_skill])) for key in data.bootstrap_units],
        [float(_record_is_solved(data.evaluations[key][no_skill])) for key in data.bootstrap_units],
        replicates=int(bootstrap_spec["replicates"]),
        confidence=float(bootstrap_spec["confidence"]),
        seed=int(bootstrap_spec["seed"]),
    )
    workflow_solve_bootstrap = paired_bootstrap(
        [float(_record_is_solved(data.evaluations[key][no_skill])) for key in data.profile_units],
        [float(_record_is_solved(data.evaluations[key][one_shot])) for key in data.profile_units],
        replicates=int(bootstrap_spec["replicates"]),
        confidence=float(bootstrap_spec["confidence"]),
        seed=int(bootstrap_spec["seed"]),
    )

    leakage_ok, leakage_detail = _attested_gate(data, "transcript_leakage_audit")
    fingerprint_ok, fingerprint_detail = _attested_gate(
        data, "fingerprint_consistency_audit"
    )
    candidate_validation_ok, candidate_validation_detail = _attested_gate(
        data, "candidate_validation_audit"
    )
    event_stream_ok, event_stream_detail = _attested_gate(
        data, "complete_candidate_event_stream"
    )
    skill_snapshot_ok, skill_snapshot_detail = _attested_gate(data, "frozen_skill_snapshot")
    matched_ok, matched_detail = _matched_budget(data)
    post_route_ok, post_route_detail = _attested_gate(data, "post_route_validation")
    post_route_spec = _require_mapping(
        _require_mapping(data.evidence.get("gate_evidence", {}), "gate_evidence").get(
            "post_route_validation", {}
        ),
        "gate_evidence.post_route_validation",
    )
    post_route_count = post_route_spec.get("stratified_winner_count", 0)
    if not isinstance(post_route_count, int) or isinstance(post_route_count, bool):
        post_route_count = 0
    if post_route_ok and post_route_count < 5:
        post_route_ok = False
        post_route_detail = f"only {post_route_count} stratified winners; five required"
    skill_gate = _require_mapping(
        _require_mapping(data.evidence.get("gate_evidence", {}), "gate_evidence").get(
            "frozen_skill_snapshot", {}
        ),
        "gate_evidence.frozen_skill_snapshot",
    )
    frozen_before = skill_gate.get("frozen_before_evaluation") is True
    no_persistence = skill_gate.get("no_evaluation_persistence") is True
    dynamic_dominates = _profile_dominates(ratios[no_skill], ratios[one_shot])
    skill_ci_improves = float(skill_bootstrap["ci_low"]) > 0.0
    skill_rate_nondecrease = skill_solve_rate + 1e-12 >= no_skill_solve_rate

    all_primary_reported_cycles_executed = all(
        rec.get("cycle_source") == "executed_rtl_cosim"
        for rec in (data.evaluations[key][primary] for key in data.headline_units.values())
        if rec.get("executed_cosim_cycles") is not None
    )
    complete_denominator = all(
        key in data.evaluations for key in set(data.profile_units) | set(data.bootstrap_units)
    )
    minimum_pairs_raw = _require_mapping(data.evidence.get("policy", {}), "policy").get(
        "minimum_valid_baseline_expert_pairs", 8
    )
    minimum_pairs = _require_positive_int(
        minimum_pairs_raw, "policy.minimum_valid_baseline_expert_pairs"
    )
    primary_recovery_count = sum(row["recovery"] is not None for row in recovery_rows)

    def decision(gates: Mapping[str, bool], details: Mapping[str, str] | None = None) -> dict[str, Any]:
        return {
            "status": "passed" if all(gates.values()) else "blocked",
            "gates": dict(gates),
            "details": dict(details or {}),
        }

    claims = {
        "headline_reference_blind_recovery": decision(
            {
                "at_least_minimum_valid_baseline_expert_pairs": valid_pair_count >= minimum_pairs,
                "at_least_one_primary_recovery_measurement": primary_recovery_count > 0,
                "every_reported_cycle_is_executed_cosim": all_primary_reported_cycles_executed,
                "failures_visible_and_in_denominator": complete_denominator,
                "transcript_leakage_audit": leakage_ok,
                "fingerprint_consistency_audit": fingerprint_ok,
            },
            {
                "transcript_leakage_audit": leakage_detail,
                "fingerprint_consistency_audit": fingerprint_detail,
            },
        ),
        "workflow_beats_one_shot": decision(
            {
                "matched_candidate_and_synthesis_budget": matched_ok,
                "fingerprint_consistency_audit": fingerprint_ok,
                "every_candidate_csim_and_only_winner_cosim_audited": candidate_validation_ok,
                "dynamic_dominates_heldout_performance_profile": dynamic_dominates,
            },
            {
                "matched_budget": matched_detail,
                "candidate_validation_audit": candidate_validation_detail,
            },
        ),
        "frozen_skill_transfer": decision(
            {
                "frozen_skill_snapshot_attested": skill_snapshot_ok,
                "library_frozen_before_evaluation": frozen_before,
                "no_evaluation_persistence": no_persistence,
                "fingerprint_consistency_audit": fingerprint_ok,
                "paired_bootstrap_ci_strictly_positive": skill_ci_improves,
                "correct_solve_rate_does_not_decrease": skill_rate_nondecrease,
            },
            {"frozen_skill_snapshot": skill_snapshot_detail},
        ),
        "compact_model_enablement": decision(
            {
                "dynamic_beats_matched_one_shot_profile": (
                    matched_ok and candidate_validation_ok and dynamic_dominates
                    and fingerprint_ok
                )
            },
            {
                "matched_budget": matched_detail,
                "candidate_validation_audit": candidate_validation_detail,
            },
        ),
        "cost_quality_tradeoff": decision(
            {
                "complete_candidate_event_stream": event_stream_ok,
                "fingerprint_consistency_audit": fingerprint_ok,
                "complete_call_token_synthesis_wall_time_attribution": _cost_complete(data),
            },
            {"complete_candidate_event_stream": event_stream_detail},
        ),
        "post_route_and_board_validation": decision(
            {"five_stratified_post_route_runs": post_route_ok},
            {"post_route_validation": post_route_detail},
        ),
    }

    profiles = {
        method.method_id: performance_profile(ratios[method.method_id], taus)
        for method in data.methods
    }
    profile_plot_taus = sorted(
        {1.0, tau_max}
        | {
            value
            for method_ratios in ratios.values()
            for value in method_ratios
            if math.isfinite(value) and 1.0 <= value <= tau_max
        }
    )
    profile_plot = {
        method.method_id: performance_profile(
            ratios[method.method_id], profile_plot_taus
        )
        for method in data.methods
    }
    profile_auc = {
        method.method_id: statistics.fmean(
            _profile_auc_scores(ratios[method.method_id], tau_max)
        )
        for method in data.methods
    }
    budget_curves = _budget_curve_analysis(data, tau_max)
    solve_rates = {
        method.method_id: _solve_rate(data, method.method_id, data.profile_units)
        for method in data.methods
    }
    primary_solve_rate = _solve_rate(
        data, primary, list(data.headline_units.values())
    )
    failures: dict[str, Counter[str]] = {}
    resource_summaries: dict[str, dict[str, Any]] = {}
    for method in data.methods:
        method_records = _records_for_method(data, method.method_id)
        failures[method.method_id] = Counter(
            record["failure_class"]
            for record in method_records
            if not _record_is_solved(record)
        )
        resource_summaries[method.method_id] = _resource_summary(method_records)
    for role in ("baseline", "expert"):
        resource_summaries[role] = _resource_summary(
            [data.frontiers[kernel][role] for kernel in data.expected_kernels]
        )

    return {
        "method_names": {key: value.display_name for key, value in methods.items()},
        "primary_method": primary,
        "recovery_rows": recovery_rows,
        "valid_pair_count": valid_pair_count,
        "minimum_pair_count": minimum_pairs,
        "ratios": ratios,
        "profile_taus": taus,
        "profiles": profiles,
        "profile_plot_taus": profile_plot_taus,
        "profile_plot": profile_plot,
        "profile_tau_max": tau_max,
        "profile_auc": profile_auc,
        "budget_curves": budget_curves,
        "solve_rates": solve_rates,
        "primary_solve_rate": primary_solve_rate,
        "failures": failures,
        "resource_rows": _resource_rows(data),
        "resource_summaries": resource_summaries,
        "skill_bootstrap": skill_bootstrap,
        "workflow_bootstrap": workflow_bootstrap,
        "skill_solve_bootstrap": skill_solve_bootstrap,
        "workflow_solve_bootstrap": workflow_solve_bootstrap,
        "claims": claims,
    }


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _tex_cycles(value: int | None) -> str:
    return "--" if value is None else f"{value:,}"


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_result_macros(data: ValidatedInput, analysis: Mapping[str, Any]) -> str:
    solve_rate = analysis["primary_solve_rate"]
    recovery_values = [
        row["recovery"] for row in analysis["recovery_rows"] if row["recovery"] is not None
    ]
    headline = analysis["claims"]["headline_reference_blind_recovery"]["status"] == "passed"
    if headline:
        abstract = (
            f"On the frozen reference-isolated run set, the system solves "
            f"{round(100 * solve_rate):d}\\% of evaluation units; failures remain in the denominator."
        )
    else:
        abstract = (
            "The frozen evidence does not satisfy every preregistered headline gate; "
            "we therefore make no aggregate expert-recovery claim."
        )
    if analysis["valid_pair_count"] >= analysis["minimum_pair_count"]:
        pair_sentence = (
            f"the gate passes with {analysis['valid_pair_count']} validated baseline/frontier pairs."
        )
    else:
        pair_sentence = (
            f"the gate is blocked: {analysis['valid_pair_count']} validated pairs are available "
            f"but {analysis['minimum_pair_count']} are required."
        )
    if recovery_values:
        summary = (
            f"Median expert recovery is {_format_float(statistics.median(recovery_values), 2)} "
            f"across {len(recovery_values)} solved units with valid frontiers."
        )
    else:
        summary = "Expert recovery is unavailable because no solved unit has a valid frontier."
    skill = analysis["claims"]["frozen_skill_transfer"]["status"]
    compact = analysis["claims"]["compact_model_enablement"]["status"]
    return (
        "% AUTO-GENERATED; DO NOT EDIT.\n"
        f"% frozen run-set sha256: {data.run_set_hash}\n"
        f"\\providecommand{{\\EvidenceAbstractSentence}}{{{abstract}}}\n"
        f"\\providecommand{{\\ValidPairGateSentence}}{{{_latex_escape(pair_sentence)}}}\n"
        f"\\providecommand{{\\FrozenRunSetHash}}{{\\texttt{{{data.run_set_hash[:12]}}}}}\n"
        f"\\providecommand{{\\PrimaryCorrectSolveRate}}{{{100 * solve_rate:.1f}\\%}}\n"
        f"\\providecommand{{\\PrimaryRecoverySummary}}{{{_latex_escape(summary)}}}\n"
        f"\\providecommand{{\\SkillTransferDecision}}{{{_latex_escape(skill)}}}\n"
        f"\\providecommand{{\\CompactModelDecision}}{{{_latex_escape(compact)}}}\n"
    )


def _render_recovery_table(data: ValidatedInput, analysis: Mapping[str, Any]) -> str:
    lines = [
        "% AUTO-GENERATED; DO NOT EDIT.",
        f"% frozen run-set sha256: {data.run_set_hash}",
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Reference-blind recovery on reference-isolated kernels. Failures remain explicit; all numeric cycle cells are executed RTL co-simulation.}",
        r"  \label{tab:recovery}",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lrrrrl}",
        r"    \hline",
        r"    Kernel & Baseline & Generated & Expert & Recovery & Status \\",
        r"    \hline",
    ]
    for row in analysis["recovery_rows"]:
        recovery = "--" if row["recovery"] is None else _format_float(row["recovery"], 2)
        lines.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(row["kernel"]),
                    _tex_cycles(row["baseline_cycles"]),
                    _tex_cycles(row["generated_cycles"]),
                    _tex_cycles(row["expert_cycles"]),
                    recovery,
                    _latex_escape(row["status"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"    \hline", r"  \end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _render_ablation_table(data: ValidatedInput, analysis: Mapping[str, Any]) -> str:
    lines = [
        "% AUTO-GENERATED; DO NOT EDIT.",
        f"% frozen run-set sha256: {data.run_set_hash}",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Matched-budget workflow outcomes over the preregistered primary profile units. Profile AUC is bounded and assigns zero to failures. Calls are mean LLM/synthesis calls.}",
        r"  \label{tab:ablation}",
        r"  \setlength{\tabcolsep}{3.5pt}",
        r"  \begin{tabular}{lccc}",
        r"    \hline",
        r"    Method & Correct solve & Profile AUC & Calls \\",
        r"    \hline",
    ]
    for method in data.methods:
        records = [data.evaluations[key][method.method_id] for key in data.profile_units]
        llm_calls = statistics.fmean(record["llm_calls"] for record in records)
        synth_calls = statistics.fmean(record["synthesis_calls"] for record in records)
        lines.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(method.display_name),
                    f"{100 * analysis['solve_rates'][method.method_id]:.1f}\\%",
                    _format_float(analysis["profile_auc"][method.method_id], 3),
                    f"{llm_calls:.1f}/{synth_calls:.1f}",
                ]
            )
            + r" \\"
        )
    confidence = 100 * analysis["skill_bootstrap"]["confidence"]
    skill = analysis["skill_bootstrap"]
    workflow = analysis["workflow_bootstrap"]
    lines.extend(
        [
            r"    \hline",
            r"  \end{tabular}",
            r"  \vspace{0.35em}",
            r"  \parbox{0.98\columnwidth}{\footnotesize Paired profile-AUC deltas "
            + f"({confidence:.1f}\\% CI): frozen skill $-$ no skill "
            + f"{skill['estimate']:.3f} [{skill['ci_low']:.3f}, {skill['ci_high']:.3f}]; "
            + f"dynamic no skill $-$ one-shot {workflow['estimate']:.3f} "
            + f"[{workflow['ci_low']:.3f}, {workflow['ci_high']:.3f}].}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _render_resource_table(data: ValidatedInput, analysis: Mapping[str, Any]) -> str:
    lines = [
        "% AUTO-GENERATED; DO NOT EDIT.",
        f"% frozen run-set sha256: {data.run_set_hash}",
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Mean Vitis CSynth Fmax and device utilization over records with passing synthesis. Failed or non-synthesized cells remain explicit in the companion CSV and are not imputed.}",
        r"  \label{tab:resource-fmax}",
        r"  \setlength{\tabcolsep}{3.5pt}",
        r"  \begin{tabular}{lrrrrrrr}",
        r"    \hline",
        "    Method/role & $n$ & Fmax (MHz) & BRAM & DSP & FF & LUT & URAM \\\\",
        r"    \hline",
    ]
    labels = [(method.method_id, method.display_name) for method in data.methods]
    labels.extend([("baseline", "Baseline"), ("expert", "Expert frontier")])
    for identifier, label in labels:
        summary = analysis["resource_summaries"][identifier]
        fmax = summary["mean_fmax_mhz"]
        values = [
            "--"
            if summary[f"mean_{key}_utilization"] is None
            else f"{100 * summary[f'mean_{key}_utilization']:.2f}\\%"
            for key in RESOURCE_KEYS
        ]
        lines.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(label),
                    str(summary["qor_records"]),
                    "--" if fmax is None else f"{fmax:.1f}",
                    *values,
                ]
            )
            + " \\\\"
        )
    lines.extend([r"    \hline", r"  \end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _svg_text(x: float, y: float, text_value: Any, **attrs: Any) -> str:
    attr_text = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(value))}"' for key, value in attrs.items()
    )
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr_text}>{html.escape(str(text_value))}</text>'


def _render_recovery_svg(data: ValidatedInput, analysis: Mapping[str, Any]) -> str:
    rows = analysis["recovery_rows"]
    width, height = 1100, 430
    left, right, top, bottom = 75, 25, 45, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    finite: list[float] = [1.0]
    normalized: list[tuple[float | None, float | None]] = []
    for row in rows:
        baseline = row["baseline_cycles"]
        generated = row["generated_cycles"]
        expert = row["expert_cycles"]
        gen_norm = generated / baseline if baseline and generated else None
        expert_norm = expert / baseline if baseline and expert else None
        normalized.append((gen_norm, expert_norm))
        finite.extend(value for value in (gen_norm, expert_norm) if value is not None)
    y_max = max(finite) * 1.12
    if y_max <= 0:
        y_max = 1.0

    def y(value: float) -> float:
        return top + plot_h * (1.0 - value / y_max)

    group_w = plot_w / max(1, len(rows))
    bar_w = min(28.0, group_w * 0.28)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="430" viewBox="0 0 1100 430">',
        '<rect width="1100" height="430" fill="white"/>',
        '<g font-family="Helvetica,Arial,sans-serif" fill="#111">',
        _svg_text(75, 22, f"Frozen run set {data.run_set_hash[:12]}", font_size=13),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111"/>',
        f'<line x1="{left}" y1="{y(1.0):.2f}" x2="{left + plot_w}" y2="{y(1.0):.2f}" stroke="#555" stroke-dasharray="5,4"/>',
        _svg_text(left - 10, y(1.0) + 4, "1.0", text_anchor="end", font_size=12),
        _svg_text(18, top + plot_h / 2, "Cycles / baseline", font_size=13, transform=f"rotate(-90 18 {top + plot_h / 2})"),
    ]
    for index, (row, values) in enumerate(zip(rows, normalized)):
        center = left + group_w * (index + 0.5)
        for offset, value, fill, stroke_dash in (
            (-bar_w * 0.55, values[0], "#555", ""),
            (bar_w * 0.55, values[1], "#d0d0d0", "3,2"),
        ):
            x = center + offset - bar_w / 2
            if value is None:
                marker_y = top + 10
                parts.extend(
                    [
                        f'<line x1="{x:.2f}" y1="{marker_y - 6}" x2="{x + bar_w:.2f}" y2="{marker_y + 6}" stroke="#111" stroke-width="2"/>',
                        f'<line x1="{x:.2f}" y1="{marker_y + 6}" x2="{x + bar_w:.2f}" y2="{marker_y - 6}" stroke="#111" stroke-width="2"/>',
                    ]
                )
            else:
                y_top = y(value)
                dash = f' stroke-dasharray="{stroke_dash}"' if stroke_dash else ""
                parts.append(
                    f'<rect x="{x:.2f}" y="{y_top:.2f}" width="{bar_w:.2f}" height="{top + plot_h - y_top:.2f}" fill="{fill}" stroke="#111"{dash}/>'
                )
        parts.append(
            _svg_text(center, top + plot_h + 19, row["kernel"], text_anchor="middle", font_size=11)
        )
    legend_y = height - 25
    parts.extend(
        [
            '<rect x="760" y="394" width="16" height="12" fill="#555" stroke="#111"/>',
            _svg_text(782, 405, "Generated", font_size=12),
            '<rect x="870" y="394" width="16" height="12" fill="#d0d0d0" stroke="#111" stroke-dasharray="3,2"/>',
            _svg_text(892, 405, "Expert frontier", font_size=12),
            _svg_text(75, legend_y, "X = failed or unavailable executed co-simulation", font_size=12),
            "</g></svg>\n",
        ]
    )
    return "\n".join(parts)


def _render_profile_svg(data: ValidatedInput, analysis: Mapping[str, Any]) -> str:
    taus = analysis["profile_taus"]
    width, height = 760, 470
    left, right, top, bottom = 70, 30, 35, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    tau_min, tau_max = 1.0, float(analysis["profile_tau_max"])

    def x(value: float) -> float:
        return left + plot_w * math.log(value / tau_min) / math.log(tau_max / tau_min)

    def y(value: float) -> float:
        return top + plot_h * (1.0 - value)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="470" viewBox="0 0 760 470">',
        '<rect width="760" height="470" fill="white"/>',
        '<g font-family="Helvetica,Arial,sans-serif" fill="#111">',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111"/>',
        _svg_text(width / 2, height - 14, "Performance ratio τ (log scale)", text_anchor="middle", font_size=13),
        _svg_text(18, top + plot_h / 2, "Fraction of all units", font_size=13, transform=f"rotate(-90 18 {top + plot_h / 2})"),
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{left - 4}" y1="{y(tick):.2f}" x2="{left}" y2="{y(tick):.2f}" stroke="#111"/>')
        parts.append(_svg_text(left - 8, y(tick) + 4, f"{tick:.2g}", text_anchor="end", font_size=11))
    for tau in taus:
        parts.append(f'<line x1="{x(tau):.2f}" y1="{top + plot_h}" x2="{x(tau):.2f}" y2="{top + plot_h + 4}" stroke="#111"/>')
        parts.append(_svg_text(x(tau), top + plot_h + 18, f"{tau:g}", text_anchor="middle", font_size=11))
    grayscale = ["#111", "#444", "#777", "#999", "#bbb", "#222", "#666"]
    dashes = ["", "8,3", "3,3", "10,3,2,3", "2,2", "12,4", "5,2,1,2"]
    for index, method in enumerate(data.methods):
        exact_taus = analysis["profile_plot_taus"]
        exact_values = analysis["profile_plot"][method.method_id]
        points = [(x(exact_taus[0]), y(exact_values[0]))]
        for tau, previous, current in zip(exact_taus[1:], exact_values, exact_values[1:]):
            points.append((x(tau), y(previous)))
            points.append((x(tau), y(current)))
        point_text = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        dash = f' stroke-dasharray="{dashes[index % len(dashes)]}"' if dashes[index % len(dashes)] else ""
        color = grayscale[index % len(grayscale)]
        parts.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>' )
        legend_y = 48 + index * 18
        parts.append(f'<line x1="500" y1="{legend_y}" x2="530" y2="{legend_y}" stroke="{color}" stroke-width="2.2"{dash}/>' )
        parts.append(_svg_text(536, legend_y + 4, method.display_name, font_size=11))
    parts.extend(
        [
            _svg_text(70, 20, f"Failures remain at infinity; run set {data.run_set_hash[:12]}", font_size=12),
            "</g></svg>\n",
        ]
    )
    return "\n".join(parts)


def _freeze_datetime(data: ValidatedInput) -> datetime_module.datetime:
    return datetime_module.datetime.fromisoformat(
        data.evidence["evidence_freeze_timestamp"].replace("Z", "+00:00")
    )


def _pdf_metadata(
    title: str, run_set_hash: str, freeze_datetime: datetime_module.datetime
) -> dict[str, Any]:
    return {
        "Title": title,
        "Author": "Anonymous",
        "Subject": "Anonymous HPCA 2027 evaluation artifact",
        "Keywords": f"C2HLS, HPCA 2027, frozen run set {run_set_hash}",
        "Creator": "C2HLS frozen-evidence artifact generator",
        "CreationDate": freeze_datetime,
        "ModDate": freeze_datetime,
    }


def _try_matplotlib() -> tuple[Any, Any] | None:
    try:
        import matplotlib  # type: ignore

        matplotlib.use("pdf", force=True)
        import matplotlib.pyplot as pyplot  # type: ignore
    except ImportError:
        return None
    return matplotlib, pyplot


def _render_pdfs_matplotlib(
    stage: Path, data: ValidatedInput, analysis: Mapping[str, Any], modules: tuple[Any, Any]
) -> None:
    matplotlib, pyplot = modules
    rc = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 9.0,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
    with matplotlib.rc_context(rc):
        rows = analysis["recovery_rows"]
        figure, axis = pyplot.subplots(figsize=(10.5, 3.45), constrained_layout=True)
        positions = list(range(len(rows)))
        width = 0.34
        normalized_generated: list[float] = []
        normalized_expert: list[float] = []
        generated_failures: list[int] = []
        expert_failures: list[int] = []
        for index, row in enumerate(rows):
            baseline = row["baseline_cycles"]
            generated = row["generated_cycles"]
            expert = row["expert_cycles"]
            if baseline is not None and generated is not None:
                normalized_generated.append(generated / baseline)
            else:
                normalized_generated.append(math.nan)
                generated_failures.append(index)
            if baseline is not None and expert is not None and row["pair_valid"]:
                normalized_expert.append(expert / baseline)
            else:
                normalized_expert.append(math.nan)
                expert_failures.append(index)
        finite = [
            value
            for value in normalized_generated + normalized_expert + [1.0]
            if math.isfinite(value)
        ]
        y_limit = max(finite) * 1.18
        axis.bar(
            [value - width / 2 for value in positions],
            normalized_generated,
            width,
            color="0.35",
            edgecolor="black",
            linewidth=0.6,
            label="Generated",
        )
        axis.bar(
            [value + width / 2 for value in positions],
            normalized_expert,
            width,
            color="white",
            edgecolor="black",
            linewidth=0.7,
            hatch="////",
            label="Expert frontier",
        )
        marker_y = y_limit * 0.94
        if generated_failures:
            axis.scatter(
                [index - width / 2 for index in generated_failures],
                [marker_y] * len(generated_failures),
                marker="x",
                color="black",
                linewidths=1.5,
                zorder=5,
            )
        if expert_failures:
            axis.scatter(
                [index + width / 2 for index in expert_failures],
                [marker_y] * len(expert_failures),
                marker="x",
                color="black",
                linewidths=1.5,
                zorder=5,
            )
        for index, row in enumerate(rows):
            if row["recovery"] is not None and math.isfinite(normalized_generated[index]):
                axis.annotate(
                    f"ρ={row['recovery']:.2f}",
                    (index - width / 2, normalized_generated[index]),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=5.8,
                    rotation=90 if len(rows) > 9 else 0,
                )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Baseline")
        axis.set_ylim(0, y_limit)
        axis.set_ylabel("Executed RTL co-sim cycles / baseline")
        axis.set_xticks(positions, [row["kernel"] for row in rows], rotation=25, ha="right")
        axis.set_title(
            "Reference-blind expert recovery "
            f"(frozen run set {data.run_set_hash[:12]}; × denotes failure)"
        )
        axis.grid(axis="y", color="0.86", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.legend(ncol=3, frameon=False, loc="upper right")
        figure.savefig(
            stage / "recovery.pdf",
            format="pdf",
            metadata=_pdf_metadata(
                "Reference-blind expert recovery", data.run_set_hash, _freeze_datetime(data)
            ),
        )
        pyplot.close(figure)

        figure = pyplot.figure(figsize=(10.5, 6.1), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.08])
        synthesis_axis = figure.add_subplot(grid[0, 0])
        token_axis = figure.add_subplot(grid[0, 1])
        ablation_axis = figure.add_subplot(grid[1, :])
        colors = ["0.05", "0.25", "0.45", "0.62", "0.78", "0.15", "0.55"]
        line_styles = ["-", "--", "-.", ":", (0, (6, 2, 1, 2)), (0, (2, 1)), (0, (8, 3))]
        markers = ["o", "s", "^", "D", "v", "P", "X"]
        for axis, budget_type, xlabel in (
            (synthesis_axis, "synthesis_evaluations", "Selection synthesis evaluations"),
            (token_axis, "tokens", "Cumulative LLM tokens"),
        ):
            for index, method in enumerate(data.methods):
                points = analysis["budget_curves"]["series"][budget_type][method.method_id]
                x_values = [point["budget"] for point in points]
                y_values = [point["qor_profile_auc"] for point in points]
                axis.plot(
                    x_values,
                    y_values,
                    label=method.display_name,
                    color=colors[index % len(colors)],
                    linestyle=line_styles[index % len(line_styles)],
                    marker=markers[index % len(markers)],
                    markersize=3.5,
                    markerfacecolor="white" if index % 2 else colors[index % len(colors)],
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
                for point in points:
                    if point["failure_count"] == 0:
                        continue
                    axis.annotate(
                        f"F{point['failure_count']}",
                        (point["budget"], point["qor_profile_auc"]),
                        xytext=(0, 4 + (index % 3) * 3),
                        textcoords="offset points",
                        ha="center",
                        fontsize=4.7,
                        color="0.1",
                    )
            axis.set_ylim(-0.02, 1.08)
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Failure-aware CSynth QoR AUC")
            axis.grid(color="0.88", linewidth=0.5)
            axis.set_axisbelow(True)
        synthesis_axis.set_title("(a) QoR versus synthesis budget; F labels mark nonzero failures")
        token_axis.set_title("(b) QoR versus token budget; F labels mark nonzero failures")
        synthesis_axis.legend(frameon=False, ncol=2, loc="lower right")

        method_positions = list(range(len(data.methods)))
        ablation_width = 0.34
        auc_values = [analysis["profile_auc"][method.method_id] for method in data.methods]
        solve_values = [analysis["solve_rates"][method.method_id] for method in data.methods]
        ablation_axis.bar(
            [value - ablation_width / 2 for value in method_positions],
            auc_values,
            ablation_width,
            color="0.35",
            edgecolor="black",
            linewidth=0.6,
            label="Executed-cosim profile AUC",
        )
        ablation_axis.bar(
            [value + ablation_width / 2 for value in method_positions],
            solve_values,
            ablation_width,
            color="white",
            edgecolor="black",
            hatch="////",
            linewidth=0.7,
            label="Correct-solve rate",
        )
        for index, method in enumerate(data.methods):
            failure_count = len(data.profile_units) - round(
                analysis["solve_rates"][method.method_id] * len(data.profile_units)
            )
            ablation_axis.annotate(
                f"F={failure_count}",
                (index, max(auc_values[index], solve_values[index])),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=6.0,
            )
        ablation_axis.set_ylim(0, 1.12)
        ablation_axis.set_ylabel("Failure-aware score")
        ablation_axis.set_xticks(
            method_positions,
            [method.display_name for method in data.methods],
            rotation=18,
            ha="right",
        )
        ablation_axis.set_title("(c) Matched-budget component ablation; failures remain in denominators")
        ablation_axis.grid(axis="y", color="0.88", linewidth=0.5)
        ablation_axis.set_axisbelow(True)
        ablation_axis.legend(frameon=False, ncol=2, loc="upper left")
        figure.suptitle(f"Frozen run set {data.run_set_hash[:12]}", fontsize=8.0)
        figure.savefig(
            stage / "budget.pdf",
            format="pdf",
            metadata=_pdf_metadata(
                "QoR, search budget, and component ablation",
                data.run_set_hash,
                _freeze_datetime(data),
            ),
        )
        pyplot.close(figure)


def _latex_pdf_info(title: str, run_set_hash: str) -> str:
    safe_title = title.replace("(", "[").replace(")", "]")
    return (
        r"\pdfinfo{" + "\n"
        f"/Title ({safe_title})\n"
        "/Author (Anonymous)\n"
        "/Subject (Anonymous HPCA 2027 evaluation artifact)\n"
        f"/Keywords (C2HLS, HPCA 2027, frozen run set {run_set_hash})\n"
        "/Creator (C2HLS frozen-evidence artifact generator)\n"
        "}\n"
    )


def _compile_standalone_pdf(
    tex_source: str, destination: Path, freeze_datetime: datetime_module.datetime
) -> None:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise ManifestError(
            "vector PDF generation requires matplotlib or a TeX installation with pdflatex/pgfplots"
        )
    with tempfile.TemporaryDirectory(prefix=".hpca-pdf-", dir=destination.parent) as raw_dir:
        workdir = Path(raw_dir)
        tex_path = workdir / "figure.tex"
        tex_path.write_text(tex_source, encoding="utf-8")
        environment = os.environ.copy()
        environment["SOURCE_DATE_EPOCH"] = str(int(freeze_datetime.timestamp()))
        environment["FORCE_SOURCE_DATE"] = "1"
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "figure.tex"],
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            check=False,
            env=environment,
        )
        pdf_path = workdir / "figure.pdf"
        if result.returncode != 0 or not pdf_path.is_file():
            excerpt = result.stdout[-3000:] if result.stdout else "no TeX output"
            raise ManifestError(f"pdflatex vector-PDF fallback failed:\n{excerpt}")
        os.replace(pdf_path, destination)


def _plot_coordinates(points: Sequence[tuple[Any, Any]]) -> str:
    return " ".join(f"({x},{y})" for x, y in points)


def _render_pdfs_latex(stage: Path, data: ValidatedInput, analysis: Mapping[str, Any]) -> None:
    rows = analysis["recovery_rows"]
    generated_points: list[tuple[int, float]] = []
    expert_points: list[tuple[int, float]] = []
    failure_points: list[tuple[float, float]] = []
    finite = [1.0]
    for index, row in enumerate(rows):
        baseline = row["baseline_cycles"]
        generated = row["generated_cycles"]
        expert = row["expert_cycles"]
        if baseline is not None and generated is not None:
            value = generated / baseline
            generated_points.append((index, value))
            finite.append(value)
        else:
            failure_points.append((index - 0.18, -1.0))
        if baseline is not None and expert is not None and row["pair_valid"]:
            value = expert / baseline
            expert_points.append((index, value))
            finite.append(value)
        else:
            failure_points.append((index + 0.18, -1.0))
    y_max = max(finite) * 1.18
    failure_points = [(x, y_max * 0.94) for x, _ in failure_points]
    ticks = ",".join(str(index) for index in range(len(rows)))
    labels = ",".join("{" + _latex_escape(row["kernel"]) + "}" for row in rows)
    recovery_tex = (
        r"\documentclass[tikz,border=2pt]{standalone}" + "\n"
        r"\usepackage[T1]{fontenc}" + "\n"
        r"\usepackage{lmodern}" + "\n"
        r"\usepackage{pgfplots}" + "\n"
        r"\usetikzlibrary{patterns}" + "\n"
        r"\pgfplotsset{compat=1.18}" + "\n"
        + _latex_pdf_info("Reference-blind expert recovery", data.run_set_hash)
        + r"\begin{document}" + "\n"
        + r"\begin{tikzpicture}" + "\n"
        + r"\begin{axis}[width=10.4in,height=3.35in,ybar,bar width=8pt," + "\n"
        + f"xmin=-0.6,xmax={len(rows) - 0.4},ymin=0,ymax={y_max:.6f},\n"
        + f"xtick={{{ticks}}},xticklabels={{{labels}}},\n"
        + r"x tick label style={rotate=25,anchor=east,font=\scriptsize}," + "\n"
        + r"ylabel={Executed RTL co-sim cycles / baseline}," + "\n"
        + f"title={{Reference-blind expert recovery (run set {data.run_set_hash[:12]}; $\\times$ is failure)}},\n"
        + r"grid=major,grid style={gray!25},legend style={draw=none,at={(0.98,0.98)},anchor=north east},legend columns=3]" + "\n"
        + r"\addplot[fill=black!65,draw=black] coordinates {"
        + _plot_coordinates([(x - 0.18, f"{y:.8f}") for x, y in generated_points])
        + r"};\addlegendentry{Generated}" + "\n"
        + r"\addplot[fill=white,draw=black,pattern=north east lines] coordinates {"
        + _plot_coordinates([(x + 0.18, f"{y:.8f}") for x, y in expert_points])
        + r"};\addlegendentry{Expert frontier}" + "\n"
        + r"\addplot[black,dashed,no marks] coordinates {"
        + _plot_coordinates([(-0.5, 1), (len(rows) - 0.5, 1)])
        + r"};\addlegendentry{Baseline}" + "\n"
        + r"\addplot[only marks,mark=x,mark size=3pt,very thick] coordinates {"
        + _plot_coordinates([(f"{x:.4f}", f"{y:.8f}") for x, y in failure_points])
        + r"};" + "\n"
        + r"\end{axis}\end{tikzpicture}" + "\n"
        + r"\end{document}" + "\n"
    )
    _compile_standalone_pdf(recovery_tex, stage / "recovery.pdf", _freeze_datetime(data))

    colors = ["black", "black!80", "black!65", "black!50", "black!35", "black!70", "black!45"]
    styles = ["solid", "dashed", "dashdotted", "dotted", "densely dashed", "loosely dotted", "densely dashdotted"]

    def curve_axis(
        axis_name: str,
        budget_type: str,
        xlabel: str,
        title: str,
        placement: str,
        legend: bool,
    ) -> str:
        output = [
            f"\\begin{{axis}}[name={axis_name},width=7.55in,height=4.15in,{placement}",
            f"xlabel={{{xlabel}}},ylabel={{Failure-aware CSynth QoR AUC}},",
            f"title={{{title}}},ymin=0,ymax=1.08,grid=major,grid style={{gray!25}},",
            "legend style={draw=none,font=\\scriptsize},legend columns=2]",
        ]
        for index, method in enumerate(data.methods):
            points = analysis["budget_curves"]["series"][budget_type][method.method_id]
            coords = _plot_coordinates(
                [(point["budget"], f"{point['qor_profile_auc']:.8f}") for point in points]
            )
            output.append(
                f"\\addplot[{colors[index % len(colors)]},{styles[index % len(styles)]},"
                f"mark={['*','square*','triangle*','diamond*','v','pentagon*','x'][index % 7]},mark size=1.8pt] coordinates {{{coords}}};"
            )
            if legend:
                output.append(f"\\addlegendentry{{{_latex_escape(method.display_name)}}}")
            for point in points:
                if point["failure_count"] == 0:
                    continue
                output.append(
                    f"\\node[font=\\tiny,anchor=south] at (axis cs:{point['budget']},{point['qor_profile_auc']:.8f}) "
                    f"{{F{point['failure_count']}}};"
                )
        output.append("\\end{axis}")
        return "\n".join(output)

    synthesis_axis = curve_axis(
        "synth", "synthesis_evaluations", "Selection synthesis evaluations",
        "(a) QoR versus synthesis budget; F labels mark nonzero failures", "", True,
    )
    token_axis = curve_axis(
        "token", "tokens", "Cumulative LLM tokens",
        "(b) QoR versus token budget; F labels mark nonzero failures",
        "at={(synth.east)},anchor=west,xshift=0.35in,", False,
    )
    method_ticks = ",".join(str(index) for index in range(len(data.methods)))
    method_labels = ",".join(
        "{" + _latex_escape(method.display_name) + "}" for method in data.methods
    )
    auc_coords = _plot_coordinates(
        [(index, f"{analysis['profile_auc'][method.method_id]:.8f}") for index, method in enumerate(data.methods)]
    )
    solve_coords = _plot_coordinates(
        [(index, f"{analysis['solve_rates'][method.method_id]:.8f}") for index, method in enumerate(data.methods)]
    )
    failure_nodes = "\n".join(
        f"\\node[font=\\tiny,anchor=south] at (axis cs:{index},{max(analysis['profile_auc'][method.method_id], analysis['solve_rates'][method.method_id]):.8f}) "
        f"{{F={len(data.profile_units) - round(analysis['solve_rates'][method.method_id] * len(data.profile_units))}}};"
        for index, method in enumerate(data.methods)
    )
    ablation_axis = (
        r"\begin{axis}[name=ablation,width=15.45in,height=4.35in," + "\n"
        r"at={(synth.south west)},anchor=north west,yshift=-0.55in," + "\n"
        + f"xtick={{{method_ticks}}},xticklabels={{{method_labels}}},\n"
        r"x tick label style={rotate=18,anchor=east,font=\scriptsize}," + "\n"
        r"ylabel={Failure-aware score},ymin=0,ymax=1.12,ybar,bar width=10pt," + "\n"
        r"title={(c) Matched-budget component ablation; failures remain in denominators}," + "\n"
        r"grid=major,grid style={gray!25},legend style={draw=none,at={(0.02,0.98)},anchor=north west},legend columns=2]" + "\n"
        r"\addplot[fill=black!65,draw=black] coordinates {" + auc_coords + r"};\addlegendentry{Executed-cosim profile AUC}" + "\n"
        r"\addplot[fill=white,draw=black,pattern=north east lines] coordinates {" + solve_coords + r"};\addlegendentry{Correct-solve rate}" + "\n"
        + failure_nodes + "\n"
        r"\end{axis}" + "\n"
    )
    budget_tex = (
        r"\documentclass[tikz,border=2pt]{standalone}" + "\n"
        r"\usepackage[T1]{fontenc}" + "\n"
        r"\usepackage{lmodern}" + "\n"
        r"\usepackage{pgfplots}" + "\n"
        r"\usetikzlibrary{patterns}" + "\n"
        r"\pgfplotsset{compat=1.18}" + "\n"
        + _latex_pdf_info("QoR, search budget, and component ablation", data.run_set_hash)
        + r"\begin{document}\begin{tikzpicture}" + "\n"
        + synthesis_axis + "\n" + token_axis + "\n" + ablation_axis
        + f"\\node[font=\\scriptsize,anchor=south west] at ([yshift=2pt]current bounding box.north west) {{Frozen run set {data.run_set_hash[:12]}}};\n"
        + r"\end{tikzpicture}\end{document}" + "\n"
    )
    _compile_standalone_pdf(budget_tex, stage / "budget.pdf", _freeze_datetime(data))


def _render_vector_pdfs(stage: Path, data: ValidatedInput, analysis: Mapping[str, Any]) -> str:
    modules = _try_matplotlib()
    if modules is not None:
        _render_pdfs_matplotlib(stage, data, analysis, modules)
        return "matplotlib-pdf-fonttype-42"
    _render_pdfs_latex(stage, data, analysis)
    return "pdflatex-pgfplots-fallback"


def _cell_provenance(data: ValidatedInput, analysis: Mapping[str, Any]) -> dict[str, list[str]]:
    cells: dict[str, list[str]] = {}
    for row in analysis["recovery_rows"]:
        kernel = row["kernel"]
        cells[f"recovery.{kernel}.baseline_cycles"] = [row["baseline_run_id"]]
        cells[f"recovery.{kernel}.generated_cycles"] = [row["generated_run_id"]]
        cells[f"recovery.{kernel}.expert_cycles"] = [row["expert_run_id"]]
        cells[f"recovery.{kernel}.recovery"] = [
            row["baseline_run_id"], row["generated_run_id"], row["expert_run_id"]
        ]
        cells[f"recovery.{kernel}.status"] = [row["generated_run_id"]]
    for method in data.methods:
        run_ids = [data.evaluations[key][method.method_id]["run_id"] for key in data.profile_units]
        for field in ("correct_solve", "profile_auc", "calls"):
            cells[f"ablation.{method.method_id}.{field}"] = run_ids
    all_profile_run_ids = [
        data.evaluations[key][method.method_id]["run_id"]
        for key in data.profile_units
        for method in data.methods
    ]
    for row in analysis["budget_curves"]["rows"]:
        prefix = f"budget.{row['budget_type']}.{row['budget']}.{row['method']}"
        cells[f"{prefix}.qor_profile_auc"] = all_profile_run_ids
        cells[f"{prefix}.failure_count"] = [
            data.evaluations[key][row["method"]]["run_id"] for key in data.profile_units
        ]
    for row in analysis["resource_rows"]:
        prefix = (
            f"resource.{row['role']}.{row['kernel']}.{row['seed'] or 'frontier'}."
            f"{row['method']}"
        )
        for field in (
            "fmax_mhz",
            *[f"{key}_{suffix}" for key in RESOURCE_KEYS for suffix in ("used", "capacity", "utilization")],
            "measurement_status",
            "failure_class",
        ):
            cells[f"{prefix}.{field}"] = [row["run_id"]]
    for identifier in [method.method_id for method in data.methods] + ["baseline", "expert"]:
        if identifier in {"baseline", "expert"}:
            run_ids = [
                data.frontiers[kernel][identifier]["run_id"]
                for kernel in data.expected_kernels
            ]
        else:
            run_ids = [record["run_id"] for record in _records_for_method(data, identifier)]
        for field in ("qor_records", "mean_fmax_mhz", *[f"mean_{key}_utilization" for key in RESOURCE_KEYS]):
            cells[f"resource_summary.{identifier}.{field}"] = run_ids
    return cells


def _claim_provenance(data: ValidatedInput, analysis: Mapping[str, Any]) -> dict[str, Any]:
    claim_methods = data.evidence["claim_methods"]

    def method_runs(*method_ids: str, units: Sequence[tuple[str, str]] | None = None) -> list[str]:
        if units is None:
            return [
                data.evaluations[(kernel, seed)][method_id]["run_id"]
                for kernel, seed, method_id in data.expected_cells
                if method_id in method_ids
            ]
        return [
            data.evaluations[key][method_id]["run_id"]
            for key in units
            for method_id in method_ids
        ]

    frontier_runs = [
        data.frontiers[kernel][role]["run_id"]
        for kernel in data.expected_kernels
        for role in ("baseline", "expert")
    ]
    all_method_runs = method_runs(*(method.method_id for method in data.methods))
    gate_evidence = _require_mapping(data.evidence.get("gate_evidence", {}), "gate_evidence")

    def artifact_for(gate: str) -> list[str]:
        spec = gate_evidence.get(gate)
        if isinstance(spec, dict) and isinstance(spec.get("artifact_id"), str):
            return [spec["artifact_id"]]
        return []

    specifications = {
        "headline_reference_blind_recovery": {
            "run_ids": frontier_runs
            + method_runs(claim_methods["primary"], units=list(data.headline_units.values())),
            "source_artifact_ids": artifact_for("transcript_leakage_audit")
            + artifact_for("fingerprint_consistency_audit"),
            "generated_outputs": [
                "result_macros.tex", "recovery_table.tex", "per_kernel_recovery.csv",
                "recovery.svg", "recovery.pdf", "performance_profiles.csv",
                "resource_table.tex", "resource_utilization_fmax.csv",
            ],
        },
        "workflow_beats_one_shot": {
            "run_ids": method_runs(
                claim_methods["one_shot"], claim_methods["dynamic_no_skill"],
                units=data.profile_units,
            ),
            "source_artifact_ids": artifact_for("candidate_validation_audit")
            + artifact_for("fingerprint_consistency_audit"),
            "generated_outputs": [
                "ablation_table.tex", "performance_profiles.csv", "performance_profile.svg",
            ],
        },
        "frozen_skill_transfer": {
            "run_ids": method_runs(
                claim_methods["dynamic_no_skill"], claim_methods["dynamic_frozen_skill"],
                units=data.bootstrap_units,
            ),
            "source_artifact_ids": artifact_for("frozen_skill_snapshot")
            + artifact_for("fingerprint_consistency_audit"),
            "generated_outputs": [
                "result_macros.tex", "ablation_table.tex", "paired_bootstrap.csv",
            ],
        },
        "compact_model_enablement": {
            "run_ids": method_runs(
                claim_methods["one_shot"], claim_methods["dynamic_no_skill"],
                units=data.profile_units,
            ),
            "source_artifact_ids": artifact_for("fingerprint_consistency_audit"),
            "generated_outputs": [
                "result_macros.tex", "ablation_table.tex", "performance_profiles.csv",
            ],
        },
        "cost_quality_tradeoff": {
            "run_ids": all_method_runs,
            "source_artifact_ids": artifact_for("complete_candidate_event_stream")
            + artifact_for("fingerprint_consistency_audit"),
            "generated_outputs": [
                "cost_summary.csv", "budget_curves.csv", "budget.pdf",
                "resource_table.tex", "resource_utilization_fmax.csv",
            ],
        },
        "post_route_and_board_validation": {
            "run_ids": [],
            "source_artifact_ids": artifact_for("post_route_validation"),
            "generated_outputs": [],
        },
    }
    return {
        claim_id: {
            "status": analysis["claims"][claim_id]["status"],
            **spec,
        }
        for claim_id, spec in specifications.items()
    }


def _write_bundle(stage: Path, data: ValidatedInput, analysis: Mapping[str, Any]) -> None:
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "result_macros.tex").write_text(_render_result_macros(data, analysis), encoding="utf-8")
    (stage / "recovery_table.tex").write_text(_render_recovery_table(data, analysis), encoding="utf-8")
    (stage / "ablation_table.tex").write_text(_render_ablation_table(data, analysis), encoding="utf-8")
    (stage / "resource_table.tex").write_text(
        _render_resource_table(data, analysis), encoding="utf-8"
    )
    (stage / "recovery.svg").write_text(_render_recovery_svg(data, analysis), encoding="utf-8")
    (stage / "performance_profile.svg").write_text(_render_profile_svg(data, analysis), encoding="utf-8")
    pdf_backend = _render_vector_pdfs(stage, data, analysis)
    _write_json(
        stage / "render_provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_set_sha256": data.run_set_hash,
            "pdf_backend": pdf_backend,
            "font_policy": (
                "embedded TrueType (PDF font type 42)"
                if pdf_backend.startswith("matplotlib")
                else "embedded TeX vector fonts"
            ),
        },
    )

    _write_csv(
        stage / "per_kernel_recovery.csv",
        [
            "kernel", "baseline_run_id", "generated_run_id", "expert_run_id",
            "baseline_cycles", "generated_cycles", "expert_cycles", "recovery",
            "pair_valid", "status",
        ],
        analysis["recovery_rows"],
    )
    resource_fields = [
        "kernel", "seed", "method", "role", "run_id", "terminal_status",
        "correctness_status", "synthesis_status", "resource_fit", "timing_met",
        "fmax_mhz", "provider_failure",
    ]
    for key in RESOURCE_KEYS:
        resource_fields.extend(
            [f"{key}_used", f"{key}_capacity", f"{key}_utilization"]
        )
    resource_fields.extend(["failure_class", "measurement_status"])
    _write_csv(
        stage / "resource_utilization_fmax.csv",
        resource_fields,
        analysis["resource_rows"],
    )
    profile_rows = []
    for index, tau in enumerate(analysis["profile_plot_taus"]):
        row: dict[str, Any] = {"tau": tau}
        for method in data.methods:
            row[method.method_id] = analysis["profile_plot"][method.method_id][index]
        profile_rows.append(row)
    _write_csv(
        stage / "performance_profiles.csv",
        ["tau"] + [method.method_id for method in data.methods],
        profile_rows,
    )
    failure_rows = []
    for method in data.methods:
        method_records = _records_for_method(data, method.method_id)
        solved = sum(_record_is_solved(record) for record in method_records)
        failure_rows.append(
            {"method": method.method_id, "outcome": "success", "count": solved, "total": len(method_records)}
        )
        for failure_class in sorted(analysis["failures"][method.method_id]):
            failure_rows.append(
                {
                    "method": method.method_id,
                    "outcome": failure_class,
                    "count": analysis["failures"][method.method_id][failure_class],
                    "total": len(method_records),
                }
            )
    for role in ("baseline", "expert"):
        records = [data.frontiers[kernel][role] for kernel in data.expected_kernels]
        solved = sum(_record_is_solved(record) for record in records)
        failure_rows.append(
            {"method": role, "outcome": "success", "count": solved, "total": len(records)}
        )
        counts = Counter(
            record["failure_class"] for record in records if not _record_is_solved(record)
        )
        for failure_class in sorted(counts):
            failure_rows.append(
                {"method": role, "outcome": failure_class, "count": counts[failure_class], "total": len(records)}
            )
    invalid_relation_count = sum(
        _record_is_solved(data.frontiers[kernel]["baseline"])
        and _record_is_solved(data.frontiers[kernel]["expert"])
        and not _valid_frontier_pair(data.frontiers[kernel])
        for kernel in data.expected_kernels
    )
    failure_rows.append(
        {
            "method": "baseline_expert_pair",
            "outcome": "invalid_expert_frontier",
            "count": invalid_relation_count,
            "total": len(data.expected_kernels),
        }
    )
    _write_csv(stage / "failure_accounting.csv", ["method", "outcome", "count", "total"], failure_rows)
    _write_csv(
        stage / "budget_curves.csv",
        [
            "budget_type", "budget", "method", "qor_profile_auc",
            "correct_solve_rate", "failure_count", "unit_count",
        ],
        analysis["budget_curves"]["rows"],
    )

    cost_rows = []
    for method in data.methods:
        records = _records_for_method(data, method.method_id)
        resource_summary = analysis["resource_summaries"][method.method_id]
        cost_rows.append(
            {
                "method": method.method_id,
                "units": len(records),
                "mean_tokens": statistics.fmean(record["tokens"] for record in records),
                "mean_llm_calls": statistics.fmean(record["llm_calls"] for record in records),
                "mean_synthesis_calls": statistics.fmean(record["synthesis_calls"] for record in records),
                "mean_selection_synthesis_evaluations": statistics.fmean(
                    record["selection_synthesis_evaluations"] for record in records
                ),
                "mean_wall_time_seconds": statistics.fmean(record["wall_time_seconds"] for record in records),
                "total_tokens": sum(record["tokens"] for record in records),
                "total_llm_calls": sum(record["llm_calls"] for record in records),
                "total_synthesis_calls": sum(record["synthesis_calls"] for record in records),
                "total_selection_synthesis_evaluations": sum(
                    record["selection_synthesis_evaluations"] for record in records
                ),
                "total_wall_time_seconds": sum(record["wall_time_seconds"] for record in records),
                **resource_summary,
            }
        )
    _write_csv(
        stage / "cost_summary.csv",
        [
            "method", "units", "mean_tokens", "mean_llm_calls", "mean_synthesis_calls",
            "mean_selection_synthesis_evaluations", "mean_wall_time_seconds", "total_tokens",
            "total_llm_calls", "total_synthesis_calls",
            "total_selection_synthesis_evaluations", "total_wall_time_seconds",
            "qor_records", "mean_fmax_mhz",
            *[f"mean_{key}_utilization" for key in RESOURCE_KEYS],
        ],
        cost_rows,
    )
    bootstrap_rows = [
        {
            "comparison": "dynamic_frozen_skill_minus_dynamic_no_skill",
            "metric": f"performance_profile_auc_tau_max_{analysis['profile_tau_max']:g}",
            **analysis["skill_bootstrap"],
        },
        {
            "comparison": "dynamic_no_skill_minus_one_shot",
            "metric": f"performance_profile_auc_tau_max_{analysis['profile_tau_max']:g}",
            **analysis["workflow_bootstrap"],
        },
        {
            "comparison": "dynamic_frozen_skill_minus_dynamic_no_skill",
            "metric": "correct_solve_indicator",
            **analysis["skill_solve_bootstrap"],
        },
        {
            "comparison": "dynamic_no_skill_minus_one_shot",
            "metric": "correct_solve_indicator",
            **analysis["workflow_solve_bootstrap"],
        },
    ]
    _write_csv(
        stage / "paired_bootstrap.csv",
        ["comparison", "metric", "n", "estimate", "ci_low", "ci_high", "confidence", "replicates", "seed"],
        bootstrap_rows,
    )
    _write_json(
        stage / "claim_decisions.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_set_sha256": data.run_set_hash,
            "claims": analysis["claims"],
        },
    )
    _write_json(
        stage / "cell_provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_set_sha256": data.run_set_hash,
            "cells": _cell_provenance(data, analysis),
        },
    )
    _write_json(
        stage / "claim_to_artifact_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_set_sha256": data.run_set_hash,
            "claims": _claim_provenance(data, analysis),
        },
    )

    output_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(stage.iterdir())
        if path.is_file()
    }
    _write_json(
        stage / "artifact_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "pdf_backend": pdf_backend,
            "run_set_sha256": data.run_set_hash,
            "evidence_manifest_sha256": data.evidence_hash,
            "source_artifacts": data.artifacts,
            "outputs": output_hashes,
        },
    )


def verify_bundle(bundle: Path) -> bool:
    bundle = bundle.resolve()
    manifest_path = bundle / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("bundle artifact manifest has unsupported schema_version")
    outputs = _require_mapping(manifest.get("outputs"), "artifact_manifest.outputs")
    expected_names = set(outputs) | {"artifact_manifest.json"}
    actual_names = {path.name for path in bundle.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise ManifestError(
            f"bundle file set mismatch; missing={sorted(expected_names - actual_names)}, extra={sorted(actual_names - expected_names)}"
        )
    for filename, expected_raw in outputs.items():
        expected = _require_sha256(
            expected_raw, f"artifact_manifest.outputs.{filename}"
        )
        actual = _sha256_file(bundle / filename)
        if actual != expected:
            raise ManifestError(
                f"bundle output hash mismatch for {filename}: expected {expected}, got {actual}"
            )
    if bundle.name != str(manifest.get("run_set_sha256")):
        raise ManifestError("bundle directory name does not match frozen run-set hash")
    return True


def generate_bundle(evidence_path: Path, output_root: Path) -> Path:
    data = load_and_validate(evidence_path)
    analysis = compute_analysis(data)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / data.run_set_hash
    if destination.exists():
        verify_bundle(destination)
        raise ManifestError(
            f"immutable output bundle already exists: {destination}; choose a new root or verify it"
        )
    stage_parent = Path(tempfile.mkdtemp(prefix=".hpca-artifacts-", dir=output_root))
    stage = stage_parent / data.run_set_hash
    try:
        _write_bundle(stage, data, analysis)
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage_parent, ignore_errors=True)
        raise
    shutil.rmtree(stage_parent, ignore_errors=True)
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path, help="frozen evidence manifest JSON")
    parser.add_argument("--output-root", required=True, type=Path, help="parent of immutable hash-named bundle")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        destination = generate_bundle(args.evidence, args.output_root)
    except ManifestError as exc:
        print(f"artifact generation refused: {exc}")
        return 2
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

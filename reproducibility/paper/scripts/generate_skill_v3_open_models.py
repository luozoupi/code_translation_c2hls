#!/usr/bin/env python3
"""Validate and render the 2026-07-31 Qwen/Gemma skill-v3 sweep.

The input is the frozen ``skill_v3_qwen_gemma_matrix_20260731`` directory.
This generator deliberately keeps single-step and multi-step results separate:
each bar is the upstream HLS reference divided by the lowest qualified
worst-case synthesis estimate among five policy configurations for one
execution mode.  A hollow diamond retains that mode's no-expertise setup.  In
particular,
the two Gemma workflows are not pooled because their raw records name different
serving revisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


BUNDLE_HASHES = {
    "README.md": "20f4e11379fbc141402478312682800db82e85f64636b9e594fea3d907495b41",
    "all_rows.csv": "33f75602fcd0e62ee3ee9e05bb08f11653411e6d96befa7427ea98682ff1c724",
    "all_rows.jsonl": "3105b36f8004889b6a562f4f2e86e1cc03c747a51563c05fa898dafe0ea50c4d",
    "manifest.json": "0858ff6940e2ac78a2ad04d3e7c6d110246230ee34330107703e94a43f1b769a",
    "setup_summary.csv": "a63469d50de1456ad319c6f2f6437632c1cff28b31d52c03f1f2fb18d8feebf9",
    "winners.csv": "84aab3cd7480839ea1feec5868c304c9f0332d05af3ad02041d7871edf8db55c",
    "worst_cycles_wide.csv": "862cc23ddb935e462563fc007f689a565b1e8c6217fd03b5e15af058650b1f63",
}

MODEL_SPECS = {
    "gemma4_31b": {
        "label": "Gemma-4-31B",
        "derived_export_revision": "google/gemma-4-31B-it@vllm-0.20.2-20260730",
        "wrapper_status": "complete",
        "wrapper_exit_code": 0,
    },
    "qwen3_6_27b": {
        "label": "Qwen3.6-27B",
        "derived_export_revision": "Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "wrapper_status": "failed",
        "wrapper_exit_code": 2,
    },
}
RAW_WORKFLOW_REVISIONS = {
    ("gemma4_31b", "flash"):
        "gemma-4-31b-it@cs-u-converge-30001-20260729",
    ("gemma4_31b", "multistep"):
        "google/gemma-4-31B-it@vllm-0.20.2-20260730",
    ("qwen3_6_27b", "flash"):
        "Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    ("qwen3_6_27b", "multistep"):
        "Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
}
MODEL_ORDER = ("gemma4_31b", "qwen3_6_27b")
STRATEGIES = ("flash", "multistep")
STRATEGY_ORDER = {value: index for index, value in enumerate(STRATEGIES)}
SKILL_MODES = (
    "skillless",
    "matched",
    "smart_best_fit",
    "smart_exhaustive",
    "all_positive",
)
SKILL_MODE_ORDER = {value: index for index, value in enumerate(SKILL_MODES)}

CANONICAL_28 = (
    "2mm",
    "3mm",
    "atax",
    "bicg",
    "cholesky",
    "correlation",
    "covariance",
    "doitgen",
    "durbin",
    "fdtd-2d",
    "floyd-warshall",
    "gemm",
    "gemver",
    "gesummv",
    "gramschmidt",
    "heat-3d",
    "jacobi-1d",
    "jacobi-2d",
    "lu",
    "ludcmp",
    "mvt",
    "nussinov",
    "seidel-2d",
    "symm",
    "syr2k",
    "syrk",
    "trisolv",
    "trmm",
)
EXPECTED_BENCHMARKS = {
    "hlsfactory_" + benchmark.replace("-", "_")
    for benchmark in CANONICAL_28
    if benchmark != "doitgen"
}

POLICY_COLORS = {
    "skillless": "#8A8A8A",
    "matched": "#56B4E9",
    "smart_best_fit": "#CC79A7",
    "smart_exhaustive": "#E69F00",
    "all_positive": "#009E73",
}
POLICY_LABELS = {
    "skillless": "No expertise",
    "matched": "Matched expertise",
    "smart_best_fit": "Focused routing",
    "smart_exhaustive": "Broad routing",
    "all_positive": "All positive expertise",
}

ALL_ROWS_FIELDS = (
    "schema_version",
    "model",
    "model_key",
    "model_revision",
    "strategy",
    "skill_mode",
    "setup_id",
    "benchmark",
    "controller_success",
    "csim_status",
    "csynth_status",
    "cosim_status",
    "valid_csim_csynth",
    "timing_met",
    "resource_fit",
    "comparison_valid",
    "selected_step",
    "selected_baseline",
    "controller_latency_cycles",
    "worst_case_latency_cycles",
    "latency_ns",
    "worst_case_latency_ns",
    "interval_cycles",
    "phase_b_controller_cycles",
    "phase_b_worst_case_cycles",
    "speedup_vs_phase_b_worst",
    "requested_clock_ns",
    "estimated_clock_ns",
    "slack_ns",
    "fmax_mhz",
    "bram",
    "dsp",
    "ff",
    "lut",
    "uram",
    "steps_attempted",
    "steps_success",
    "llm_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "selected_rendered_skill_count",
    "selected_rendered_skill_ids",
    "selected_declared_applied_skill_count",
    "selected_declared_applied_skill_ids",
    "selected_verified_applied_skill_count",
    "selected_verified_applied_skill_ids",
    "selected_synthesized_skill_count",
    "selected_synthesized_skill_ids",
    "candidate_max_rendered_skill_count",
    "reference_isolation_audit_passed",
    "reference_isolation_finding_count",
    "run_fingerprint_sha256",
    "summary_row_index",
    "source_summary_path",
    "source_result_path",
    "wrapper_state_status",
    "wrapper_state_exit_code",
    "wrapper_state_path",
    "reference_worst_case_cycles",
    "reference_source_kind",
    "reference_source_path",
    "reference_source_line",
    "speedup_vs_reference_worst",
    "paired_skillless_worst_case_cycles",
    "speedup_vs_paired_skillless_worst",
)

WINNER_FIELDS = (
    "model",
    "model_key",
    "benchmark",
    "winning_setup_id",
    "winning_strategy",
    "winning_skill_mode",
    "winning_selected_step",
    "winning_worst_case_cycles",
    "winning_controller_cycles",
    "reference_worst_case_cycles",
    "speedup_vs_reference_worst",
    "selected_rendered_skill_count",
    "selected_verified_applied_skill_count",
    "tie_count",
    "candidate_count",
    "source_result_path",
)

SUMMARY_FIELDS = (
    "setup_id",
    "model",
    "model_key",
    "strategy",
    "skill_mode",
    "completed",
    "valid_csim_csynth_timing_resource",
    "baseline_selected",
    "best_of_10_wins",
    "geomean_speedup_vs_reference_worst",
    "geomean_speedup_vs_paired_skillless_worst",
    "wins_vs_paired_skillless_over_1pct",
    "ties_vs_paired_skillless_within_1pct",
    "losses_vs_paired_skillless_over_1pct",
    "median_selected_rendered_skill_count",
    "reference_isolation_audit_failures",
    "wrapper_state_status",
    "wrapper_state_exit_code",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_bundle(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in BUNDLE_HASHES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[name] = sha256(path)
        if observed[name] != expected:
            raise ValueError(
                f"{name}: SHA-256 mismatch; expected {expected}, "
                f"observed {observed[name]}"
            )
    return observed


def read_csv(path: Path, fields: tuple[str, ...] | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if fields is not None and tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"{path.name}: unexpected columns {reader.fieldnames}")
        return list(reader)


def parse_bool(value: Any, label: str) -> bool:
    if value is True or value == "true":
        return True
    if value is False or value == "false":
        return False
    raise ValueError(f"{label}: expected a boolean, observed {value!r}")


def finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or value is None or value == "":
        raise ValueError(f"{label}: expected a number, observed {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: expected a number, observed {value!r}") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label}: invalid number {result!r}")
    return result


def exact_int(value: Any, label: str) -> int:
    result = finite_float(value, label)
    integer = int(result)
    if result != integer:
        raise ValueError(f"{label}: expected an integer, observed {value!r}")
    return integer


def geometric_mean(values: Iterable[float]) -> float:
    sequence = list(values)
    if not sequence or any(value <= 0 or not math.isfinite(value) for value in sequence):
        raise ValueError(f"invalid geometric-mean input: {sequence}")
    return math.exp(math.fsum(math.log(value) for value in sequence) / len(sequence))


def close(left: Any, right: Any, label: str, *, rel_tol: float = 2e-10) -> None:
    left_number = finite_float(left, f"{label} left")
    right_number = finite_float(right, f"{label} right")
    if not math.isclose(left_number, right_number, rel_tol=rel_tol, abs_tol=1e-9):
        raise ValueError(f"{label}: {left_number} != {right_number}")


def validate_source_manifest(
    root: Path, bundle_hashes: dict[str, str]
) -> dict[str, Any]:
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != "c2hls.skill-v3-sweep-table-manifest.v1":
        raise ValueError("unexpected source-manifest schema")
    if payload.get("toolchain") != {
        "vitis_version": "2023.2",
        "part": "xcu280-fsvh2892-2L-e",
        "clock_ns": 3.33,
        "cosim": False,
    }:
        raise ValueError("unexpected source-manifest toolchain")
    if payload.get("skill_catalog") != {
        "id": "skill_v3_no_rmw_fc27133",
        "catalog_count": 123,
        "positive_count": 84,
        "avoid_count": 39,
    }:
        raise ValueError("unexpected skill catalog")
    expected_invariants = {
        "row_count": 540,
        "benchmark_count": 27,
        "setup_count": 20,
        "winner_count": 54,
        "valid_comparison_count": 540,
        "cosim_run_count": 0,
        "reference_count": 27,
        "website_schema_reference_count": 25,
        "cache_only_reference_count": 2,
        "reference_isolation_audit_failures": 540,
    }
    if payload.get("invariants") != expected_invariants:
        raise ValueError("unexpected source-manifest invariants")

    expected_inputs = {
        (spec["label"], key, spec["derived_export_revision"], strategy)
        for key, spec in MODEL_SPECS.items()
        for strategy in STRATEGIES
    }
    observed_inputs = {
        (
            item.get("model"),
            item.get("model_key"),
            item.get("model_revision"),
            item.get("strategy"),
        )
        for item in payload.get("inputs", [])
    }
    if observed_inputs != expected_inputs or len(payload.get("inputs", [])) != 4:
        raise ValueError("unexpected model inputs in source manifest")
    for item in payload["inputs"]:
        for field in ("summary_sha256", "state_sha256"):
            value = item.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"source manifest has invalid {field}")

    outputs = payload.get("outputs")
    expected_output_names = set(BUNDLE_HASHES) - {"manifest.json"}
    if not isinstance(outputs, dict) or set(outputs) != expected_output_names:
        raise ValueError("source manifest has an unexpected output set")
    for name in expected_output_names:
        record = outputs[name]
        path = root / name
        if record.get("sha256") != bundle_hashes[name]:
            raise ValueError(f"source manifest hash mismatch for {name}")
        if exact_int(record.get("bytes"), f"{name} bytes") != path.stat().st_size:
            raise ValueError(f"source manifest size mismatch for {name}")
    return payload


def load_rows(root: Path) -> list[dict[str, str]]:
    rows = read_csv(root / "all_rows.csv", ALL_ROWS_FIELDS)
    if len(rows) != 540:
        raise ValueError(f"all_rows.csv: expected 540 rows, observed {len(rows)}")
    keys: set[tuple[str, str, str, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        prefix = f"all_rows.csv:{row_number}"
        model_key = row["model_key"]
        if model_key not in MODEL_SPECS:
            raise ValueError(f"{prefix}: unexpected model key {model_key!r}")
        spec = MODEL_SPECS[model_key]
        if (
            row["model"] != spec["label"]
            or row["model_revision"] != spec["derived_export_revision"]
        ):
            raise ValueError(f"{prefix}: model identity mismatch")
        strategy = row["strategy"]
        mode = row["skill_mode"]
        benchmark = row["benchmark"]
        if strategy not in STRATEGIES or mode not in SKILL_MODES:
            raise ValueError(f"{prefix}: unexpected setup")
        if benchmark not in EXPECTED_BENCHMARKS:
            raise ValueError(f"{prefix}: unexpected benchmark {benchmark!r}")
        key = (model_key, strategy, mode, benchmark)
        if key in keys:
            raise ValueError(f"{prefix}: duplicate matrix cell {key}")
        keys.add(key)
        expected_setup = f"{model_key}__{strategy}__{mode}"
        if row["setup_id"] != expected_setup:
            raise ValueError(f"{prefix}: setup ID mismatch")
        if row["schema_version"] != "c2hls.skill-v3-sweep-table.v1":
            raise ValueError(f"{prefix}: row schema mismatch")
        if not all(
            parse_bool(row[field], f"{prefix}:{field}")
            for field in (
                "controller_success",
                "valid_csim_csynth",
                "timing_met",
                "resource_fit",
                "comparison_valid",
            )
        ):
            raise ValueError(f"{prefix}: selected result fails an eligibility gate")
        if not (
            row["csim_status"] == "pass"
            and row["csynth_status"] == "pass"
            and row["cosim_status"] == "not_run"
        ):
            raise ValueError(f"{prefix}: unexpected tool status")
        if parse_bool(
            row["reference_isolation_audit_passed"],
            f"{prefix}:reference_isolation_audit_passed",
        ):
            raise ValueError(f"{prefix}: expected the advisory audit flag")
        if exact_int(
            row["reference_isolation_finding_count"],
            f"{prefix}:reference_isolation_finding_count",
        ) <= 0:
            raise ValueError(f"{prefix}: audit flag has no findings")
        if row["run_fingerprint_sha256"]:
            raise ValueError(f"{prefix}: expected a missing run fingerprint")
        if not math.isclose(
            finite_float(row["requested_clock_ns"], f"{prefix}:clock"),
            3.33,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{prefix}: unexpected target period")
        if (
            row["wrapper_state_status"] != spec["wrapper_status"]
            or exact_int(row["wrapper_state_exit_code"], f"{prefix}:exit code")
            != spec["wrapper_exit_code"]
        ):
            raise ValueError(f"{prefix}: unexpected wrapper status")
        cycles = finite_float(
            row["worst_case_latency_cycles"], f"{prefix}:selected cycles", positive=True
        )
        reference = finite_float(
            row["reference_worst_case_cycles"], f"{prefix}:reference", positive=True
        )
        paired = finite_float(
            row["paired_skillless_worst_case_cycles"],
            f"{prefix}:paired no-skills",
            positive=True,
        )
        close(
            row["speedup_vs_reference_worst"],
            reference / cycles,
            f"{prefix}:reference ratio",
        )
        close(
            row["speedup_vs_paired_skillless_worst"],
            paired / cycles,
            f"{prefix}:no-skills ratio",
        )
        for resource in ("bram", "dsp", "ff", "lut", "uram"):
            if finite_float(row[resource], f"{prefix}:{resource}") < 0:
                raise ValueError(f"{prefix}: negative resource usage")

    expected_keys = {
        (model, strategy, mode, benchmark)
        for model in MODEL_SPECS
        for strategy in STRATEGIES
        for mode in SKILL_MODES
        for benchmark in EXPECTED_BENCHMARKS
    }
    if keys != expected_keys:
        raise ValueError("all_rows.csv does not contain the exact 2x2x5x27 matrix")

    json_rows = [
        json.loads(line)
        for line in (root / "all_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(json_rows) != 540 or any(set(row) != set(ALL_ROWS_FIELDS) for row in json_rows):
        raise ValueError("all_rows.jsonl has unexpected rows or fields")
    json_index = {
        (row["model_key"], row["strategy"], row["skill_mode"], row["benchmark"]): row
        for row in json_rows
    }
    if set(json_index) != keys:
        raise ValueError("CSV/JSONL matrix keys disagree")
    for row in rows:
        key = (row["model_key"], row["strategy"], row["skill_mode"], row["benchmark"])
        peer = json_index[key]
        for field in (
            "model",
            "model_revision",
            "setup_id",
            "csim_status",
            "csynth_status",
            "cosim_status",
            "wrapper_state_status",
            "reference_source_kind",
        ):
            if row[field] != str(peer[field]):
                raise ValueError(f"CSV/JSONL mismatch for {key}/{field}")
        for field in (
            "worst_case_latency_cycles",
            "reference_worst_case_cycles",
            "speedup_vs_reference_worst",
        ):
            close(row[field], peer[field], f"CSV/JSONL {key}/{field}")
        if peer["run_fingerprint_sha256"] is not None:
            raise ValueError(f"JSONL {key}: expected null run fingerprint")
    return rows


def validate_raw_sources(
    source_manifest: dict[str, Any], rows: list[dict[str, str]]
) -> dict[str, Any]:
    """Audit raw result provenance omitted or flattened by the table export."""

    derived = {
        (row["model_key"], row["strategy"], row["skill_mode"], row["benchmark"]): row
        for row in rows
    }
    workflows: dict[str, dict[str, Any]] = {}
    raw_fingerprint_count = 0
    raw_independent_golden_pass_count = 0
    derived_revision_mismatch_count = 0
    observed_workflows: set[tuple[str, str]] = set()

    for item in source_manifest["inputs"]:
        model_key = item["model_key"]
        strategy = item["strategy"]
        workflow = (model_key, strategy)
        if workflow in observed_workflows or workflow not in RAW_WORKFLOW_REVISIONS:
            raise ValueError(f"unexpected or duplicate raw workflow {workflow}")
        observed_workflows.add(workflow)
        summary_path = Path(item["summary_path"])
        state_path = Path(item["state_path"])
        for field, source_path in (
            ("summary_sha256", summary_path),
            ("state_sha256", state_path),
        ):
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if sha256(source_path) != item[field]:
                raise ValueError(f"raw source hash mismatch: {source_path}")

        state = json.loads(state_path.read_text(encoding="utf-8"))
        spec = MODEL_SPECS[model_key]
        if (
            state.get("status") != spec["wrapper_status"]
            or exact_int(state.get("exit_code"), f"{workflow}:state exit code")
            != spec["wrapper_exit_code"]
        ):
            raise ValueError(f"{workflow}: raw wrapper state mismatch")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_rows = summary.get("rows")
        if not isinstance(source_rows, list) or len(source_rows) != 135:
            raise ValueError(f"{workflow}: expected 135 source-summary rows")
        source_keys: set[tuple[str, str]] = set()
        endpoint_hashes: set[str] = set()
        workflow_fingerprints = 0
        workflow_golden_passes = 0
        for row_number, source_row in enumerate(source_rows):
            benchmark = str(source_row.get("bench") or "").replace("-", "_")
            mode = str(source_row.get("skill_mode") or "")
            if benchmark not in EXPECTED_BENCHMARKS or mode not in SKILL_MODES:
                raise ValueError(f"{workflow}: malformed source row {row_number}")
            source_key = (mode, benchmark)
            if source_key in source_keys:
                raise ValueError(f"{workflow}: duplicate source row {source_key}")
            source_keys.add(source_key)
            current = source_row.get("current")
            if not isinstance(current, dict) or current.get("success") is not True:
                raise ValueError(f"{workflow}/{source_key}: unsuccessful source row")
            result_path = Path(str(current.get("json") or ""))
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            run = result.get("run")
            if not isinstance(run, dict):
                raise ValueError(f"{workflow}/{source_key}: missing run record")
            revision = run.get("model_revision")
            if not (
                isinstance(revision, dict)
                and revision.get("value") == RAW_WORKFLOW_REVISIONS[workflow]
                and revision.get("resolved") is True
            ):
                raise ValueError(f"{workflow}/{source_key}: raw revision mismatch")

            fingerprint = result.get("run_fingerprint")
            raw_sha = fingerprint.get("sha256") if isinstance(fingerprint, dict) else None
            summary_sha = current.get("run_fingerprint_sha256")
            if not (
                isinstance(raw_sha, str)
                and len(raw_sha) == 64
                and summary_sha == raw_sha
            ):
                raise ValueError(f"{workflow}/{source_key}: raw fingerprint mismatch")
            workflow_fingerprints += 1
            raw_fingerprint_count += 1

            evaluation = result.get("evaluation_status")
            feasibility = result.get("candidate_feasibility")
            independent_golden = result.get("independent_golden")
            if not (
                isinstance(evaluation, dict)
                and evaluation.get("correctness_status") == "passed"
                and evaluation.get("synthesis_status") == "passed"
                and evaluation.get("cosim_execution_status") == "not_run"
                and isinstance(feasibility, dict)
                and feasibility.get("feasible") is True
                and feasibility.get("timing_met") is True
                and feasibility.get("resource_fit") is True
            ):
                raise ValueError(f"{workflow}/{source_key}: raw eligibility mismatch")
            if not (
                isinstance(independent_golden, dict)
                and independent_golden.get("status") == "passed"
            ):
                raise ValueError(f"{workflow}/{source_key}: golden status mismatch")
            workflow_golden_passes += 1
            raw_independent_golden_pass_count += 1

            payload = fingerprint.get("payload")
            model_payload = payload.get("model") if isinstance(payload, dict) else None
            endpoint = (
                model_payload.get("endpoint")
                if isinstance(model_payload, dict) else None
            )
            endpoint_sha = (
                endpoint.get("endpoint_sha256") if isinstance(endpoint, dict) else None
            )
            if not isinstance(endpoint_sha, str) or len(endpoint_sha) != 64:
                raise ValueError(f"{workflow}/{source_key}: endpoint hash missing")
            endpoint_hashes.add(endpoint_sha)

            exported = derived[(model_key, strategy, mode, benchmark)]
            if exported["source_summary_path"] != str(summary_path.resolve()):
                raise ValueError(f"{workflow}/{source_key}: summary path mismatch")
            if exported["source_result_path"] != str(result_path.resolve()):
                raise ValueError(f"{workflow}/{source_key}: result path mismatch")
            if exported["run_fingerprint_sha256"]:
                raise ValueError(f"{workflow}/{source_key}: export fingerprint is not blank")
            if exported["model_revision"] != RAW_WORKFLOW_REVISIONS[workflow]:
                derived_revision_mismatch_count += 1

        expected_source_keys = {
            (mode, benchmark)
            for mode in SKILL_MODES
            for benchmark in EXPECTED_BENCHMARKS
        }
        if source_keys != expected_source_keys or len(endpoint_hashes) != 1:
            raise ValueError(f"{workflow}: incomplete raw source matrix")
        workflow_name = f"{model_key}__{strategy}"
        workflows[workflow_name] = {
            "model": MODEL_SPECS[model_key]["label"],
            "model_key": model_key,
            "strategy": strategy,
            "raw_recorded_revision": RAW_WORKFLOW_REVISIONS[workflow],
            "derived_export_revision": MODEL_SPECS[model_key]["derived_export_revision"],
            "revision_source": "C2HLS_MODEL_REVISION",
            "externally_authenticated_revision": False,
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": item["summary_sha256"],
            "state_path": str(state_path.resolve()),
            "state_sha256": item["state_sha256"],
            "row_count": len(source_rows),
            "matching_raw_fingerprint_count": workflow_fingerprints,
            "raw_independent_golden_pass_count": workflow_golden_passes,
            "endpoint_sha256": next(iter(endpoint_hashes)),
        }

    if observed_workflows != set(RAW_WORKFLOW_REVISIONS):
        raise ValueError("raw workflow coverage mismatch")
    if raw_fingerprint_count != 540 or raw_independent_golden_pass_count != 540:
        raise ValueError("raw provenance coverage mismatch")
    if derived_revision_mismatch_count != 135:
        raise ValueError(
            "expected exactly the 135 Gemma single-step rows to have a "
            "flattened revision mismatch"
        )
    return {
        "workflows": workflows,
        "raw_fingerprint_count": raw_fingerprint_count,
        "raw_independent_golden_pass_count": raw_independent_golden_pass_count,
        "derived_export_blank_fingerprint_count": 540,
        "derived_export_revision_mismatch_count": derived_revision_mismatch_count,
    }


def validate_references(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by_benchmark: dict[str, dict[str, Any]] = {}
    for benchmark in EXPECTED_BENCHMARKS:
        relevant = [row for row in rows if row["benchmark"] == benchmark]
        signatures = {
            (
                row["reference_worst_case_cycles"],
                row["reference_source_kind"],
                row["reference_source_path"],
                row["reference_source_line"],
            )
            for row in relevant
        }
        if len(relevant) != 20 or len(signatures) != 1:
            raise ValueError(f"{benchmark}: inconsistent reference across setups")
        cycles, source_kind, source_path, source_line = next(iter(signatures))
        by_benchmark[benchmark] = {
            "cycles": finite_float(cycles, f"{benchmark}:reference", positive=True),
            "source_kind": source_kind,
            "source_path": source_path,
            "source_line": source_line,
        }
    kinds = Counter(record["source_kind"] for record in by_benchmark.values())
    if kinds != Counter({"website_schema": 25, "reference_validation_cache_not_schema": 2}):
        raise ValueError(f"unexpected reference-source split: {kinds}")
    cache = {
        benchmark for benchmark, record in by_benchmark.items()
        if record["source_kind"] == "reference_validation_cache_not_schema"
    }
    if cache != {"hlsfactory_heat_3d", "hlsfactory_seidel_2d"}:
        raise ValueError(f"unexpected cache-only references: {cache}")
    return by_benchmark


def rank_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            finite_float(row["worst_case_latency_cycles"], "candidate cycles"),
            STRATEGY_ORDER[row["strategy"]],
            SKILL_MODE_ORDER[row["skill_mode"]],
        ),
    )


def select_rows(
    rows: list[dict[str, str]], references: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_key"], row["strategy"], row["benchmark"])].append(row)
    snapshot: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for model_key in MODEL_ORDER:
        for strategy in STRATEGIES:
            workflow_rows: list[dict[str, Any]] = []
            for benchmark in sorted(EXPECTED_BENCHMARKS):
                candidates = rank_candidates(groups[(model_key, strategy, benchmark)])
                no_skills = [
                    row for row in candidates if row["skill_mode"] == "skillless"
                ]
                if len(candidates) != 5 or len(no_skills) != 1:
                    raise ValueError(
                        f"{model_key}/{strategy}/{benchmark}: incomplete setup set"
                    )
                selected = candidates[0]
                off = no_skills[0]
                cycles = finite_float(
                    selected["worst_case_latency_cycles"], "selected", positive=True
                )
                off_cycles = finite_float(
                    off["worst_case_latency_cycles"], "no skills", positive=True
                )
                reference = references[benchmark]["cycles"]
                tie_count = sum(
                    finite_float(row["worst_case_latency_cycles"], "tie cycles")
                    == cycles
                    for row in candidates
                )
                record = {
                    "model": MODEL_SPECS[model_key]["label"],
                    "model_key": model_key,
                    "workflow": strategy,
                    "raw_recorded_model_revision": RAW_WORKFLOW_REVISIONS[
                        (model_key, strategy)
                    ],
                    "derived_export_model_revision": MODEL_SPECS[model_key][
                        "derived_export_revision"
                    ],
                    "benchmark": benchmark,
                    "reference_worst_case_cycles": reference,
                    "reference_source_kind": references[benchmark]["source_kind"],
                    "selected_setup_id": selected["setup_id"],
                    "selected_skill_mode": selected["skill_mode"],
                    "selected_step": selected["selected_step"],
                    "selected_baseline": parse_bool(
                        selected["selected_baseline"], "selected baseline"
                    ),
                    "selected_worst_case_cycles": cycles,
                    "reference_over_selected": reference / cycles,
                    "no_skills_setup_id": off["setup_id"],
                    "no_skills_worst_case_cycles": off_cycles,
                    "reference_over_no_skills": reference / off_cycles,
                    "no_skills_over_selected": off_cycles / cycles,
                    "selected_rendered_skill_count": exact_int(
                        selected["selected_rendered_skill_count"], "rendered skills"
                    ),
                    "selected_verified_applied_skill_count": exact_int(
                        selected["selected_verified_applied_skill_count"],
                        "verified skills",
                    ),
                    "tie_count": tie_count,
                    "candidate_count": len(candidates),
                    "selected_source_result_path": selected["source_result_path"],
                    "no_skills_source_result_path": off["source_result_path"],
                }
                snapshot.append(record)
                workflow_rows.append(record)
            selected_policies = Counter(
                row["selected_skill_mode"] for row in workflow_rows
            )
            summary_key = f"{model_key}__{strategy}"
            summaries[summary_key] = {
                "model": MODEL_SPECS[model_key]["label"],
                "model_key": model_key,
                "workflow": strategy,
                "raw_recorded_model_revision": RAW_WORKFLOW_REVISIONS[
                    (model_key, strategy)
                ],
                "externally_authenticated_revision": False,
                "kernel_count": len(workflow_rows),
                "posthoc_best_of_setup_count": 5,
                "reference_over_selected_geomean": geometric_mean(
                    row["reference_over_selected"] for row in workflow_rows
                ),
                "reference_over_no_skills_geomean": geometric_mean(
                    row["reference_over_no_skills"] for row in workflow_rows
                ),
                "no_skills_over_selected_geomean": geometric_mean(
                    row["no_skills_over_selected"] for row in workflow_rows
                ),
                "expertise_policy_setup_selections": sum(
                    count
                    for mode, count in selected_policies.items()
                    if mode != "skillless"
                ),
                "no_expertise_setup_selections": selected_policies["skillless"],
                "selected_baseline_fallbacks": sum(
                    row["selected_baseline"] for row in workflow_rows
                ),
                "selected_policy_counts": dict(sorted(selected_policies.items())),
            }
    return snapshot, summaries


def validate_winners(root: Path, matrix_rows: list[dict[str, str]]) -> None:
    """Cross-check the bundled pooled-winner companion without using it."""

    rows = read_csv(root / "winners.csv", WINNER_FIELDS)
    if len(rows) != 54:
        raise ValueError("winners.csv: expected 54 rows")
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for matrix_row in matrix_rows:
        groups[(matrix_row["model_key"], matrix_row["benchmark"])].append(matrix_row)
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for key, candidates in groups.items():
        ranked = rank_candidates(candidates)
        if len(ranked) != 10:
            raise ValueError(f"winners.csv:{key}: expected ten candidates")
        selected = ranked[0]
        cycles = finite_float(
            selected["worst_case_latency_cycles"], f"winners.csv:{key}:cycles"
        )
        expected[key] = {
            "selected": selected,
            "tie_count": sum(
                finite_float(row["worst_case_latency_cycles"], "winner tie")
                == cycles
                for row in ranked
            ),
        }
    observed: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["model_key"], row["benchmark"])
        if key in observed or key not in expected:
            raise ValueError(f"winners.csv: duplicate or unexpected row {key}")
        observed.add(key)
        peer = expected[key]
        selected = peer["selected"]
        exact = {
            "model": selected["model"],
            "winning_setup_id": selected["setup_id"],
            "winning_strategy": selected["strategy"],
            "winning_skill_mode": selected["skill_mode"],
            "winning_selected_step": selected["selected_step"],
            "source_result_path": selected["source_result_path"],
        }
        for field, value in exact.items():
            if row[field] != str(value):
                raise ValueError(f"winners.csv: {key}/{field} mismatch")
        for field, value in {
            "winning_worst_case_cycles": selected["worst_case_latency_cycles"],
            "reference_worst_case_cycles": selected["reference_worst_case_cycles"],
            "speedup_vs_reference_worst": selected["speedup_vs_reference_worst"],
        }.items():
            close(row[field], value, f"winners.csv:{key}/{field}")
        if exact_int(row["candidate_count"], f"winners.csv:{key}/candidate_count") != 10:
            raise ValueError(f"winners.csv: {key} has a non-ten candidate set")
        if exact_int(row["tie_count"], f"winners.csv:{key}/tie_count") != peer["tie_count"]:
            raise ValueError(f"winners.csv: {key} tie count mismatch")
    if observed != set(expected):
        raise ValueError("winners.csv coverage mismatch")


def validate_setup_summary(
    root: Path, rows: list[dict[str, str]]
) -> None:
    summaries = read_csv(root / "setup_summary.csv", SUMMARY_FIELDS)
    if len(summaries) != 20:
        raise ValueError("setup_summary.csv: expected 20 rows")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["setup_id"]].append(row)
    pooled_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pooled_groups[(row["model_key"], row["benchmark"])].append(row)
    winner_counts = Counter(
        rank_candidates(candidates)[0]["setup_id"]
        for candidates in pooled_groups.values()
    )
    observed: set[str] = set()
    for summary in summaries:
        setup_id = summary["setup_id"]
        if setup_id in observed or setup_id not in grouped:
            raise ValueError(f"setup_summary.csv: duplicate or unknown {setup_id}")
        observed.add(setup_id)
        setup_rows = grouped[setup_id]
        first = setup_rows[0]
        for field in ("model", "model_key", "strategy", "skill_mode"):
            if summary[field] != first[field]:
                raise ValueError(f"setup_summary.csv:{setup_id}/{field} mismatch")
        expected_integers = {
            "completed": 27,
            "valid_csim_csynth_timing_resource": 27,
            "baseline_selected": sum(
                parse_bool(row["selected_baseline"], f"{setup_id}:selected_baseline")
                for row in setup_rows
            ),
            "best_of_10_wins": winner_counts[setup_id],
            "reference_isolation_audit_failures": 27,
        }
        speedups_reference = [
            finite_float(row["speedup_vs_reference_worst"], f"{setup_id}:reference")
            for row in setup_rows
        ]
        speedups_no_skills = [
            finite_float(row["speedup_vs_paired_skillless_worst"], f"{setup_id}:no skills")
            for row in setup_rows
        ]
        expected_integers.update(
            {
                "wins_vs_paired_skillless_over_1pct": sum(
                    value > 1.01 for value in speedups_no_skills
                ),
                "ties_vs_paired_skillless_within_1pct": sum(
                    1 / 1.01 <= value <= 1.01 for value in speedups_no_skills
                ),
                "losses_vs_paired_skillless_over_1pct": sum(
                    value < 1 / 1.01 for value in speedups_no_skills
                ),
            }
        )
        for field, value in expected_integers.items():
            if exact_int(summary[field], f"setup_summary.csv:{setup_id}/{field}") != value:
                raise ValueError(f"setup_summary.csv:{setup_id}/{field} mismatch")
        close(
            summary["geomean_speedup_vs_reference_worst"],
            geometric_mean(speedups_reference),
            f"setup_summary.csv:{setup_id}/reference geomean",
        )
        close(
            summary["geomean_speedup_vs_paired_skillless_worst"],
            geometric_mean(speedups_no_skills),
            f"setup_summary.csv:{setup_id}/no-skills geomean",
        )
        rendered = [exact_int(row["selected_rendered_skill_count"], "rendered") for row in setup_rows]
        close(
            summary["median_selected_rendered_skill_count"],
            statistics.median(rendered),
            f"setup_summary.csv:{setup_id}/median rendered",
        )
        spec = MODEL_SPECS[first["model_key"]]
        if (
            summary["wrapper_state_status"] != spec["wrapper_status"]
            or exact_int(summary["wrapper_state_exit_code"], f"{setup_id}:wrapper code")
            != spec["wrapper_exit_code"]
        ):
            raise ValueError(f"setup_summary.csv:{setup_id} wrapper mismatch")
    if observed != set(grouped):
        raise ValueError("setup_summary.csv coverage mismatch")


def validate_wide(root: Path, rows: list[dict[str, str]]) -> None:
    wide = read_csv(root / "worst_cycles_wide.csv")
    setup_ids = sorted(
        {row["setup_id"] for row in rows},
        key=lambda setup: (
            setup.split("__")[0],
            STRATEGY_ORDER[setup.split("__")[1]],
            SKILL_MODE_ORDER[setup.split("__")[2]],
        ),
    )
    expected_fields = (
        "benchmark",
        "reference_worst_case_cycles",
        "reference_source_kind",
        *setup_ids,
    )
    with (root / "worst_cycles_wide.csv").open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError("worst_cycles_wide.csv columns mismatch")
    if len(wide) != 27 or {row["benchmark"] for row in wide} != EXPECTED_BENCHMARKS:
        raise ValueError("worst_cycles_wide.csv benchmark coverage mismatch")
    lookup = {(row["benchmark"], row["setup_id"]): row for row in rows}
    for wide_row in wide:
        benchmark = wide_row["benchmark"]
        sample = lookup[(benchmark, setup_ids[0])]
        close(
            wide_row["reference_worst_case_cycles"],
            sample["reference_worst_case_cycles"],
            f"wide:{benchmark}/reference",
        )
        if wide_row["reference_source_kind"] != sample["reference_source_kind"]:
            raise ValueError(f"wide:{benchmark}/reference source mismatch")
        for setup_id in setup_ids:
            close(
                wide_row[setup_id],
                lookup[(benchmark, setup_id)]["worst_case_latency_cycles"],
                f"wide:{benchmark}/{setup_id}",
            )


def write_snapshot(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = tuple(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_macros(
    path: Path,
    source_manifest_hash: str,
    matrix_rows: list[dict[str, str]],
    snapshot: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "% AUTO-GENERATED by scripts/generate_skill_v3_open_models.py.",
        f"\\providecommand{{\\SkillVThreeOpenEvidenceHash}}{{{source_manifest_hash[:12]}}}",
        "\\providecommand{\\SkillVThreeOpenCellCount}{540}",
        "\\providecommand{\\SkillVThreeOpenGemmaCellCount}{270}",
        "\\providecommand{\\SkillVThreeOpenQwenCellCount}{270}",
        "\\providecommand{\\SkillVThreeOpenKernelCount}{27}",
        "\\providecommand{\\SkillVThreeOpenWorkflowCount}{4}",
        "\\providecommand{\\SkillVThreeOpenSetupCount}{5}",
        "\\providecommand{\\SkillVThreeOpenAuditFlagCount}{540}",
        "\\providecommand{\\SkillVThreeOpenExportMissingFingerprintCount}{540}",
        "\\providecommand{\\SkillVThreeOpenRawFingerprintCount}{540}",
        "\\providecommand{\\SkillVThreeOpenSchemaReferenceCount}{25}",
        "\\providecommand{\\SkillVThreeOpenCacheReferenceCount}{2}",
        f"\\providecommand{{\\SkillVThreeOpenBaselineFallbackCount}}{{{sum(parse_bool(row['selected_baseline'], 'selected baseline') for row in matrix_rows)}}}",
        f"\\providecommand{{\\SkillVThreeOpenWinnerBaselineFallbackCount}}{{{sum(row['selected_baseline'] for row in snapshot)}}}",
    ]
    macro_prefixes = {
        "gemma4_31b__flash": "GemmaFlash",
        "gemma4_31b__multistep": "GemmaMultistep",
        "qwen3_6_27b__flash": "QwenFlash",
        "qwen3_6_27b__multistep": "QwenMultistep",
    }
    for key, suffix in macro_prefixes.items():
        summary = summaries[key]
        lines.extend(
            [
                f"\\providecommand{{\\SkillVThreeOpen{suffix}ReferenceReduction}}{{{summary['reference_over_selected_geomean']:.2f}}}",
                f"\\providecommand{{\\SkillVThreeOpen{suffix}NoExpertiseReferenceReduction}}{{{summary['reference_over_no_skills_geomean']:.2f}}}",
                f"\\providecommand{{\\SkillVThreeOpen{suffix}OracleOverNoExpertise}}{{{summary['no_skills_over_selected_geomean']:.2f}}}",
                f"\\providecommand{{\\SkillVThreeOpen{suffix}ExpertiseEnabledSelections}}{{{summary['expertise_policy_setup_selections']}}}",
                f"\\providecommand{{\\SkillVThreeOpen{suffix}NoExpertiseSelections}}{{{summary['no_expertise_setup_selections']}}}",
            ]
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def ratio_tick(exponent: int) -> str:
    if exponent == 0:
        return r"$1\times$"
    if exponent > 0:
        return rf"${2 ** exponent:g}\times$"
    return rf"$1/{2 ** (-exponent):g}\times$"


def display_benchmark(value: str) -> str:
    return value.removeprefix("hlsfactory_").replace("_", "-")


def display_ratio(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}×"
    if value >= 10:
        return f"{value:.1f}×"
    if value >= 1:
        return f"{value:.1f}×"
    return f"{value:.2f}×"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 6.3,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.4,
            "xtick.labelsize": 5.0,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.32,
            "hatch.linewidth": 0.42,
        }
    )


def plot_figure(
    path: Path,
    snapshot: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> None:
    configure_matplotlib()
    lookup = {
        (row["model_key"], row["workflow"], display_benchmark(row["benchmark"])): row
        for row in snapshot
    }
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.08, 3.13),
        sharex=True,
        gridspec_kw={
            "left": 0.115,
            "right": 0.995,
            "top": 0.86,
            "bottom": 0.22,
            "hspace": 0.20,
        },
    )
    strategy_offsets = {"flash": -0.19, "multistep": 0.19}
    for axis, model_key in zip(axes, MODEL_ORDER):
        for index, benchmark in enumerate(CANONICAL_28):
            if benchmark == "doitgen":
                axis.text(
                    index,
                    -7.55,
                    "not run",
                    ha="center",
                    va="bottom",
                    fontsize=4.2,
                    color="0.48",
                    rotation=90,
                )
                continue
            for strategy in STRATEGIES:
                row = lookup[(model_key, strategy, benchmark)]
                position = index + strategy_offsets[strategy]
                height = math.log2(row["reference_over_selected"])
                axis.bar(
                    position,
                    height,
                    width=0.34,
                    color=POLICY_COLORS[row["selected_skill_mode"]],
                    hatch="///" if strategy == "multistep" else "",
                    edgecolor="0.18",
                    linewidth=0.32,
                    zorder=3,
                )
                axis.text(
                    position,
                    height + (0.16 if height >= 0 else -0.16),
                    display_ratio(row["reference_over_selected"]),
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    rotation=90,
                    fontsize=2.8,
                    color="0.18",
                    clip_on=True,
                    zorder=7,
                )
                axis.scatter(
                    position,
                    math.log2(row["reference_over_no_skills"]),
                    marker="D",
                    facecolors="white",
                    edgecolors="black",
                    linewidths=0.42,
                    s=7.5,
                    zorder=6,
                )
        flash = summaries[f"{model_key}__flash"]
        multistep = summaries[f"{model_key}__multistep"]
        axis.set_title(
            MODEL_SPECS[model_key]["label"],
            loc="left",
            pad=1.5,
            fontweight="bold",
        )
        axis.text(
            0.995,
            1.025,
            f"Single-step: {flash['reference_over_selected_geomean']:.2f}× reference, "
            f"{flash['no_skills_over_selected_geomean']:.2f}× over no expertise; "
            f"multi-step: {multistep['reference_over_selected_geomean']:.2f}× reference, "
            f"{multistep['no_skills_over_selected_geomean']:.2f}× over no expertise",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=4.7,
            color="0.28",
        )
        axis.axhline(0, color="0.12", linewidth=0.65, zorder=4)
        axis.grid(axis="y", color="0.86", zorder=0)
        axis.set_facecolor("#F7F8FA")
        axis.set_ylim(-8.2, 9.5)
        ticks = (-8, -4, 0, 4, 8)
        axis.set_yticks(ticks, [ratio_tick(value) for value in ticks])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].tick_params(axis="x", which="both", bottom=False)
    figure.supylabel(
        "Upstream HLS reference / selected synthesis cycles", x=0.012, fontsize=6.3
    )
    axes[-1].set_xticks(range(len(CANONICAL_28)))
    axes[-1].set_xticklabels(
        CANONICAL_28,
        rotation=56,
        ha="right",
        rotation_mode="anchor",
    )
    axes[-1].set_xlabel("HLSFactory kernel (numeric/alphabetic order)")

    policy_handles = [
        Patch(
            facecolor=POLICY_COLORS[mode],
            edgecolor="0.18",
            label=POLICY_LABELS[mode],
        )
        for mode in SKILL_MODES
    ]
    figure.legend(
        handles=policy_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=5,
        frameon=False,
        columnspacing=0.65,
        handlelength=1.25,
    )
    figure.text(
        0.5,
        0.910,
        "Paired bars: solid = single-step, hatched = multi-step; "
        "hollow diamond = that mode's no-expertise setup; Doitgen = not run",
        ha="center",
        va="center",
        fontsize=5.0,
        color="0.35",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.015,
        metadata={
            "Title": "Open-weight architectural-expertise setup sweep",
            "Author": "Anonymous",
            "Creator": "generate_skill_v3_open_models.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def write_output_manifest(
    path: Path,
    source_root: Path,
    source_manifest: dict[str, Any],
    bundle_hashes: dict[str, str],
    raw_provenance: dict[str, Any],
    matrix_rows: list[dict[str, str]],
    snapshot: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    outputs: list[Path],
) -> None:
    payload = {
        "schema_version": "c2hls.paper.skill-v3-open-models.v1",
        "generator": {
            "path": "scripts/generate_skill_v3_open_models.py",
            "sha256": sha256(Path(__file__).resolve()),
        },
        "source_bundle": {
            "path": str(source_root.resolve()),
            "manifest_schema": source_manifest["schema_version"],
            "files": {
                name: {"sha256": bundle_hashes[name]}
                for name in sorted(bundle_hashes)
            },
        },
        "validated_matrix": {
            "cell_count": 540,
            "model_count": 2,
            "strategy_count": 2,
            "workflow_count": 4,
            "skill_policy_count": 5,
            "kernel_count": 27,
            "setups_per_workflow_kernel": 5,
            "csim_pass_count": 540,
            "csynth_pass_count": 540,
            "timing_pass_count": 540,
            "resource_fit_count": 540,
            "cosim_not_run_count": 540,
            "reference_isolation_audit_flag_count": 540,
            "derived_export_blank_fingerprint_count": 540,
            "matching_raw_fingerprint_count": 540,
            "raw_independent_golden_pass_count": 540,
            "selected_baseline_fallback_count": sum(
                parse_bool(row["selected_baseline"], "selected baseline")
                for row in matrix_rows
            ),
            "within_workflow_winner_baseline_fallback_count": sum(
                row["selected_baseline"] for row in snapshot
            ),
            "website_schema_reference_count": 25,
            "cache_only_reference_count": 2,
            "missing_kernel": "hlsfactory_doitgen",
            "qwen_wrapper_failed_exit_2_rows": 270,
            "gemma_wrapper_complete_exit_0_rows": 270,
            "skill_catalog": source_manifest["skill_catalog"],
            "toolchain": source_manifest["toolchain"],
        },
        "raw_provenance_audit": raw_provenance,
        "workflow_summaries": summaries,
        "figure_definition": {
            "bar": "upstream_hls_reference_worst_case_cycles / within_mode_posthoc_best_of_5_qualified_worst_case_csynth_cycles",
            "diamond": "upstream_hls_reference_worst_case_cycles / same_mode_no_expertise_qualified_worst_case_csynth_cycles",
            "color": "selected_skill_policy",
            "paired_bar_offset": "single_step_left_multi_step_right",
            "hatch": "multi_step",
            "displayed_kernel_slots": 28,
            "recorded_kernel_slots": 27,
        },
        "evidence_boundaries": [
            "Each setup is one deterministic sample; no repeated-seed variance is available.",
            "Each per-kernel bar is a within-mode post-hoc best-of-five HLS-synthesis oracle, not an online-router or matched-cost deployment result.",
            "Single-step and multi-step results are not pooled; the two Gemma modes name distinct, externally unauthenticated serving revisions.",
            "Raw selected-result records label the independent-golden, CSim, CSynth, timing, and resource gates as passed; the figure reports only HLS-synthesis cycles.",
            "Generated-design RTL simulation was not run.",
            "Every row carries an advisory reference-isolation failure; the bundled README attributes these to unlabeled_reference_metric and says classification remains pending.",
            "The derived CSV/JSONL fingerprint column is blank because of an export bug; all 540 manifest-linked raw summary/result pairs contain the same nonempty fingerprint.",
            "The derived export flattens the Gemma multi-step revision onto its single-step rows; raw result records preserve the two distinct revisions.",
            "The Qwen wrapper reports exit code 2 after producing the complete row bundle; the Gemma wrapper reports completion.",
            "Heat-3d and Seidel-2d references come from the validation cache rather than the website schema.",
            "Stored references are comparison denominators, not independently established expert frontiers.",
            "The four execution-mode studies are reported separately and must not be pooled with commercial-model, older skill-library, SFT, or RTL studies as matched model comparisons.",
            "The bundled pooled winners.csv is cross-checked for source integrity but is not used to select or summarize the plotted within-mode winners.",
        ],
        "snapshot_row_count": len(snapshot),
        "outputs": {},
    }
    for output in outputs:
        payload["outputs"][str(output)] = {
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def redraw_validated_snapshot(
    snapshot_path: Path,
    figure_path: Path,
    manifest_path: Path,
) -> None:
    """Redraw only the figure after verifying the previously audited snapshot."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "c2hls.paper.skill-v3-open-models.v1":
        raise ValueError("unexpected skill-v3 manifest schema")
    snapshot_key = str(snapshot_path)
    snapshot_record = manifest.get("outputs", {}).get(snapshot_key)
    if not snapshot_record or snapshot_record.get("sha256") != sha256(snapshot_path):
        raise ValueError("validated snapshot hash does not match the manifest")

    with snapshot_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != manifest.get("snapshot_row_count"):
        raise ValueError("validated snapshot row count does not match the manifest")
    required = {
        "model_key",
        "workflow",
        "benchmark",
        "selected_skill_mode",
        "reference_over_selected",
        "reference_over_no_skills",
    }
    for row in rows:
        if not required.issubset(row):
            raise ValueError("validated snapshot lacks a plotting field")
        row["reference_over_selected"] = float(row["reference_over_selected"])
        row["reference_over_no_skills"] = float(row["reference_over_no_skills"])

    summaries = manifest.get("workflow_summaries")
    if not isinstance(summaries, dict) or set(summaries) != {
        "gemma4_31b__flash",
        "gemma4_31b__multistep",
        "qwen3_6_27b__flash",
        "qwen3_6_27b__multistep",
    }:
        raise ValueError("validated manifest lacks workflow summaries")
    plot_figure(figure_path, rows, summaries)

    manifest["generator"] = {
        "path": "scripts/generate_skill_v3_open_models.py",
        "sha256": sha256(Path(__file__)),
    }
    manifest["outputs"][str(figure_path)] = {
        "bytes": figure_path.stat().st_size,
        "sha256": sha256(figure_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bundle_dir",
        type=Path,
        nargs="?",
        help="path to skill_v3_qwen_gemma_matrix_20260731",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("generated/skill_v3_open_models_snapshot.csv"),
    )
    parser.add_argument(
        "--macros",
        type=Path,
        default=Path("generated/skill_v3_open_models_macros.tex"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/skill_v3_open_models.pdf"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("generated/skill_v3_open_models_manifest.json"),
    )
    parser.add_argument(
        "--plot-only-from-validated-snapshot",
        action="store_true",
        help="redraw from the snapshot whose hash is already recorded in --manifest",
    )
    parser.add_argument(
        "--rtl-evidence-dir", type=Path,
        help="render the separate SIGMETRICS RTL figure from its portable evidence bundle",
    )
    args = parser.parse_args()

    if args.rtl_evidence_dir:
        from generate_skill_v3_rtl import generate

        generate(args.rtl_evidence_dir, Path(__file__).resolve().parents[1])
        return 0

    if args.plot_only_from_validated_snapshot:
        redraw_validated_snapshot(args.snapshot, args.figure, args.manifest)
        print(f"redrew {args.figure} from checksum-validated {args.snapshot}")
        return 0

    if args.bundle_dir is None:
        parser.error("bundle_dir is required unless redrawing a validated snapshot or RTL evidence")
    source_root = args.bundle_dir.resolve()
    bundle_hashes = require_bundle(source_root)
    source_manifest = validate_source_manifest(source_root, bundle_hashes)
    rows = load_rows(source_root)
    raw_provenance = validate_raw_sources(source_manifest, rows)
    references = validate_references(rows)
    snapshot, summaries = select_rows(rows, references)
    validate_winners(source_root, rows)
    validate_setup_summary(source_root, rows)
    validate_wide(source_root, rows)

    write_snapshot(args.snapshot, snapshot)
    write_macros(
        args.macros,
        bundle_hashes["manifest.json"],
        rows,
        snapshot,
        summaries,
    )
    plot_figure(args.figure, snapshot, summaries)
    write_output_manifest(
        args.manifest,
        source_root,
        source_manifest,
        bundle_hashes,
        raw_provenance,
        rows,
        snapshot,
        summaries,
        [args.snapshot, args.macros, args.figure],
    )

    print(
        json.dumps(
            {
                "bundle_sha256": bundle_hashes["manifest.json"],
                "cells": len(rows),
                "snapshot_rows": len(snapshot),
                "models": summaries,
                "figure": str(args.figure),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

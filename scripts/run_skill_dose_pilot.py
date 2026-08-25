#!/usr/bin/env python3
"""Run the frozen-baseline Sonnet skill-cardinality/guard-policy pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import c2hls


SCHEMA_VERSION = "c2hls.skill-dose-outcome.v1"
COUNTS = (0, 1, 2, 3, 5, 8, 42)
GUARD_ABLATION_COUNTS = (3, 42)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _environment(updates: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _experiment_cells(
    seed_manifest: dict[str, Any],
    *,
    repeats: int,
    problems: set[str] | None,
    counts: tuple[int, ...] = COUNTS,
    guard_ablation_counts: tuple[int, ...] = GUARD_ABLATION_COUNTS,
) -> list[dict[str, Any]]:
    cells = []
    for benchmark, entry in sorted(seed_manifest["entries"].items()):
        problem = str(entry["problem"])
        if problems is not None and problem not in problems:
            continue
        ranking = list(entry["router_ranking_skill_ids"])
        if len(ranking) < 42:
            raise ValueError(
                f"{benchmark} has only {len(ranking)} ranked positive skills"
            )
        policies = [
            (count, "action_only")
            for count in counts
        ] + [
            (count, "positive_with_preconditions")
            for count in guard_ablation_counts
        ]
        for count, prompt_policy in policies:
            for repeat in range(repeats):
                skill_ids = ranking[:count]
                identity = {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark": benchmark,
                    "problem": problem,
                    "count": count,
                    "prompt_policy": prompt_policy,
                    "repeat": repeat,
                    "phase_b_code_sha256": entry["code_sha256"],
                    "router_ranking_sha256": entry[
                        "router_ranking_sha256"
                    ],
                    "skill_ids": skill_ids,
                }
                cells.append(
                    {
                        **identity,
                        "cell_id": _canonical_sha256(identity),
                    }
                )
    return cells


def _candidate_event(result: dict[str, Any]) -> dict[str, Any]:
    events = (
        (result.get("synthesis_evaluations") or {}).get("events") or []
    )
    optimization = [
        event
        for event in events
        if isinstance(event, dict)
        and str(event.get("label") or "").startswith("[Step:")
    ]
    usable = optimization or [
        event for event in events if isinstance(event, dict)
    ]
    return usable[-1] if usable else {}


def _skill_prompt(result: dict[str, Any], event: dict[str, Any]) -> dict:
    if isinstance(event.get("skill_prompt"), dict):
        return dict(event["skill_prompt"])
    for step in reversed(result.get("steps") or []):
        if isinstance(step, dict) and isinstance(
            step.get("skill_prompt"), dict
        ):
            return dict(step["skill_prompt"])
    return {}


def _positive_cycles(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _candidate_report(
    result: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any]:
    target_cycles = _positive_cycles(event.get("synthesized_latency_cycles"))
    reports: list[dict[str, Any]] = []
    for step in result.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for key in ("report", "rejected_report"):
            value = step.get(key)
            if isinstance(value, dict) and value:
                reports.append(value)
        for attempt in step.get("attempt_results") or []:
            if isinstance(attempt, dict) and isinstance(
                attempt.get("report"), dict
            ):
                reports.append(attempt["report"])
    if target_cycles is not None:
        for report in reports:
            if _positive_cycles(
                report.get("latency_cycles_worst")
                or report.get("latency_cycles")
            ) == target_cycles:
                return report
    return reports[-1] if reports else {}


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "latency_cycles",
            "latency_cycles_worst",
            "latency_ns",
            "latency_ns_worst",
            "interval",
            "bram",
            "dsp",
            "ff",
            "lut",
            "uram",
            "fmax_mhz",
            "estimated_clock_period_ns",
            "requested_clock_period_ns",
            "slack_ns",
        )
        if report.get(key) is not None
    }


def _compact_record(
    *,
    cell: dict[str, Any],
    result: dict[str, Any],
    result_path: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    event = _candidate_event(result)
    skill_prompt = _skill_prompt(result, event)
    candidate_report = _candidate_report(result, event)
    cycles = _positive_cycles(event.get("synthesized_latency_cycles"))
    valid = (
        event.get("correctness_status") == "passed"
        and event.get("synthesis_status") == "passed"
        and event.get("resource_fit") is True
        and event.get("timing_met") is True
        and cycles is not None
    )
    baseline_report = result.get("baseline_report") or {}
    final_report = result.get("final_report") or {}
    reference_report = result.get("ground_truth_report") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell["cell_id"],
        "completed_at": _utc_now(),
        "benchmark": cell["benchmark"],
        "problem": cell["problem"],
        "strategy": "flash",
        "model": "claude-sonnet-4-6",
        "sample_index": cell["repeat"],
        "requested_positive_skill_count": cell["count"],
        "requested_positive_skill_ids": cell["skill_ids"],
        "prompt_policy": cell["prompt_policy"],
        "avoid_rule_skills_excluded": True,
        "one_optimization_candidate": True,
        "valid": valid,
        "candidate_status": (
            "passed"
            if valid
            else str(
                event.get("failure_class")
                or event.get("status")
                or "not_evaluated"
            )
        ),
        "candidate_latency_cycles": cycles if valid else None,
        "candidate_latency_source": (
            event.get("latency_source") if valid else "none"
        ),
        "candidate_csynth_report": (
            _compact_report(candidate_report) if valid else None
        ),
        "candidate_csim": {
            "status": (
                "passed"
                if event.get("correctness_status") == "passed"
                else "failed"
                if event.get("correctness_status") == "failed"
                else "not_run"
            ),
            "ran": event.get("correctness_status") in {"passed", "failed"},
            "passed": event.get("correctness_status") == "passed",
        },
        "selected_final_latency_cycles": _positive_cycles(
            final_report.get("latency_cycles_worst")
            or final_report.get("latency_cycles")
        ),
        "phase_b_latency_cycles": _positive_cycles(
            baseline_report.get("latency_cycles_worst")
            or baseline_report.get("latency_cycles")
        ),
        "phase_b_code_sha256": cell["phase_b_code_sha256"],
        "router_ranking_sha256": cell["router_ranking_sha256"],
        "skill_telemetry": {
            key: skill_prompt.get(key)
            for key in (
                "catalog_skill_ids",
                "catalog_skill_count",
                "routed_skill_ids",
                "routed_skill_count",
                "rendered_skill_ids",
                "rendered_skill_count",
                "rendered_prompt_characters",
                "declared_applied_skill_ids",
                "declared_applied_skill_count",
                "verified_applied_skill_ids",
                "verified_applied_skill_count",
                "synthesized_candidate_skill_ids",
                "synthesized_candidate_skill_count",
                "skill_declaration_status",
                "verification_method",
            )
        },
        "input_tokens": int(
            (result.get("llm_usage") or {}).get("input_tokens") or 0
        ),
        "output_tokens": int(
            (result.get("llm_usage") or {}).get("output_tokens") or 0
        ),
        "reference_reporting_only": {
            "available": bool(reference_report),
            "latency_cycles": _positive_cycles(
                reference_report.get("latency_cycles_worst")
                or reference_report.get("latency_cycles")
            ),
            "unavailable_to_generation_routing_and_selection": True,
        },
        "toolchain": {
            "vitis_version": "2023.2",
            "part": "xcu280-fsvh2892-2L-e",
            "clock_ns": 3.33,
            "flow_target": "vitis",
            "cosim_executed": False,
        },
        "elapsed_seconds": elapsed_seconds,
        "result_path": str(result_path),
        "result_success": bool(result.get("success")),
        "result_error": result.get("error"),
    }


def _run_cell(
    cell: dict[str, Any],
    configuration: dict[str, str],
) -> dict[str, Any]:
    raw_root = Path(configuration["raw_root"])
    repo_root = Path(configuration["repo_root"])
    benchmark_dir = (
        repo_root
        / "benchmarks_external"
        / "HLSFactory"
        / "polybench_float_small"
        / cell["benchmark"]
    )
    cell_dir = raw_root / cell["problem"] / cell["cell_id"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = raw_root / "_vitis_work" / cell["cell_id"]
    vitis_home = tmp_root / "vitis_user_home"
    tmp_root.mkdir(parents=True, exist_ok=True)
    vitis_home.mkdir(parents=True, exist_ok=True)

    skillless = cell["count"] == 0
    environment = {
        "C2HLS_TMP_ROOT": str(tmp_root),
        "C2HLS_VITIS_USER_HOME": str(vitis_home),
        "TMPDIR": str(tmp_root),
        "XILINX_LOCAL_USER_DATA": str(vitis_home),
        "C2HLS_VITIS_VERSION": "2023.2",
        "C2HLS_PART": "xcu280-fsvh2892-2L-e",
        "C2HLS_CLOCK_NS": "3.33",
        "C2HLS_FLOW_TARGET": "vitis",
        "C2HLS_PHASE_B_SEED_MANIFEST": configuration[
            "phase_b_manifest"
        ],
        "C2HLS_STRATEGY": "flash",
        "C2HLS_PHASEB_MODE": "functional",
        "C2HLS_SKILL_MODE": "skillless" if skillless else "default",
        "C2HLS_FORCE_SKILL_PROMPTS": "1",
        "C2HLS_SKILL_PROMPT_SCOPE": (
            "skillless" if skillless else "matched_positive"
        ),
        "C2HLS_SKILL_PROMPT_MODE": cell["prompt_policy"],
        "C2HLS_SKILL_EXPLICIT_IDS": ",".join(cell["skill_ids"]),
        "C2HLS_SKILL_USAGE_DECLARATION": "1",
        "C2HLS_SKILL_LIBRARY_FROZEN": "1",
        "C2HLS_SKILL_LIBRARY_PATH": configuration["skill_library"],
        "C2HLS_SKILL_LIBRARY_PERSIST": "0",
        "C2HLS_SKILL_UPDATE_STATS": "0",
        "C2HLS_CANDIDATES_PER_STEP": "1",
        "C2HLS_ATTEMPTS_PER_CANDIDATE": "1",
        "C2HLS_EXHAUSTIVE_CANDIDATE_ATTEMPTS": "0",
        "C2HLS_LLM_CANDIDATE_BUDGET": "1",
        "C2HLS_SYNTHESIS_EVAL_BUDGET": "1",
        "C2HLS_REFERENCE_BLIND": "1",
        "C2HLS_ORACLE_MODE": "0",
        "C2HLS_GT_COMPARISON_IN_CONTROL": "0",
        "C2HLS_REFERENCE_METRICS_IN_PROMPTS": "0",
        "C2HLS_REFERENCE_CODE_IN_PROMPTS": "0",
        "C2HLS_PHASE8_BASELINE_ALIGN": "0",
        "C2HLS_PHASE5_GT_PREPOP": "0",
        "C2HLS_FEASIBILITY_SELECTION": "1",
        "C2HLS_CORRECTNESS_BEFORE_SYNTH": "1",
        "C2HLS_COSIM_REQUIRED": "0",
        "C2HLS_COSIM_SELECTED_ONLY": "0",
        "C2HLS_FORCE_SELECTED_COSIM": "0",
        "C2HLS_REFERENCE_COSIM": "0",
        "C2HLS_HW_EMU_FINAL": "0",
        "C2HLS_REFERENCE_VALIDATE_MODE": "trusted_external",
        "C2HLS_REFERENCE_CACHE_DIR": configuration["reference_cache"],
        "C2HLS_REFERENCE_CACHE_REQUIRE_COSIM": "0",
        "C2HLS_SYNTH_TIMEOUT": configuration["synth_timeout"],
        "C2HLS_CSIM_TIMEOUT": configuration["csim_timeout"],
        "C2HLS_LLM_TIMEOUT": configuration["llm_timeout"],
        "C2HLS_LLM_TEMPERATURE": "0.2",
        "C2HLS_LLM_TOP_P": "0.95",
        "C2HLS_MAX_COMPLETION_TOKENS": "8192",
        "C2HLS_MODEL_REVISION": "claude-sonnet-4-6",
        "C2HLS_TRANSLATOR_MODEL": "claude-sonnet-4-6",
        "C2HLS_SYNTHESIS_MODEL": "claude-sonnet-4-6",
        "C2HLS_QUALITY_REPAIR_MODEL": "claude-sonnet-4-6",
        "C2HLS_FEEDBACK_MODEL": "claude-sonnet-4-6",
    }
    started = time.monotonic()
    try:
        with _environment(environment):
            result = c2hls.run_benchmark_multistep(
                str(benchmark_dir),
                output_dir=str(cell_dir),
                gpt_model="claude-sonnet-4-6",
                turns_limitation=1,
            )
        result_path = (
            cell_dir / f"{cell['benchmark']}_multistep_results.json"
        )
        return _compact_record(
            cell=cell,
            result=result,
            result_path=result_path,
            elapsed_seconds=time.monotonic() - started,
        )
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "cell_id": cell["cell_id"],
            "completed_at": _utc_now(),
            "benchmark": cell["benchmark"],
            "problem": cell["problem"],
            "strategy": "flash",
            "model": "claude-sonnet-4-6",
            "sample_index": cell["repeat"],
            "requested_positive_skill_count": cell["count"],
            "requested_positive_skill_ids": cell["skill_ids"],
            "prompt_policy": cell["prompt_policy"],
            "avoid_rule_skills_excluded": True,
            "one_optimization_candidate": True,
            "valid": False,
            "candidate_status": "runner_exception",
            "candidate_latency_cycles": None,
            "elapsed_seconds": time.monotonic() - started,
            "result_path": str(cell_dir),
            "result_success": False,
            "result_error": f"{type(exc).__name__}: {exc}",
        }


def _write_status(
    path: Path,
    *,
    total: int,
    completed: int,
    records: list[dict[str, Any]],
    started_at: str,
) -> None:
    counts = {
        "valid": sum(record.get("valid") is True for record in records),
        "invalid": sum(record.get("valid") is not True for record in records),
        "runner_exception": sum(
            record.get("candidate_status") == "runner_exception"
            for record in records
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "total_cells": total,
        "completed_cells": completed,
        "remaining_cells": total - completed,
        "counts": counts,
        "status": "complete" if completed == total else "running",
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    control_dir = args.control_dir.resolve()
    control_dir.mkdir(parents=True, exist_ok=True)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    seed_manifest = json.loads(
        args.phase_b_manifest.read_text(encoding="utf-8")
    )
    problems = (
        {item.strip() for item in args.problems.split(",") if item.strip()}
        if args.problems
        else None
    )
    cells = _experiment_cells(
        seed_manifest,
        repeats=args.repeats,
        problems=problems,
        counts=tuple(
            int(item) for item in args.counts.split(",") if item.strip()
        ),
        guard_ablation_counts=tuple(
            int(item)
            for item in args.guard_ablation_counts.split(",")
            if item.strip()
        ),
    )
    records_path = control_dir / "skill_dose_records.jsonl"
    existing: list[dict[str, Any]] = []
    if args.resume and records_path.is_file():
        existing = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    complete_ids = {record["cell_id"] for record in existing}
    pending = [cell for cell in cells if cell["cell_id"] not in complete_ids]
    configuration = {
        "raw_root": str(args.raw_root.resolve()),
        "repo_root": str(REPO_ROOT),
        "phase_b_manifest": str(args.phase_b_manifest.resolve()),
        "skill_library": str(args.skill_library.resolve()),
        "reference_cache": str(args.reference_cache.resolve()),
        "synth_timeout": str(args.synth_timeout),
        "csim_timeout": str(args.csim_timeout),
        "llm_timeout": str(args.llm_timeout),
    }
    experiment_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "model": "claude-sonnet-4-6",
        "benchmarks": sorted(
            {cell["problem"] for cell in cells}
        ),
        "positive_skill_counts": [
            int(item) for item in args.counts.split(",") if item.strip()
        ],
        "guard_policy_ablations": {
            "counts": [
                int(item)
                for item in args.guard_ablation_counts.split(",")
                if item.strip()
            ],
            "policies": [
                "action_only",
                "positive_with_preconditions",
            ],
        },
        "repeats": args.repeats,
        "total_cells": len(cells),
        "workers": args.workers,
        "phase_b_manifest": str(args.phase_b_manifest.resolve()),
        "skill_library": str(args.skill_library.resolve()),
        "raw_root": str(args.raw_root.resolve()),
        "reference_blind": True,
        "cosim": False,
        "selection_measurement": (
            "optimization candidate event before any regression fallback"
        ),
    }
    (control_dir / "experiment_manifest.json").write_text(
        json.dumps(experiment_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    started_at = _utc_now()
    all_records = list(existing)
    _write_status(
        control_dir / "status.json",
        total=len(cells),
        completed=len(all_records),
        records=all_records,
        started_at=started_at,
    )
    if not pending:
        return

    with records_path.open("a", encoding="utf-8") as output:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(_run_cell, cell, configuration): cell
                for cell in pending
            }
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                all_records.append(record)
                _write_status(
                    control_dir / "status.json",
                    total=len(cells),
                    completed=len(all_records),
                    records=all_records,
                    started_at=started_at,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-b-manifest", type=Path, required=True)
    parser.add_argument(
        "--skill-library",
        type=Path,
        default=REPO_ROOT / "skill_v2" / "skills.json",
    )
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reference_validation_cache",
    )
    parser.add_argument("--problems", default="")
    parser.add_argument(
        "--counts",
        default=",".join(str(value) for value in COUNTS),
    )
    parser.add_argument(
        "--guard-ablation-counts",
        default=",".join(str(value) for value in GUARD_ABLATION_COUNTS),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--synth-timeout", type=int, default=900)
    parser.add_argument("--csim-timeout", type=int, default=240)
    parser.add_argument("--llm-timeout", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("--workers must be 1 or 2")
    requested_counts = [
        int(item)
        for raw in (args.counts, args.guard_ablation_counts)
        for item in raw.split(",")
        if item.strip()
    ]
    if any(value < 0 or value > 42 for value in requested_counts):
        parser.error("skill counts must be between 0 and 42")
    return args


if __name__ == "__main__":
    run(parse_args())

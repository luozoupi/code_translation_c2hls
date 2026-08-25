#!/usr/bin/env python3
"""Run the corrected ten-setup matrix with two isolated Vitis workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from argparse import Namespace
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_setup_tournament import _run_live
from setup_router import CORRECTED_VERSION


SCHEMA_VERSION = "c2hls.corrected-setup-matrix.v1"


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


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    feasibility = candidate.get("tournament_feasibility") or {}
    return {
        "setup_id": candidate.get("setup_id"),
        "setup_fingerprint": candidate.get("setup_fingerprint"),
        "valid": feasibility.get("feasible") is True,
        "latency_cycles": feasibility.get("latency_cycles"),
        "failure_reasons": feasibility.get("reasons") or [],
        "code_sha256": candidate.get("code_sha256"),
        "result_path": (
            candidate.get("result_path")
            or candidate.get("source_result_path")
        ),
    }


def _entry_metadata(
    benchmark: str,
    entries: dict[str, dict[str, Any]],
) -> dict[str, str]:
    entry = entries.get(benchmark) or {}
    default_problem = benchmark.removeprefix("hlsfactory_")
    return {
        "problem": str(entry.get("problem") or default_problem),
        "benchmark_lineage": str(
            entry.get("benchmark_lineage")
            or f"polybench:{default_problem}"
        ),
        "split": str(entry.get("split") or ""),
    }


def _select_benchmarks(
    available: list[str],
    requested: str,
) -> list[str]:
    if not requested.strip():
        return list(available)
    available_set = set(available)
    selected: set[str] = set()
    unknown: list[str] = []
    for raw in requested.split(","):
        item = raw.strip()
        if not item:
            continue
        if item in available_set:
            selected.add(item)
            continue
        legacy_name = f"hlsfactory_{item}"
        if legacy_name in available_set:
            selected.add(legacy_name)
            continue
        suffix_matches = [
            benchmark
            for benchmark in available
            if benchmark.endswith(f"_{item}")
        ]
        if len(suffix_matches) == 1:
            selected.add(suffix_matches[0])
            continue
        unknown.append(item)
    if unknown:
        raise ValueError(
            "unknown or ambiguous benchmark selector(s): "
            + ", ".join(sorted(unknown))
        )
    return [benchmark for benchmark in available if benchmark in selected]


def _run_benchmark(
    benchmark: str,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    raw_root = Path(configuration["raw_root"])
    output_dir = raw_root / benchmark
    work_root = raw_root / "_vitis_work" / benchmark
    user_home = work_root / "vitis_user_home"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    user_home.mkdir(parents=True, exist_ok=True)
    environment = {
        "C2HLS_TMP_ROOT": str(work_root),
        "C2HLS_VITIS_USER_HOME": str(user_home),
        "TMPDIR": str(work_root),
        "XILINX_LOCAL_USER_DATA": str(user_home),
        "C2HLS_REFERENCE_CACHE_DIR": configuration["reference_cache"],
        "C2HLS_REFERENCE_CACHE_REQUIRE_COSIM": "0",
        "C2HLS_REFERENCE_VALIDATE_MODE": configuration[
            "reference_validate_mode"
        ],
        "C2HLS_SYNTH_TIMEOUT": str(configuration["synth_timeout"]),
        "C2HLS_CSIM_TIMEOUT": str(configuration["csim_timeout"]),
        "C2HLS_LLM_TIMEOUT": str(configuration["llm_timeout"]),
        "C2HLS_LLM_TEMPERATURE": "0.2",
        "C2HLS_LLM_TOP_P": "0.95",
        "C2HLS_MAX_COMPLETION_TOKENS": "8192",
        "C2HLS_MODEL_REVISION": "claude-sonnet-4-6",
        "C2HLS_TRANSLATOR_MODEL": "claude-sonnet-4-6",
        "C2HLS_SYNTHESIS_MODEL": "claude-sonnet-4-6",
        "C2HLS_QUALITY_REPAIR_MODEL": "claude-sonnet-4-6",
        "C2HLS_FEEDBACK_MODEL": "claude-sonnet-4-6",
    }
    arguments = Namespace(
        benchmark_dir=Path(configuration["benchmarks_dir"]) / benchmark,
        output_dir=output_dir,
        measurements_jsonl=None,
        policy="exhaustive",
        predictions=None,
        setup_ids=configuration["setup_ids"],
        registry_version=CORRECTED_VERSION,
        phase_b_manifest=Path(configuration["phase_b_manifest"]),
        skill_library=Path(configuration["skill_library"]),
        model="claude-sonnet-4-6",
        turns=int(configuration["turns"]),
        vitis_version="2023.2",
        part="xcu280-fsvh2892-2L-e",
        clock_ns=3.33,
    )
    metadata = dict(configuration["entry_metadata"].get(benchmark) or {})
    started = time.monotonic()
    try:
        with _environment(environment):
            outcome = _run_live(arguments)
        destination = output_dir / "tournament_result.json"
        destination.write_text(
            json.dumps(outcome, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark": benchmark,
            "problem": metadata.get(
                "problem", benchmark.removeprefix("hlsfactory_")
            ),
            "benchmark_lineage": metadata.get("benchmark_lineage"),
            "split": metadata.get("split"),
            "completed_at": _utc_now(),
            "success": outcome.get("success") is True,
            "winner_setup_id": (
                (outcome.get("winner") or {}).get("setup_id")
                if isinstance(outcome.get("winner"), dict)
                else None
            ),
            "winner_setup_fingerprint": (
                (outcome.get("winner") or {}).get("setup_fingerprint")
                if isinstance(outcome.get("winner"), dict)
                else None
            ),
            "winner_explanation": outcome.get("winner_explanation"),
            "candidates": [
                _candidate_summary(candidate)
                for candidate in outcome.get("candidate_measurements") or []
            ],
            "elapsed_seconds": time.monotonic() - started,
            "result_path": str(destination),
        }
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark": benchmark,
            "problem": metadata.get(
                "problem", benchmark.removeprefix("hlsfactory_")
            ),
            "benchmark_lineage": metadata.get("benchmark_lineage"),
            "split": metadata.get("split"),
            "completed_at": _utc_now(),
            "success": False,
            "winner_setup_id": None,
            "candidates": [],
            "elapsed_seconds": time.monotonic() - started,
            "result_path": str(output_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _status(path: Path, records: list[dict], total: int) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _utc_now(),
        "total_benchmarks": total,
        "completed_benchmarks": len(records),
        "remaining_benchmarks": total - len(records),
        "successful_benchmarks": sum(
            record.get("success") is True for record in records
        ),
        "failed_benchmarks": sum(
            record.get("success") is not True for record in records
        ),
        "status": "complete" if len(records) == total else "running",
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    args.control_dir.mkdir(parents=True, exist_ok=True)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        args.phase_b_manifest.read_text(encoding="utf-8")
    )
    entries = manifest["entries"]
    benchmarks = _select_benchmarks(
        sorted(entries),
        args.benchmarks,
    )
    if not benchmarks:
        raise ValueError("no benchmarks selected")
    missing_dirs = [
        benchmark
        for benchmark in benchmarks
        if not (args.benchmarks_dir / benchmark).is_dir()
    ]
    if missing_dirs:
        raise FileNotFoundError(
            "benchmark directories are missing under "
            f"{args.benchmarks_dir}: {missing_dirs}"
        )
    records_path = args.control_dir / "corrected_matrix_records.jsonl"
    records = []
    if args.resume and records_path.is_file():
        prior_records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        latest_by_benchmark = {
            record["benchmark"]: record for record in prior_records
        }
        records = list(latest_by_benchmark.values())
    complete = {
        record["benchmark"]
        for record in records
        if record.get("success") is True
    }
    pending = [item for item in benchmarks if item not in complete]
    configuration = {
        "raw_root": str(args.raw_root.resolve()),
        "benchmarks_dir": str(args.benchmarks_dir.resolve()),
        "phase_b_manifest": str(args.phase_b_manifest.resolve()),
        "skill_library": str(args.skill_library.resolve()),
        "reference_cache": str(args.reference_cache.resolve()),
        "reference_validate_mode": args.reference_validate_mode,
        "turns": args.turns,
        "synth_timeout": args.synth_timeout,
        "csim_timeout": args.csim_timeout,
        "llm_timeout": args.llm_timeout,
        "setup_ids": args.setup_ids,
        "entry_metadata": {
            benchmark: _entry_metadata(benchmark, entries)
            for benchmark in benchmarks
        },
    }
    (args.control_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": _utc_now(),
                "registry_version": CORRECTED_VERSION,
                "setups": (
                    len(
                        [
                            item
                            for item in args.setup_ids.split(",")
                            if item.strip()
                        ]
                    )
                    if args.setup_ids
                    else 10
                ),
                "setup_ids": [
                    item.strip()
                    for item in args.setup_ids.split(",")
                    if item.strip()
                ],
                "benchmarks": benchmarks,
                "benchmark_metadata": configuration["entry_metadata"],
                "benchmarks_dir": str(args.benchmarks_dir.resolve()),
                "workers": args.workers,
                "model": "claude-sonnet-4-6",
                "reference_blind": True,
                "reference_validate_mode": args.reference_validate_mode,
                "cosim": False,
                "phase_b_manifest": str(
                    args.phase_b_manifest.resolve()
                ),
                "raw_root": str(args.raw_root.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _status(args.control_dir / "status.json", records, len(benchmarks))
    with records_path.open("a", encoding="utf-8") as output:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            futures = {
                executor.submit(
                    _run_benchmark, benchmark, configuration
                ): benchmark
                for benchmark in pending
            }
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                records = [
                    prior
                    for prior in records
                    if prior.get("benchmark") != record.get("benchmark")
                ]
                records.append(record)
                _status(
                    args.control_dir / "status.json",
                    records,
                    len(benchmarks),
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-b-manifest", type=Path, required=True)
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "benchmarks_external"
            / "HLSFactory"
            / "polybench_float_small"
        ),
    )
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--skill-library",
        type=Path,
        default=REPO_ROOT / "skill_v2" / "skills.json",
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reference_validation_cache",
    )
    parser.add_argument(
        "--reference-validate-mode",
        choices=(
            "all",
            "selected",
            "preferred",
            "baseline",
            "external",
            "trusted_external",
        ),
        default="trusted_external",
    )
    parser.add_argument("--benchmarks", default="")
    parser.add_argument(
        "--setup-ids",
        default="",
        help=(
            "Optional comma-separated corrected-v2 setup IDs for bounded "
            "integration smokes. The default evaluates all ten setups."
        ),
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--synth-timeout", type=int, default=900)
    parser.add_argument("--csim-timeout", type=int, default=240)
    parser.add_argument("--llm-timeout", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("--workers must be 1 or 2")
    return args


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python3
"""Generate validated, hash-locked Phase-B seeds for external benchmarks."""

from __future__ import annotations

import argparse
import concurrent.futures
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
from phase_b_manifest import (
    SCHEMA_VERSION,
    canonical_json_sha256,
    file_sha256,
    text_sha256,
    toolchain_fingerprint,
)
from skill_library import load_frozen_library
from smart_skill_router import route_smart_skills


STATUS_SCHEMA_VERSION = "c2hls.external-phase-b-freeze-status.v1"


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


def _compact_csim(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keep = {
        "status",
        "supported",
        "ran",
        "success",
        "passed",
        "correctness",
        "golden_output_sha256",
        "error",
        "log_excerpt",
    }
    return {key: value[key] for key in keep if key in value}


def _load_split(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("split manifest must contain a non-empty entries object")
    lineages_by_split: dict[str, set[str]] = {}
    for benchmark, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError(f"split entry for {benchmark} must be an object")
        lineage = str(entry.get("benchmark_lineage") or "")
        split = str(entry.get("split") or "")
        if not lineage or not split:
            raise ValueError(
                f"split entry for {benchmark} lacks benchmark_lineage or split"
            )
        lineages_by_split.setdefault(split, set()).add(lineage)
    primary_splits = ("train", "validation", "test")
    for first_index, first in enumerate(primary_splits):
        for second in primary_splits[first_index + 1 :]:
            overlap = lineages_by_split.get(first, set()) & lineages_by_split.get(
                second, set()
            )
            if overlap:
                raise ValueError(
                    f"lineage leakage between {first} and {second}: "
                    f"{sorted(overlap)}"
                )
    return payload


def _select_entries(
    split_manifest: dict[str, Any],
    requested: str,
) -> dict[str, dict[str, Any]]:
    entries = {
        str(benchmark): dict(entry)
        for benchmark, entry in split_manifest["entries"].items()
        if isinstance(entry, dict) and entry.get("representative") is True
    }
    if not requested.strip():
        return entries
    selected: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []
    for raw in requested.split(","):
        selector = raw.strip()
        if not selector:
            continue
        if selector in entries:
            selected[selector] = entries[selector]
            continue
        matches = [
            benchmark
            for benchmark, entry in entries.items()
            if selector
            in {
                str(entry.get("problem") or ""),
                str(entry.get("benchmark_lineage") or ""),
            }
            or benchmark.endswith(f"_{selector}")
        ]
        if len(matches) == 1:
            selected[matches[0]] = entries[matches[0]]
        else:
            unknown.append(selector)
    if unknown:
        raise ValueError(
            "unknown or ambiguous representative selector(s): "
            + ", ".join(sorted(unknown))
        )
    return selected


def _failure(
    benchmark: str,
    metadata: dict[str, Any],
    started: float,
    error: str,
    *,
    phase: str,
) -> dict[str, Any]:
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "benchmark": benchmark,
        "problem": metadata.get("problem"),
        "benchmark_lineage": metadata.get("benchmark_lineage"),
        "split": metadata.get("split"),
        "success": False,
        "phase": phase,
        "error": error,
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": _utc_now(),
    }


def _freeze_one(
    benchmark: str,
    metadata: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    raw_root = Path(configuration["raw_root"])
    raw_dir = raw_root / benchmark
    work_root = raw_root / "_vitis_work" / benchmark
    user_home = work_root / "vitis_user_home"
    raw_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    user_home.mkdir(parents=True, exist_ok=True)
    environment = {
        "C2HLS_TMP_ROOT": str(work_root),
        "C2HLS_VITIS_USER_HOME": str(user_home),
        "TMPDIR": str(work_root),
        "XILINX_LOCAL_USER_DATA": str(user_home),
        "C2HLS_REFERENCE_CACHE_DIR": configuration["reference_cache"],
        "C2HLS_REFERENCE_CACHE_REQUIRE_COSIM": "0",
        "C2HLS_REFERENCE_VALIDATE_MODE": "selected",
        "C2HLS_REFERENCE_COSIM": "0",
        "C2HLS_REFERENCE_COSIM_SELECTED_ONLY": "0",
        "C2HLS_REFERENCE_COSIM_BASELINE": "0",
        "C2HLS_SYNTH_TIMEOUT": str(configuration["synth_timeout"]),
        "C2HLS_CSIM_TIMEOUT": str(configuration["csim_timeout"]),
        "C2HLS_LLM_TIMEOUT": str(configuration["llm_timeout"]),
        "C2HLS_LLM_TEMPERATURE": "0.2",
        "C2HLS_LLM_TOP_P": "0.95",
        "C2HLS_MAX_COMPLETION_TOKENS": "8192",
        "C2HLS_MODEL_REVISION": configuration["model"],
        "C2HLS_TRANSLATOR_MODEL": configuration["model"],
        "C2HLS_SYNTHESIS_MODEL": configuration["model"],
        "C2HLS_QUALITY_REPAIR_MODEL": configuration["model"],
        "C2HLS_FEEDBACK_MODEL": configuration["model"],
        "C2HLS_PHASEB_MODE": "functional",
        "C2HLS_PHASE_B_SEED_MANIFEST": "",
        "C2HLS_SKILL_MODE": "skillless",
        "C2HLS_SKILL_PROMPT_SCOPE": "skillless",
        "C2HLS_FORCE_SKILL_PROMPTS": "0",
        "C2HLS_SKILL_LIBRARY_PERSIST": "0",
        "C2HLS_SKILL_UPDATE_STATS": "0",
        "C2HLS_REFERENCE_BLIND": "1",
        "C2HLS_ORACLE_MODE": "0",
        "C2HLS_GT_COMPARISON_IN_CONTROL": "0",
        "C2HLS_REFERENCE_METRICS_IN_PROMPTS": "0",
        "C2HLS_REFERENCE_CODE_IN_PROMPTS": "0",
        "C2HLS_COSIM_REQUIRED": "0",
        "C2HLS_COSIM_SELECTED_ONLY": "0",
        "C2HLS_FORCE_SELECTED_COSIM": "0",
        "C2HLS_HW_EMU_FINAL": "0",
        "C2HLS_VITIS_VERSION": configuration["vitis_version"],
        "C2HLS_VITIS_SETTINGS": configuration["vitis_settings"],
        "C2HLS_FLOW_TARGET": configuration["flow_target"],
        "C2HLS_PART": configuration["part"],
        "C2HLS_CLOCK_NS": str(configuration["clock_ns"]),
    }
    benchmark_dir = Path(configuration["benchmarks_dir"]) / benchmark
    try:
        with _environment(environment):
            inputs = c2hls._load_benchmark_inputs(str(benchmark_dir))
            independent_golden = c2hls._prepare_independent_golden(inputs)
            if independent_golden.get("success") is not True:
                return _failure(
                    benchmark,
                    metadata,
                    started,
                    str(
                        independent_golden.get("error")
                        or "independent golden generation failed"
                    ),
                    phase="independent_golden",
                )
            inputs["independent_golden_output"] = independent_golden.get(
                "output", ""
            )
            inputs["independent_golden_specs"] = independent_golden.get(
                "specs", {}
            )
            inputs["independent_golden_provenance"] = independent_golden.get(
                "provenance", {}
            )
            reference_validation = c2hls.validate_gold_reference(inputs)
            if reference_validation.get("benchmark_ready") is not True:
                return _failure(
                    benchmark,
                    metadata,
                    started,
                    str(
                        reference_validation.get("invalid_reason")
                        or "reference CSim/CSynth validation failed"
                    ),
                    phase="reference",
                )

            orchestrator = c2hls.C2HLSOrchestrator(
                gpt_model=configuration["model"],
                turns_limitation=int(configuration["turns"]),
                quality_repair_turns=0,
            )
            orchestrator.testbench_code = inputs.get("testbench_code", "")
            orchestrator.configure_benchmark(
                extra_files=inputs.get("extra_files", []),
                translated_hls_top=inputs["meta"].get(
                    "translated_hls_top", "workload"
                ),
                reference_hls_top=inputs["meta"].get(
                    "hls_top", "workload"
                ),
                part=inputs["meta"].get("part", configuration["part"]),
                clock_ns=inputs["meta"].get(
                    "clock_ns", configuration["clock_ns"]
                ),
                supports_cosim=False,
                cosim_depths={},
                benchmark_name=benchmark,
                benchmark_context=inputs.get("benchmark_context", ""),
                independent_golden_output=inputs[
                    "independent_golden_output"
                ],
                independent_golden_specs=inputs[
                    "independent_golden_specs"
                ],
                independent_golden_provenance=inputs[
                    "independent_golden_provenance"
                ],
            )
            if not orchestrator.run_phase_a(
                inputs["c_code"],
                inputs["header_code"],
                inputs["header_name"] or "kernel.h",
            ):
                return _failure(
                    benchmark,
                    metadata,
                    started,
                    "plain-C validation failed",
                    phase="phase_a",
                )
            if not orchestrator.run_phase_b(multistep=True):
                return _failure(
                    benchmark,
                    metadata,
                    started,
                    "Phase-B translation did not produce a valid baseline",
                    phase="phase_b",
                )

            csim = _compact_csim(orchestrator.generated_csim)
            report = dict(orchestrator.synth_report or {})
            code = str(orchestrator.hls_code or "")
            cycles = report.get("latency_cycles")
            if (
                not code
                or csim.get("ran") is not True
                or csim.get("passed") is not True
                or not isinstance(cycles, int)
                or isinstance(cycles, bool)
                or cycles <= 0
            ):
                return _failure(
                    benchmark,
                    metadata,
                    started,
                    "Phase-B result lacks code, passing CSim, or exact "
                    "positive CSynth cycles",
                    phase="phase_b_contract",
                )

            code_path = (
                Path(configuration["manifest_code_dir"])
                / f"{benchmark}.cpp"
            )
            code_path.parent.mkdir(parents=True, exist_ok=True)
            code_path.write_text(code, encoding="utf-8")
            history_path = raw_dir / "phase_b_history.json"
            history_path.write_text(
                json.dumps(
                    {
                        "benchmark": benchmark,
                        "event_history": orchestrator.history,
                        "llm_usage": orchestrator._llm_usage_summary(),
                        "synthesis_evaluations": (
                            orchestrator._synthesis_evaluation_summary()
                        ),
                    },
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            library = load_frozen_library(
                Path(configuration["skill_library"])
            )
            route = route_smart_skills(
                library,
                scope="smart_exhaustive_v2",
                step_name="flash",
                current_code=code,
                synth_report=report,
                vitis_version=configuration["vitis_version"],
                fpga=configuration["part"],
                max_skills=42,
                min_score=-1.0e9,
            )
            ranking = [
                {"rank": index + 1, **score}
                for index, score in enumerate(route.audit["scores"])
            ]
            relative_code_path = code_path.relative_to(
                Path(configuration["manifest_root"])
            )
            entry = {
                "benchmark": benchmark,
                "problem": metadata["problem"],
                "benchmark_lineage": metadata["benchmark_lineage"],
                "split": metadata["split"],
                "dataset": metadata.get("dataset", "MachSuite"),
                "input_c_sha256": text_sha256(inputs["c_code"]),
                "header_sha256": text_sha256(inputs["header_code"]),
                "code_path": str(relative_code_path),
                "code_sha256": text_sha256(code),
                "csim": csim,
                "csim_sha256": canonical_json_sha256(csim),
                "csynth_report": report,
                "csynth_report_sha256": canonical_json_sha256(report),
                "router_ranking": ranking,
                "router_ranking_skill_ids": [
                    item["skill_id"] for item in ranking
                ],
                "router_ranking_sha256": canonical_json_sha256(ranking),
                "independent_golden": independent_golden.get(
                    "provenance", {}
                ),
                "reference_validation": {
                    "reference_source": reference_validation.get(
                        "reference_source"
                    ),
                    "synthesis_status": (
                        reference_validation.get("synthesis") or {}
                    ).get("status"),
                    "csim_status": (
                        reference_validation.get("csim") or {}
                    ).get("status"),
                    "reference_cache": reference_validation.get(
                        "reference_cache", {}
                    ),
                },
                "source_artifacts": {
                    "benchmark_dir": str(benchmark_dir.resolve()),
                    "history_path": str(history_path.resolve()),
                    "history_sha256": file_sha256(history_path),
                },
            }
            result = {
                "schema_version": STATUS_SCHEMA_VERSION,
                "benchmark": benchmark,
                "problem": metadata["problem"],
                "benchmark_lineage": metadata["benchmark_lineage"],
                "split": metadata["split"],
                "success": True,
                "phase": "complete",
                "elapsed_seconds": time.monotonic() - started,
                "completed_at": _utc_now(),
                "entry": entry,
            }
            result_path = raw_dir / "phase_b_result.json"
            result_path.write_text(
                json.dumps(result, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            entry["source_artifacts"].update(
                {
                    "result_path": str(result_path.resolve()),
                }
            )
            result["entry"] = entry
            result_path.write_text(
                json.dumps(result, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            return result
    except Exception as exc:
        result = _failure(
            benchmark,
            metadata,
            started,
            f"{type(exc).__name__}: {exc}",
            phase="exception",
        )
        (raw_dir / "phase_b_result.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return result


def _load_resumable(
    benchmark: str,
    configuration: dict[str, Any],
) -> dict[str, Any] | None:
    path = (
        Path(configuration["raw_root"])
        / benchmark
        / "phase_b_result.json"
    )
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entry = result.get("entry") or {}
    code_path = (
        Path(configuration["manifest_root"])
        / str(entry.get("code_path") or "")
    )
    if (
        result.get("success") is True
        and code_path.is_file()
        and entry.get("code_sha256")
        == text_sha256(code_path.read_text(encoding="utf-8"))
        and entry.get("csim_sha256")
        == canonical_json_sha256(entry.get("csim"))
        and entry.get("csynth_report_sha256")
        == canonical_json_sha256(entry.get("csynth_report"))
    ):
        return result
    return None


def _write_manifest(
    args: argparse.Namespace,
    split_manifest: dict[str, Any],
    selected: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
    failures: dict[str, dict[str, Any]],
) -> None:
    complete = len(records) == len(selected) and not failures
    terminal_failure = (
        bool(failures)
        and len(records) + len(failures) == len(selected)
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": args.created_at,
        "updated_at": _utc_now(),
        "status": (
            "complete"
            if complete
            else "failed"
            if terminal_failure
            else "running"
        ),
        "purpose": "external_setup_router_phase_b_seed",
        "dataset": split_manifest.get("dataset", "MachSuite"),
        "toolchain": toolchain_fingerprint(
            vitis_version=args.vitis_version,
            part=args.part,
            clock_ns=args.clock_ns,
            flow_target=args.flow_target,
        ),
        "skill_library": {
            "path": str(args.skill_library.resolve()),
            "sha256": file_sha256(args.skill_library),
            "skill_count": len(
                load_frozen_library(args.skill_library).all()
            ),
        },
        "split_manifest": {
            "path": str(args.split_manifest.resolve()),
            "sha256": file_sha256(args.split_manifest),
        },
        "benchmarks": [
            metadata["problem"] for metadata in selected.values()
        ],
        "entries": {
            benchmark: records[benchmark]["entry"]
            for benchmark in sorted(records)
        },
        "failures": {
            benchmark: failures[benchmark]
            for benchmark in sorted(failures)
        },
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    status = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "updated_at": payload["updated_at"],
        "status": payload["status"],
        "total_benchmarks": len(selected),
        "completed_benchmarks": len(records),
        "failed_benchmarks": len(failures),
        "remaining_benchmarks": len(selected) - len(records) - len(failures),
        "failures": {
            benchmark: record.get("error")
            for benchmark, record in sorted(failures.items())
        },
    }
    (args.output.parent / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    split_manifest = _load_split(args.split_manifest)
    selected = _select_entries(split_manifest, args.benchmarks)
    if not selected:
        raise ValueError("no representative benchmarks selected")
    missing = [
        benchmark
        for benchmark in selected
        if not (args.benchmarks_dir / benchmark).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing benchmark directories: {sorted(missing)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "raw_root": str(args.raw_root.resolve()),
        "benchmarks_dir": str(args.benchmarks_dir.resolve()),
        "manifest_root": str(args.output.parent.resolve()),
        "manifest_code_dir": str(
            (args.output.parent / "code").resolve()
        ),
        "reference_cache": str(args.reference_cache.resolve()),
        "skill_library": str(args.skill_library.resolve()),
        "model": args.model,
        "turns": args.turns,
        "synth_timeout": args.synth_timeout,
        "csim_timeout": args.csim_timeout,
        "llm_timeout": args.llm_timeout,
        "vitis_version": args.vitis_version,
        "vitis_settings": str(args.vitis_settings.resolve()),
        "part": args.part,
        "clock_ns": args.clock_ns,
        "flow_target": args.flow_target,
    }
    records: dict[str, dict[str, Any]] = {}
    if args.resume:
        for benchmark in selected:
            resumed = _load_resumable(benchmark, configuration)
            if resumed is not None:
                records[benchmark] = resumed
    failures: dict[str, dict[str, Any]] = {}
    pending = [
        benchmark for benchmark in selected if benchmark not in records
    ]
    _write_manifest(
        args, split_manifest, selected, records, failures
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {
            executor.submit(
                _freeze_one,
                benchmark,
                selected[benchmark],
                configuration,
            ): benchmark
            for benchmark in pending
        }
        for future in concurrent.futures.as_completed(futures):
            benchmark = futures[future]
            result = future.result()
            if result.get("success") is True:
                records[benchmark] = result
                failures.pop(benchmark, None)
            else:
                failures[benchmark] = result
            _write_manifest(
                args, split_manifest, selected, records, failures
            )
    return 0 if len(records) == len(selected) and not failures else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--benchmarks", default="")
    parser.add_argument(
        "--skill-library",
        type=Path,
        default=REPO_ROOT / "skill_v2" / "skills.json",
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "reference_validation_cache_machsuite"
        ),
    )
    parser.add_argument(
        "--vitis-settings",
        type=Path,
        default=Path(
            "/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh"
        ),
    )
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--synth-timeout", type=int, default=900)
    parser.add_argument("--csim-timeout", type=int, default=240)
    parser.add_argument("--llm-timeout", type=int, default=900)
    parser.add_argument("--vitis-version", default="2023.2")
    parser.add_argument("--part", default="xcu280-fsvh2892-2L-e")
    parser.add_argument("--clock-ns", type=float, default=3.33)
    parser.add_argument("--flow-target", default="vitis")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("--workers must be 1 or 2")
    if not args.vitis_settings.is_file():
        parser.error(
            f"--vitis-settings does not exist: {args.vitis_settings}"
        )
    args.created_at = _utc_now()
    if args.output.is_file():
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            args.created_at = str(
                existing.get("created_at") or args.created_at
            )
        except (OSError, json.JSONDecodeError):
            pass
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

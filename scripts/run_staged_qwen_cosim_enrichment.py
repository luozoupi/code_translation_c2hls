#!/usr/bin/env python3
"""Run resource-gated RTL COSIM for staged Qwen agentic kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = (
    REPO
    / "artifacts"
    / "qwen_cosim_enrichment"
    / "strong_gains_20260725_staged"
    / "manifest.jsonl"
)
DEFAULT_OUTPUT = (
    REPO
    / "artifacts"
    / "qwen_cosim_enrichment"
    / "strong_gains_20260725_cosim"
)
DEFAULT_TMP = Path("/mnt/data/luo00466/tmp/qwen_strong_gains_cosim_20260725")
DEFAULT_ARCHIVE = Path(
    "/mnt/data2/luo00466/c2hls_rl/vitis_work/qwen_strong_gains_cosim_20260725"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _status(result: dict[str, Any]) -> str:
    if result.get("success") and result.get("passed"):
        return "pass"
    error = str(result.get("error") or "").lower()
    if "timed out" in error or "timeout" in error:
        return "timeout"
    return "fail"


def _active_vitis_processes() -> list[str]:
    command = [
        "pgrep",
        "-u",
        str(os.getuid()),
        "-af",
        "vitis-run|vitis_hls|xsim",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _wait_for_resources(
    *,
    tmp_root: Path,
    min_free_gib: float,
    max_load_ratio: float,
    poll_seconds: int,
) -> None:
    while True:
        probe = tmp_root if tmp_root.exists() else tmp_root.parent
        free_gib = shutil.disk_usage(probe).free / 2**30
        cpu_count = os.cpu_count() or 1
        load_one = os.getloadavg()[0]
        load_ratio = load_one / cpu_count
        active_vitis = _active_vitis_processes()
        if (
            free_gib >= min_free_gib
            and load_ratio <= max_load_ratio
            and not active_vitis
        ):
            print(
                f"RESOURCE_GATE_PASS free_gib={free_gib:.1f} "
                f"load1={load_one:.2f} load_ratio={load_ratio:.3f}",
                flush=True,
            )
            return
        print(
            f"RESOURCE_GATE_WAIT free_gib={free_gib:.1f}/{min_free_gib:.1f} "
            f"load1={load_one:.2f} load_ratio={load_ratio:.3f}/"
            f"{max_load_ratio:.3f} active_vitis={len(active_vitis)}",
            flush=True,
        )
        time.sleep(poll_seconds)


def _read_optional(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""


def _extra_files(bench_dir: Path, metadata: dict[str, Any], header_name: str) -> list[str]:
    extras: list[str] = []
    for relative in metadata.get("support_files") or []:
        path = bench_dir / relative
        if path.exists():
            extras.append(str(path))
    for path in bench_dir.glob("*.h"):
        if path.name != header_name and str(path) not in extras:
            extras.append(str(path))
    return extras


def _archive_work_dir(work_dir: Path, archive_dir: Path) -> tuple[str, str | None]:
    if not work_dir.is_dir():
        return "", "COSIM work directory was not created"
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["rsync", "-a", f"{work_dir}/", f"{archive_dir}/"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return str(work_dir), (completed.stderr or completed.stdout).strip()
    source_kernel = work_dir / "kernel.cpp"
    archived_kernel = archive_dir / "kernel.cpp"
    if source_kernel.is_file() and (
        not archived_kernel.is_file()
        or _sha256(source_kernel) != _sha256(archived_kernel)
    ):
        return str(work_dir), "archived kernel verification failed"
    shutil.rmtree(work_dir)
    return str(archive_dir), None


def _write_outputs(
    *,
    output_dir: Path,
    cases: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    schema_records: list[dict[str, Any]],
    validate_jsonl,
) -> None:
    details_path = output_dir / "results.jsonl"
    schema_path = output_dir / "schema_records.jsonl"
    with details_path.open("w") as handle:
        for case in cases:
            result = results.get(case["case_id"])
            if result:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
    with schema_path.open("w") as handle:
        for record in schema_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    validation = validate_jsonl(schema_path, verbose=False) if schema_records else None

    status_counts: dict[str, int] = {}
    for result in results.values():
        status = str((result.get("cosim") or {}).get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "schema_version": "c2hls.qwen_cosim_enrichment_summary.v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "selected_cases": len(cases),
        "completed_cases": len(results),
        "status_counts": status_counts,
        "details_jsonl": str(details_path),
        "schema_jsonl": str(schema_path),
        "schema_validation": validation,
        "rows": [results[case["case_id"]] for case in cases if case["case_id"] in results],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Qwen Strong-Gain RTL COSIM Enrichment",
        "",
        f"- Selected cases: {len(cases)}",
        f"- Completed cases: {len(results)}",
        f"- Status counts: `{json.dumps(status_counts, sort_keys=True)}`",
        f"- Schema validation: `{json.dumps(validation, sort_keys=True)}`",
        "",
        "| case | benchmark | treatment | strategy | skills | csynth cycles | cosim | cosim cycles |",
        "|---|---|---|---|---|---:|---|---:|",
    ]
    for case in cases:
        result = results.get(case["case_id"])
        if not result:
            continue
        cosim = result.get("cosim") or {}
        lines.append(
            "| {case_id} | {benchmark} | {training} | {strategy} | {skill_mode} | "
            "{csynth_cycles} | {status} | {cycles} |".format(
                **case,
                status=cosim.get("status"),
                cycles=cosim.get("kernel_runtime_cycles") or "",
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tmp-root", type=Path, default=DEFAULT_TMP)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--min-free-gib", type=float, default=150.0)
    parser.add_argument("--max-load-ratio", type=float, default=0.65)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retain-work", action="store_true")
    args = parser.parse_args()

    cases = _read_jsonl(args.manifest.resolve())
    if args.limit > 0:
        cases = cases[: args.limit]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    args.archive_root.mkdir(parents=True, exist_ok=True)

    os.environ["C2HLS_TMP_ROOT"] = str(args.tmp_root.resolve())
    os.environ["C2HLS_SWEEP_TMP_ROOT"] = str(args.tmp_root.resolve())
    os.environ.setdefault(
        "C2HLS_VITIS_SETTINGS",
        "/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh",
    )
    os.environ["C2HLS_VITIS_VERSION"] = "2023.2"
    os.environ["C2HLS_PART"] = "xcu280-fsvh2892-2L-e"
    os.environ["C2HLS_CLOCK_NS"] = "3.33"
    os.environ["C2HLS_FLOW_TARGET"] = "vitis"
    os.environ["C2HLS_COSIM_TRACE_LEVEL"] = "none"
    os.environ["C2HLS_COSIM_TIMEOUT"] = str(args.timeout)

    sys.path.insert(0, str(REPO))
    from c2hls_temp import configure_temp_env
    from export_schema_jsonl import (
        SCHEMA_VERSION,
        TARGET_COSIM,
        _build_implementation,
        _build_problem,
        _build_run,
        validate_jsonl,
    )
    from hls_eval import run_cosim

    configure_temp_env(create=True)
    results: dict[str, dict[str, Any]] = {}
    result_files = sorted((output_dir / "cases").glob("*.result.json"))
    for path in result_files:
        value = json.loads(path.read_text())
        if isinstance(value, dict) and value.get("case_id"):
            results[value["case_id"]] = value

    measurement_by_target: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results.values():
        cosim = result.get("cosim") or {}
        if cosim.get("ran") and cosim.get("target_code_sha256"):
            measurement_by_target[
                (result["benchmark"], cosim["target_code_sha256"])
            ] = result

    schema_records: list[dict[str, Any]] = []
    for case in cases:
        existing = results.get(case["case_id"])
        if existing:
            schema_records.append(existing["schema_record"])
            continue

        case_result: dict[str, Any] = {
            **case,
            "schema_version": "c2hls.qwen_cosim_enrichment_result.v1",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        staged_kernel = Path(case["staged_kernel"])
        if not staged_kernel.is_file() or _sha256(staged_kernel) != case["kernel_sha256"]:
            raise RuntimeError(f"staged kernel verification failed: {staged_kernel}")

        reference_cycles = case.get("reference_cycles")
        csynth_cycles = case.get("csynth_cycles")
        if (
            reference_cycles is not None
            and csynth_cycles is not None
            and int(csynth_cycles) > int(reference_cycles)
        ):
            cosim_result = {
                "status": "not_run",
                "ran": False,
                "passed": False,
                "skip_reason": "predicted_longer_than_gold",
                "predicted_cycles": csynth_cycles,
                "reference_cycles": reference_cycles,
                "kernel_runtime_cycles": None,
                "kernel_runtime_us": None,
                "kernel_clock_freq_mhz": None,
                "target_code_sha256": case["kernel_sha256"],
                "error": "",
            }
        elif not case.get("supports_cosim"):
            cosim_result = {
                "status": "not_run",
                "ran": False,
                "passed": False,
                "skip_reason": "benchmark_does_not_support_cosim",
                "kernel_runtime_cycles": None,
                "kernel_runtime_us": None,
                "kernel_clock_freq_mhz": None,
                "target_code_sha256": case["kernel_sha256"],
                "error": "",
            }
        else:
            target_key = (case["benchmark"], case["kernel_sha256"])
            reused = measurement_by_target.get(target_key)
            if reused:
                prior = reused["cosim"]
                cosim_result = {
                    **prior,
                    "reused_from_case_id": reused["case_id"],
                    "work_dir": prior.get("work_dir"),
                }
            else:
                _wait_for_resources(
                    tmp_root=args.tmp_root,
                    min_free_gib=args.min_free_gib,
                    max_load_ratio=args.max_load_ratio,
                    poll_seconds=args.poll_seconds,
                )
                bench_dir = Path(case["benchmark_dir"])
                metadata = json.loads((bench_dir / "metadata.json").read_text())
                header_name = metadata.get("header_file") or "kernel.h"
                top_function = (
                    metadata.get("translated_hls_top")
                    or metadata.get("hls_top")
                    or "workload"
                )
                testbench_name = metadata.get("testbench_file") or "testbench.cpp"
                print(
                    f"COSIM_START case={case['case_id']} benchmark={case['benchmark']} "
                    f"timeout={args.timeout}",
                    flush=True,
                )
                started = time.monotonic()
                raw = run_cosim(
                    staged_kernel.read_text(errors="replace"),
                    _read_optional(bench_dir / testbench_name),
                    _read_optional(bench_dir / header_name),
                    header_name=header_name,
                    top_function=top_function,
                    part="xcu280-fsvh2892-2L-e",
                    clock_ns=3.33,
                    extra_files=_extra_files(bench_dir, metadata, header_name),
                    interface_depths=metadata.get("cosim_depths") or {},
                )
                runtime_seconds = round(time.monotonic() - started, 3)
                work_dir = Path(raw.get("work_dir") or "")
                archive_error = None
                durable_work_dir = str(work_dir) if work_dir else ""
                if work_dir and not args.retain_work:
                    durable_work_dir, archive_error = _archive_work_dir(
                        work_dir, args.archive_root / case["case_id"]
                    )
                cosim_result = {
                    "status": _status(raw),
                    "ran": True,
                    "success": bool(raw.get("success")),
                    "passed": bool(raw.get("passed")),
                    "kernel_runtime_cycles": raw.get("kernel_runtime_cycles"),
                    "kernel_runtime_us": raw.get("kernel_runtime_us"),
                    "kernel_clock_freq_mhz": raw.get("kernel_clock_freq_mhz"),
                    "runtime_seconds": runtime_seconds,
                    "work_dir": durable_work_dir,
                    "archive_error": archive_error,
                    "target_code_sha256": case["kernel_sha256"],
                    "error": raw.get("error") or "",
                }
                measurement_by_target[target_key] = {
                    **case_result,
                    "cosim": cosim_result,
                }
                print(
                    f"COSIM_DONE case={case['case_id']} "
                    f"status={cosim_result['status']} "
                    f"cycles={cosim_result.get('kernel_runtime_cycles')}",
                    flush=True,
                )

        metadata = json.loads(Path(case["metadata_path"]).read_text())
        origin_meta = {
            "model": case.get("model"),
            "training": case.get("training"),
            "strategy": case.get("strategy"),
            "skills": case.get("skill_mode"),
            "skill_injected_count": case.get("skill_injected_count"),
            "source_summary": case.get("source_summary"),
            "source_result": case.get("source_result"),
            "source_code_sha256": case.get("kernel_sha256"),
            "csynth_cycles": case.get("csynth_cycles"),
            "reference_cycles": case.get("reference_cycles"),
            "reference_source_kind": case.get("reference_source_kind"),
            "reference_isolation_audit_passed": case.get(
                "reference_isolation_audit_passed"
            ),
            "cosim_work_dir": cosim_result.get("work_dir"),
            "cosim_skip_reason": cosim_result.get("skip_reason"),
            "error": (cosim_result.get("error") or "")[:300] or None,
        }
        schema_record = {
            "schema_version": SCHEMA_VERSION,
            "report_type": "rtl_sim",
            "run": _build_run(
                TARGET_COSIM,
                "xcu280-fsvh2892-2L-e",
                cosim_result.get("runtime_seconds"),
                {
                    "vitis_version": "2023.2",
                    "flow_target": "vitis",
                    "clock_ns": 3.33,
                },
            ),
            "problem": _build_problem(metadata),
            "implementation": _build_implementation(
                metadata,
                variant_name=str(case.get("selected_step_name") or "selected"),
                origin_override="c2hls_orchestrator",
                origin_version=(
                    f"{case.get('model')}__{case.get('training')}__"
                    f"{case.get('strategy')}__{case.get('skill_mode')}"
                ),
                origin_meta=origin_meta,
            ),
            "rtl_sim": {
                "status": cosim_result["status"],
                "ran": cosim_result.get("ran"),
                "passed": cosim_result.get("passed"),
                "kernel_runtime_cycles": cosim_result.get("kernel_runtime_cycles"),
                "kernel_runtime_us": cosim_result.get("kernel_runtime_us"),
                "kernel_clock_freq_mhz": cosim_result.get(
                    "kernel_clock_freq_mhz"
                ),
                "target_code_sha256": case["kernel_sha256"],
                "skip_reason": cosim_result.get("skip_reason"),
                "error": (cosim_result.get("error") or "")[:300] or None,
            },
        }
        case_result.update(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "cosim": cosim_result,
                "schema_record": schema_record,
            }
        )
        case_dir = output_dir / "cases"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / f"{case['case_id']}.result.json").write_text(
            json.dumps(case_result, indent=2, sort_keys=True) + "\n"
        )
        results[case["case_id"]] = case_result
        schema_records.append(schema_record)
        _write_outputs(
            output_dir=output_dir,
            cases=cases,
            results=results,
            schema_records=schema_records,
            validate_jsonl=validate_jsonl,
        )

    _write_outputs(
        output_dir=output_dir,
        cases=cases,
        results=results,
        schema_records=schema_records,
        validate_jsonl=validate_jsonl,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

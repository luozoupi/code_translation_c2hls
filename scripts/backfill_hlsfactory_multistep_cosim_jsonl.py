#!/usr/bin/env python3
"""Backfill `rtl_sim` JSONL records for HLSFactory multistep sweep steps.

The normal multistep sweep can run with cosim disabled, producing hls_synth and
sw_run records but no rtl_sim records. This script replays Vitis cosim for each
successful step kernel and emits matching schema-1.0 rtl_sim rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import export_schema_jsonl as schema  # noqa: E402
from build_hlsfactory_revision_set_jsonl import SourceSpec, _retarget_source_record  # noqa: E402


DEFAULT_STAMP = "hlsfactory_multistep_sonnet46_no_skills_20260615"
DEFAULT_SUMMARY = REPO / "artifacts" / f"agentic_no_streamcluster_{DEFAULT_STAMP}.summary.json"
DEFAULT_RAW_JSONL = REPO / "artifacts" / f"agentic_no_streamcluster_{DEFAULT_STAMP}.jsonl"
DEFAULT_OUT_JSONL = REPO / "artifacts" / f"{DEFAULT_STAMP}_cosim_backfill_20260615.schema.jsonl"
DEFAULT_OUT_CSV = REPO / "artifacts" / f"{DEFAULT_STAMP}_cosim_backfill_20260615.csv"
DEFAULT_ORIGIN_VERSION = "1164311__multistep__no_skills"
DEFAULT_MODE = "multistep"
DEFAULT_SKILLS = "off"
DEFAULT_SKILLS_VARIANT = "none"


def _read_optional(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def _load_metadata(bench_dir: Path) -> dict[str, Any]:
    return json.loads((bench_dir / "metadata.json").read_text())


def _extra_files(bench_dir: Path, meta: dict[str, Any], header_file: str) -> list[str]:
    extras: list[str] = []
    for rel in meta.get("support_files") or []:
        path = bench_dir / rel
        if path.exists():
            extras.append(str(path))
    for header in bench_dir.glob("*.h"):
        if header.name != header_file and str(header) not in extras:
            extras.append(str(header))
    return extras


def _step_name(record: dict[str, Any]) -> str:
    impl = record.get("implementation") or {}
    meta = impl.get("origin_meta") or {}
    variant = impl.get("variant") or {}
    return str(meta.get("step") or variant.get("name") or "").strip()


def _bench_name(record: dict[str, Any]) -> str:
    group_path = (record.get("problem") or {}).get("group_path") or []
    return str(group_path[0]) if group_path else ""


def _load_hls_records(raw_jsonl: Path, source: SourceSpec) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with raw_jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = schema._strict_json_loads(line)
            if record.get("report_type") != "hls_synth":
                continue
            step = _step_name(record)
            bench = _bench_name(record)
            if not bench or not step:
                continue
            records[(bench, step)] = _retarget_source_record(record, source)
    return records


def _completed_keys(out_jsonl: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not out_jsonl.exists():
        return keys
    with out_jsonl.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = schema._strict_json_loads(line)
            meta = (record.get("implementation") or {}).get("origin_meta") or {}
            bench = meta.get("source_bench") or _bench_name(record)
            step = meta.get("multistep_step") or _step_name(record)
            if bench and step:
                keys.add((str(bench), str(step)))
    return keys


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(schema._strict_json_dumps(record) + "\n")


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    fieldnames = [
        "bench", "step", "variant_index", "variant_name", "status", "passed",
        "kernel_runtime_cycles", "kernel_runtime_us", "kernel_clock_freq_mhz",
        "elapsed_sec", "code_path", "work_dir", "error",
    ]
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def _kernel_path_for_step(step: dict[str, Any]) -> Path | None:
    containers = [step.get("report") or {}, step.get("cosim") or {}]
    for attempt in step.get("attempt_results") or []:
        containers.extend([attempt.get("report") or {}, attempt.get("cosim") or {}])
    for container in containers:
        work_dir = container.get("work_dir")
        if not work_dir:
            continue
        candidate = Path(work_dir) / "kernel.cpp"
        if candidate.exists():
            return candidate
    return None


def _rtl_record_from_template(
    template: dict[str, Any],
    *,
    bench: str,
    step: str,
    status: str,
    cycles: int | None,
    runtime_us: float | None,
    clock_mhz: float | None,
    elapsed_sec: float | None,
    code_path: str,
    work_dir: str,
    error: str,
) -> dict[str, Any]:
    impl = schema._json_safe(template.get("implementation") or {})
    meta = impl.get("origin_meta")
    if not isinstance(meta, dict):
        meta = {}
    meta.update({
        "source_bench": bench,
        "multistep_step": step,
        "cosim_backfill": True,
        "cosim_code_path": code_path,
        "cosim_work_dir": work_dir,
    })
    if error:
        meta["cosim_error"] = error[:1000]
    impl["origin_meta"] = meta
    return {
        "schema_version": "1.0",
        "report_type": "rtl_sim",
        "run": {
            "target": "vitis.cosim",
            "device": os.getenv("C2HLS_PART", "xcu280-fsvh2892-2L-e"),
            "vitis_version": os.getenv("C2HLS_VITIS_VERSION", "2023.2"),
            "runtime_seconds": elapsed_sec,
        },
        "problem": schema._json_safe(template.get("problem") or {}),
        "implementation": impl,
        "rtl_sim": {
            "status": status,
            "kernel_runtime_cycles": cycles,
            "kernel_runtime_us": runtime_us,
            "kernel_clock_freq_mhz": clock_mhz,
        },
    }


def _iter_cases(summary: dict[str, Any], benches: set[str]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    cases: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in summary.get("rows") or []:
        bench = row.get("bench")
        if benches and bench not in benches:
            continue
        cur = row.get("current") or {}
        result_path = Path(cur.get("json") or "")
        if not cur.get("success") or not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        for step in result.get("steps") or []:
            if step.get("success") and step.get("step_name"):
                cases.append((row, step))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--raw-jsonl", type=Path, default=DEFAULT_RAW_JSONL)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--origin-version", default=DEFAULT_ORIGIN_VERSION)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--skills", default=DEFAULT_SKILLS)
    parser.add_argument("--skills-variant", default=DEFAULT_SKILLS_VARIANT)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--benches", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("C2HLS_TMP_ROOT", "/mnt/data/luo00466/tmp")
    os.environ.setdefault("C2HLS_VITIS_SETTINGS", "/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh")
    os.environ.setdefault("C2HLS_VITIS_VERSION", "2023.2")
    os.environ.setdefault("C2HLS_PART", "xcu280-fsvh2892-2L-e")
    os.environ.setdefault("C2HLS_CLOCK_NS", "3.33")
    os.environ.setdefault("C2HLS_FLOW_TARGET", "vitis")
    os.environ.setdefault("C2HLS_COSIM_TRACE_LEVEL", "none")
    os.environ["C2HLS_COSIM_TIMEOUT"] = str(args.timeout)

    from c2hls_temp import configure_temp_env
    from hls_eval import run_cosim

    configure_temp_env(create=True)
    source = SourceSpec(
        path=args.raw_jsonl,
        origin_version=args.origin_version,
        mode=args.mode,
        skills=args.skills,
        skills_variant=args.skills_variant,
    )
    hls_records = _load_hls_records(args.raw_jsonl, source)
    completed = _completed_keys(args.out_jsonl)
    requested = {item.strip() for item in args.benches.split(",") if item.strip()}
    summary = json.loads(args.summary_json.read_text())
    cases = [
        (row, step)
        for row, step in _iter_cases(summary, requested)
        if (row.get("bench"), step.get("step_name")) not in completed
    ]
    if args.limit > 0:
        cases = cases[: args.limit]

    print(f"CASES {len(cases)} completed={len(completed)} timeout={args.timeout}", flush=True)
    for row, step in cases:
        bench = str(row.get("bench"))
        step_name = str(step.get("step_name"))
        template = hls_records.get((bench, step_name))
        if template is None:
            print(f"SKIP {bench} {step_name}: no hls_synth template", flush=True)
            continue
        bench_dir = Path(row.get("bench_dir") or "")
        meta = _load_metadata(bench_dir)
        header_file = meta.get("header_file") or "kernel.h"
        testbench_file = meta.get("testbench_file") or "testbench.cpp"
        top_function = meta.get("translated_hls_top") or meta.get("hls_top") or "workload"
        kernel_path = _kernel_path_for_step(step)
        if kernel_path is None:
            cosim = {
                "passed": False,
                "error": "step kernel.cpp could not be located",
                "work_dir": "",
                "kernel_runtime_cycles": None,
                "kernel_runtime_us": None,
                "kernel_clock_freq_mhz": None,
            }
            elapsed = 0.0
        else:
            print(f"START {bench} step={step_name} code={kernel_path}", flush=True)
            start = time.time()
            cosim = run_cosim(
                kernel_path.read_text(errors="replace"),
                _read_optional(bench_dir / testbench_file),
                _read_optional(bench_dir / header_file),
                header_name=header_file,
                top_function=top_function,
                part=os.getenv("C2HLS_PART", "xcu280-fsvh2892-2L-e"),
                clock_ns=float(os.getenv("C2HLS_CLOCK_NS", "3.33")),
                extra_files=_extra_files(bench_dir, meta, header_file),
                interface_depths=meta.get("cosim_depths") or {},
            )
            elapsed = round(time.time() - start, 3)
        error = cosim.get("error") or ""
        status = "pass" if cosim.get("passed") else ("timeout" if "timed out" in error.lower() else "fail")
        variant = (template.get("implementation") or {}).get("variant") or {}
        record = _rtl_record_from_template(
            template,
            bench=bench,
            step=step_name,
            status=status,
            cycles=cosim.get("kernel_runtime_cycles"),
            runtime_us=cosim.get("kernel_runtime_us"),
            clock_mhz=cosim.get("kernel_clock_freq_mhz"),
            elapsed_sec=elapsed,
            code_path=str(kernel_path or ""),
            work_dir=str(cosim.get("work_dir") or ""),
            error=error,
        )
        _append_jsonl(args.out_jsonl, record)
        _append_csv(args.out_csv, {
            "bench": bench,
            "step": step_name,
            "variant_index": variant.get("index"),
            "variant_name": variant.get("name"),
            "status": status,
            "passed": bool(cosim.get("passed")),
            "kernel_runtime_cycles": cosim.get("kernel_runtime_cycles"),
            "kernel_runtime_us": cosim.get("kernel_runtime_us"),
            "kernel_clock_freq_mhz": cosim.get("kernel_clock_freq_mhz"),
            "elapsed_sec": elapsed,
            "code_path": str(kernel_path or ""),
            "work_dir": str(cosim.get("work_dir") or ""),
            "error": error,
        })
        print(
            f"DONE {bench} step={step_name} status={status} "
            f"cycles={cosim.get('kernel_runtime_cycles')} elapsed={elapsed}s",
            flush=True,
        )

    validation = schema.validate_jsonl(args.out_jsonl) if args.out_jsonl.exists() else {"invalid": 0}
    print(schema._strict_json_dumps({
        "out_jsonl": str(args.out_jsonl),
        "out_csv": str(args.out_csv),
        "schema_invalid": validation.get("invalid"),
    }, sort_keys=True), flush=True)
    return 1 if validation.get("invalid") else 0


if __name__ == "__main__":
    raise SystemExit(main())

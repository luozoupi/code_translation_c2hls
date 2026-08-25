#!/usr/bin/env python3
"""Backfill cosim cycles for selected HLSFactory multistep sweep kernels."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_STAMP = "hlsfactory_multistep_sonnet46_skill_on_trace_cosim1800_20260613"
DEFAULT_SUMMARY = REPO / "artifacts" / f"agentic_no_streamcluster_{DEFAULT_STAMP}.summary.json"
DEFAULT_COMPARISON = REPO / "artifacts" / f"hlsfactory_multistep_vs_flash_skill_trace_{DEFAULT_STAMP}.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _selected_kernel_path(result: dict[str, Any], bench_dir: Path, meta: dict[str, Any], best_step: str) -> Path | None:
    if best_step == "baseline":
        return bench_dir / (meta.get("kernel_file") or meta.get("gold_hls_source_file") or "hls_baseline.cpp")
    for step in result.get("steps") or []:
        if step.get("step_name") != best_step or not step.get("success"):
            continue
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


def _run_case(row: dict[str, Any], timeout: int, run_cosim) -> dict[str, Any]:
    bench = row.get("bench")
    skill_mode = row.get("skill_mode")
    cur = row.get("current") or {}
    best = cur.get("best") or {}
    best_step = best.get("step") or ""
    result_path = Path(cur.get("json") or "")
    bench_dir = Path(row.get("bench_dir") or "")

    base = {
        "bench": bench,
        "skill_mode": skill_mode,
        "best_step": best_step,
        "best_cycles": best.get("cycles"),
        "timeout_sec": timeout,
        "source_result_json": str(result_path),
        "code_path": "",
        "work_dir": "",
        "status": "skipped",
        "passed": False,
        "kernel_runtime_cycles": None,
        "kernel_runtime_us": None,
        "kernel_clock_freq_mhz": None,
        "error": "",
    }

    if not cur.get("success"):
        base["error"] = cur.get("error") or "multistep run did not succeed"
        return base
    if not result_path.exists():
        base["error"] = "result JSON not found"
        return base
    if not bench_dir.exists():
        base["error"] = "benchmark directory not found"
        return base

    result = json.loads(result_path.read_text())
    meta = _load_metadata(bench_dir)
    header_file = meta.get("header_file") or "kernel.h"
    testbench_file = meta.get("testbench_file") or "testbench.cpp"
    top_function = meta.get("translated_hls_top") or meta.get("hls_top") or "workload"
    kernel_path = _selected_kernel_path(result, bench_dir, meta, best_step)
    if not kernel_path or not kernel_path.exists():
        base["error"] = "selected best-step kernel.cpp could not be located"
        base["code_path"] = str(kernel_path or "")
        return base

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
    passed = bool(cosim.get("passed"))
    status = "pass" if passed else ("timeout" if "timed out" in (cosim.get("error") or "").lower() else "fail")
    base.update({
        "status": status,
        "passed": passed,
        "kernel_runtime_cycles": cosim.get("kernel_runtime_cycles"),
        "kernel_runtime_us": cosim.get("kernel_runtime_us"),
        "kernel_clock_freq_mhz": cosim.get("kernel_clock_freq_mhz"),
        "code_path": str(kernel_path),
        "work_dir": cosim.get("work_dir", ""),
        "error": cosim.get("error", ""),
    })
    return base


def _ratio(old_value: Any, new_value: Any) -> float | None:
    try:
        old = float(old_value)
        new = float(new_value)
    except (TypeError, ValueError):
        return None
    if new == 0:
        return None
    return old / new


def _merge_comparison(comparison_csv: Path, supplement_rows: list[dict[str, Any]], out_csv: Path, out_md: Path) -> None:
    rows = _read_csv(comparison_csv)
    supp_by_key = {(row["bench"], row["skill_mode"]): row for row in supplement_rows}
    merged: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = dict(row)
        supp = supp_by_key.get((row.get("bench"), row.get("skill_mode")))
        out.setdefault("multistep_best_cosim", "")
        out.setdefault("multistep_best_cosim_cycles", "")
        out.setdefault("multistep_cosim_supplement_status", "")
        out.setdefault("multistep_cosim_supplement_error", "")
        out.setdefault("multistep_cosim_supplement_work_dir", "")
        out.setdefault("cosim_speedup_flash_over_multistep_best", "")
        if supp:
            out["multistep_best_cosim"] = "True" if supp.get("passed") else "False"
            out["multistep_best_cosim_cycles"] = supp.get("kernel_runtime_cycles") or ""
            out["multistep_cosim_supplement_status"] = supp.get("status") or ""
            out["multistep_cosim_supplement_error"] = supp.get("error") or ""
            out["multistep_cosim_supplement_work_dir"] = supp.get("work_dir") or ""
            ratio = _ratio(row.get("flash_agent_cosim_cycles"), supp.get("kernel_runtime_cycles"))
            out["cosim_speedup_flash_over_multistep_best"] = ratio if ratio is not None else ""
        merged.append(out)
    _write_csv(out_csv, merged)

    success = sum(1 for row in supplement_rows if row.get("passed"))
    timeout = sum(1 for row in supplement_rows if row.get("status") == "timeout")
    skipped = sum(1 for row in supplement_rows if row.get("status") == "skipped")
    lines = [
        "# HLSFactory Multistep Cosim Supplement",
        "",
        f"- comparison source: `{comparison_csv}`",
        f"- output csv: `{out_csv}`",
        f"- supplement rows: {len(supplement_rows)}",
        f"- pass: {success}",
        f"- timeout: {timeout}",
        f"- skipped: {skipped}",
        "",
        "| bench | skill | best step | status | multistep cosim cycles | flash cosim cycles | cosim speedup flash/multistep | error |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in merged:
        if not row.get("multistep_cosim_supplement_status"):
            continue
        lines.append(
            f"| {row.get('bench')} | {row.get('skill_mode')} | {row.get('multistep_best_step') or '-'} | "
            f"{row.get('multistep_cosim_supplement_status')} | {row.get('multistep_best_cosim_cycles') or '-'} | "
            f"{row.get('flash_agent_cosim_cycles') or '-'} | {row.get('cosim_speedup_flash_over_multistep_best') or '-'} | "
            f"{(row.get('multistep_cosim_supplement_error') or '').replace('|', '/')} |"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--benches", default="", help="Comma-separated benchmark names to rerun")
    args = parser.parse_args()

    os.environ.setdefault("C2HLS_TMP_ROOT", "/mnt/data/luo00466/tmp")
    os.environ.setdefault("C2HLS_VITIS_SETTINGS", "/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh")
    os.environ.setdefault("C2HLS_VITIS_VERSION", "2023.2")
    os.environ.setdefault("C2HLS_PART", "xcu280-fsvh2892-2L-e")
    os.environ.setdefault("C2HLS_CLOCK_NS", "3.33")
    os.environ.setdefault("C2HLS_FLOW_TARGET", "vitis")
    os.environ.setdefault("C2HLS_COSIM_TRACE_LEVEL", "none")
    os.environ["C2HLS_COSIM_TIMEOUT"] = str(args.timeout)

    sys.path.insert(0, str(REPO))
    from c2hls_temp import configure_temp_env
    from hls_eval import run_cosim

    configure_temp_env(create=True)
    summary = json.loads(args.summary_json.read_text())
    candidates = [row for row in summary.get("rows") or [] if (row.get("current") or {}).get("success")]
    if args.benches.strip():
        requested = {item.strip() for item in args.benches.split(",") if item.strip()}
        candidates = [row for row in candidates if row.get("bench") in requested]
    if args.limit > 0:
        candidates = candidates[: args.limit]

    prefix = REPO / "artifacts" / f"hlsfactory_multistep_cosim10800_supplement_{args.stamp}"
    supplement_rows: list[dict[str, Any]] = []
    for row in candidates:
        bench = row.get("bench")
        best = ((row.get("current") or {}).get("best") or {}).get("step")
        print(f"START {bench} best={best} timeout={args.timeout}", flush=True)
        result = _run_case(row, args.timeout, run_cosim)
        print(
            f"DONE {bench} status={result['status']} cycles={result.get('kernel_runtime_cycles')} "
            f"error={result.get('error')}",
            flush=True,
        )
        supplement_rows.append(result)
        _write_csv(prefix.with_suffix(".csv"), supplement_rows)
        prefix.with_suffix(".json").write_text(json.dumps(supplement_rows, indent=2) + "\n")
        _merge_comparison(
            args.comparison_csv,
            supplement_rows,
            prefix.with_name(prefix.name + ".comparison.csv"),
            prefix.with_name(prefix.name + ".comparison.md"),
        )

    print(f"SUPPLEMENT_CSV {prefix.with_suffix('.csv')}", flush=True)
    print(f"SUPPLEMENT_JSON {prefix.with_suffix('.json')}", flush=True)
    print(f"COMPARISON_CSV {prefix.with_name(prefix.name + '.comparison.csv')}", flush=True)
    print(f"COMPARISON_MD {prefix.with_name(prefix.name + '.comparison.md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

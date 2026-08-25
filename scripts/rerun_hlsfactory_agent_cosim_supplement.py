#!/usr/bin/env python3
"""Rerun missing HLSFactory agentic cosim rows from an existing merged table.

This is intentionally a Vitis-only supplement. It reuses the selected kernels
already produced by the agentic sweep and reruns cosim with a longer timeout,
instead of spending another LLM sweep.
"""

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
DEFAULT_MERGED = REPO / "artifacts" / "hlsfactory_flash_agentic_vs_direct_merged_cosim10800_20260601.csv"
DEFAULT_SUMMARY = REPO / "artifacts" / "agentic_no_streamcluster_hlsfactory_flash_sonnet46_cosim_skill_onoff_20260528_cosim1800.summary.json"


def _truthy_cosim(value: str) -> bool:
    return value.strip().lower() in {"true", "pass", "passed", "1", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_summary(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text())
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("rows", []):
        bench = row.get("bench")
        skill = row.get("skill_mode")
        if bench and skill:
            out[(bench, skill)] = row
    return out


def _load_metadata(bench_dir: Path) -> dict[str, Any]:
    return json.loads((bench_dir / "metadata.json").read_text())


def _read_optional(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def _extra_files(bench_dir: Path, meta: dict[str, Any], header_file: str) -> list[str]:
    extras: list[str] = []
    for rel in meta.get("support_files") or []:
        p = bench_dir / rel
        if p.exists():
            extras.append(str(p))
    for candidate in bench_dir.glob("*.h"):
        if candidate.name != header_file and str(candidate) not in extras:
            extras.append(str(candidate))
    return extras


def _selected_step(result: dict[str, Any], best_step: str) -> dict[str, Any] | None:
    for step in result.get("steps") or []:
        if step.get("step_name") == best_step and step.get("success"):
            return step
    return None


def _kernel_from_step(step: dict[str, Any] | None) -> Path | None:
    if not step:
        return None
    for container_key in ("cosim", "report"):
        container = step.get(container_key) or {}
        work_dir = container.get("work_dir")
        if work_dir:
            candidate = Path(work_dir) / "kernel.cpp"
            if candidate.exists():
                return candidate
    for attempt in step.get("attempt_results") or []:
        for container_key in ("cosim", "report"):
            container = attempt.get(container_key) or {}
            work_dir = container.get("work_dir")
            if work_dir:
                candidate = Path(work_dir) / "kernel.cpp"
                if candidate.exists():
                    return candidate
    return None


def _needs_agent_cosim(row: dict[str, str]) -> bool:
    cycles = (row.get("agent_step_cosim_cycles") or "").strip()
    cosim = (row.get("agent_step_cosim") or "").strip()
    return not cycles or not _truthy_cosim(cosim)


def _as_bool_cell(value: Any) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    return ""


def _run_case(
    row: dict[str, str],
    summary_row: dict[str, Any],
    timeout: int,
    run_cosim,
) -> dict[str, Any]:
    bench = row["bench"]
    skill = row["skill_mode"]
    result_path = Path(summary_row["current"]["json"])
    result = json.loads(result_path.read_text())
    bench_dir = Path(summary_row["bench_dir"])
    meta = _load_metadata(bench_dir)
    best_step = row.get("best_step") or ((summary_row.get("current") or {}).get("best") or {}).get("step") or ""

    existing_baseline = result.get("baseline_cosim") or {}
    if best_step == "baseline" and existing_baseline.get("passed") and existing_baseline.get("kernel_runtime_cycles") is not None:
        return {
            "bench": bench,
            "skill_mode": skill,
            "best_step": best_step,
            "action": "reused_existing_baseline_cosim",
            "status": "pass",
            "passed": True,
            "kernel_runtime_cycles": existing_baseline.get("kernel_runtime_cycles"),
            "kernel_runtime_us": existing_baseline.get("kernel_runtime_us"),
            "kernel_clock_freq_mhz": existing_baseline.get("kernel_clock_freq_mhz"),
            "timeout_sec": timeout,
            "source_result_json": str(result_path),
            "code_path": str(bench_dir / (meta.get("kernel_file") or meta.get("gold_hls_source_file") or "hls_baseline.cpp")),
            "work_dir": existing_baseline.get("work_dir", ""),
            "error": "",
        }

    if not result.get("success"):
        return {
            "bench": bench,
            "skill_mode": skill,
            "best_step": best_step,
            "action": "skipped",
            "status": "skipped",
            "passed": False,
            "kernel_runtime_cycles": None,
            "kernel_runtime_us": None,
            "kernel_clock_freq_mhz": None,
            "timeout_sec": timeout,
            "source_result_json": str(result_path),
            "code_path": "",
            "work_dir": "",
            "error": "agentic run did not produce a successful selected implementation",
        }

    header_file = meta.get("header_file") or "kernel.h"
    top_function = meta.get("translated_hls_top") or meta.get("hls_top") or "workload"
    testbench_file = meta.get("testbench_file") or "testbench.cpp"
    header_code = _read_optional(bench_dir / header_file)
    testbench_code = _read_optional(bench_dir / testbench_file)
    extra_files = _extra_files(bench_dir, meta, header_file)
    cosim_depths = meta.get("cosim_depths") or {}

    if best_step == "baseline":
        kernel_path = bench_dir / (meta.get("kernel_file") or meta.get("gold_hls_source_file") or "hls_baseline.cpp")
        action = "rerun_baseline_cosim"
    else:
        step = _selected_step(result, best_step)
        kernel_path = _kernel_from_step(step) if step else None
        action = "rerun_selected_step_cosim"

    if not kernel_path or not kernel_path.exists():
        return {
            "bench": bench,
            "skill_mode": skill,
            "best_step": best_step,
            "action": "skipped",
            "status": "skipped",
            "passed": False,
            "kernel_runtime_cycles": None,
            "kernel_runtime_us": None,
            "kernel_clock_freq_mhz": None,
            "timeout_sec": timeout,
            "source_result_json": str(result_path),
            "code_path": str(kernel_path or ""),
            "work_dir": "",
            "error": "selected kernel.cpp could not be located",
        }

    result_cosim = run_cosim(
        kernel_path.read_text(errors="replace"),
        testbench_code,
        header_code,
        header_name=header_file,
        top_function=top_function,
        part=os.getenv("C2HLS_PART", "xcu280-fsvh2892-2L-e"),
        clock_ns=float(os.getenv("C2HLS_CLOCK_NS", "3.33")),
        extra_files=extra_files,
        interface_depths=cosim_depths,
    )
    passed = bool(result_cosim.get("passed"))
    return {
        "bench": bench,
        "skill_mode": skill,
        "best_step": best_step,
        "action": action,
        "status": "pass" if passed else ("timeout" if "timed out" in (result_cosim.get("error") or "").lower() else "fail"),
        "passed": passed,
        "kernel_runtime_cycles": result_cosim.get("kernel_runtime_cycles"),
        "kernel_runtime_us": result_cosim.get("kernel_runtime_us"),
        "kernel_clock_freq_mhz": result_cosim.get("kernel_clock_freq_mhz"),
        "timeout_sec": timeout,
        "source_result_json": str(result_path),
        "code_path": str(kernel_path),
        "work_dir": result_cosim.get("work_dir", ""),
        "error": result_cosim.get("error", ""),
    }


def _merge_rows(original_rows: list[dict[str, str]], supplement_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    supplement_by_key = {(row["bench"], row["skill_mode"]): row for row in supplement_rows}
    merged: list[dict[str, Any]] = []
    for row in original_rows:
        out: dict[str, Any] = dict(row)
        supp = supplement_by_key.get((row["bench"], row["skill_mode"]))
        out["agent_cosim_supplement_status"] = ""
        out["agent_cosim_supplement_timeout_sec"] = ""
        out["agent_cosim_supplement_work_dir"] = ""
        out["agent_cosim_supplement_error"] = ""
        if supp:
            out["agent_cosim_supplement_status"] = supp["status"]
            out["agent_cosim_supplement_timeout_sec"] = supp["timeout_sec"]
            out["agent_cosim_supplement_work_dir"] = supp["work_dir"]
            out["agent_cosim_supplement_error"] = supp["error"]
            if supp["passed"] and supp.get("kernel_runtime_cycles") is not None:
                out["agent_step_cosim"] = "True"
                out["agent_step_cosim_cycles"] = supp["kernel_runtime_cycles"]
        merged.append(out)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-csv", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-failed-agent-runs", action="store_true")
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

    original_rows = _read_csv(args.merged_csv)
    summary_by_key = _load_summary(args.summary_json)
    candidate_rows = [row for row in original_rows if _needs_agent_cosim(row)]
    if not args.include_failed_agent_runs:
        candidate_rows = [row for row in candidate_rows if row.get("agent_success") == "True"]
    if args.limit > 0:
        candidate_rows = candidate_rows[: args.limit]

    out_prefix = REPO / "artifacts" / f"hlsfactory_agent_cosim10800_supplement_{args.stamp}"
    supplement_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        key = (row["bench"], row["skill_mode"])
        summary_row = summary_by_key.get(key)
        if not summary_row:
            supplement = {
                "bench": row["bench"],
                "skill_mode": row["skill_mode"],
                "best_step": row.get("best_step", ""),
                "action": "skipped",
                "status": "skipped",
                "passed": False,
                "kernel_runtime_cycles": None,
                "kernel_runtime_us": None,
                "kernel_clock_freq_mhz": None,
                "timeout_sec": args.timeout,
                "source_result_json": "",
                "code_path": "",
                "work_dir": "",
                "error": "summary row not found",
            }
        else:
            print(f"START {row['bench']} {row['skill_mode']} best={row.get('best_step')} timeout={args.timeout}", flush=True)
            supplement = _run_case(row, summary_row, args.timeout, run_cosim)
            print(
                f"DONE {row['bench']} {row['skill_mode']} status={supplement['status']} "
                f"cycles={supplement.get('kernel_runtime_cycles')} error={supplement.get('error')}",
                flush=True,
            )
        supplement_rows.append(supplement)
        _write_csv(out_prefix.with_suffix(".csv"), supplement_rows)
        out_prefix.with_suffix(".json").write_text(json.dumps(supplement_rows, indent=2) + "\n")
        merged_rows = _merge_rows(original_rows, supplement_rows)
        _write_csv(out_prefix.with_name(out_prefix.name + ".merged.csv"), merged_rows)

    print(f"SUPPLEMENT_CSV {out_prefix.with_suffix('.csv')}", flush=True)
    print(f"SUPPLEMENT_JSON {out_prefix.with_suffix('.json')}", flush=True)
    print(f"MERGED_CSV {out_prefix.with_name(out_prefix.name + '.merged.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

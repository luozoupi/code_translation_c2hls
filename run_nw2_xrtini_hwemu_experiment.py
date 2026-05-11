#!/usr/bin/env python3
"""Run nw_2_pipeline hw_emu with xrt.ini debug waveform disabled.

This is a focused experiment for XSIM/WDB crash diagnosis. It stages the Nova
variant, writes xrt.ini in the staged benchmark working directory, runs
`make check TARGET=hw_emu`, and emits one canonical schema-1.0 rtl_sim record.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

os.environ.setdefault(
    "C2HLS_VITIS_SETTINGS",
    "/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh",
)
os.environ.setdefault("C2HLS_EMU_ENV_SCRIPT", str(REPO / "scripts" / "setup_emu_env.sh"))
os.environ.setdefault("C2HLS_DEVICE_PLATFORM", "xilinx_u280_gen3x16_xdma_1_202211_1")
os.environ.setdefault("C2HLS_VITIS_VERSION", "2023.2")

import hls_eval  # noqa: E402
from export_schema_jsonl import SCHEMA_VERSION, validate_jsonl  # noqa: E402

DEVICE = os.getenv("C2HLS_DEVICE_PLATFORM", "xilinx_u280_gen3x16_xdma_1_202211_1")
VITIS_VERSION = os.getenv("C2HLS_VITIS_VERSION", "2023.2")
NOVA_VARIANT = Path(os.getenv(
    "C2HLS_NW2_VARIANT_DIR",
    "/home/luo00466/rodinia-hls-nova/Benchmarks/nw/nw_2_pipeline",
))
OUT_JSONL = Path(os.getenv(
    "C2HLS_NW2_XRT_JSONL",
    str(REPO / "artifacts" / "nw2_pipeline_hwemu_xrt_debug_off.jsonl"),
))
SUMMARY_MD = OUT_JSONL.with_suffix(".md")
TIMEOUT = int(os.getenv("C2HLS_HW_EMU_TIMEOUT", "86400"))
XRT_INI = """[Emulation]
debug_mode=off

[Debug]
timeline_trace = true
profile = true
"""


def _status(result: dict) -> str:
    if result.get("success"):
        return "pass"
    if "timed out" in (result.get("error") or "").lower():
        return "timeout"
    return "fail"


def _make_record(result: dict, elapsed: float) -> dict:
    status = _status(result)
    rtl = {
        "status": status,
        "kernel_runtime_cycles": result.get("kernel_runtime_cycles"),
        "kernel_runtime_us": result.get("kernel_runtime_us"),
        "kernel_clock_freq_mhz": result.get("kernel_clock_freq_mhz"),
    }
    if status != "pass":
        rtl["error"] = (result.get("error") or "hw_emu failed")[:300]
    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "rtl_sim",
        "run": {
            "target": "vitis.hw_emu",
            "device": DEVICE,
            "vitis_version": VITIS_VERSION,
            "runtime_seconds": round(elapsed, 6),
        },
        "problem": {"suite": "rodinia_hls", "group_path": ["nw"]},
        "implementation": {
            "origin": "rodinia_hls_benchmark",
            "origin_version": "2023_port",
            "origin_meta": {
                "source_variant": "nw_2_pipeline",
                "kernel_basename": "nw",
                "experiment": "xrt_debug_mode_off_profile_on",
                "xrt_ini_path": result.get("xrt_ini_path") or None,
                "xrt_ini_content": XRT_INI,
                "work_dir": result.get("work_dir") or None,
                "make_check_log": result.get("log_path") or None,
                "profile_csv": result.get("profile_csv") or None,
                "profile_compute_unit_rows": result.get("profile_compute_unit_rows"),
                "system_diagram_model": result.get("system_diagram_model") or None,
                "crash_log": result.get("crash_log") or None,
                "crash_summary": result.get("crash_summary") or None,
                "clock_source": result.get("clock_source") or None,
                "clock_fallback": result.get("clock_fallback"),
                "direct_script": Path(__file__).name,
            },
            "variant": {"index": 2, "name": "pipeline"},
        },
        "rtl_sim": rtl,
    }


def _write_summary(record: dict) -> None:
    meta = record["implementation"]["origin_meta"]
    rtl = record["rtl_sim"]
    lines = [
        "# NW 2 Pipeline hw_emu xrt.ini Experiment",
        "",
        f"variant: `nw:2:pipeline`",
        f"xrt.ini: `{meta.get('xrt_ini_path')}`",
        f"status: `{rtl.get('status')}`",
        f"cycles: `{rtl.get('kernel_runtime_cycles')}`",
        f"runtime_us: `{rtl.get('kernel_runtime_us')}`",
        f"clock_mhz: `{rtl.get('kernel_clock_freq_mhz')}`",
        f"profile_csv: `{meta.get('profile_csv')}`",
        f"crash_log: `{meta.get('crash_log')}`",
        f"make_check_log: `{meta.get('make_check_log')}`",
        "",
        "```ini",
        XRT_INI.rstrip(),
        "```",
    ]
    if rtl.get("error"):
        lines.extend(["", f"error: `{rtl['error']}`"])
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def _run() -> dict:
    staged, info = hls_eval._stage_nova_workdir(str(NOVA_VARIANT), kernel_basename="nw")
    if staged is None:
        return {
            "ran": False,
            "passed": False,
            "success": False,
            "kernel_runtime_us": None,
            "kernel_runtime_cycles": None,
            "kernel_clock_freq_mhz": None,
            "profile_csv": "",
            "profile_compute_unit_rows": 0,
            "system_diagram_model": "",
            "crash_log": "",
            "crash_summary": "",
            "clock_source": "not_run",
            "clock_fallback": False,
            "error": info,
            "work_dir": "",
            "log_path": "",
            "xrt_ini_path": "",
        }

    xrt_path = staged / "xrt.ini"
    xrt_path.write_text(XRT_INI)
    output, timed_out = hls_eval._run_make_check_emu(staged, "hw_emu", TIMEOUT)
    log_path = staged / "c2hls_hw_emu_xrt_debug_off_make_check.log"
    log_path.write_text(output or "", encoding="utf-8", errors="ignore")

    log_lower = output.lower()
    passed = (
        "finished checking data: correct" in log_lower
        and re.search(r"^Success\.\s*$", output, re.MULTILINE) is not None
    )
    profile_csv = hls_eval._choose_latest(list(staged.glob(".run/**/profile_kernels.csv")))
    kernel_runtime_us, profile_rows = hls_eval._parse_runtime_us(profile_csv)
    crash_log, crash_summary = hls_eval._find_hw_emu_crash_marker(staged)
    kernel_clock_freq_mhz, system_diagram_model, clock_fallback, clock_source = (
        hls_eval._resolve_hw_emu_clock(staged)
    )
    kernel_runtime_cycles = None
    if kernel_runtime_us is not None and kernel_clock_freq_mhz is not None:
        kernel_runtime_cycles = round(kernel_runtime_us * kernel_clock_freq_mhz)

    success = (not timed_out) and (kernel_runtime_us is not None) and passed
    error = ""
    if timed_out:
        error = f"hw_emu timed out after {TIMEOUT}s"
    elif not success:
        for line in reversed(output.splitlines()[-200:]):
            line = line.strip()
            if line.lower().startswith("error:") or "ERROR" in line[:6]:
                error = line[:300]
                break
        if not error:
            if kernel_runtime_us is None:
                if crash_summary:
                    error = f"hw_emu simulator crash before profile_kernels.csv: {crash_summary}"
                else:
                    error = "hw_emu did not produce profile_kernels.csv"
            elif not passed:
                error = "testbench check failed"
            elif kernel_clock_freq_mhz is None:
                error = "hw_emu clock unavailable from systemDiagramModel.json"

    return {
        "ran": True,
        "passed": passed,
        "success": success,
        "kernel_runtime_us": kernel_runtime_us,
        "kernel_runtime_cycles": kernel_runtime_cycles,
        "kernel_clock_freq_mhz": kernel_clock_freq_mhz,
        "profile_csv": str(profile_csv) if profile_csv else "",
        "profile_compute_unit_rows": profile_rows,
        "system_diagram_model": system_diagram_model or "",
        "crash_log": crash_log,
        "crash_summary": crash_summary,
        "clock_fallback": clock_fallback,
        "clock_source": clock_source,
        "error": error,
        "work_dir": str(staged),
        "log_path": str(log_path),
        "xrt_ini_path": str(xrt_path),
    }


def main() -> int:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    print(f"variant={NOVA_VARIANT}", flush=True)
    print(f"output={OUT_JSONL}", flush=True)
    print("xrt.ini content:", flush=True)
    print(XRT_INI.rstrip(), flush=True)
    t0 = time.time()
    result = _run()
    elapsed = time.time() - t0
    record = _make_record(result, elapsed)
    OUT_JSONL.write_text(json.dumps(record) + "\n")
    _write_summary(record)
    validation = validate_jsonl(OUT_JSONL)
    print(json.dumps(validation), flush=True)
    print(
        f"DONE status={record['rtl_sim']['status']} "
        f"cycles={record['rtl_sim'].get('kernel_runtime_cycles')} "
        f"work_dir={record['implementation']['origin_meta'].get('work_dir')} "
        f"log={record['implementation']['origin_meta'].get('make_check_log')}",
        flush=True,
    )
    return 1 if validation["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

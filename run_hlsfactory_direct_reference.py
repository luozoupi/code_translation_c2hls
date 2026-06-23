#!/usr/bin/env python3
"""Direct Vitis validation of materialized HLSFactory reference kernels.

This runner is intentionally no-LLM: it synthesizes each HLSFactory
`hls_baseline.cpp` reference variant and, when a testbench is available,
runs Vitis C simulation and C/RTL co-simulation. The output is schema-1.0
JSONL so these records can be compared against agentic C2HLS outputs without
overloading the word "baseline".

Default corpus:
  benchmarks_external/HLSFactory/polybench_float_small/*

Outputs:
  artifacts/hlsfactory_direct_reference_<stamp>.jsonl
  artifacts/hlsfactory_direct_reference_<stamp>.summary.json
  artifacts/hlsfactory_direct_reference_<stamp>.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from c2hls_paths import BENCHMARKS_EXTERNAL_DIR, apply_runtime_defaults, configure_site

configure_site()
apply_runtime_defaults()

import hls_eval  # noqa: E402
from c2hls import _ground_truth_candidates, _load_benchmark_inputs  # noqa: E402
from export_schema_jsonl import (  # noqa: E402
    SCHEMA_VERSION,
    TARGET_CSIM,
    TARGET_COSIM,
    TARGET_CSYNTH,
    _build_hls_synth_payload,
    _build_implementation,
    _build_problem,
    _build_run,
    validate_jsonl,
)

STAMP = os.getenv("C2HLS_HLSFACTORY_DIRECT_STAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
HLSFACTORY_ROOT = Path(
    os.getenv(
        "C2HLS_HLSFACTORY_ROOT",
        str(BENCHMARKS_EXTERNAL_DIR / "HLSFactory" / "polybench_float_small"),
    )
)
OUT_JSONL = Path(
    os.getenv(
        "C2HLS_HLSFACTORY_DIRECT_JSONL",
        str(REPO / "artifacts" / f"hlsfactory_direct_reference_{STAMP}.jsonl"),
    )
)
OUT_SUMMARY = Path(
    os.getenv(
        "C2HLS_HLSFACTORY_DIRECT_SUMMARY",
        str(OUT_JSONL.with_suffix(".summary.json")),
    )
)
OUT_MD = Path(
    os.getenv(
        "C2HLS_HLSFACTORY_DIRECT_MD",
        str(OUT_JSONL.with_suffix(".md")),
    )
)


def _selected_bench_dirs() -> list[Path]:
    raw = os.getenv("C2HLS_HLSFACTORY_BENCHES", "").strip()
    all_dirs = sorted(p for p in HLSFACTORY_ROOT.iterdir() if (p / "metadata.json").is_file())
    if raw:
        wanted = {item.strip() for item in raw.split(",") if item.strip()}
        all_dirs = [
            p for p in all_dirs
            if p.name in wanted or p.name.removeprefix("hlsfactory_") in wanted
        ]
    max_raw = os.getenv("C2HLS_HLSFACTORY_MAX_BENCHES", "").strip()
    if max_raw:
        all_dirs = all_dirs[: max(0, int(max_raw))]
    return all_dirs


def _status_from_result(result: dict[str, Any] | None, *, passed_key: bool = False) -> str:
    if not result:
        return "not_run"
    if passed_key:
        if result.get("success") and result.get("passed"):
            return "pass"
    elif result.get("success"):
        return "pass"
    err = str(result.get("error") or "")
    if "timed out" in err.lower() or "timeout" in err.lower():
        return "timeout"
    return "fail"


def _top_model_payload(report: dict[str, Any], status: str, top: str) -> dict[str, Any]:
    payload = _build_hls_synth_payload(
        report,
        hls_eval.DEFAULT_PART,
        hls_eval.DEFAULT_CLOCK_NS,
        status=status,
    )
    ua = payload.get("UserAssignments")
    if isinstance(ua, dict):
        ua["TopModelName"] = top
    return payload


def _jsonl_record(
    *,
    report_type: str,
    target: str,
    runtime_seconds: float | None,
    meta: dict[str, Any],
    variant_name: str,
    origin_meta: dict[str, Any],
    payload_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": report_type,
        "run": _build_run(
            target,
            hls_eval.DEFAULT_PART,
            runtime_seconds,
            {
                "vitis_version": os.getenv("C2HLS_VITIS_VERSION", "2023.2"),
                "flow_target": hls_eval.DEFAULT_FLOW_TARGET,
                "clock_ns": hls_eval.DEFAULT_CLOCK_NS,
            },
        ),
        "problem": _build_problem(meta),
        "implementation": _build_implementation(
            meta,
            variant_name=variant_name,
            origin_version="direct_vitis_2023.2",
            origin_meta=origin_meta,
        ),
        payload_key: payload,
    }


def _write_outputs(rows: list[dict[str, Any]], records_written: int) -> None:
    OUT_SUMMARY.write_text(
        json.dumps(
            {
                "stamp": STAMP,
                "root": str(HLSFACTORY_ROOT),
                "jsonl": str(OUT_JSONL),
                "jsonl_records": records_written,
                "vitis": {
                    "settings": os.getenv("C2HLS_VITIS_SETTINGS"),
                    "version": os.getenv("C2HLS_VITIS_VERSION"),
                    "part": hls_eval.DEFAULT_PART,
                    "clock_ns": hls_eval.DEFAULT_CLOCK_NS,
                    "flow_target": hls_eval.DEFAULT_FLOW_TARGET,
                },
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )

    lines = [
        "# HLSFactory Direct Reference Vitis Results",
        "",
        f"- JSONL: `{OUT_JSONL}`",
        f"- Vitis: `{os.getenv('C2HLS_VITIS_VERSION')}` / `{hls_eval.DEFAULT_PART}` / `{hls_eval.DEFAULT_CLOCK_NS} ns`",
        "",
        "| bench | synth | csim | cosim | cycles | cosim cycles | latency ns | fmax MHz | BRAM | DSP | FF | LUT | error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        report = row.get("report") or {}
        err = row.get("error") or row.get("csim_error") or row.get("cosim_error") or ""
        if len(err) > 120:
            err = err[:117] + "..."
        lines.append(
            "| {bench} | {synth_status} | {csim_status} | {cosim_status} | "
            "{cycles} | {cosim_cycles} | {lat_ns} | {fmax} | {bram} | {dsp} | "
            "{ff} | {lut} | {err} |".format(
                bench=row.get("bench", ""),
                synth_status=row.get("synth_status", "-"),
                csim_status=row.get("csim_status", "-"),
                cosim_status=row.get("cosim_status", "-"),
                cycles=report.get("latency_cycles", "-"),
                cosim_cycles=row.get("cosim_cycles", "-"),
                lat_ns=report.get("latency_ns", "-"),
                fmax=report.get("fmax_mhz", "-"),
                bram=report.get("bram", "-"),
                dsp=report.get("dsp", "-"),
                ff=report.get("ff", "-"),
                lut=report.get("lut", "-"),
                err=err.replace("|", "/"),
            )
        )
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    bench_dirs = _selected_bench_dirs()
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.write_text("")
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    run_csim = os.getenv("C2HLS_HLSFACTORY_DIRECT_CSIM", "1") != "0"
    run_cosim = os.getenv("C2HLS_HLSFACTORY_DIRECT_COSIM", "1") != "0"
    rows: list[dict[str, Any]] = []
    records_written = 0

    print(f"root={HLSFACTORY_ROOT}", flush=True)
    print(f"benches={len(bench_dirs)} jsonl={OUT_JSONL}", flush=True)
    print(
        f"vitis={os.getenv('C2HLS_VITIS_VERSION')} part={hls_eval.DEFAULT_PART} "
        f"clock={hls_eval.DEFAULT_CLOCK_NS} csim={run_csim} cosim={run_cosim}",
        flush=True,
    )

    for idx, bench_dir in enumerate(bench_dirs, 1):
        inputs = _load_benchmark_inputs(str(bench_dir))
        meta = inputs["meta"]
        top = meta.get("hls_top", "workload")
        candidates = _ground_truth_candidates(inputs)
        if not candidates:
            row = {
                "bench": bench_dir.name,
                "synth_status": "fail",
                "csim_status": "not_run",
                "cosim_status": "not_run",
                "error": "missing ground-truth HLS candidate",
            }
            rows.append(row)
            _write_outputs(rows, records_written)
            continue

        preferred = meta.get("preferred_gt_file") or meta.get("gold_hls_baseline_file")
        candidate = next((c for c in candidates if c.get("file") == preferred), candidates[0])
        variant_name = candidate.get("variant_name") or f"{bench_dir.name}_0_baseline"
        print(f"[{idx}/{len(bench_dirs)}] {bench_dir.name} variant={variant_name}", flush=True)

        synth_t0 = time.time()
        synth = hls_eval.run_hls_synthesis(
            candidate["code"],
            candidate.get("header_code", inputs.get("header_code", "")),
            header_name=candidate.get("header_name") or inputs.get("header_name") or "kernel.h",
            top_function=top,
            part=hls_eval.DEFAULT_PART,
            clock_ns=hls_eval.DEFAULT_CLOCK_NS,
            extra_files=inputs.get("extra_files", []),
        )
        synth_elapsed = round(time.time() - synth_t0, 3)
        synth_status = _status_from_result(synth)
        report = synth.get("report") or {}

        origin_meta = {
            "direct_script": Path(__file__).name,
            "source_repo": meta.get("source_repo"),
            "source_path": candidate.get("source_path"),
            "source_file": candidate.get("file"),
            "reference_role": "hlsfactory_ground_truth",
            "synth_work_dir": synth.get("work_dir"),
            "error": (synth.get("error") or "")[:300] if synth_status != "pass" else None,
        }
        synth_record = _jsonl_record(
            report_type="hls_synth",
            target=TARGET_CSYNTH,
            runtime_seconds=synth_elapsed,
            meta=meta,
            variant_name=variant_name,
            origin_meta=origin_meta,
            payload_key="hls_synth",
            payload=_top_model_payload(report, synth_status, top),
        )
        with OUT_JSONL.open("a") as f:
            f.write(json.dumps(synth_record) + "\n")
        records_written += 1

        csim = None
        csim_elapsed = None
        csim_status = "not_run"
        if synth_status == "pass" and run_csim and inputs.get("testbench_code") and meta.get("supports_csim"):
            csim_t0 = time.time()
            csim = hls_eval.run_csim(
                candidate["code"],
                inputs.get("testbench_code", ""),
                candidate.get("header_code", inputs.get("header_code", "")),
                header_name=candidate.get("header_name") or inputs.get("header_name") or "kernel.h",
                top_function=top,
                part=hls_eval.DEFAULT_PART,
                clock_ns=hls_eval.DEFAULT_CLOCK_NS,
                extra_files=inputs.get("extra_files", []),
            )
            csim_elapsed = round(time.time() - csim_t0, 3)
            csim_status = _status_from_result(csim, passed_key=True)
            csim_record = _jsonl_record(
                report_type="sw_run",
                target=TARGET_CSIM,
                runtime_seconds=csim_elapsed,
                meta=meta,
                variant_name=variant_name,
                origin_meta={
                    **origin_meta,
                    "csim_work_dir": csim.get("work_dir") if isinstance(csim, dict) else None,
                    "error": (csim.get("error") or "")[:300] if csim_status != "pass" else None,
                },
                payload_key="sw_run",
                payload={
                    "status": csim_status,
                    "error": (csim.get("error") or "")[:300] if csim_status != "pass" else None,
                },
            )
            with OUT_JSONL.open("a") as f:
                f.write(json.dumps(csim_record) + "\n")
            records_written += 1

        cosim = None
        cosim_elapsed = None
        cosim_status = "not_run"
        if synth_status == "pass" and run_cosim and inputs.get("testbench_code") and meta.get("supports_cosim"):
            cosim_t0 = time.time()
            cosim = hls_eval.run_cosim(
                candidate["code"],
                inputs.get("testbench_code", ""),
                candidate.get("header_code", inputs.get("header_code", "")),
                header_name=candidate.get("header_name") or inputs.get("header_name") or "kernel.h",
                top_function=top,
                part=hls_eval.DEFAULT_PART,
                clock_ns=hls_eval.DEFAULT_CLOCK_NS,
                extra_files=inputs.get("extra_files", []),
                interface_depths=meta.get("cosim_depths") or {},
            )
            cosim_elapsed = round(time.time() - cosim_t0, 3)
            cosim_status = _status_from_result(cosim, passed_key=True)
            cosim_record = _jsonl_record(
                report_type="rtl_sim",
                target=TARGET_COSIM,
                runtime_seconds=cosim_elapsed,
                meta=meta,
                variant_name=variant_name,
                origin_meta={
                    **origin_meta,
                    "cosim_work_dir": cosim.get("work_dir") if isinstance(cosim, dict) else None,
                    "error": (cosim.get("error") or "")[:300] if cosim_status != "pass" else None,
                },
                payload_key="rtl_sim",
                payload={
                    "status": cosim_status,
                    "kernel_runtime_cycles": cosim.get("kernel_runtime_cycles") if isinstance(cosim, dict) else None,
                    "kernel_runtime_us": cosim.get("kernel_runtime_us") if isinstance(cosim, dict) else None,
                    "kernel_clock_freq_mhz": cosim.get("kernel_clock_freq_mhz") if isinstance(cosim, dict) else None,
                    "error": (cosim.get("error") or "")[:300] if cosim_status != "pass" else None,
                },
            )
            with OUT_JSONL.open("a") as f:
                f.write(json.dumps(cosim_record) + "\n")
            records_written += 1

        row = {
            "bench": bench_dir.name,
            "variant": variant_name,
            "synth_status": synth_status,
            "synth_runtime_seconds": synth_elapsed,
            "csim_status": csim_status,
            "csim_runtime_seconds": csim_elapsed,
            "cosim_status": cosim_status,
            "cosim_runtime_seconds": cosim_elapsed,
            "cosim_cycles": (
                cosim.get("kernel_runtime_cycles")
                if isinstance(cosim, dict)
                else None
            ),
            "report": report,
            "error": (synth.get("error") or "")[:300] if synth_status != "pass" else "",
            "csim_error": (
                (csim.get("error") or "")[:300]
                if isinstance(csim, dict) and csim_status != "pass"
                else ""
            ),
            "cosim_error": (
                (cosim.get("error") or "")[:300]
                if isinstance(cosim, dict) and cosim_status != "pass"
                else ""
            ),
            "synth_work_dir": synth.get("work_dir"),
            "csim_work_dir": csim.get("work_dir") if isinstance(csim, dict) else None,
            "cosim_work_dir": cosim.get("work_dir") if isinstance(cosim, dict) else None,
        }
        rows.append(row)
        _write_outputs(rows, records_written)
        print(
            f"  synth={synth_status} csim={csim_status} cosim={cosim_status} "
            f"cycles={report.get('latency_cycles')} ns={report.get('latency_ns')} "
            f"cosim_cycles={row.get('cosim_cycles')} fmax={report.get('fmax_mhz')}",
            flush=True,
        )

    validation = validate_jsonl(OUT_JSONL)
    if validation.get("invalid"):
        print(f"ERROR invalid_jsonl={validation['invalid']} path={OUT_JSONL}", file=sys.stderr)
        return 1
    _write_outputs(rows, records_written)
    print(f"validated records={records_written} path={OUT_JSONL}", flush=True)
    print(f"summary={OUT_SUMMARY}", flush=True)
    print(f"markdown={OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

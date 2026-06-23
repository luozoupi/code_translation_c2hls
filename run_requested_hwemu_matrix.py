#!/usr/bin/env python3
"""Run the requested six-benchmark direct hw_emu matrix.

This is intentionally direct/no-LLM: each upstream Nova/Rodinia variant is
staged and run with `make check TARGET=hw_emu`. Every attempted variant emits
one schema-1.0 `rtl_sim` JSONL record. The runner is resumable: existing
records in the output JSONL are skipped unless C2HLS_FORCE_RERUN=1.
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

from c2hls_paths import active_site, apply_runtime_defaults, configure_site, rodinia_nova_benchmarks_dir

configure_site()
apply_runtime_defaults()

import hls_eval  # noqa: E402
from export_schema_jsonl import SCHEMA_VERSION, validate_jsonl  # noqa: E402

_nova = rodinia_nova_benchmarks_dir()
if _nova is None or not _nova.is_dir():
    if active_site() == "pc2":
        raise SystemExit("Set C2HLS_RODINIA_NOVA_DIR in local.env (see local.env.example).")
    raise SystemExit(f"Nova benchmarks not found: {_nova}")
NOVA_ROOT = _nova
REF_HW = REPO / "results" / "references_philip" / "hw_emu_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl"
OUT_JSONL = Path(os.getenv("C2HLS_HWEMU_MATRIX_JSONL", str(REPO / "artifacts" / "requested_hwemu_matrix.jsonl")))
SUMMARY_MD = OUT_JSONL.with_suffix(".md")

DEVICE = os.getenv("C2HLS_DEVICE_PLATFORM", "xilinx_u280_gen3x16_xdma_1_202211_1")
VITIS_VERSION = os.getenv("C2HLS_VITIS_VERSION", "2023.2")
FORCE_RERUN = os.getenv("C2HLS_FORCE_RERUN", "").lower() in {"1", "true", "yes"}
ORDER = os.getenv("C2HLS_HWEMU_MATRIX_ORDER", "reference_runtime")

BENCHES = [
    (("knn",), NOVA_ROOT / "knn", "knn"),
    (("lud",), NOVA_ROOT / "lud", "lud"),
    (("pathfinder",), NOVA_ROOT / "pathfinder", "pathfinder"),
    (("cfd", "cfd_step_factor"), NOVA_ROOT / "cfd" / "cfd_step_factor", "cfd_step_factor"),
    (("leukocyte", "lc_dilate"), NOVA_ROOT / "leukocyte" / "lc_dilate", "dilate"),
    (("nw",), NOVA_ROOT / "nw", "nw"),
]


def _variant_identity(variant_name: str) -> tuple[int, str]:
    match = re.match(r"^.+_(\d+)_(.+)$", variant_name)
    if not match:
        return 0, variant_name or "implementation"
    short = match.group(2)
    short = short.replace("unrolling", "unroll").replace("double_buffer", "doublebuffer")
    return int(match.group(1)), short


def _record_key(record: dict) -> tuple:
    variant = record["implementation"]["variant"]
    return (
        tuple(record["problem"]["group_path"]),
        int(variant["index"]),
        variant["name"],
        record["report_type"],
    )


def _load_reference() -> dict:
    out = {}
    if not REF_HW.exists():
        return out
    for line in REF_HW.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        out[_record_key(record)] = record
    return out


def _existing_keys(path: Path) -> set[tuple]:
    if FORCE_RERUN or not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            keys.add(_record_key(json.loads(line)))
        except (KeyError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return keys


def _jobs(reference: dict) -> list[dict]:
    jobs = []
    bench_order = {group_path: index for index, (group_path, _, _) in enumerate(BENCHES)}
    for group_path, parent, kernel_basename in BENCHES:
        if not parent.is_dir():
            jobs.append({
                "missing": True,
                "group_path": group_path,
                "parent": parent,
                "kernel_basename": kernel_basename,
                "variant_dir": None,
                "variant_name": f"{kernel_basename}_0_missing",
                "variant_index": 0,
                "variant_short": "missing",
                "ref_runtime_seconds": None,
            })
            continue
        for variant_dir in sorted(p for p in parent.iterdir() if p.is_dir() and (p / "Makefile").exists()):
            index, short = _variant_identity(variant_dir.name)
            ref = reference.get((group_path, index, short, "rtl_sim"))
            jobs.append({
                "missing": False,
                "group_path": group_path,
                "parent": parent,
                "kernel_basename": kernel_basename,
                "variant_dir": variant_dir,
                "variant_name": variant_dir.name,
                "variant_index": index,
                "variant_short": short,
                "ref_runtime_seconds": (ref or {}).get("run", {}).get("runtime_seconds"),
                "ref_record": ref,
            })
    if ORDER == "reference_runtime":
        jobs.sort(key=lambda job: (
            job.get("ref_runtime_seconds") is None,
            job.get("ref_runtime_seconds") or 10**12,
            bench_order.get(job["group_path"], 999),
            job["variant_index"],
        ))
    return jobs


def _status(result: dict) -> str:
    if result.get("success"):
        return "pass"
    if "timed out" in (result.get("error") or "").lower():
        return "timeout"
    return "fail"


def _record_for_job(job: dict, result: dict, elapsed: float) -> dict:
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
        "problem": {"suite": "rodinia_hls", "group_path": list(job["group_path"])},
        "implementation": {
            "origin": "rodinia_hls_benchmark",
            "origin_version": "2023_port",
            "origin_meta": {
                "source_variant": job["variant_name"],
                "kernel_basename": job["kernel_basename"],
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
            "variant": {
                "index": int(job["variant_index"]),
                "name": job["variant_short"],
            },
        },
        "rtl_sim": rtl,
    }


def _missing_record(job: dict) -> dict:
    return _record_for_job(
        job,
        {
            "success": False,
            "kernel_runtime_cycles": None,
            "kernel_runtime_us": None,
            "kernel_clock_freq_mhz": None,
            "error": f"benchmark directory missing: {job['parent']}",
            "clock_source": "not_run",
            "clock_fallback": False,
        },
        0.0,
    )


def _append_record(record: dict) -> None:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()


def _write_summary(records: list[dict], reference: dict) -> None:
    lines = [
        "# Requested Direct hw_emu Matrix",
        "",
        f"Vitis {VITIS_VERSION} / {DEVICE}",
        "",
        "| problem | variant | status | ref_status | cycles | ref_cycles | delta_cycles | clock_mhz |",
        "|---|---:|:---:|:---:|---:|---:|---:|---:|",
    ]
    for record in records:
        key = _record_key(record)
        ref = reference.get(key)
        rtl = record["rtl_sim"]
        ref_rtl = (ref or {}).get("rtl_sim", {})
        cycles = rtl.get("kernel_runtime_cycles")
        ref_cycles = ref_rtl.get("kernel_runtime_cycles")
        delta = cycles - ref_cycles if isinstance(cycles, int) and isinstance(ref_cycles, int) else None
        lines.append(
            f"| {'/'.join(key[0])} | {key[1]} {key[2]} | {rtl.get('status')} | "
            f"{ref_rtl.get('status') or '-'} | {cycles if cycles is not None else '-'} | "
            f"{ref_cycles if ref_cycles is not None else '-'} | {delta if delta is not None else '-'} | "
            f"{rtl.get('kernel_clock_freq_mhz') if rtl.get('kernel_clock_freq_mhz') is not None else '-'} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    reference = _load_reference()
    existing = _existing_keys(OUT_JSONL)
    jobs = _jobs(reference)
    print(f"output={OUT_JSONL}", flush=True)
    print(f"jobs={len(jobs)} existing={len(existing)} force_rerun={FORCE_RERUN} order={ORDER}", flush=True)
    if not FORCE_RERUN:
        OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSONL.touch(exist_ok=True)
    else:
        OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSONL.write_text("")

    emitted_records = []
    for job in jobs:
        key = (job["group_path"], job["variant_index"], job["variant_short"], "rtl_sim")
        label = f"{'/'.join(job['group_path'])}:{job['variant_index']}:{job['variant_short']}"
        if key in existing:
            print(f"SKIP existing {label}", flush=True)
            continue
        if job.get("missing"):
            print(f"MISSING {label}: {job['parent']}", flush=True)
            record = _missing_record(job)
            _append_record(record)
            emitted_records.append(record)
            continue

        print(f"START {label} dir={job['variant_dir']}", flush=True)
        t0 = time.time()
        try:
            result = hls_eval.run_hw_emu_via_nova(
                str(job["variant_dir"]),
                kernel_basename=job["kernel_basename"],
                timeout=int(os.getenv("C2HLS_HW_EMU_TIMEOUT", "86400")),
            )
        except Exception as exc:
            result = {
                "success": False,
                "passed": False,
                "kernel_runtime_cycles": None,
                "kernel_runtime_us": None,
                "kernel_clock_freq_mhz": None,
                "error": f"exception: {exc}",
                "clock_source": "exception",
                "clock_fallback": False,
            }
        elapsed = time.time() - t0
        record = _record_for_job(job, result, elapsed)
        _append_record(record)
        emitted_records.append(record)
        rtl = record["rtl_sim"]
        print(
            f"DONE {label} status={rtl.get('status')} us={rtl.get('kernel_runtime_us')} "
            f"cycles={rtl.get('kernel_runtime_cycles')} clock={rtl.get('kernel_clock_freq_mhz')} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        validation = validate_jsonl(OUT_JSONL)
        if validation["invalid"]:
            print(f"ERROR invalid_jsonl={validation['invalid']} path={OUT_JSONL}", file=sys.stderr, flush=True)
            return 1

    records = []
    for line in OUT_JSONL.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    _write_summary(records, reference)
    validation = validate_jsonl(OUT_JSONL)
    print(f"validated total={validation['total']} invalid={validation['invalid']}", flush=True)
    print(f"summary={SUMMARY_MD}", flush=True)
    return 1 if validation["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Targeted direct hw_emu rerun for reference/local status mismatches.

This script is intentionally smaller than the full matrix runner. It reads the
current direct-matrix JSONL, compares it to results/references_philip, and
reruns only rows whose status differs unless C2HLS_RERUN_VARIANTS is supplied.
Outputs are written to a separate JSONL artifact and never overwrite the
reference files.
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
import run_requested_hwemu_matrix as matrix  # noqa: E402
from export_schema_jsonl import SCHEMA_VERSION, validate_jsonl  # noqa: E402

OUT_JSONL = Path(os.getenv(
    "C2HLS_HWEMU_RERUN_JSONL",
    str(REPO / "artifacts" / "requested_hwemu_mismatch_rerun.jsonl"),
))
SUMMARY_MD = OUT_JSONL.with_suffix(".md")
DIRECT_JSONL = Path(os.getenv(
    "C2HLS_HWEMU_MATRIX_JSONL",
    str(REPO / "artifacts" / "requested_hwemu_matrix.jsonl"),
))
DEVICE = os.getenv("C2HLS_DEVICE_PLATFORM", "xilinx_u280_gen3x16_xdma_1_202211_1")
VITIS_VERSION = os.getenv("C2HLS_VITIS_VERSION", "2023.2")
FORCE_RERUN = os.getenv("C2HLS_FORCE_RERUN", "").lower() in {"1", "true", "yes"}


def _record_key(record: dict) -> tuple:
    variant = record["implementation"]["variant"]
    return (
        tuple(record["problem"]["group_path"]),
        int(variant["index"]),
        variant["name"],
        record["report_type"],
    )


def _load_jsonl(path: Path) -> dict[tuple, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        out[_record_key(record)] = record
    return out


def _parse_variant_specs(raw: str) -> set[tuple]:
    specs = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        match = re.match(r"^(.+):(\d+):([^:]+)$", item)
        if not match:
            raise ValueError(
                "C2HLS_RERUN_VARIANTS entries must look like "
                "problem/path:variant_index:variant_name"
            )
        group_path = tuple(part for part in match.group(1).split("/") if part)
        specs.add((group_path, int(match.group(2)), match.group(3), "rtl_sim"))
    return specs


def _select_jobs(reference: dict, current: dict) -> list[dict]:
    explicit = os.getenv("C2HLS_RERUN_VARIANTS", "").strip()
    selected_keys: set[tuple]
    if explicit:
        selected_keys = _parse_variant_specs(explicit)
    else:
        selected_keys = {
            key for key, record in current.items()
            if key in reference
            and record.get("rtl_sim", {}).get("status")
            != reference[key].get("rtl_sim", {}).get("status")
        }

    jobs_by_key = {
        (job["group_path"], job["variant_index"], job["variant_short"], "rtl_sim"): job
        for job in matrix._jobs(reference)
    }
    jobs = []
    for key in sorted(selected_keys):
        job = jobs_by_key.get(key)
        if job is None:
            print(f"WARN no job found for {key}", file=sys.stderr, flush=True)
            continue
        if job.get("missing"):
            print(f"WARN benchmark dir missing for {key}", file=sys.stderr, flush=True)
            continue
        jobs.append(job)
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


def _write_summary(records: list[dict], reference: dict, current: dict) -> None:
    lines = [
        "# Requested Direct hw_emu Mismatch Rerun",
        "",
        f"Vitis {VITIS_VERSION} / {DEVICE}",
        "",
        "| problem | variant | previous | reference | rerun | cycles | log |",
        "|---|---:|:---:|:---:|:---:|---:|---|",
    ]
    for record in records:
        key = _record_key(record)
        ref = reference.get(key, {})
        prev = current.get(key, {})
        rtl = record["rtl_sim"]
        meta = record["implementation"].get("origin_meta") or {}
        lines.append(
            f"| {'/'.join(key[0])} | {key[1]} {key[2]} | "
            f"{prev.get('rtl_sim', {}).get('status') or '-'} | "
            f"{ref.get('rtl_sim', {}).get('status') or '-'} | "
            f"{rtl.get('status')} | "
            f"{rtl.get('kernel_runtime_cycles') if rtl.get('kernel_runtime_cycles') is not None else '-'} | "
            f"{meta.get('make_check_log') or '-'} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def _existing_keys(path: Path) -> set[tuple]:
    if FORCE_RERUN or not path.exists():
        return set()
    return set(_load_jsonl(path))


def main() -> int:
    reference = matrix._load_reference()
    current = _load_jsonl(DIRECT_JSONL)
    jobs = _select_jobs(reference, current)
    existing = _existing_keys(OUT_JSONL)
    print(f"output={OUT_JSONL}", flush=True)
    print(f"direct_jsonl={DIRECT_JSONL}", flush=True)
    print(f"jobs={len(jobs)} existing={len(existing)} force_rerun={FORCE_RERUN}", flush=True)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    if FORCE_RERUN:
        OUT_JSONL.write_text("")
    else:
        OUT_JSONL.touch(exist_ok=True)

    emitted = []
    for job in jobs:
        key = (job["group_path"], job["variant_index"], job["variant_short"], "rtl_sim")
        label = f"{'/'.join(job['group_path'])}:{job['variant_index']}:{job['variant_short']}"
        if key in existing:
            print(f"SKIP existing {label}", flush=True)
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
        with OUT_JSONL.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        emitted.append(record)
        rtl = record["rtl_sim"]
        print(
            f"DONE {label} status={rtl.get('status')} us={rtl.get('kernel_runtime_us')} "
            f"cycles={rtl.get('kernel_runtime_cycles')} elapsed={elapsed:.1f}s "
            f"log={(record['implementation']['origin_meta'] or {}).get('make_check_log')}",
            flush=True,
        )
        validation = validate_jsonl(OUT_JSONL)
        if validation["invalid"]:
            print(f"ERROR invalid_jsonl={validation['invalid']} path={OUT_JSONL}", file=sys.stderr)
            return 1

    records = list(_load_jsonl(OUT_JSONL).values())
    _write_summary(records, reference, current)
    validation = validate_jsonl(OUT_JSONL)
    print(f"validated total={validation['total']} invalid={validation['invalid']}", flush=True)
    print(f"summary={SUMMARY_MD}", flush=True)
    return 1 if validation["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

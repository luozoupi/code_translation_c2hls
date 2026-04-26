#!/usr/bin/env python3
"""Export c2hls run results as JSONL records conforming to jsonl_schema.md.

Reads existing pipeline outputs and emits one record per line in the
schema-defined format. Supports three sources, all read-only:

  1. results/<bench>/<bench>_results.json
       - sw_run    record (when csim ran)
       - hls_synth record (the orchestrator's final synth_report, generated)
       - hls_synth record (the validated ground-truth report, when present)
       - rtl_sim   record (when cosim ran with kernel_runtime_cycles)

  2. results_multistep/<run>/<bench>_multistep_results.json
       - hls_synth record per validated optimization step

  3. artifacts/stability/<bench>.json
       - hls_synth record per repeat run inside a stability variant

Output: artifacts/schema_records.jsonl (default) plus a manifest counter.

This emitter does NOT re-run anything. It builds canonical XML-shaped
sections from our existing flat synth reports — string fidelity is best-
effort because we don't preserve raw XML strings end-to-end yet. New
runs going forward will pass through this emitter the same way.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_MULTISTEP_DIR = REPO_ROOT / "results_multistep"
STABILITY_DIR = REPO_ROOT / "artifacts" / "stability"
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"

SCHEMA_VERSION = "1.0"

# Map our metadata's source_repo to the schema's suite/origin pair.
SOURCE_REPO_MAP = {
    "rodinia-hls":       ("rodinia_hls",     "rodinia_hls_benchmark"),
    "ML4Accel-Dataset":  ("ml4accel",        "ml4accel_benchmark"),
}

# Vitis target naming. Our pipeline runs csynth + (optionally) csim + cosim;
# each maps onto a schema target string.
TARGET_CSYNTH = "vitis.csynth"
TARGET_CSIM   = "vitis.csim"
TARGET_COSIM  = "vitis.cosim"


# ─── Common helpers ─────────────────────────────────────────────────────────

def _suite_and_origin(source_repo: Optional[str]) -> tuple[str, str]:
    if not source_repo:
        return ("unknown", "unknown")
    return SOURCE_REPO_MAP.get(source_repo, (source_repo, source_repo))


def _vitis_version_from_env() -> str:
    """Best-effort match of verify_corpus_stability._vitis_version."""
    for env in ("C2HLS_VITIS_VERSION", "C2HLS_VITIS_SETTINGS",
                "XILINX_VITIS", "XILINX_HLS"):
        v = os.getenv(env, "")
        if v:
            for token in v.split("/"):
                if token.count(".") == 1 and token and token[0].isdigit():
                    return token
            if env == "C2HLS_VITIS_VERSION":
                return v
    return "unknown"


def _device_for_part(part: Optional[str]) -> Optional[str]:
    """Return the schema's device id. Schema examples are platform XSA names
    (xilinx_u50_gen3x16_xdma_5_202210_1) but we run with bare part ids
    (xcu50-fsvh2104-2-e). Use the part id verbatim — it uniquely identifies
    the synthesis target."""
    return part or None


def _build_run(target: str, part: Optional[str], runtime_seconds: Optional[float]) -> dict:
    return {
        "target": target,
        "device": _device_for_part(part),
        "vitis_version": _vitis_version_from_env(),
        "runtime_seconds": runtime_seconds,
    }


def _build_problem(meta: dict) -> dict:
    suite, _ = _suite_and_origin(meta.get("source_repo"))
    bench = meta.get("benchmark", "")
    return {
        "suite": suite,
        "group_path": [bench] if bench else [],
    }


def _variant_index_from_name(name: str) -> int:
    """Variant names look like 'nw_4_doublebuffer'; pull the integer."""
    m = re.search(r"_(\d+)_", name or "")
    if m:
        return int(m.group(1))
    return 0


def _build_implementation(meta: dict, variant_name: str = "",
                          variant_index: Optional[int] = None,
                          origin_override: Optional[str] = None,
                          origin_version: Optional[str] = None,
                          origin_meta: Optional[dict] = None) -> dict:
    _, default_origin = _suite_and_origin(meta.get("source_repo"))
    name = variant_name or "implementation"
    # Strip the benchmark prefix and integer index for the schema's variant.name.
    short = re.sub(r"^[A-Za-z]+_\d+_", "", name) if "_" in name else name
    if variant_index is None:
        variant_index = _variant_index_from_name(name)
    return {
        "origin": origin_override or default_origin,
        "origin_version": origin_version,
        "origin_meta": origin_meta,
        "variant": {
            "index": int(variant_index),
            "name": short or name or "implementation",
        },
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── hls_synth payload reconstruction ───────────────────────────────────────

def _stringify(v: Any) -> Optional[str]:
    """The schema preserves XML strings verbatim (incl. units). Our internal
    parser already coerced to ints/floats, so we stringify back. For ints we
    emit "1115" not "1115.0"; floats keep their string form."""
    if v is None:
        return None
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # If the float is a whole number (typical for cycles) drop the .0
        if v.is_integer():
            return str(int(v))
        return str(v)
    return str(v)


def _ns_to_real_time_string(ns: Any) -> Optional[str]:
    """Format a latency_ns float as the schema's '<n> ms'/'<n> us'/'<n> ns'
    string. Vitis reports use ms/us/ns depending on magnitude."""
    if ns is None:
        return None
    try:
        x = float(ns)
    except (TypeError, ValueError):
        return None
    if x >= 1e6:
        return f"{x/1e6:.3f} ms"
    if x >= 1e3:
        return f"{x/1e3:.3f} us"
    return f"{x:.3f} ns"


def _available_resources_for(part: Optional[str]) -> Optional[dict]:
    """Pull device totals from rubric._DEVICE_TABLE so AvailableResources
    matches what Vitis would have reported."""
    if not part:
        return None
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from rubric import _DEVICE_TABLE  # type: ignore
    except Exception:
        return None
    part_lc = part.lower()
    for key, limits in _DEVICE_TABLE.items():
        if part_lc.startswith(key):
            return {
                "BRAM_18K": _stringify(limits.get("bram")),
                "DSP":      _stringify(limits.get("dsp")),
                "FF":       _stringify(limits.get("ff")),
                "LUT":      _stringify(limits.get("lut")),
                "URAM":     _stringify(limits.get("uram")),
            }
    return None


def _build_hls_synth_payload(report: dict, part: Optional[str],
                             clock_ns: Optional[float],
                             status: str = "pass") -> dict:
    """Reconstruct the schema's hls_synth payload from our flat synth_report.

    Best-effort. We don't preserve original XML strings end-to-end, so:
      - latency_ns -> reformatted as 'X.XXX ms' (or us/ns).
      - resources kept as int->string.
      - missing fields show as null.
    """
    if status != "pass" or not report:
        return {
            "status": status,
            "ReportVersion": None,
            "UserAssignments": None,
            "PerformanceEstimates": None,
            "AreaEstimates": None,
        }

    vitis_version = _vitis_version_from_env()

    user_assignments = {
        "unit": "ns",
        "ProductFamily": None,  # not tracked in our flat report
        "Part": part,
        "TopModelName": "workload",
        "TargetClockPeriod": _stringify(report.get("requested_clock_period_ns")
                                        or clock_ns),
        "ClockUncertainty": None,
        "FlowTarget": "vitis",
    }

    timing = {
        "unit": "ns",
        "EstimatedClockPeriod": _stringify(report.get("estimated_clock_period_ns")),
    }

    lat_cycles_str = _stringify(report.get("latency_cycles"))
    lat_realtime_str = _ns_to_real_time_string(report.get("latency_ns"))
    interval_cycles_str = _stringify(report.get("interval"))
    overall_latency = {
        "unit": "clock cycles",
        "Best-caseLatency": lat_cycles_str,
        "Average-caseLatency": lat_cycles_str,
        "Worst-caseLatency": lat_cycles_str,
        "Best-caseRealTimeLatency": lat_realtime_str,
        "Average-caseRealTimeLatency": lat_realtime_str,
        "Worst-caseRealTimeLatency": lat_realtime_str,
        "Interval-min": interval_cycles_str,
        "Interval-max": interval_cycles_str,
    }

    resources = {
        "BRAM_18K": _stringify(report.get("bram")),
        "DSP":      _stringify(report.get("dsp")),
        "FF":       _stringify(report.get("ff")),
        "LUT":      _stringify(report.get("lut")),
        "URAM":     _stringify(report.get("uram")),
    }
    avail = _available_resources_for(part)

    return {
        "status": "pass",
        "ReportVersion": {"Version": vitis_version},
        "UserAssignments": user_assignments,
        "PerformanceEstimates": {
            "SummaryOfTimingAnalysis": timing,
            "SummaryOfOverallLatency": overall_latency,
        },
        "AreaEstimates": {
            "Resources": resources,
            "AvailableResources": avail,
        },
    }


# ─── Record builders ────────────────────────────────────────────────────────

def _records_from_results_json(bench_dir: Path, results_json: Path,
                               default_part: str, default_clock_ns: float) -> list[dict]:
    """One pipeline result file → up to 3 records (sw_run, hls_synth, rtl_sim).

    Conventions:
      - The orchestrator's `synth_report` is the AI-generated kernel result.
        origin = c2hls_orchestrator.
      - The validated `ground_truth_report` is the GT variant's synthesis.
        origin = the rodinia/ML4Accel benchmark origin.
    """
    try:
        data = json.loads(results_json.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    meta_path = bench_dir / "metadata.json"
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return []

    part = default_part
    clock_ns = default_clock_ns
    bench = meta.get("benchmark", bench_dir.name)
    records: list[dict] = []

    # ── AI-generated hls_synth (the orchestrator's final synth) ────────────
    synth_report = data.get("synth_report") or {}
    if synth_report:
        records.append({
            "schema_version": SCHEMA_VERSION,
            "report_type": "hls_synth",
            "run": _build_run(TARGET_CSYNTH, part, None),
            "problem": _build_problem(meta),
            "implementation": _build_implementation(
                meta,
                variant_name="generated",
                variant_index=0,
                origin_override="c2hls_orchestrator",
                origin_version=os.getenv("C2HLS_MODEL") or None,
                origin_meta={"phase": data.get("phase", "")},
            ),
            "hls_synth": _build_hls_synth_payload(synth_report, part, clock_ns,
                                                  status="pass"),
        })

    # ── Generated csim → sw_run ────────────────────────────────────────────
    csim = data.get("csim") or {}
    if csim and csim.get("ran"):
        status = "pass" if csim.get("passed") else (
            "timeout" if "timed out" in (csim.get("error") or "").lower() else "fail"
        )
        records.append({
            "schema_version": SCHEMA_VERSION,
            "report_type": "sw_run",
            "run": _build_run(TARGET_CSIM, part, None),
            "problem": _build_problem(meta),
            "implementation": _build_implementation(
                meta,
                variant_name="generated",
                variant_index=0,
                origin_override="c2hls_orchestrator",
                origin_version=os.getenv("C2HLS_MODEL") or None,
            ),
            "sw_run": {"status": status},
        })

    # ── Generated cosim → rtl_sim ──────────────────────────────────────────
    cosim = data.get("cosim") or {}
    if cosim and cosim.get("ran"):
        status = "pass" if cosim.get("passed") else (
            "timeout" if "timed out" in (cosim.get("error") or "").lower() else "fail"
        )
        records.append({
            "schema_version": SCHEMA_VERSION,
            "report_type": "rtl_sim",
            "run": _build_run(TARGET_COSIM, part, None),
            "problem": _build_problem(meta),
            "implementation": _build_implementation(
                meta,
                variant_name="generated",
                variant_index=0,
                origin_override="c2hls_orchestrator",
                origin_version=os.getenv("C2HLS_MODEL") or None,
            ),
            "rtl_sim": {
                "status": status,
                "kernel_runtime_cycles": cosim.get("kernel_runtime_cycles"),
                "kernel_runtime_us": None,  # we use cosim, not hw_emu
                "kernel_clock_freq_mhz": None,
            },
        })

    # ── Ground-truth hls_synth (selected variant) ──────────────────────────
    rv = data.get("reference_validation") or {}
    selected_file = rv.get("selected_variant_file") or rv.get("selected_file") or ""
    selected_name = rv.get("selected_variant_name") or rv.get("selected_step_name") or ""
    gt_report = rv.get("report") or {}
    if gt_report and selected_file:
        # Pull the specific workflow entry (it has the full synth metrics).
        workflow_entry = next(
            (e for e in (rv.get("workflow") or []) if e.get("file") == selected_file),
            None,
        )
        gt_full_report = (workflow_entry or {}).get("report") or gt_report
        records.append({
            "schema_version": SCHEMA_VERSION,
            "report_type": "hls_synth",
            "run": _build_run(TARGET_CSYNTH, part, None),
            "problem": _build_problem(meta),
            "implementation": _build_implementation(
                meta,
                variant_name=selected_name,
            ),
            "hls_synth": _build_hls_synth_payload(gt_full_report, part, clock_ns),
        })

    return records


def _records_from_stability_json(bench_dir: Path, stability_json: Path,
                                 part_default: str,
                                 clock_default: float) -> list[dict]:
    """One stability file → one hls_synth record per variant per repeat run."""
    try:
        data = json.loads(stability_json.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    meta_path = bench_dir / "metadata.json"
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return []

    part = data.get("part") or part_default
    clock_ns = data.get("clock_ns") or clock_default
    records: list[dict] = []

    for variant in data.get("variants", []):
        v_name = variant.get("variant_name", "")
        v_index = _variant_index_from_name(v_name)
        runs = variant.get("runs") or []
        for run_idx, run in enumerate(runs):
            status = "pass" if run.get("success") else "fail"
            report = run.get("report") or {}
            records.append({
                "schema_version": SCHEMA_VERSION,
                "report_type": "hls_synth",
                "run": _build_run(
                    TARGET_CSYNTH, part,
                    runtime_seconds=variant.get("elapsed_sec"),
                ),
                "problem": _build_problem(meta),
                "implementation": _build_implementation(
                    meta,
                    variant_name=v_name,
                    variant_index=v_index,
                    origin_meta={"stability_run_index": run_idx,
                                 "stability_n_runs": variant.get("n_runs")},
                ),
                "hls_synth": _build_hls_synth_payload(report, part, clock_ns,
                                                      status=status),
            })
    return records


def _records_from_multistep(bench_dir: Path, multistep_json: Path,
                            default_part: str, default_clock_ns: float) -> list[dict]:
    """Multistep results → one hls_synth record per validated step."""
    try:
        data = json.loads(multistep_json.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    meta_path = bench_dir / "metadata.json"
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return []

    records: list[dict] = []
    steps = data.get("steps") or []
    for step in steps:
        if not step.get("success"):
            continue
        report = step.get("report") or {}
        if not report:
            continue
        step_name = step.get("step_name", "")
        records.append({
            "schema_version": SCHEMA_VERSION,
            "report_type": "hls_synth",
            "run": _build_run(TARGET_CSYNTH, default_part, None),
            "problem": _build_problem(meta),
            "implementation": _build_implementation(
                meta,
                variant_name=step_name,
                origin_override="c2hls_orchestrator",
                origin_version=os.getenv("C2HLS_MODEL") or None,
                origin_meta={"multistep": True},
            ),
            "hls_synth": _build_hls_synth_payload(report, default_part,
                                                  default_clock_ns),
        })
    return records


# ─── Driver ─────────────────────────────────────────────────────────────────

def _validate_record(record: dict) -> list[str]:
    """Return a list of schema violations (empty list = valid)."""
    errors = []
    rt = record.get("report_type")
    if rt not in {"sw_run", "hls_synth", "rtl_sim"}:
        errors.append(f"unknown report_type: {rt!r}")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version != {SCHEMA_VERSION!r}")
    impl = record.get("implementation") or {}
    variant = impl.get("variant")
    if not isinstance(variant, dict) or "index" not in variant or "name" not in variant:
        errors.append("implementation.variant missing index/name")
    if rt and rt not in record:
        errors.append(f"missing payload key {rt!r}")
    return errors


def export(results_dir: Path, multistep_dir: Path, stability_dir: Path,
           benchmarks_dir: Path, output_dir: Path,
           default_part: str, default_clock_ns: float,
           verbose: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "schema_records.jsonl"

    counts = {"sw_run": 0, "hls_synth": 0, "rtl_sim": 0}
    counts_by_source = {"results": 0, "results_multistep": 0, "stability": 0}
    invalid = 0

    with out_path.open("w") as f:
        # 1. main pipeline results/
        if results_dir.is_dir():
            for bench_dir in sorted(b for b in results_dir.iterdir() if b.is_dir()):
                rj = bench_dir / f"{bench_dir.name}_results.json"
                if not rj.exists():
                    continue
                bench_meta_dir = benchmarks_dir / bench_dir.name
                recs = _records_from_results_json(bench_meta_dir, rj,
                                                  default_part, default_clock_ns)
                for r in recs:
                    errs = _validate_record(r)
                    if errs:
                        invalid += 1
                        if verbose:
                            print(f"  invalid record: {errs}", file=sys.stderr)
                        continue
                    f.write(json.dumps(r) + "\n")
                    counts[r["report_type"]] += 1
                    counts_by_source["results"] += 1

        # 2. multistep results
        if multistep_dir.is_dir():
            for run_dir in sorted(d for d in multistep_dir.iterdir() if d.is_dir()):
                for jpath in run_dir.glob("*_multistep_results.json"):
                    bench_name = jpath.stem.replace("_multistep_results", "")
                    bench_meta_dir = benchmarks_dir / bench_name
                    recs = _records_from_multistep(bench_meta_dir, jpath,
                                                   default_part, default_clock_ns)
                    for r in recs:
                        errs = _validate_record(r)
                        if errs:
                            invalid += 1
                            continue
                        f.write(json.dumps(r) + "\n")
                        counts[r["report_type"]] += 1
                        counts_by_source["results_multistep"] += 1

        # 3. stability records (per-run hls_synth)
        if stability_dir.is_dir():
            for spath in sorted(stability_dir.glob("*.json")):
                bench_meta_dir = benchmarks_dir / spath.stem
                recs = _records_from_stability_json(bench_meta_dir, spath,
                                                    default_part, default_clock_ns)
                for r in recs:
                    errs = _validate_record(r)
                    if errs:
                        invalid += 1
                        continue
                    f.write(json.dumps(r) + "\n")
                    counts[r["report_type"]] += 1
                    counts_by_source["stability"] += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "vitis_version": _vitis_version_from_env(),
        "default_part": default_part,
        "default_clock_ns": default_clock_ns,
        "counts_by_report_type": counts,
        "counts_by_source": counts_by_source,
        "invalid_records": invalid,
        "output_file": out_path.name,
    }
    (output_dir / "schema_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--multistep-dir", default=str(RESULTS_MULTISTEP_DIR))
    p.add_argument("--stability-dir", default=str(STABILITY_DIR))
    p.add_argument("--benchmarks-dir", default=str(BENCHMARKS_DIR))
    p.add_argument("--output", default=str(REPO_ROOT / "artifacts"))
    p.add_argument("--part", default=os.getenv("C2HLS_PART", "xcu50-fsvh2104-2-e"))
    p.add_argument("--clock-ns", type=float,
                   default=float(os.getenv("C2HLS_CLOCK_NS", "3.33")))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Load .env for the sake of vitis_version detection consistency.
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    manifest = export(
        Path(args.results_dir),
        Path(args.multistep_dir),
        Path(args.stability_dir),
        Path(args.benchmarks_dir),
        Path(args.output),
        default_part=args.part,
        default_clock_ns=args.clock_ns,
        verbose=args.verbose,
    )
    total = sum(manifest["counts_by_report_type"].values())
    print(f"wrote {total} schema records to {args.output}/schema_records.jsonl")
    print(f"  by type:   {manifest['counts_by_report_type']}")
    print(f"  by source: {manifest['counts_by_source']}")
    if manifest["invalid_records"]:
        print(f"  WARN: {manifest['invalid_records']} records failed schema validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

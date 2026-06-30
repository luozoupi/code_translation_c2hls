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
    "rodinia-hls-nova":  ("rodinia_hls",     "rodinia_hls_benchmark"),
    "ML4Accel-Dataset":  ("ml4accel",        "ml4accel_benchmark"),
    "HLSFactory":        ("hlsfactory",      "hlsfactory_benchmark"),
    "hls-eval":          ("hls_eval",        "hls_eval_benchmark"),
}

# Vitis target naming. Our pipeline runs csynth + (optionally) csim + cosim;
# each maps onto a schema target string.
TARGET_CSYNTH = "vitis.csynth"
TARGET_CSIM   = "vitis.csim"
TARGET_COSIM  = "vitis.cosim"
TARGET_HW_EMU = "vitis.hw_emu"


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


def _hw_emu_device_platform(run_meta: Optional[dict] = None) -> str:
    return (
        (run_meta or {}).get("device")
        or os.getenv("C2HLS_DEVICE_PLATFORM")
        or "xilinx_u280_gen3x16_xdma_1_202211_1"
    )


def _build_run(target: str, part: Optional[str], runtime_seconds: Optional[float],
               run_meta: Optional[dict] = None) -> dict:
    """Compose the JSONL `run` section.

    `run_meta` is the orchestrator-saved attribution dict (model, vitis_version,
    flow_target, …). When provided, it overrides env-derived defaults so that
    historical runs export with the tooling/model they actually used, not what
    the current shell happens to be.
    """
    device = _hw_emu_device_platform(run_meta) if target == TARGET_HW_EMU else _device_for_part(part)
    out = {
        "target": target,
        "device": device,
        "vitis_version": (run_meta or {}).get("vitis_version") or _vitis_version_from_env(),
        "runtime_seconds": runtime_seconds,
    }
    if run_meta:
        for key in ("flow_target", "clock_ns"):
            if run_meta.get(key) is not None:
                out[key] = run_meta[key]
    return out


def _build_problem(meta: dict) -> dict:
    suite, _ = _suite_and_origin(meta.get("source_repo"))
    bench = meta.get("benchmark", "")
    group_path = _problem_group_path(meta)
    return {
        "suite": suite,
        "group_path": group_path or ([bench] if bench else []),
    }


def _problem_group_path(meta: dict) -> list[str]:
    explicit = meta.get("group_path")
    if isinstance(explicit, list) and all(isinstance(p, str) and p for p in explicit):
        return explicit

    source_path = str(meta.get("gold_hls_source_path") or "")
    marker = "/Benchmarks/"
    if marker in source_path:
        rel_parts = source_path.split(marker, 1)[1].split("/")
        if len(rel_parts) >= 2:
            if rel_parts[0] in {"cfd", "leukocyte"}:
                return rel_parts[:2]
            return [rel_parts[0]]
    bench = meta.get("benchmark", "")
    return [bench] if bench else []


_VARIANT_RE = re.compile(r"^(?P<prefix>.+?)_(?P<index>\d+)_(?P<name>.+)$")


def _split_variant(name: str) -> tuple[int, str]:
    """Split a variant name like 'spmv_crs_0_baseline' into (0, 'baseline').

    Handles multi-underscore bench prefixes (spmv_crs, gemm_ncubed, etc.).
    Falls back to (0, name) when the pattern doesn't match.
    """
    if not name:
        return (0, "implementation")
    m = _VARIANT_RE.match(name)
    if m:
        return (int(m.group("index")), m.group("name"))
    return (0, name)


def _variant_index_from_name(name: str) -> int:
    return _split_variant(name)[0]


def _normalize_step_name(name: str) -> str:
    short = _split_variant(name or "")[1]
    short = short.replace("double_buffer", "doublebuffer")
    short = short.replace("doublebuffer", "doublebuffer")
    short = short.replace("unrolll", "unroll")
    short = short.replace("unrolling", "unroll")
    return short or "implementation"


def _variant_identity_for_step(meta: dict, step_name: str) -> tuple[int, str]:
    normalized = _normalize_step_name(step_name)
    for variant in meta.get("variants", []) or []:
        raw = variant.get("name") if isinstance(variant, dict) else str(variant)
        idx, short = _split_variant(raw or "")
        if _normalize_step_name(short) == normalized:
            return idx, _normalize_step_name(short)
    return _split_variant(step_name or "implementation")


def _build_implementation(meta: dict, variant_name: str = "",
                          variant_index: Optional[int] = None,
                          origin_override: Optional[str] = None,
                          origin_version: Optional[str] = None,
                          origin_meta: Optional[dict] = None) -> dict:
    _, default_origin = _suite_and_origin(meta.get("source_repo"))
    raw = variant_name or "implementation"
    parsed_index, short = _split_variant(raw)
    if variant_index is None:
        variant_index = parsed_index
    return {
        "origin": origin_override or default_origin,
        "origin_version": origin_version,
        "origin_meta": origin_meta,
        "variant": {
            "index": int(variant_index),
            "name": short or raw or "implementation",
        },
    }


def _compact_origin_meta(value: Any) -> Any:
    """Remove generated code blobs before embedding telemetry in JSONL."""
    if isinstance(value, dict):
        return {
            k: _compact_origin_meta(v)
            for k, v in value.items()
            if k not in {"code", "hls_code", "current_code"}
        }
    if isinstance(value, list):
        return [_compact_origin_meta(item) for item in value]
    return value


def _test_status_for_schema(summary: dict) -> str:
    raw = (summary or {}).get("status", "")
    if raw == "passed":
        return "pass"
    if raw == "failed":
        return "timeout" if "timed out" in ((summary or {}).get("error") or "").lower() else "fail"
    return raw


def _emit_sw_run_record(records: list[dict], *, meta: dict, run_meta: dict,
                        part: str, model_id: Optional[str], variant_name: str,
                        variant_index: int, origin_meta: dict,
                        csim: dict) -> None:
    status = _test_status_for_schema(csim)
    if status in ("not_run", "not_supported", ""):
        return
    records.append({
        "schema_version": SCHEMA_VERSION,
        "report_type": "sw_run",
        "run": _build_run(TARGET_CSIM, part, None, run_meta),
        "problem": _build_problem(meta),
        "implementation": _build_implementation(
            meta,
            variant_name=variant_name,
            variant_index=variant_index,
            origin_override="c2hls_orchestrator",
            origin_version=model_id,
            origin_meta=origin_meta,
        ),
        "sw_run": {
            "status": status,
            "error": (csim.get("error") or "")[:300] if status != "pass" else None,
        },
    })


def _emit_cosim_record(records: list[dict], *, meta: dict, run_meta: dict,
                       part: str, model_id: Optional[str], variant_name: str,
                       variant_index: int, origin_meta: dict,
                       cosim: dict) -> None:
    status = _test_status_for_schema(cosim)
    if status in ("not_run", "not_supported", ""):
        return
    records.append({
        "schema_version": SCHEMA_VERSION,
        "report_type": "rtl_sim",
        "run": _build_run(TARGET_COSIM, part, None, run_meta),
        "problem": _build_problem(meta),
        "implementation": _build_implementation(
            meta,
            variant_name=variant_name,
            variant_index=variant_index,
            origin_override="c2hls_orchestrator",
            origin_version=model_id,
            origin_meta=origin_meta,
        ),
        "rtl_sim": {
            "status": status,
            "kernel_runtime_cycles": cosim.get("kernel_runtime_cycles"),
            "kernel_runtime_us": cosim.get("kernel_runtime_us"),
            "kernel_clock_freq_mhz": cosim.get("kernel_clock_freq_mhz"),
            "error": (cosim.get("error") or "")[:300] if status != "pass" else None,
        },
    })


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hw_emu_clock_freq_mhz(hw_emu: dict, clock_ns: Optional[float],
                           run_meta: Optional[dict] = None) -> Optional[float]:
    explicit = hw_emu.get("kernel_clock_freq_mhz")
    if explicit is not None:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            pass
    env_clock = os.getenv("C2HLS_HW_EMU_CLOCK_MHZ")
    if env_clock:
        try:
            return float(env_clock)
        except ValueError:
            pass
    device = _hw_emu_device_platform(run_meta)
    if "u280" in device.lower():
        return 300.0
    if clock_ns:
        return round(1000.0 / clock_ns, 2)
    return None


def _hw_emu_emittable(hw_emu: dict) -> bool:
    """Return True when a hw_emu object should produce an explicit JSONL row.

    Successful and failed runs are obvious. Skips are also emitted when the
    orchestrator recorded a skip_reason/profile_required marker so sweep rows
    do not silently disappear from intended-result JSONL.
    """
    return bool(
        hw_emu
        and (
            hw_emu.get("ran")
            or hw_emu.get("skip_reason")
            or hw_emu.get("profile_required")
        )
    )


def _hw_emu_status(hw_emu: dict) -> str:
    if hw_emu.get("success"):
        return "pass"
    text = " ".join(
        str(hw_emu.get(key) or "")
        for key in ("error", "skip_reason")
    ).lower()
    if "timed out" in text or "timeout" in text:
        return "timeout"
    return "fail"


def _hw_emu_error(hw_emu: dict, status: str) -> Optional[str]:
    if status == "pass":
        return None
    return (hw_emu.get("error") or hw_emu.get("skip_reason") or "hw_emu did not run")[:300]


def _metric_fallbacks(meta: dict, part: Optional[str],
                      run_meta: Optional[dict] = None) -> list[str]:
    fallbacks: list[str] = []
    source_repo = meta.get("source_repo")
    if not source_repo or source_repo not in SOURCE_REPO_MAP:
        fallbacks.append("unknown_source_repo")
    vitis_version = (run_meta or {}).get("vitis_version") or _vitis_version_from_env()
    if not vitis_version or vitis_version == "unknown":
        fallbacks.append("unknown_vitis_version")
    if _available_resources_for(part) is None:
        fallbacks.append("unknown_device_table")
    return fallbacks


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


_DEVICE_PRODUCT_FAMILY = {
    "xcu280-fsvh2892-2L-e": "virtexuplusHBM",
    "xcu50-fsvh2104-2-e": "virtexuplusHBM",
}


def _product_family_for_part(part: Optional[str]) -> Optional[str]:
    if not part:
        return None
    return _DEVICE_PRODUCT_FAMILY.get(part) or _DEVICE_PRODUCT_FAMILY.get(part.lower())


def _clock_uncertainty_ns(clock_ns: Optional[float]) -> Optional[str]:
    """Vitis default: 27% of target period (3.33 ns -> 0.90)."""
    if clock_ns is None:
        return None
    try:
        return f"{float(clock_ns) * 0.27:.2f}"
    except (TypeError, ValueError):
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
    target_clock = report.get("requested_clock_period_ns") or clock_ns

    user_assignments = {
        "unit": "ns",
        "ProductFamily": _product_family_for_part(part),
        "Part": part,
        "TopModelName": "workload",
        "TargetClockPeriod": _stringify(target_clock),
        "ClockUncertainty": _clock_uncertainty_ns(target_clock),
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

    # Run attribution recorded by the orchestrator (model id, vitis version,
    # flow target, …). Empty for results predating the attribution change.
    run_meta = data.get("run") or {}
    model_id = run_meta.get("model") or os.getenv("C2HLS_MODEL") or None
    # Per-task model overrides, recorded so the JSONL captures multi-agent setups.
    model_translator = run_meta.get("model_translator") or model_id
    model_synthesis = run_meta.get("model_synthesis") or model_id
    model_quality_repair = run_meta.get("model_quality_repair") or model_id
    # When the run records its own part/clock, prefer those over CLI defaults so
    # historical results don't get re-tagged with whatever device is configured now.
    part = run_meta.get("part") or default_part
    clock_ns = run_meta.get("clock_ns") or default_clock_ns

    bench = meta.get("benchmark", bench_dir.name)
    records: list[dict] = []

    gen_origin_meta = {
        "phase": data.get("phase", ""),
        "model": model_id,
        "model_translator": model_translator,
        "model_synthesis": model_synthesis,
        "model_quality_repair": model_quality_repair,
        "skill_mode": run_meta.get("skill_mode"),
        "skill_prompts": run_meta.get("skill_prompts"),
        "generated_at": run_meta.get("generated_at"),
    }
    gt_variant = data.get("ground_truth_variant") or {}
    selected_step = gt_variant.get("step") or ""
    if selected_step:
        gen_variant_index, gen_variant_name = _variant_identity_for_step(meta, selected_step)
        gen_origin_meta["selected_variant_step"] = selected_step
    else:
        gen_variant_index, gen_variant_name = (0, "generated")
    if gt_variant.get("fallback_used"):
        gen_origin_meta["gt_fallback_used"] = True
        gen_origin_meta["gt_fallback_reason"] = gt_variant.get("fallback_reason") or "ground truth fallback used"
    if data.get("csim_status"):
        gen_origin_meta["csim_status"] = data.get("csim_status")
    if data.get("cosim_status"):
        gen_origin_meta["cosim_status"] = data.get("cosim_status")
    hw_meta = data.get("hw_emu") or {}
    if hw_meta and not hw_meta.get("ran"):
        gen_origin_meta["hw_emu_skip_reason"] = hw_meta.get("skip_reason") or "hw_emu not run"
        gen_origin_meta["hw_emu_profile_required"] = hw_meta.get("profile_required")
    fallbacks = _metric_fallbacks(meta, part, run_meta)
    if fallbacks:
        gen_origin_meta["profiled_fallbacks"] = fallbacks
    # Drop None/empty values so records stay tidy.
    gen_origin_meta = {k: v for k, v in gen_origin_meta.items() if v not in (None, "")}

    # ── AI-generated hls_synth (the orchestrator's final synth) ────────────
    synth_report = data.get("synth_report") or {}
    if synth_report:
        records.append({
            "schema_version": SCHEMA_VERSION,
            "report_type": "hls_synth",
            "run": _build_run(TARGET_CSYNTH, part, None, run_meta),
            "problem": _build_problem(meta),
            "implementation": _build_implementation(
                meta,
                variant_name=gen_variant_name,
                variant_index=gen_variant_index,
                origin_override="c2hls_orchestrator",
                origin_version=model_id,
                origin_meta=gen_origin_meta,
            ),
            "hls_synth": _build_hls_synth_payload(synth_report, part, clock_ns,
                                                  status="pass"),
        })

    # ── Generated csim → sw_run ────────────────────────────────────────────
    # Emit a record whenever csim was attempted (status != "not_run" /
    # "not_supported"). A timeout or fail is information too — silence on
    # those states is what made earlier exports look like nothing happened.
    csim = data.get("csim") or {}
    csim_status_field = csim.get("status", "")
    if csim and csim_status_field not in ("not_run", "not_supported", ""):
        # Map orchestrator status field to schema status. Already canonical
        # in most cases (pass/fail/timeout), but normalise just in case.
        s = csim_status_field
        if s == "passed":
            status = "pass"
        elif s == "failed":
            status = (
                "timeout" if "timed out" in (csim.get("error") or "").lower()
                else "fail"
            )
        else:
            status = s
        records.append({
            "schema_version": SCHEMA_VERSION,
            "report_type": "sw_run",
            "run": _build_run(TARGET_CSIM, part, None, run_meta),
            "problem": _build_problem(meta),
            "implementation": _build_implementation(
                meta,
                variant_name=gen_variant_name,
                variant_index=gen_variant_index,
                origin_override="c2hls_orchestrator",
                origin_version=model_id,
                origin_meta=gen_origin_meta,
            ),
            "sw_run": {"status": status,
                       "error": (csim.get("error") or "")[:300] if status != "pass" else None},
        })

    # ── Generated cosim → rtl_sim ──────────────────────────────────────────
    # Same rule: emit on any attempted cosim, including timeouts. Cosim is
    # expensive; an emitted "timeout" is more useful than a missing record.
    cosim = data.get("cosim") or {}
    cosim_status_field = cosim.get("status", "")
    if cosim and cosim_status_field not in ("not_run", "not_supported", ""):
        s = cosim_status_field
        if s == "passed":
            status = "pass"
        elif s == "failed":
            status = (
                "timeout" if "timed out" in (cosim.get("error") or "").lower()
                else "fail"
            )
        else:
            status = s
        records.append({
            "schema_version": SCHEMA_VERSION,
            "report_type": "rtl_sim",
            "run": _build_run(TARGET_COSIM, part, None, run_meta),
            "problem": _build_problem(meta),
            "implementation": _build_implementation(
                meta,
                variant_name=gen_variant_name,
                variant_index=gen_variant_index,
                origin_override="c2hls_orchestrator",
                origin_version=model_id,
                origin_meta=gen_origin_meta,
            ),
            "rtl_sim": {
                "status": status,
                "kernel_runtime_cycles": cosim.get("kernel_runtime_cycles"),
                "kernel_runtime_us": cosim.get("kernel_runtime_us"),
                "kernel_clock_freq_mhz": cosim.get("kernel_clock_freq_mhz"),
                "error": (cosim.get("error") or "")[:300] if status != "pass" else None,
            },
        })

    # ── Generated hw_emu → rtl_sim ─────────────────────────────────────────
    # Authoritative cycle count from XSIM RTL simulation via nova `make
    # check TARGET=hw_emu`. Also emit explicit fail rows for profiled skips
    # so intended-result JSONL does not silently drop failed/unsupported cases.
    hw_emu = data.get("hw_emu") or {}
    if _hw_emu_emittable(hw_emu):
        hw_status = _hw_emu_status(hw_emu)
        records.append({
            "schema_version": SCHEMA_VERSION,
            "report_type": "rtl_sim",
            "run": _build_run(TARGET_HW_EMU, part, None, run_meta),
            "problem": _build_problem(meta),
            "implementation": _build_implementation(
                meta,
                variant_name=hw_emu.get("variant_name") or gen_variant_name,
                variant_index=hw_emu.get("variant_index", gen_variant_index),
                origin_override="c2hls_orchestrator",
                origin_version=model_id,
                origin_meta=dict(
                    gen_origin_meta,
                    hw_emu_passed=hw_emu.get("passed"),
                    hw_emu_profile_csv=hw_emu.get("profile_csv") or None,
                    hw_emu_clock_source=hw_emu.get("clock_source") or None,
                    hw_emu_clock_fallback=hw_emu.get("clock_fallback"),
                    hw_emu_debug_symbols_disabled=hw_emu.get("debug_symbols_disabled"),
                    hw_emu_debug_symbols_note=hw_emu.get("debug_symbols_note") or None,
                    requested_variant_step=hw_emu.get("requested_variant_step") or selected_step or None,
                    hw_emu_skip_reason=hw_emu.get("skip_reason") or None,
                    hw_emu_profile_required=hw_emu.get("profile_required"),
                    interface_mismatch=hw_emu.get("interface_mismatch"),
                    wide_abi_markers=hw_emu.get("wide_abi_markers") or None,
                ),
            ),
            "rtl_sim": {
                "status": hw_status,
                "kernel_runtime_cycles": hw_emu.get("kernel_runtime_cycles"),
                "kernel_runtime_us": hw_emu.get("kernel_runtime_us"),
                "kernel_clock_freq_mhz": _hw_emu_clock_freq_mhz(hw_emu, clock_ns, run_meta),
                "error": _hw_emu_error(hw_emu, hw_status),
            },
        })

    # Ground-truth records (rodinia_hls_benchmark / ml4accel_benchmark) are
    # *not* emitted here. They'd duplicate (or contradict) the canonical
    # reference JSONL the user maintains alongside the corpus, and the
    # orchestrator's GT-variant selection is not always the same one the
    # reference picked, which made those records inconsistent.

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

    run_meta = data.get("run") or {}
    model_id = run_meta.get("model") or os.getenv("C2HLS_MODEL") or None
    part = run_meta.get("part") or default_part
    clock_ns = run_meta.get("clock_ns") or default_clock_ns

    base_origin_meta = {
        "multistep": True,
        "model": model_id,
        "model_translator":     run_meta.get("model_translator") or model_id,
        "model_synthesis":      run_meta.get("model_synthesis") or model_id,
        "model_quality_repair": run_meta.get("model_quality_repair") or model_id,
        "skill_mode": run_meta.get("skill_mode"),
        "skill_prompts": run_meta.get("skill_prompts"),
        "generated_at": run_meta.get("generated_at"),
    }
    gt_variant = data.get("ground_truth_variant") or {}
    if gt_variant.get("fallback_used"):
        base_origin_meta["gt_fallback_used"] = True
        base_origin_meta["gt_fallback_reason"] = gt_variant.get("fallback_reason") or "ground truth fallback used"
    if data.get("coverage"):
        base_origin_meta["validation_coverage"] = data.get("coverage")
    hw_meta = data.get("hw_emu") or {}
    if hw_meta and not hw_meta.get("ran"):
        base_origin_meta["hw_emu_skip_reason"] = hw_meta.get("skip_reason") or "hw_emu not run"
        base_origin_meta["hw_emu_profile_required"] = hw_meta.get("profile_required")
    fallbacks = _metric_fallbacks(meta, part, run_meta)
    if fallbacks:
        base_origin_meta["profiled_fallbacks"] = fallbacks

    records: list[dict] = []
    steps = data.get("steps") or []
    for step in steps:
        if not step.get("success"):
            continue
        report = step.get("report") or {}
        step_name = step.get("step_name", "")
        step_index, step_short_name = _variant_identity_for_step(meta, step_name)
        step_origin_meta = dict(base_origin_meta, step=step_name)
        for key in (
            "candidate_search",
            "candidate_attempts",
            "attempt_stats",
            "attempt_results",
            "selected_candidate_index",
            "selected_attempt_index",
            "successful_attempt_count",
            "attempt_count",
            "candidate_count",
            "routing_decision",
            "skill_update",
        ):
            value = step.get(key)
            if value not in (None, "", []):
                step_origin_meta[key] = _compact_origin_meta(value)

        # AI-generated record at this step.
        if report:
            records.append({
                "schema_version": SCHEMA_VERSION,
                "report_type": "hls_synth",
                "run": _build_run(TARGET_CSYNTH, part, None, run_meta),
                "problem": _build_problem(meta),
                "implementation": _build_implementation(
                    meta,
                    variant_name=step_short_name,
                    variant_index=step_index,
                    origin_override="c2hls_orchestrator",
                    origin_version=model_id,
                    origin_meta=step_origin_meta,
                ),
                "hls_synth": _build_hls_synth_payload(report, part, clock_ns),
            })

        csim = step.get("csim") or {}
        if csim:
            _emit_sw_run_record(
                records,
                meta=meta,
                run_meta=run_meta,
                part=part,
                model_id=model_id,
                variant_name=step_short_name,
                variant_index=step_index,
                origin_meta=step_origin_meta,
                csim=csim,
            )

        cosim = step.get("cosim") or {}
        if cosim:
            _emit_cosim_record(
                records,
                meta=meta,
                run_meta=run_meta,
                part=part,
                model_id=model_id,
                variant_name=step_short_name,
                variant_index=step_index,
                origin_meta=step_origin_meta,
                cosim=cosim,
            )

        # Same-step GT record. The orchestrator synthesises the upstream
        # benchmark variant whose name matches this step (`tiling`, `pipeline`,
        # …) and stashes the result under `gt_report`. Emitting it here lets
        # downstream tools compare AI-vs-GT *at the same optimisation step*,
        # which is the only meaningful comparison in multi-step mode (final
        # variant != intermediate variant).
        gt_report = step.get("gt_report") or {}
        if gt_report:
            _, default_origin = _suite_and_origin(meta.get("source_repo"))
            step_index, step_short_name = _variant_identity_for_step(meta, step_name)
            records.append({
                "schema_version": SCHEMA_VERSION,
                "report_type": "hls_synth",
                "run": _build_run(TARGET_CSYNTH, part, None, run_meta),
                "problem": _build_problem(meta),
                "implementation": _build_implementation(
                    meta,
                    variant_name=step_short_name,  # same step name for direct comparison
                    variant_index=step_index,
                    origin_override=default_origin,  # rodinia / ml4accel
                    origin_version="upstream",
                    origin_meta={"step": step_name, "comparison_pair": "gt"},
                ),
                "hls_synth": _build_hls_synth_payload(gt_report, part, clock_ns),
            })

    # Final-stage hw_emu (one per multistep run, after the last step succeeds).
    hw_emu = data.get("hw_emu") or {}
    if _hw_emu_emittable(hw_emu):
        hw_status = _hw_emu_status(hw_emu)
        last_step_name = (
            hw_emu.get("variant_step")
            or hw_emu.get("requested_variant_step")
            or (steps[-1].get("step_name", "baseline") if steps else "baseline")
        )
        last_step_index, last_step_short_name = _variant_identity_for_step(meta, last_step_name)
        records.append({
            "schema_version": SCHEMA_VERSION,
            "report_type": "rtl_sim",
            "run": _build_run(TARGET_HW_EMU, part, None, run_meta),
            "problem": _build_problem(meta),
            "implementation": _build_implementation(
                meta,
                variant_name=hw_emu.get("variant_name") or last_step_short_name,
                variant_index=hw_emu.get("variant_index", last_step_index),
                origin_override="c2hls_orchestrator",
                origin_version=model_id,
                origin_meta=dict(base_origin_meta, step=last_step_name,
                                 hw_emu_passed=hw_emu.get("passed"),
                                 hw_emu_profile_csv=hw_emu.get("profile_csv") or None,
                                 hw_emu_clock_source=hw_emu.get("clock_source") or None,
                                 hw_emu_clock_fallback=hw_emu.get("clock_fallback"),
                                 hw_emu_debug_symbols_disabled=hw_emu.get("debug_symbols_disabled"),
                                 hw_emu_debug_symbols_note=hw_emu.get("debug_symbols_note") or None,
                                 requested_variant_step=hw_emu.get("requested_variant_step") or None,
                                 hw_emu_skip_reason=hw_emu.get("skip_reason") or None,
                                 hw_emu_profile_required=hw_emu.get("profile_required"),
                                 interface_mismatch=hw_emu.get("interface_mismatch"),
                                 wide_abi_markers=hw_emu.get("wide_abi_markers") or None),
            ),
            "rtl_sim": {
                "status": hw_status,
                "kernel_runtime_cycles": hw_emu.get("kernel_runtime_cycles"),
                "kernel_runtime_us": hw_emu.get("kernel_runtime_us"),
                "kernel_clock_freq_mhz": _hw_emu_clock_freq_mhz(hw_emu, clock_ns, run_meta),
                "error": _hw_emu_error(hw_emu, hw_status),
            },
        })
    return records


# ─── Driver ─────────────────────────────────────────────────────────────────

def _validate_record(record: dict) -> list[str]:
    """Return a list of schema violations (empty list = valid)."""
    errors = []
    if not isinstance(record, dict):
        return ["record is not an object"]
    rt = record.get("report_type")
    if rt not in {"sw_run", "hls_synth", "rtl_sim"}:
        errors.append(f"unknown report_type: {rt!r}")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version != {SCHEMA_VERSION!r}")
    run = record.get("run")
    if not isinstance(run, dict):
        errors.append("run missing/object")
    else:
        if not run.get("target"):
            errors.append("run.target missing")
        if not run.get("device"):
            errors.append("run.device missing")
        if not run.get("vitis_version"):
            errors.append("run.vitis_version missing")
        target = run.get("target")
        if rt == "sw_run" and target not in {TARGET_CSIM, "vitis.sw_emu"}:
            errors.append(f"sw_run target mismatch: {target!r}")
        if rt == "hls_synth" and target != TARGET_CSYNTH:
            errors.append(f"hls_synth target mismatch: {target!r}")
        if rt == "rtl_sim" and target not in {TARGET_COSIM, TARGET_HW_EMU}:
            errors.append(f"rtl_sim target mismatch: {target!r}")
    problem = record.get("problem")
    if not isinstance(problem, dict):
        errors.append("problem missing/object")
    else:
        group_path = problem.get("group_path")
        if not isinstance(group_path, list) or not group_path or not all(isinstance(p, str) and p for p in group_path):
            errors.append("problem.group_path missing/non-empty string list required")
        if not problem.get("suite"):
            errors.append("problem.suite missing")
    impl = record.get("implementation") or {}
    if not isinstance(impl, dict):
        errors.append("implementation missing/object")
        impl = {}
    if not impl.get("origin"):
        errors.append("implementation.origin missing")
    variant = impl.get("variant")
    if not isinstance(variant, dict) or "index" not in variant or "name" not in variant:
        errors.append("implementation.variant missing index/name")
    else:
        if variant.get("index") is None:
            errors.append("implementation.variant.index is null")
        elif not isinstance(variant.get("index"), int):
            errors.append("implementation.variant.index must be int")
        if not isinstance(variant.get("name"), str) or not variant.get("name"):
            errors.append("implementation.variant.name missing")
    payload_keys = [k for k in ("sw_run", "hls_synth", "rtl_sim") if k in record]
    if len(payload_keys) != 1:
        errors.append(f"expected exactly one payload, found {payload_keys}")
    elif rt and payload_keys[0] != rt:
        errors.append(f"report_type {rt!r} does not match payload {payload_keys[0]!r}")
    if rt and rt in record:
        payload = record.get(rt)
        if not isinstance(payload, dict):
            errors.append(f"{rt} payload is not an object")
        else:
            status = payload.get("status")
            if status not in {"pass", "fail", "timeout", "missing"}:
                errors.append(f"{rt}.status invalid: {status!r}")
            if rt == "hls_synth" and "metrics" in payload:
                errors.append("hls_synth.metrics is non-canonical")
            if rt == "rtl_sim":
                for key in ("kernel_runtime_cycles", "kernel_runtime_us", "kernel_clock_freq_mhz"):
                    if key not in payload:
                        errors.append(f"rtl_sim.{key} missing")
    return errors


def validate_jsonl(path: Path, verbose: bool = False) -> dict:
    total = 0
    invalid = 0
    by_type = {"sw_run": 0, "hls_synth": 0, "rtl_sim": 0}
    errors_by_line: list[dict] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        return {
            "path": str(path),
            "total": 0,
            "invalid": 1,
            "errors": [{"line": 0, "errors": [str(exc)]}],
            "by_type": by_type,
        }
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        total += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid += 1
            errors_by_line.append({"line": lineno, "errors": [f"invalid json: {exc}"]})
            continue
        errs = _validate_record(record)
        if errs:
            invalid += 1
            errors_by_line.append({"line": lineno, "errors": errs})
            if verbose:
                print(f"{path}:{lineno}: {errs}", file=sys.stderr)
            continue
        by_type[record["report_type"]] += 1
    return {
        "path": str(path),
        "total": total,
        "invalid": invalid,
        "errors": errors_by_line,
        "by_type": by_type,
    }


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
    p.add_argument("--validate-jsonl", action="append", default=[],
                   help="Validate an existing schema-1.0 JSONL file and exit nonzero on malformed records.")
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

    if args.validate_jsonl:
        overall_invalid = 0
        for item in args.validate_jsonl:
            summary = validate_jsonl(Path(item), verbose=args.verbose)
            overall_invalid += summary["invalid"]
            print(json.dumps({
                "path": summary["path"],
                "total": summary["total"],
                "invalid": summary["invalid"],
                "by_type": summary["by_type"],
            }, sort_keys=True))
            if args.verbose and summary["errors"]:
                for err in summary["errors"][:20]:
                    print(f"  line {err['line']}: {err['errors']}", file=sys.stderr)
        return 1 if overall_invalid else 0

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

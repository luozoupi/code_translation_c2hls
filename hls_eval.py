"""
HLS evaluation utilities: run Vitis HLS synthesis and parse reports.
"""

import os
import re
import shlex
import signal
import subprocess
import tempfile
import logging
import json
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s'
)

# === Configuration ===========================================================
# Override any of these via environment variables. See README for details.
#   C2HLS_VITIS_SETTINGS  Path to the Vitis HLS settings64.sh sourced before
#                         every Vitis invocation.
#   C2HLS_PART            Target FPGA part id (e.g. xc7a100t-csg324-1).
#   C2HLS_CLOCK_NS        Target clock period in nanoseconds.
#   C2HLS_SYNTH_TIMEOUT   Max seconds for csynth_design.
#   C2HLS_CSIM_TIMEOUT    Max seconds for csim_design.
#   C2HLS_COSIM_TIMEOUT   Max seconds for cosim_design.
VITIS_SETTINGS = os.getenv(
    "C2HLS_VITIS_SETTINGS",
    "/mnt/data/luo00466/Xilinx/2025.2/Vitis/settings64.sh",
)
DEFAULT_PART = os.getenv("C2HLS_PART", "xc7a100t-csg324-1")
DEFAULT_CLOCK_NS = float(os.getenv("C2HLS_CLOCK_NS", "4"))
SYNTH_TIMEOUT = int(os.getenv("C2HLS_SYNTH_TIMEOUT", "1200"))  # 20 minutes
CSIM_TIMEOUT = int(os.getenv("C2HLS_CSIM_TIMEOUT", "180"))     # 3 minutes
COSIM_TIMEOUT = int(os.getenv("C2HLS_COSIM_TIMEOUT", "1200"))  # 20 minutes
# =============================================================================


def _descendant_pids(root_pid: int) -> set:
    """Best-effort snapshot of descendant PIDs for a process tree."""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=", "ppid="],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return set()

    children = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        pid, ppid = map(int, parts)
        children.setdefault(ppid, set()).add(pid)

    descendants = set()
    stack = [root_pid]
    while stack:
        parent = stack.pop()
        for child in children.get(parent, ()):
            if child not in descendants:
                descendants.add(child)
                stack.append(child)
    return descendants


def _signal_pids(pids: set, sig) -> None:
    for pid in sorted(pids, reverse=True):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue


def _run_vitis_cmd(cmd: str, timeout: int) -> tuple:
    """Run a shell command with Vitis sourced. Returns (stdout+stderr, timed_out).

    We deliberately do NOT `exec` the command string: the callers chain
    `cd <work_dir> && vitis-run ...`, and `cd` is a shell builtin that cannot
    be exec'd. Process-tree cleanup on timeout is handled via start_new_session
    + killpg, so losing the exec replacement does not leak processes.
    """
    full_cmd = f"source {shlex.quote(VITIS_SETTINGS)} && {cmd}"
    proc = subprocess.Popen(
        ["bash", "-lc", full_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    def _as_text(blob):
        # Popen was opened with text=True, but TimeoutExpired.stdout may still
        # carry bytes in some Python versions when the decoder was interrupted.
        if blob is None:
            return ""
        if isinstance(blob, bytes):
            return blob.decode("utf-8", errors="replace")
        return blob

    try:
        output, _ = proc.communicate(timeout=timeout)
        return _as_text(output), False
    except subprocess.TimeoutExpired as exc:
        output = _as_text(exc.stdout)
        tree_pids = _descendant_pids(proc.pid)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        _signal_pids(tree_pids, signal.SIGTERM)

        try:
            tail, _ = proc.communicate(timeout=5)
            output += _as_text(tail)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _signal_pids(tree_pids, signal.SIGKILL)
            tail, _ = proc.communicate()
            output += _as_text(tail)

        _signal_pids(tree_pids, signal.SIGKILL)
        return output, True


def _extract_vitis_failure_reason(log: str, fallback: str) -> str:
    if not log:
        return fallback

    interesting_lines = []
    for line in log.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if (
            ("error" in lowered and "0 errors" not in lowered)
            or "simulation failed" in lowered
            or "segmentation violation" in lowered
            or "child killed" in lowered
            or "undefined symbol" in lowered
            or "did you mean to declare" in lowered
            or "ld.lld" in lowered
            or lowered.startswith("@e ")
        ):
            if stripped not in interesting_lines:
                interesting_lines.append(stripped)
        if len(interesting_lines) >= 4:
            break

    if interesting_lines:
        return "\n".join(interesting_lines)

    tail = [line.strip() for line in log.splitlines() if line.strip()]
    if tail:
        return "\n".join(tail[-3:])
    return fallback


def _normalize_extra_files(extra_files) -> list:
    if not extra_files:
        return []
    normalized = []
    for item in extra_files:
        if isinstance(item, dict):
            rel_path = item.get("path")
            content = item.get("content", "")
        else:
            rel_path, content = item
        if not rel_path:
            continue
        normalized.append((rel_path, content))
    return normalized


def _materialize_inputs(work_dir: str, hls_code: str, header_code: str, header_name: str,
                       testbench_code: str = "", extra_files=None, interface_depths=None) -> dict:
    os.makedirs(work_dir, exist_ok=True)

    if interface_depths:
        hls_code = _inject_interface_depths(hls_code, interface_depths)

    src_file = os.path.join(work_dir, "kernel.cpp")
    with open(src_file, "w") as f:
        f.write(hls_code)

    tb_file = ""
    if testbench_code:
        tb_file = os.path.join(work_dir, "testbench.cpp")
        with open(tb_file, "w") as f:
            f.write(testbench_code)

    hdr_file = ""
    if header_code:
        hdr_file = os.path.join(work_dir, header_name)
        os.makedirs(os.path.dirname(hdr_file), exist_ok=True)
        with open(hdr_file, "w") as f:
            f.write(header_code)

    materialized = []
    for rel_path, content in _normalize_extra_files(extra_files):
        out_path = os.path.join(work_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(content)
        materialized.append(out_path)

    return {
        "src_file": src_file,
        "tb_file": tb_file,
        "hdr_file": hdr_file,
        "extra_files": materialized,
    }


def _inject_interface_depths(hls_code: str, interface_depths: dict) -> str:
    if not interface_depths:
        return hls_code
    lines = []
    for line in hls_code.splitlines():
        match = re.search(r'#pragma\s+HLS\s+INTERFACE\s+m_axi\b.*?\bport\s*=\s*([A-Za-z_][A-Za-z0-9_]*)', line)
        if match:
            port = match.group(1)
            depth = interface_depths.get(port)
            if depth is not None and 'depth=' not in line:
                line = line.rstrip() + f' depth={depth}'
        lines.append(line)
    return "\n".join(lines)


def run_hls_synthesis(
    hls_code: str,
    header_code: str = "",
    header_name: str = "kernel.h",
    top_function: str = "workload",
    part: str = DEFAULT_PART,
    clock_ns: float = DEFAULT_CLOCK_NS,
    work_dir: str = None,
    extra_files=None,
) -> dict:
    """
    Run Vitis HLS C-synthesis on the given code.
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="hls_synth_")

    inputs = _materialize_inputs(
        work_dir, hls_code, header_code, header_name,
        extra_files=extra_files,
    )
    src_file = inputs["src_file"]
    hdr_file = inputs["hdr_file"]

    tcl_file = os.path.join(work_dir, "run_synth.tcl")
    proj_name = "hls_proj"
    tcl_content = f"""open_project {proj_name}
set_top {top_function}
add_files {src_file}
"""
    if hdr_file:
        tcl_content += f"add_files {hdr_file}\n"
    tcl_content += f"""open_solution "sol1" -flow_target vivado
set_part {{{part}}}
create_clock -period {clock_ns} -name default
csynth_design
exit
"""
    with open(tcl_file, "w") as f:
        f.write(tcl_content)

    cmd = f"cd {work_dir} && vitis-run --tcl --input_file {tcl_file}"
    log, timed_out = _run_vitis_cmd(cmd, SYNTH_TIMEOUT)
    if timed_out:
        return {
            "success": False,
            "error": f"Synthesis timed out after {SYNTH_TIMEOUT}s",
            "report": {},
            "report_raw": "",
            "log": log,
            "work_dir": work_dir,
        }

    if "Pre-synthesis failed" in log or "ERROR" in log:
        errors = [l for l in log.split("\n") if "ERROR" in l]
        return {
            "success": False,
            "error": "\n".join(errors) if errors else "Synthesis failed (unknown error)",
            "report": {},
            "report_raw": log,
            "log": log,
        }

    report_dir = os.path.join(work_dir, proj_name, "sol1", "syn", "report")
    xml_path = os.path.join(report_dir, "csynth.xml")
    rpt_path = os.path.join(report_dir, "csynth.rpt")

    report_raw = ""
    if os.path.exists(rpt_path):
        with open(rpt_path, "r") as f:
            report_raw = f.read()

    if os.path.exists(xml_path):
        report = parse_synthesis_xml(xml_path)
    elif report_raw:
        report = parse_synthesis_report(report_raw)
    else:
        return {
            "success": False,
            "error": "Synthesis report not found",
            "report": {},
            "report_raw": "",
            "log": log,
        }

    if report.get("fmax_mhz") is None:
        fmax_match = re.search(r'Estimated Fmax:\s*([\d.]+)\s*MHz', log)
        if fmax_match:
            report["fmax_mhz"] = round(float(fmax_match.group(1)), 2)

    if report.get("latency_cycles") is None and report_raw:
        report["latency_cycles"], report["latency_ns"] = _extract_max_loop_latency(report_raw)

    if report.get("requested_clock_period_ns") is None:
        report["requested_clock_period_ns"] = float(clock_ns)
    if report.get("estimated_clock_period_ns") is not None and report.get("slack_ns") is None:
        report["slack_ns"] = round(
            float(report["requested_clock_period_ns"]) - float(report["estimated_clock_period_ns"]), 3
        )

    # Sanitize BEFORE attaching work_dir so downstream canonical_report can
    # also drop work_dir without having to re-run numeric coercion.
    report = sanitize_report(report)
    report["work_dir"] = work_dir

    return {
        "success": True,
        "error": "",
        "report": report,
        "report_raw": report_raw,
        "log": log,
    }


# === Repeat-N synthesis for stability measurement ============================

import math as _math
import statistics as _statistics

# Default: no repeat, preserves single-run behavior. Set C2HLS_VERIFY_RUNS=3
# (or higher) to enable variance measurement on every synth invocation.
VERIFY_RUNS = int(os.getenv("C2HLS_VERIFY_RUNS", "1"))

# A report is "stable" when the coefficient of variation (stdev / mean) of
# latency_ns stays under this threshold. 5% tolerates normal Vitis jitter on
# modern CPUs; values above that usually indicate a mid-run server change
# (CPU contention, thermal throttling) or a Vitis-version skew worth
# investigating.
STABILITY_CV_THRESHOLD = float(os.getenv("C2HLS_STABILITY_CV", "0.05"))


def _summarize_metric(values):
    """Compute mean/stdev/cv for a list of numeric values (None-safe)."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "stdev": None, "cv": None, "n": 0}
    mean = sum(clean) / len(clean)
    stdev = _statistics.pstdev(clean) if len(clean) > 1 else 0.0
    cv = (stdev / mean) if mean not in (0, None) else None
    return {
        "mean": round(mean, 6) if isinstance(mean, float) else mean,
        "stdev": round(stdev, 6) if isinstance(stdev, float) else stdev,
        "cv": round(cv, 6) if cv is not None else None,
        "n": len(clean),
    }


def summarize_repeated_reports(reports: list) -> dict:
    """Aggregate a list of sanitized reports into mean/stdev/stability flags.

    Returns a dict containing per-metric stats plus an overall is_stable flag
    based on latency_ns coefficient of variation.
    """
    metrics_to_track = ("latency_ns", "latency_cycles", "fmax_mhz",
                        "bram", "dsp", "ff", "lut", "slack_ns")
    summary = {}
    for key in metrics_to_track:
        summary[key] = _summarize_metric([r.get(key) for r in reports])

    lat_cv = summary["latency_ns"]["cv"]
    summary["is_stable"] = (
        lat_cv is not None and lat_cv <= STABILITY_CV_THRESHOLD
    )
    summary["stability_threshold_cv"] = STABILITY_CV_THRESHOLD
    summary["n_runs"] = len(reports)
    return summary


def run_hls_synthesis_repeated(
    hls_code: str,
    header_code: str = "",
    header_name: str = "kernel.h",
    top_function: str = "workload",
    part: str = DEFAULT_PART,
    clock_ns: float = DEFAULT_CLOCK_NS,
    extra_files=None,
    n_runs: int = None,
) -> dict:
    """Run run_hls_synthesis N times on fresh temp dirs; return per-run
    reports plus aggregate stability metrics.

    Each run uses a new work_dir so no state leaks between runs. A failed
    run is recorded with success=False and kept in the runs list; the
    summary is computed from successful runs only.

    Returns:
        {
            "success": bool,               # all runs synthesized
            "n_runs": int,
            "runs": [ {success, error, report, work_dir}, ... ],
            "summary": { per-metric stats, is_stable, ... },
            "canonical_report": dict,       # mean-valued, for hashing/cache
        }
    """
    n = int(n_runs if n_runs is not None else VERIFY_RUNS)
    if n < 1:
        n = 1

    runs = []
    all_success = True
    for i in range(n):
        r = run_hls_synthesis(
            hls_code, header_code,
            header_name=header_name,
            top_function=top_function,
            part=part,
            clock_ns=clock_ns,
            extra_files=extra_files,
        )
        # run_hls_synthesis creates its own tempdir when work_dir is None.
        runs.append({
            "success": r.get("success", False),
            "error": r.get("error", ""),
            "report": r.get("report", {}),
            "work_dir": r.get("report", {}).get("work_dir"),
        })
        if not r.get("success"):
            all_success = False

    successful_reports = [run["report"] for run in runs if run["success"]]
    summary = summarize_repeated_reports(successful_reports) if successful_reports else {
        "n_runs": n, "is_stable": False, "stability_threshold_cv": STABILITY_CV_THRESHOLD,
    }

    # Canonical report: use the MEAN of each numeric field so downstream
    # rubric/pref-pair code can feed one representative report. For non-
    # numeric fields, take the first successful run's value.
    canonical = {}
    if successful_reports:
        canonical = canonical_report(successful_reports[0])
        for key in ("latency_ns", "latency_cycles", "fmax_mhz",
                    "bram", "dsp", "ff", "lut", "interval",
                    "estimated_clock_period_ns", "slack_ns"):
            stats = summary.get(key)
            if stats and stats.get("mean") is not None:
                canonical[key] = stats["mean"]

    return {
        "success": all_success,
        "n_runs": n,
        "runs": runs,
        "summary": summary,
        "canonical_report": canonical,
    }


# =============================================================================


def run_csim(
    hls_code: str,
    testbench_code: str,
    header_code: str = "",
    header_name: str = "kernel.h",
    top_function: str = "workload",
    part: str = DEFAULT_PART,
    clock_ns: float = DEFAULT_CLOCK_NS,
    work_dir: str = None,
    extra_files=None,
) -> dict:
    """Run Vitis HLS C-simulation (csim)."""
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="hls_csim_")

    inputs = _materialize_inputs(
        work_dir, hls_code, header_code, header_name,
        testbench_code=testbench_code,
        extra_files=extra_files,
    )
    src_file = inputs["src_file"]
    tb_file = inputs["tb_file"]
    hdr_file = inputs["hdr_file"]

    proj_name = "hls_proj"
    tcl_content = f"""open_project {proj_name}
set_top {top_function}
add_files {src_file}
"""
    if hdr_file:
        tcl_content += f"add_files {hdr_file}\n"
    tcl_content += f"""add_files -tb {tb_file}
open_solution "sol1" -flow_target vivado
set_part {{{part}}}
create_clock -period {clock_ns} -name default
csim_design
exit
"""
    tcl_file = os.path.join(work_dir, "run_csim.tcl")
    with open(tcl_file, "w") as f:
        f.write(tcl_content)

    cmd = f"cd {work_dir} && vitis-run --tcl --input_file {tcl_file}"
    log, timed_out = _run_vitis_cmd(cmd, CSIM_TIMEOUT)
    if timed_out:
        return {
            "success": False,
            "passed": False,
            "error": f"Csim timed out after {CSIM_TIMEOUT}s",
            "log": "",
            "work_dir": work_dir,
        }

    passed = "CSim done with 0 errors" in log or "csim_design finished successfully" in log.lower()
    log_lower = log.lower()
    has_error = (
        ("ERROR" in log and "0 errors" not in log_lower)
        or "simulation failed" in log_lower
        or "segmentation violation" in log_lower
        or "child killed" in log_lower
        or "undefined symbol" in log_lower
        or "ld.lld" in log_lower
    )
    success = passed and not has_error

    return {
        "success": success,
        "passed": passed,
        "error": "" if success else _extract_vitis_failure_reason(log, "Csim failed"),
        "log": log,
        "work_dir": work_dir,
    }


def run_cosim(
    hls_code: str,
    testbench_code: str,
    header_code: str = "",
    header_name: str = "kernel.h",
    top_function: str = "workload",
    part: str = DEFAULT_PART,
    clock_ns: float = DEFAULT_CLOCK_NS,
    work_dir: str = None,
    extra_files=None,
    interface_depths=None,
) -> dict:
    """Run Vitis HLS co-simulation (cosim)."""
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="hls_cosim_")

    inputs = _materialize_inputs(
        work_dir, hls_code, header_code, header_name,
        testbench_code=testbench_code,
        extra_files=extra_files,
        interface_depths=interface_depths,
    )
    src_file = inputs["src_file"]
    tb_file = inputs["tb_file"]
    hdr_file = inputs["hdr_file"]

    proj_name = "hls_proj"
    tcl_content = f"""open_project {proj_name}
set_top {top_function}
add_files {src_file}
"""
    if hdr_file:
        tcl_content += f"add_files {hdr_file}\n"
    tcl_content += f"""add_files -tb {tb_file}
open_solution "sol1" -flow_target vivado
set_part {{{part}}}
create_clock -period {clock_ns} -name default
csynth_design
cosim_design
exit
"""
    tcl_file = os.path.join(work_dir, "run_cosim.tcl")
    with open(tcl_file, "w") as f:
        f.write(tcl_content)

    cmd = f"cd {work_dir} && vitis-run --tcl --input_file {tcl_file}"
    log, timed_out = _run_vitis_cmd(cmd, COSIM_TIMEOUT)
    if timed_out:
        return {
            "success": False,
            "passed": False,
            "error": f"Cosim timed out after {COSIM_TIMEOUT}s",
            "log": "",
            "work_dir": work_dir,
        }

    log_lower = log.lower()
    passed = (
        "cosim done with 0 errors" in log_lower
        or "cosim_design finished successfully" in log_lower
        or "c/rtl co-simulation finished: pass" in log_lower
    )
    has_error = (
        ("ERROR" in log and "0 errors" not in log_lower)
        or "simulation failed" in log_lower
        or "segmentation violation" in log_lower
        or "child killed" in log_lower
        or "undefined symbol" in log_lower
        or "ld.lld" in log_lower
    )
    success = passed and not has_error

    return {
        "success": success,
        "passed": passed,
        "error": "" if success else _extract_vitis_failure_reason(log, "Cosim failed"),
        "log": log,
        "work_dir": work_dir,
    }


# === Report sanitization =====================================================
# Centralized post-parse cleanup so every synth report goes through the same
# hygiene checks before reaching rubric.py / downstream tooling.
#
# Two responsibilities:
#   (1) Coerce numeric fields that arrived as strings ("14" -> 14).
#   (2) Drop implausibly large latency values that Vitis can produce when a
#       loop's trip count is bounded by a runtime variable. Example: lud's
#       tiling variant reports 2.7e14 cycles because the inner loop bound is
#       `matrix_dim` which Vitis can't prove < 256 at synth time. Letting
#       2.7e14 into the rubric's latency_ratio makes gen-vs-gt comparisons
#       meaningless. We return None for these and rely on the rubric's
#       None-guarding to skip the metric.

# Caps chosen an order of magnitude above "realistic" HLS numbers on
# data-center parts. 1 second of latency at 1 GHz is 1e9 cycles; 1 day is
# 8.6e13. Anything above 1e12 cycles or 1e12 ns is almost certainly a
# Vitis "undef" artifact — a real kernel at that scale wouldn't be
# benchmarkable anyway.
_LATENCY_CYCLES_CAP = 1e12
_LATENCY_NS_CAP = 1e12

_NUMERIC_RESOURCE_FIELDS = ("bram", "dsp", "ff", "lut", "uram")
_NUMERIC_INT_FIELDS = ("latency_cycles", "interval", *_NUMERIC_RESOURCE_FIELDS)
_NUMERIC_FLOAT_FIELDS = ("latency_ns", "fmax_mhz", "estimated_clock_period_ns",
                         "requested_clock_period_ns", "slack_ns")


def _coerce_int(value):
    """Best-effort int coercion; keeps None for "undef"/"?" and non-numeric strings."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == value else None  # NaN check
    s = str(value).strip()
    if not s or s.lower() in {"undef", "?", "nan", "inf", "-inf", "infinity", "-infinity"}:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _coerce_float(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return None if value != value else float(value)  # NaN check
    s = str(value).strip()
    if not s or s.lower() in {"undef", "?", "nan", "inf", "-inf", "infinity", "-infinity"}:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def sanitize_report(report: dict) -> dict:
    """Return a cleaned copy of a parsed synthesis report.

    - Coerces numeric fields from strings to ints/floats.
    - Nulls out latency values above the plausibility cap (Vitis "undef"
      artifact masquerading as a huge number).
    - Preserves unknown fields so callers can carry extra data.
    """
    if not report:
        return {}
    out = dict(report)

    sanitized_note = []

    lat_cyc = _coerce_int(out.get("latency_cycles"))
    if lat_cyc is not None and lat_cyc > _LATENCY_CYCLES_CAP:
        sanitized_note.append(f"latency_cycles={lat_cyc:.3e} exceeds cap {_LATENCY_CYCLES_CAP:.0e}")
        lat_cyc = None
    out["latency_cycles"] = lat_cyc

    lat_ns = _coerce_float(out.get("latency_ns"))
    if lat_ns is not None and lat_ns > _LATENCY_NS_CAP:
        sanitized_note.append(f"latency_ns={lat_ns:.3e} exceeds cap {_LATENCY_NS_CAP:.0e}")
        lat_ns = None
    out["latency_ns"] = lat_ns

    for key in _NUMERIC_INT_FIELDS:
        if key == "latency_cycles":
            continue  # already handled above with cap
        out[key] = _coerce_int(out.get(key))

    for key in _NUMERIC_FLOAT_FIELDS:
        if key == "latency_ns":
            continue  # already handled above with cap
        out[key] = _coerce_float(out.get(key))

    if sanitized_note:
        existing = out.get("sanitized_notes") or []
        out["sanitized_notes"] = existing + sanitized_note
        logging.info("sanitize_report: %s", "; ".join(sanitized_note))

    return out


def canonical_report(report: dict) -> dict:
    """Return a report suitable for hashing / cross-run comparison.

    Drops non-deterministic fields (work_dir, raw log paths) and sanitizes the
    numeric fields. Callers use this as the input to sha256-based GT cache
    keys so that two runs producing identical synthesis numbers hash the same.
    """
    r = sanitize_report(report)
    for junk in ("work_dir", "log_path"):
        r.pop(junk, None)
    return r


# =============================================================================


def parse_synthesis_xml(xml_path: str) -> dict:
    """Parse key metrics from a Vitis HLS csynth.xml file (primary parser)."""
    report = {
        "latency_cycles": None,
        "latency_ns": None,
        "bram": None,
        "dsp": None,
        "ff": None,
        "lut": None,
        "uram": None,
        "fmax_mhz": None,
        "interval": None,
        "estimated_clock_period_ns": None,
        "requested_clock_period_ns": None,
        "slack_ns": None,
    }

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError):
        return report

    # Resources
    resources = root.find(".//AreaEstimates/Resources")
    if resources is not None:
        report["bram"] = _xml_text(resources, "BRAM_18K", "0")
        report["dsp"] = _xml_text(resources, "DSP", "0")
        report["ff"] = _xml_text(resources, "FF", "0")
        report["lut"] = _xml_text(resources, "LUT", "0")
        report["uram"] = _xml_text(resources, "URAM", "0")

    # Latency
    latency = root.find(".//PerformanceEstimates/SummaryOfOverallLatency")
    if latency is not None:
        worst = _xml_text(latency, "Worst-caseLatency")
        if worst and worst != "undef":
            report["latency_cycles"] = int(float(worst))
        worst_ns = _xml_text(latency, "Worst-caseRealTimeLatency")
        if worst_ns and worst_ns != "undef":
            report["latency_ns"] = _parse_ns_value(worst_ns)
        interval_max = _xml_text(latency, "Interval-max")
        if interval_max and interval_max != "undef":
            report["interval"] = int(float(interval_max))

    # Fmax from estimated clock period
    timing = root.find(".//PerformanceEstimates/SummaryOfTimingAnalysis")
    if timing is not None:
        estimated_period = _parse_float(_xml_text(timing, "EstimatedClockPeriod"))
        target_period = _parse_float(_xml_text(timing, "TargetClockPeriod"))
        if estimated_period is not None:
            report["estimated_clock_period_ns"] = estimated_period
            if estimated_period > 0:
                report["fmax_mhz"] = round(1000.0 / estimated_period, 2)
        if target_period is not None:
            report["requested_clock_period_ns"] = target_period
        if estimated_period is not None and target_period is not None:
            report["slack_ns"] = round(target_period - estimated_period, 3)

    return report


def _xml_text(parent, tag, default=None):
    """Get text content of an XML child element."""
    el = parent.find(tag)
    if el is not None and el.text:
        return el.text.strip()
    return default


def _parse_ns_value(s: str) -> float:
    """Parse a latency value that may have unit suffixes like 'ms', 'us', 'ns', 'sec'."""
    s = s.strip()
    multipliers = {"sec": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            num_str = s[:-len(suffix)].strip()
            try:
                return float(num_str) * mult
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _extract_max_loop_latency(report_text: str) -> tuple:
    """
    Extract the largest loop/sub-module latency from the text report table.
    Used as fallback when top-level latency is 'undef'.
    Returns (max_cycles, max_ns) or (None, None).
    """
    max_cycles = None
    max_ns = None
    lines = report_text.split("\n")
    for line in lines:
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        if len(parts) < 14:
            continue
        # Column layout: Name(0) | Issue(1) | Violation(2) | IterLat(3) | Interval(4) | Trip(5) | Pipelined(6) | Lat_cycles(7) | Lat_ns(8) | Slack(9) | BRAM(10) | DSP(11) | FF(12) | LUT(13) | URAM(14)
        cycles = _parse_int(parts[7])
        ns = _parse_float(parts[8])
        if cycles is not None:
            if max_cycles is None or cycles > max_cycles:
                max_cycles = cycles
                max_ns = ns
    return max_cycles, max_ns


def parse_synthesis_report(report_text: str) -> dict:
    """Parse key metrics from a Vitis HLS csynth.rpt text file (fallback parser)."""
    report = {
        "latency_cycles": None,
        "latency_ns": None,
        "bram": None,
        "dsp": None,
        "ff": None,
        "lut": None,
        "uram": None,
        "fmax_mhz": None,
        "interval": None,
        "estimated_clock_period_ns": None,
        "requested_clock_period_ns": None,
        "slack_ns": None,
    }

    # Column layout (0-indexed after split+filter):
    # 0:Name | 1:Issue | 2:Violation | 3:IterLat | 4:Interval | 5:Trip | 6:Pipelined | 7:Lat_cycles | 8:Lat_ns | 9:Slack | 10:BRAM | 11:DSP | 12:FF | 13:LUT | 14:URAM
    lines = report_text.split("\n")
    for line in lines:
        if line.startswith("|+"):
            parts = [p.strip() for p in line.split("|")]
            parts = [p for p in parts if p]
            if len(parts) >= 14:
                try:
                    report["latency_cycles"] = _parse_int(parts[7])
                    report["latency_ns"] = _parse_float(parts[8])
                    report["interval"] = _parse_int(parts[4])
                    report["bram"] = _parse_resource(parts[10])
                    report["dsp"] = _parse_resource(parts[11])
                    report["ff"] = _parse_resource(parts[12])
                    report["lut"] = _parse_resource(parts[13])
                    if len(parts) > 14:
                        report["uram"] = _parse_resource(parts[14])
                except (IndexError, ValueError):
                    pass
            break  # Only need the first |+ line (top-level)

    # Fmax from report text
    fmax_match = re.search(r'Estimated Fmax:\s*([\d.]+)\s*MHz', report_text)
    if fmax_match:
        report["fmax_mhz"] = round(float(fmax_match.group(1)), 2)

    return report


def _parse_int(s: str) -> int:
    """Parse integer from report field, handling '-' and scientific notation."""
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if s == "-" or not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_float(s: str) -> float:
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if s == "-" or not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_resource(s: str) -> str:
    """Parse resource field like '2 (~0%)' or '1399 (1%)' or '-'."""
    s = s.strip()
    if s == "-" or not s:
        return "0"
    # Extract just the number
    m = re.match(r'(\d+)', s)
    if m:
        return m.group(1)
    return s


def compare_reports(generated: dict, ground_truth: dict) -> dict:
    """Compare two synthesis reports. Returns comparison metrics.

    For resource metrics (lower is better): ratio = gen/gt, so <1.0 means generated is better.
    For fmax (higher is better): ratio = gen/gt, so >1.0 means generated is better.
    """
    comparison = {}
    for key in ["latency_cycles", "latency_ns", "interval", "bram", "dsp", "ff", "lut"]:
        gen_val = generated.get(key)
        gt_val = ground_truth.get(key)
        if gen_val is not None and gt_val is not None:
            try:
                gen_n = float(gen_val)
                gt_n = float(gt_val)
                if gt_n > 0:
                    ratio = gen_n / gt_n
                elif gen_n == 0:
                    ratio = 1.0
                else:
                    ratio = float("inf")
                comparison[key] = {
                    "generated": gen_n,
                    "ground_truth": gt_n,
                    "ratio": round(ratio, 3),
                }
            except (ValueError, TypeError):
                comparison[key] = {"generated": gen_val, "ground_truth": gt_val, "ratio": None}

    # Fmax comparison (higher is better)
    gen_fmax = generated.get("fmax_mhz")
    gt_fmax = ground_truth.get("fmax_mhz")
    if gen_fmax and gt_fmax:
        comparison["fmax_mhz"] = {
            "generated": gen_fmax,
            "ground_truth": gt_fmax,
            "ratio": round(gen_fmax / gt_fmax, 3),
        }

    return comparison


def format_report_summary(report: dict) -> str:
    """Format a synthesis report dict as a readable string."""
    lines = []
    for key in [
        "latency_cycles", "latency_ns", "interval",
        "requested_clock_period_ns", "estimated_clock_period_ns", "slack_ns",
        "bram", "dsp", "ff", "lut", "uram", "fmax_mhz"
    ]:
        val = report.get(key)
        if val is not None:
            lines.append(f"  {key}: {val}")
    return "\n".join(lines) if lines else "  (no data)"


if __name__ == "__main__":
    # Quick test with a simple kernel
    test_code = """
void add(int a[100], int b[100], int c[100]) {
    for (int i = 0; i < 100; i++) {
        c[i] = a[i] + b[i];
    }
}

extern "C" {
void workload(int a[100], int b[100], int c[100]) {
#pragma HLS INTERFACE m_axi port=a bundle=gmem
#pragma HLS INTERFACE m_axi port=b bundle=gmem
#pragma HLS INTERFACE m_axi port=c bundle=gmem
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=c bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    add(a, b, c);
}
}
"""
    print("Running test synthesis...")
    result = run_hls_synthesis(test_code)
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Report:\n{format_report_summary(result['report'])}")
    else:
        print(f"Error: {result['error']}")

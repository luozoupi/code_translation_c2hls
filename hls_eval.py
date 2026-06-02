"""
HLS evaluation utilities: run Vitis HLS synthesis and parse reports.
"""

import os
import re
import csv
import shlex
import shutil
import signal
import subprocess
import logging
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from c2hls_temp import C2HLS_TMP_ROOT_ENV, configure_temp_env, make_tempdir

# Pillar 1: fine-grained per-scope HLS feedback (per-loop II / Slack / Issue,
# scheduler-blame, typed bottleneck records). Imported lazily inside
# run_hls_synthesis so unit tests that don't touch synthesis still load.
from hls_feedback import attach_feedback as _hf_attach_feedback

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
#   C2HLS_FLOW_TARGET     "vitis" (kernel/v++ flow, default) or "vivado"
#                         (raw IP flow). Vitis flow generates the AXI-lite
#                         control + AXI-master kernel wrapper that v++
#                         would emit for U50/U280 deployment; Vivado flow
#                         omits it. Use "vitis" for cross-tool comparisons.
#   C2HLS_SYNTH_TIMEOUT   Max seconds for csynth_design.
#   C2HLS_CSIM_TIMEOUT    Max seconds for csim_design.
#   C2HLS_COSIM_TIMEOUT   Max seconds for cosim_design.
#   C2HLS_VITIS_USER_HOME Writable HOME for Vitis/Vivado subprocesses.
#   C2HLS_COSIM_TRACE_LEVEL
#                         Vitis cosim trace level. Default "none" avoids
#                         simulator waveform/debug setup for batch sweeps.
VITIS_SETTINGS = os.getenv(
    "C2HLS_VITIS_SETTINGS",
    "/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh",
)
DEFAULT_PART = os.getenv("C2HLS_PART", "xcu280-fsvh2892-2L-e")
DEFAULT_CLOCK_NS = float(os.getenv("C2HLS_CLOCK_NS", "3.33"))
DEFAULT_FLOW_TARGET = os.getenv("C2HLS_FLOW_TARGET", "vitis")
DEFAULT_COSIM_TRACE_LEVEL = os.getenv("C2HLS_COSIM_TRACE_LEVEL", "none").strip()
SYNTH_TIMEOUT = int(os.getenv("C2HLS_SYNTH_TIMEOUT", "1200"))  # 20 minutes
CSIM_TIMEOUT = int(os.getenv("C2HLS_CSIM_TIMEOUT", "180"))     # 3 minutes
COSIM_TIMEOUT = int(os.getenv("C2HLS_COSIM_TIMEOUT", "1200"))  # 20 minutes
KERNEL_CLOCK_ID = 0
VITIS_USER_HOME_ENV = "C2HLS_VITIS_USER_HOME"
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


def _vitis_shell_exports(temp_root: Path) -> str:
    """Return exports that keep Vitis intermediates on writable storage."""
    raw_home = os.getenv(VITIS_USER_HOME_ENV, str(temp_root / "vitis_user_home")).strip()
    vitis_home = Path(raw_home or str(temp_root / "vitis_user_home")).expanduser()
    if not vitis_home.is_absolute():
        raise RuntimeError(f"{VITIS_USER_HOME_ENV} must be absolute, got: {vitis_home}")
    vitis_home.mkdir(parents=True, exist_ok=True)
    (vitis_home / ".Xilinx").mkdir(parents=True, exist_ok=True)
    return (
        f"export {C2HLS_TMP_ROOT_ENV}={shlex.quote(str(temp_root))} "
        f"TMPDIR={shlex.quote(str(temp_root))} "
        f"TEMP={shlex.quote(str(temp_root))} "
        f"TMP={shlex.quote(str(temp_root))} "
        f"HOME={shlex.quote(str(vitis_home))}"
    )


def _run_vitis_cmd(cmd: str, timeout: int) -> tuple:
    """Run a shell command with Vitis sourced. Returns (stdout+stderr, timed_out).

    We deliberately do NOT `exec` the command string: the callers chain
    `cd <work_dir> && vitis-run ...`, and `cd` is a shell builtin that cannot
    be exec'd. Process-tree cleanup on timeout is handled via start_new_session
    + killpg, so losing the exec replacement does not leak processes.
    """
    temp_root = configure_temp_env(create=True)
    temp_exports = _vitis_shell_exports(temp_root)
    full_cmd = f"{temp_exports} && source {shlex.quote(VITIS_SETTINGS)} && {temp_exports} && {cmd}"
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


def _read_log_tail(path: Path, max_bytes: int = 65536) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _terminate_process_group(proc: subprocess.Popen, sig) -> None:
    tree_pids = _descendant_pids(proc.pid)
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass
    _signal_pids(tree_pids, sig)


def _run_vitis_cmd_logged(
    cmd: str,
    timeout: int,
    log_path: Path,
    *,
    terminal_markers: "list[str] | None" = None,
    terminal_settle_s: int = 10,
) -> tuple:
    """Run a Vitis command with stdout/stderr redirected to a real log file.

    Some hw_emu failures leave descendant processes holding stdout open after
    make has printed a terminal failure line. Using a pipe plus communicate()
    can then block until the full timeout. This variant polls the process and
    the log file, allowing us to stop after explicit terminal failure markers.
    """
    temp_root = configure_temp_env(create=True)
    temp_exports = _vitis_shell_exports(temp_root)
    full_cmd = f"{temp_exports} && source {shlex.quote(VITIS_SETTINGS)} && {temp_exports} && {cmd}"
    markers = [m.lower() for m in (terminal_markers or [])]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    marker_seen_at = None
    marker_seen = ""
    timed_out = False
    stopped_on_marker = False

    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        proc = subprocess.Popen(
            ["bash", "-lc", full_cmd],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        while True:
            if proc.poll() is not None:
                break

            now = time.monotonic()
            if now - start > timeout:
                timed_out = True
                _terminate_process_group(proc, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _terminate_process_group(proc, signal.SIGKILL)
                    proc.wait()
                break

            if markers:
                tail = _read_log_tail(log_path).lower()
                hit = next((marker for marker in markers if marker in tail), "")
                if hit:
                    if marker_seen_at is None or hit != marker_seen:
                        marker_seen_at = now
                        marker_seen = hit
                    elif now - marker_seen_at >= terminal_settle_s:
                        stopped_on_marker = True
                        _terminate_process_group(proc, signal.SIGTERM)
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            _terminate_process_group(proc, signal.SIGKILL)
                            proc.wait()
                        break
                else:
                    marker_seen_at = None
                    marker_seen = ""

            time.sleep(1)

        log_handle.flush()

    output = _read_log_tail(log_path, max_bytes=128 * 1024 * 1024)
    if stopped_on_marker:
        note = (
            "\n[C2HLS] stopped emulation command after terminal log marker "
            f"{marker_seen!r} remained for {terminal_settle_s}s.\n"
        )
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(note)
        except OSError:
            pass
        output += note
    return output, timed_out


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
        work_dir = make_tempdir(prefix="hls_synth_")

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
    tcl_content += f"""open_solution "sol1" -flow_target {DEFAULT_FLOW_TARGET}
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

    # Pillar 1: attach fine-grained per-scope feedback (per-loop II / Slack /
    # Issue / Violation / scheduler-blame / typed bottlenecks). Best-effort:
    # if any of the inputs are unparseable, the feedback dict will simply
    # have empty lists.
    try:
        _hf_attach_feedback(
            report,
            xml_path=xml_path if os.path.exists(xml_path) else None,
            rpt_text=report_raw or None,
            log_text=log or None,
            # Phase 7a: pass the work_dir so the harvester can find
            # burst.xml / fe_messages.xml / be_messages.xml /
            # csynth_design_size.rpt under hls_proj/sol1/.
            work_dir=work_dir,
        )
    except Exception as exc:  # pragma: no cover — feedback is non-critical
        logging.warning("attach_feedback failed: %s", exc)

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
        work_dir = make_tempdir(prefix="hls_csim_")

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
open_solution "sol1" -flow_target {DEFAULT_FLOW_TARGET}
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
        work_dir = make_tempdir(prefix="hls_cosim_")

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
    cosim_cmd = "cosim_design"
    if DEFAULT_COSIM_TRACE_LEVEL:
        cosim_cmd += f" -trace_level {DEFAULT_COSIM_TRACE_LEVEL}"

    tcl_content += f"""add_files -tb {tb_file}
open_solution "sol1" -flow_target {DEFAULT_FLOW_TARGET}
set_part {{{part}}}
create_clock -period {clock_ns} -name default
csynth_design
if {{[info exists ::env(LIBRARY_PATH)]}} {{ unset ::env(LIBRARY_PATH) }}
{cosim_cmd}
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

    # Vitis writes the cycle count to sim/report/verilog/lat.rpt. We pull
    # $TOTAL_EXECUTE_TIME from there per the JSONL schema's rtl_sim contract;
    # without this the cosim result is just pass/fail and downstream tools
    # have no way to compare RTL-level performance.
    kernel_runtime_cycles = _parse_lat_rpt_cycles(work_dir, proj_name)
    kernel_clock_freq_mhz = 1000.0 / clock_ns if clock_ns else None
    kernel_runtime_us = None
    if kernel_runtime_cycles is not None and kernel_clock_freq_mhz:
        kernel_runtime_us = kernel_runtime_cycles / kernel_clock_freq_mhz

    return {
        "success": success,
        "passed": passed,
        "error": "" if success else _extract_vitis_failure_reason(log, "Cosim failed"),
        "log": log,
        "work_dir": work_dir,
        "kernel_runtime_cycles": kernel_runtime_cycles,
        "kernel_runtime_us": kernel_runtime_us,
        "kernel_clock_freq_mhz": kernel_clock_freq_mhz,
    }


def _stage_nova_workdir(nova_bench_dir: str,
                        hls_code: "str | None" = None,
                        kernel_basename: "str | None" = None) -> "tuple[Path, Path] | tuple[None, str]":
    """Materialise a private copy of a rodinia-hls-nova variant dir suitable
    for `make check TARGET=sw_emu/hw_emu`. Replicates the relative-path
    hierarchy the Makefile expects (../common.mk one up, ../../common two up)
    and mirrors Benchmarks/common via a symlink.

    When `hls_code` is provided, the variant's kernel cpp is replaced by it
    (with c2hls' `support/common/X` includes rewritten back to upstream's
    `../../../common/X` form). Pass `hls_code=None` to leave the upstream
    cpp unchanged — useful for direct (no-LLM) baseline runs.

    Returns (staged_bench_path, src_dir) on success, or (None, error_msg).
    """
    nova_path = Path(nova_bench_dir)
    if not nova_path.is_dir():
        return None, f"nova_bench_dir not found: {nova_bench_dir}"

    # Find the Benchmarks root by walking up from the variant until we hit a
    # dir whose `common/libs/xcl2/xcl2.mk` exists. Variants live at:
    #   <Benchmarks>/<bench>/<variant>/                 (top-level: pathfinder)
    #   <Benchmarks>/<group>/<bench>/<variant>/         (nested: cfd, leukocyte)
    # Group dirs sometimes carry an INCOMPLETE common/ (e.g. leukocyte/common
    # has just `harness.cpp` + headers but no `libs/`). Use the xcl2.mk probe
    # to skip those and reach the canonical Benchmarks/common.
    canonical_marker = Path("common") / "libs" / "xcl2" / "xcl2.mk"
    bench_root = nova_path.parent.parent
    while bench_root.parent != bench_root and not (bench_root / canonical_marker).is_file():
        bench_root = bench_root.parent
    if not (bench_root / canonical_marker).is_file():
        return None, f"could not locate Benchmarks/common above {nova_bench_dir}"

    rel = nova_path.parent.relative_to(bench_root)  # e.g. "cfd/cfd_flux" or "pathfinder"
    work_root = make_tempdir(prefix="emu_")
    staged_parent = Path(work_root) / rel
    staged_parent.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(nova_path.parent, staged_parent, symlinks=True,
                    ignore=shutil.ignore_patterns("_x.*", ".run", "xclbin",
                                                  "host", "*.xclbin", "*.xo"))
    try:
        (Path(work_root) / "common").symlink_to(bench_root / "common",
                                                target_is_directory=True)
    except FileExistsError:
        pass
    # Some benches (e.g. leukocyte/lc_dilate) include `../../../common/X`
    # from a header inside src/, which resolves to a *group-level* common
    # (Benchmarks/leukocyte/common/) — not the canonical Benchmarks/common.
    # Mirror those intermediate group `common/` dirs so the include path
    # still resolves under our staged tree.
    cur = nova_path.parent.parent  # group dir, e.g. Benchmarks/leukocyte
    while cur != bench_root and cur.parent != cur:
        group_common = cur / "common"
        if group_common.is_dir():
            staged_group = Path(work_root) / cur.relative_to(bench_root)
            staged_group.mkdir(parents=True, exist_ok=True)
            try:
                (staged_group / "common").symlink_to(group_common,
                                                    target_is_directory=True)
            except FileExistsError:
                pass
        cur = cur.parent

    staged_bench = staged_parent / nova_path.name
    src_dir = staged_bench / "src"

    # Pillar 9 (MVP): always drop an xrt.ini that disables xsim's debug-mode
    # bring-up. Without this, hw_emu on some benches (notably nw on the
    # current u280 toolchain) crashes inside xsim before profile_kernels.csv
    # is produced. The user validated this content directly and upstreamed it
    # to rodinia-hls-nova; we replicate it here so the agentic flow doesn't
    # depend on every benchmark dir already carrying the file.
    try:
        xrt_ini = staged_bench / "xrt.ini"
        if not xrt_ini.exists():
            xrt_ini.write_text(
                "[Emulation]\n"
                "debug_mode=off\n"
                "\n"
                "[Debug]\n"
                "timeline_trace=true\n"
                "profile=true\n",
                encoding="utf-8",
            )
    except OSError as exc:  # pragma: no cover — best-effort
        logging.warning("failed to inject xrt.ini into %s: %s", staged_bench, exc)

    if hls_code is not None and kernel_basename:
        target_cpp = src_dir / f"{kernel_basename}.cpp"
        if not target_cpp.exists():
            return None, f"missing kernel cpp at {target_cpp}"
        nova_kernel_code = re.sub(
            r'#include\s+"support/common/([^"]+)"',
            r'#include "../../../common/\1"',
            hls_code,
        )
        target_cpp.write_text(nova_kernel_code)

    return staged_bench, src_dir


def _hw_emu_disable_debug_symbols_enabled() -> bool:
    return os.getenv("C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS", "").lower() in ("1", "true", "yes")


def _disable_hw_emu_debug_symbols(staged_bench: Path) -> "tuple[bool, str]":
    """Remove Nova's v++ `-g` flag in the temporary staged copy.

    The xrt.ini `debug_mode=off` setting alone does not stop Vitis 2023.2 from
    launching behav_waveform/xsim with WDB/protoinst setup when Nova common.mk
    keeps `CLFLAGS += -g ...`. This edit is confined to the temp emulation
    workdir and is recorded in the returned hw_emu result.
    """
    common_mk = staged_bench.parent / "common.mk"
    if not common_mk.is_file():
        return False, f"common.mk not found at {common_mk}"
    try:
        text = common_mk.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return False, f"failed to read {common_mk}: {exc}"

    updated = re.sub(r"(^\s*CLFLAGS\s*\+=\s*)-g\s+", r"\1", text, count=1, flags=re.MULTILINE)
    if updated == text:
        return False, f"no CLFLAGS '-g' entry found in {common_mk}"
    try:
        common_mk.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return False, f"failed to write {common_mk}: {exc}"
    return True, str(common_mk)


def _run_make_check_emu(staged_bench: Path, target: str, timeout: int) -> "tuple[str, bool]":
    """Drive `make check TARGET=<sw_emu|hw_emu>` inside a staged nova variant.
    Returns (combined_stdout, timed_out)."""
    emu_env_script = os.getenv(
        "C2HLS_EMU_ENV_SCRIPT",
        str(Path(__file__).resolve().parent / "scripts" / "setup_emu_env.sh"),
    )
    device = os.getenv("C2HLS_DEVICE_PLATFORM", "xilinx_u280_gen3x16_xdma_1_202211_1")
    cmd = (
        f"source {shlex.quote(emu_env_script)} && "
        f"cd {shlex.quote(str(staged_bench))} && "
        f"make clean > /dev/null 2>&1 && "
        f"make check TARGET={target} DEVICE={shlex.quote(device)}"
    )
    # Keep the live log outside staged_bench: Nova `make clean` may remove
    # `*.log` in the benchmark directory after Python has opened the file,
    # which unlinks the watcher log and prevents marker polling.
    log_path = staged_bench.parent / f".{staged_bench.name}.c2hls_{target}_make_check.live.log"
    terminal_markers = [
        "make: ***",
        "benchmark results are incorrect",
        "xsimk: *e",
        "child killed",
        "segmentation violation",
    ]
    return _run_vitis_cmd_logged(
        cmd,
        timeout=timeout,
        log_path=log_path,
        terminal_markers=terminal_markers,
        terminal_settle_s=int(os.getenv("C2HLS_EMU_TERMINAL_SETTLE_S", "10")),
    )


def _choose_latest(paths: "list[Path]") -> "Path | None":
    candidates = []
    for path in paths:
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _parse_runtime_us(profile_path: "Path | None") -> "tuple[float | None, int]":
    """Sum CU runtimes from the Nova hw_emu profile section."""
    if profile_path is None or not profile_path.is_file():
        return None, 0

    try:
        with profile_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None, 0

    in_section = False
    header = None
    runtime_index = None
    total_runtime = 0.0
    row_count = 0

    for raw_line in lines:
        line = raw_line.strip()

        if line == "Compute Units: Running Time and Stalls":
            in_section = True
            continue

        if not in_section:
            continue

        if not line:
            if header is not None:
                break
            continue

        if header is not None and "," not in line and ":" in line:
            break

        row = [cell.strip() for cell in next(csv.reader([raw_line]))]
        if header is None:
            header = row
            for candidate in ("Running Time (us)", "Time (us)"):
                if candidate in header:
                    runtime_index = header.index(candidate)
                    break
            if runtime_index is None:
                return None, 0
            continue

        try:
            total_runtime += float(row[runtime_index])
        except (IndexError, TypeError, ValueError):
            continue
        row_count += 1

    if row_count == 0:
        return None, 0
    return total_runtime, row_count


def _find_hw_emu_crash_marker(staged_bench: Path) -> "tuple[str, str]":
    """Return (path, summary) for simulator/runtime crash evidence, if any."""
    candidates = (
        list(staged_bench.glob(".run/**/hs_err_pid*.log"))
        + list(staged_bench.glob(".run/**/xsimcrash.log"))
        + list(staged_bench.glob(".run/**/simulate.log"))
    )
    for path in sorted(candidates, key=lambda item: item.as_posix()):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:80]:
            lower = line.lower()
            if (
                "unexpected error has occurred" in lower
                or lower.startswith("fatal:")
                or lower.startswith("error:")
                or "segmentation" in lower
                or "xsim" in path.name.lower() and "crash" in lower
            ):
                return str(path), line[:300]
    return "", ""


def _system_diagram_model_score(path: Path) -> "tuple[int, int, str]":
    parts = path.parts
    preferred_suffix = ("link", "int", "systemDiagramModel.json")
    suffix_penalty = 0 if parts[-3:] == preferred_suffix else 1
    temp_penalty = 1 if "temp" in parts or "sys_link" in parts else 0
    return (suffix_penalty, temp_penalty, path.as_posix())


def _choose_canonical_system_diagram_model(paths: "list[Path]") -> "Path | None":
    if not paths:
        return None
    return min(paths, key=_system_diagram_model_score)


def _parse_number(raw_value) -> "float | None":
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _parse_kernel_clock_freq_mhz(system_diagram_model_path: "Path | None") -> "float | None":
    if system_diagram_model_path is None or not system_diagram_model_path.is_file():
        return None
    try:
        payload = json.loads(system_diagram_model_path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return None

    metadata = payload.get("system_diagram_metadata")
    if not isinstance(metadata, dict):
        return None
    xsa = metadata.get("xsa")
    if not isinstance(xsa, dict):
        return None
    clocks = xsa.get("clocks")
    if not isinstance(clocks, list):
        return None

    for clock in clocks:
        if not isinstance(clock, dict):
            continue
        if _parse_number(clock.get("id")) == KERNEL_CLOCK_ID:
            return _parse_number(clock.get("spec_frequency"))
    return None


def _resolve_hw_emu_clock(staged_bench: Path) -> "tuple[float | None, str | None, bool, str]":
    system_model = _choose_canonical_system_diagram_model(
        list(staged_bench.glob("_x.hw_emu.*/**/systemDiagramModel.json"))
        + list(staged_bench.glob("**/systemDiagramModel.json"))
    )
    clock_mhz = _parse_kernel_clock_freq_mhz(system_model)
    if clock_mhz is not None:
        return clock_mhz, str(system_model), False, "systemDiagramModel.json"

    env_clock_mhz = os.getenv("C2HLS_HW_EMU_CLOCK_MHZ")
    if env_clock_mhz:
        parsed = _parse_number(env_clock_mhz)
        if parsed is not None:
            return parsed, str(system_model) if system_model else None, True, "env:C2HLS_HW_EMU_CLOCK_MHZ"

    env_clock_ns = os.getenv("C2HLS_HW_EMU_CLOCK_NS")
    if env_clock_ns:
        parsed_ns = _parse_number(env_clock_ns)
        if parsed_ns:
            return 1000.0 / parsed_ns, str(system_model) if system_model else None, True, "env:C2HLS_HW_EMU_CLOCK_NS"

    device = os.getenv("C2HLS_DEVICE_PLATFORM", "xilinx_u280_gen3x16_xdma_1_202211_1")
    if "u280" in device.lower():
        return 300.0, str(system_model) if system_model else None, True, "u280_default"
    return None, str(system_model) if system_model else None, True, "missing_systemDiagramModel_clock"


def run_sw_emu_via_nova(
    nova_bench_dir: str,
    hls_code: "str | None" = None,
    *,
    kernel_basename: "str | None" = None,
    timeout: int = 1800,
) -> dict:
    """Run `make check TARGET=sw_emu` on a rodinia-hls-nova variant.

    sw_emu compiles the kernel for software emulation, runs the host program
    against a CPU model of the kernel, and reports whether the testbench
    matched the golden output. Fast (~30s/variant) — used for correctness
    validation across all variants. No kernel-cycle measurement is produced;
    use hw_emu for that.

    Returns:
      {ran, passed, success, error, work_dir, log}
        passed   = testbench `Finished checking data: correct` AND a final
                   `Success.` marker appeared
        success  = passed AND build+run completed without timeout/error
    """
    staged, info = _stage_nova_workdir(nova_bench_dir, hls_code=hls_code,
                                       kernel_basename=kernel_basename)
    if staged is None:
        return {"ran": False, "passed": False, "success": False,
                "error": info, "work_dir": "", "log": ""}

    output, timed_out = _run_make_check_emu(staged, "sw_emu", timeout)
    log_lower = output.lower()
    passed = (
        "finished checking data: correct" in log_lower
        # Match the literal "Success." line v++ host emits on a clean run.
        and re.search(r"^Success\.\s*$", output, re.MULTILINE) is not None
    )
    success = (not timed_out) and passed

    error = ""
    if timed_out:
        error = f"sw_emu timed out after {timeout}s"
    elif not success:
        for line in reversed(output.splitlines()[-200:]):
            line = line.strip()
            if line.lower().startswith("error:") or "ERROR" in line[:6]:
                error = line[:300]
                break
        if not error and not passed:
            error = "testbench did not report Success."

    return {
        "ran": True, "passed": passed, "success": success,
        "error": error,
        "work_dir": str(staged),
        "log": output[-4000:] if output else "",
    }


def run_hw_emu_via_nova(
    nova_bench_dir: str,
    hls_code: "str | None" = None,
    *,
    kernel_basename: "str | None" = None,
    timeout: int = 7200,
) -> dict:
    """Run `make check TARGET=hw_emu` against a rodinia-hls-nova variant.

    hw_emu compiles to RTL via v++ and runs the host program against XSIM —
    authoritative kernel-cycle count from `profile_kernels.csv`. Slow:
    typical runtime is 15-90 minutes per variant.

    Pass `hls_code=None` to run the upstream variant cpp unchanged (direct
    reference measurement); pass an LLM-generated cpp for agentic validation.

    Returns:
      {ran, passed, success, kernel_runtime_us, kernel_runtime_cycles,
       kernel_clock_freq_mhz, profile_csv, error, work_dir, log}
        passed   = testbench `Finished checking data: correct` AND a final
                   `Success.` marker appeared
        success  = passed AND profile_kernels.csv produced AND no timeout
    """
    staged, info = _stage_nova_workdir(nova_bench_dir, hls_code=hls_code,
                                       kernel_basename=kernel_basename)
    if staged is None:
        return {
            "ran": False, "passed": False, "success": False,
            "kernel_runtime_us": None, "kernel_runtime_cycles": None,
            "kernel_clock_freq_mhz": None, "profile_csv": "",
            "profile_compute_unit_rows": 0, "clock_source": "not_run",
            "clock_fallback": False, "error": info, "work_dir": "", "log": "",
        }
    debug_symbols_disabled = False
    debug_symbols_note = ""
    if _hw_emu_disable_debug_symbols_enabled():
        debug_symbols_disabled, debug_symbols_note = _disable_hw_emu_debug_symbols(staged)
        if not debug_symbols_disabled:
            logging.warning(
                "C2HLS_HW_EMU_DISABLE_DEBUG_SYMBOLS requested but not applied: %s",
                debug_symbols_note,
            )
    output, timed_out = _run_make_check_emu(staged, "hw_emu", timeout)
    log_path = staged / "c2hls_hw_emu_make_check.log"
    try:
        log_path.write_text(output or "", encoding="utf-8", errors="ignore")
    except OSError:
        log_path = None
    log_lower = output.lower()
    passed = (
        "finished checking data: correct" in log_lower
        and re.search(r"^Success\.\s*$", output, re.MULTILINE) is not None
    )

    # Parse profile_kernels.csv with the same section semantics used by
    # rodinia-hls-nova's extractor: sum all compute-unit runtime rows.
    profile_csv = _choose_latest(list(staged.glob(".run/**/profile_kernels.csv")))
    kernel_runtime_us, profile_rows = _parse_runtime_us(profile_csv)
    crash_log, crash_summary = _find_hw_emu_crash_marker(staged)

    kernel_clock_freq_mhz, system_diagram_model, clock_fallback, clock_source = _resolve_hw_emu_clock(staged)
    kernel_runtime_cycles = None
    if kernel_runtime_us is not None and kernel_clock_freq_mhz is not None:
        kernel_runtime_cycles = round(kernel_runtime_us * kernel_clock_freq_mhz)

    # success := compile + link + sim finished AND testbench validated. We
    # use kernel_runtime_us as the proxy for "sim ran to completion" since
    # profile_kernels.csv is only emitted after a clean XSIM exit.
    success = (not timed_out) and (kernel_runtime_us is not None) and passed

    error = ""
    if timed_out:
        error = f"hw_emu timed out after {timeout}s"
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
        "debug_symbols_disabled": debug_symbols_disabled,
        "debug_symbols_note": debug_symbols_note,
        "error": error,
        "work_dir": str(staged),
        "log_path": str(log_path) if log_path else "",
        "log": output[-4000:] if output else "",
    }


def _parse_lat_rpt_cycles(work_dir: str, proj_name: str = "hls_proj") -> "int | None":
    """Pull $TOTAL_EXECUTE_TIME from a Vitis cosim lat.rpt.

    Vitis writes lat.rpt at sim/report/verilog/lat.rpt under the solution
    directory; on some flows it ends up under .../verilog/ or .../vhdl/.
    We search the work_dir tree for any lat.rpt and return the cycle count
    from the first one that parses cleanly. Returns None if not found.
    """
    if not work_dir:
        return None
    root = os.path.join(work_dir, proj_name) if proj_name else work_dir
    if not os.path.isdir(root):
        root = work_dir
    pattern = re.compile(r'\$TOTAL_EXECUTE_TIME\s*=\s*"([^"]+)"')
    for cur, _dirs, files in os.walk(root):
        if "lat.rpt" not in files:
            continue
        try:
            with open(os.path.join(cur, "lat.rpt"), "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        m = pattern.search(text)
        if not m:
            continue
        try:
            return round(float(m.group(1)))
        except ValueError:
            continue
    return None


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

    # Worst-case companions are optional — coerce + cap when present.
    lat_cyc_w = _coerce_int(out.get("latency_cycles_worst"))
    if lat_cyc_w is not None and lat_cyc_w > _LATENCY_CYCLES_CAP:
        lat_cyc_w = None
    if "latency_cycles_worst" in out:
        out["latency_cycles_worst"] = lat_cyc_w
    lat_ns_w = _coerce_float(out.get("latency_ns_worst"))
    if lat_ns_w is not None and lat_ns_w > _LATENCY_NS_CAP:
        lat_ns_w = None
    if "latency_ns_worst" in out:
        out["latency_ns_worst"] = lat_ns_w

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
    # Drop derived / non-deterministic / large fields before hashing.
    # `feedback` is recomputed from xml/rpt/log at synth time and would
    # invalidate every cache entry on a schema bump; we want the cache
    # key to track only the synthesis numerics that drive scoring.
    for junk in ("work_dir", "log_path", "feedback"):
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

    # Latency. Prefer Average-case to mirror what v++/HW-emu deployment numbers
    # report — for data-independent designs Average == Worst, but for designs
    # with conditional flags (e.g. double-buffered prologue/epilogue) the worst
    # case includes warm-up/drain iterations that don't reflect steady-state.
    # Fall back to Worst when Average is undef.
    latency = root.find(".//PerformanceEstimates/SummaryOfOverallLatency")
    if latency is not None:
        for tag in ("Average-caseLatency", "Worst-caseLatency"):
            val = _xml_text(latency, tag)
            if val and val != "undef":
                report["latency_cycles"] = int(float(val))
                break
        for tag in ("Average-caseRealTimeLatency", "Worst-caseRealTimeLatency"):
            val = _xml_text(latency, tag)
            if val and val != "undef":
                report["latency_ns"] = _parse_ns_value(val)
                break
        # Keep the worst-case figures available for callers that need them.
        worst_cycles = _xml_text(latency, "Worst-caseLatency")
        if worst_cycles and worst_cycles != "undef":
            report["latency_cycles_worst"] = int(float(worst_cycles))
        worst_ns = _xml_text(latency, "Worst-caseRealTimeLatency")
        if worst_ns and worst_ns != "undef":
            report["latency_ns_worst"] = _parse_ns_value(worst_ns)
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
    """Format a synthesis report dict as a readable string.

    Includes top-level metrics plus per-loop bottleneck records from Pillar 1
    (II violations, issue types, source locations) so the LLM optimization
    prompt can see WHERE latency comes from, not just the total.
    """
    lines = []
    for key in [
        "latency_cycles", "latency_ns", "interval",
        "requested_clock_period_ns", "estimated_clock_period_ns", "slack_ns",
        "bram", "dsp", "ff", "lut", "uram", "fmax_mhz"
    ]:
        val = report.get(key)
        if val is not None:
            lines.append(f"  {key}: {val}")

    # Pillar 1: surface per-loop II violations and bottleneck types so the
    # LLM knows which loops are limiting performance and why, not just that
    # total latency is high.
    feedback = report.get("feedback") or {}
    bottlenecks = (feedback.get("bottlenecks") or [])[:6]
    if bottlenecks:
        lines.append("  per_loop_bottlenecks (top issues limiting performance):")
        for bn in bottlenecks:
            scope = bn.get("scope_id", "?")
            kind = bn.get("kind", "?")
            evidence = bn.get("evidence", "")
            loc = bn.get("source_location", "")
            loc_str = f" [{loc}]" if loc else ""
            lines.append(f"    - {scope}: {kind} | {evidence}{loc_str}")

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

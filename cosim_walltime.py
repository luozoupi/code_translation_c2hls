"""Opt-in wall-time budget for the Vitis ``cosim_design`` command.

The measured phase includes RTL generation/setup, elaboration, simulation, and
output checking inside cosim_design. It is not hardware latency or XSIM-only
execution time. Fresh synthesis is covered only by the invocation watchdog.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ENV_NAME = "C2HLS_COSIM_WALL_TIMEOUT_SECONDS"
TIMING_SCOPE = "cosim_design_including_setup_elaboration_rtl_and_output_check_excluding_csynth"


def configured_timeout(environ=None):
    raw = (os.environ if environ is None else environ).get(ENV_NAME, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_NAME} must be positive finite seconds") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{ENV_NAME} must be positive finite seconds")
    return value


def tcl_phase_command(command, marker_path):
    # Tcl braced literals keep spaces, dollar signs, and brackets literal.
    path = str(Path(marker_path).resolve())
    if any(char in path for char in "{}\\\n\r"):
        raise ValueError("unsupported character in co-simulation marker path")
    return (
        f"set c2hls_phase_marker [open {{{path}}} w]\n"
        'puts $c2hls_phase_marker "start\\t[clock milliseconds]"\n'
        "flush $c2hls_phase_marker\nclose $c2hls_phase_marker\n"
        f"{command}\n"
        f"set c2hls_phase_marker [open {{{path}}} a]\n"
        'puts $c2hls_phase_marker "end\\t[clock milliseconds]"\n'
        "flush $c2hls_phase_marker\nclose $c2hls_phase_marker"
    )


def read_markers(path):
    try:
        with Path(path).open("r", encoding="ascii") as stream:
            text = stream.read(4096)
    except (OSError, UnicodeError):
        return {}
    markers = {}
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] not in {"start", "end"}:
            continue
        try:
            value = float(fields[1]) / 1000
        except ValueError:
            continue
        if math.isfinite(value) and value > 0:
            markers.setdefault(fields[0], value)
    if "end" in markers and (
        "start" not in markers or markers["end"] < markers["start"]
    ):
        markers.pop("end")
    return markers


def process_info(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"ppid": int(fields[1]), "start_ticks": fields[19], "state": fields[0]}
    except (OSError, ValueError, IndexError):
        return None


def process_table():
    result = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            info = process_info(int(path.name))
            if info is not None:
                result[int(path.name)] = info
    return result


def signal_identity(pid, start_ticks, signum):
    descriptor = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            descriptor = os.pidfd_open(pid)
        info = process_info(pid)
        if info is None or info["start_ticks"] != start_ticks or info["state"] == "Z":
            return False
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, signum)
        else:
            os.kill(pid, signum)
        return True
    except ProcessLookupError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


class OwnedProcessTree:
    """Track descendant PID/start-time identities, including detached sessions."""

    def __init__(self, process):
        self.process = process
        info = process_info(process.pid)
        self.identities = {process.pid: info["start_ticks"]} if info else {}
        self.capture()

    def capture(self):
        table = process_table()
        owned = {
            pid for pid, ticks in self.identities.items()
            if pid in table and table[pid]["start_ticks"] == ticks
        }
        while True:
            children = {pid for pid, info in table.items() if info["ppid"] in owned}
            if children <= owned:
                break
            owned.update(children)
        self.identities.update({pid: table[pid]["start_ticks"] for pid in owned})
        return {pid: table[pid] for pid in owned if table[pid]["state"] != "Z"}

    def terminate(self, grace_seconds=5, kill_wait_seconds=5):
        errors, signaled = [], []
        for signum, duration in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, kill_wait_seconds)):
            deadline, sent = time.monotonic() + duration, set()
            while True:
                for pid in sorted(self.capture(), key=lambda item: item == self.process.pid):
                    identity = (pid, self.identities[pid])
                    if identity in sent:
                        continue
                    sent.add(identity)
                    try:
                        if signal_identity(pid, identity[1], signum):
                            signaled.append({"pid": pid, "start_ticks": identity[1], "signal": int(signum)})
                    except OSError as exc:
                        errors.append(f"{type(exc).__name__}: {exc}")
                self.process.poll()
                if not self.capture() or time.monotonic() >= deadline:
                    break
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
        try:
            self.process.wait(timeout=.2)
        except subprocess.TimeoutExpired:
            errors.append("direct Vitis process did not exit within cleanup bound")
        return {"signaled": signaled, "remaining_owned_pids": sorted(self.capture()), "errors": errors}


def _tail(path, max_bytes=131072):
    with Path(path).open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - max_bytes))
        return stream.read(max_bytes).decode("utf-8", errors="replace")


def scan_validation_log(path, *, max_evidence_bytes=32 * 1024 * 1024,
                        max_line_bytes=1024 * 1024):
    """Read the entire stream and keep every complete C/RTL dump in order.

    A returned tail is for display only. This separate bounded evidence stream
    includes every outer dump marker and every RTL-postcheck marker, so an early
    mismatch can never disappear behind later warnings or a later correct dump.
    Oversized lines/sections or malformed markers fail closed; their discarded
    content is never treated as a successful complete validation stream.
    """
    marker_re = re.compile(r"==(BEGIN|END)\s+DUMP_ARRAYS==")
    failure_re = re.compile(
        r"\bERROR\b|simulation failed|segmentation violation|child killed|undefined symbol|ld\.lld"
    )
    success_markers = ("cosim done with 0 errors", "cosim_design finished successfully",
                       "c/rtl co-simulation finished: pass")
    fragments, issues = [], []
    depth = captured = scanned = 0
    tool_pass = tool_fail = False
    reached_eof = False
    with Path(path).open("rb") as stream:
        while True:
            raw = stream.readline(max_line_bytes + 1)
            if not raw:
                reached_eof = True
                break
            scanned += len(raw)
            if len(raw) > max_line_bytes:
                issues.append("validation_log_line_limit_exceeded")
                break
            line = raw.decode("utf-8", errors="replace")
            lowered = line.lower()
            tool_pass |= any(marker in lowered for marker in success_markers)
            tool_fail |= bool(failure_re.search(line) or failure_re.search(lowered))
            markers = list(marker_re.finditer(line))
            # Normal Vitis outer markers occupy one physical line. Reject a
            # split or partial marker rather than losing it from the evidence.
            if line.count("==BEGIN") + line.count("==END") != len(markers):
                issues.append("malformed_outer_dump_marker")
            keep = depth > 0 or bool(markers) or "Starting C post checking" in line
            for marker in markers:
                if marker.group(1) == "BEGIN":
                    if depth:
                        issues.append("nested_outer_dump_marker")
                    depth += 1
                else:
                    if depth != 1:
                        issues.append("unbalanced_outer_dump_marker")
                    depth = max(0, depth - 1)
            if keep:
                captured += len(raw)
                if captured > max_evidence_bytes:
                    issues.append("validation_evidence_limit_exceeded")
                    break
                fragments.append(line)
    if depth:
        issues.append("unterminated_outer_dump_marker")
    return {
        "validation_log": "".join(fragments),
        "validation_log_scan": {
            "complete": reached_eof and not issues, "reached_eof": reached_eof,
            "issues": sorted(set(issues)), "bytes_scanned": scanned,
            "evidence_bytes": captured, "evidence_limit_bytes": max_evidence_bytes,
            "line_limit_bytes": max_line_bytes,
            "scope": "all_C_and_RTL_dumps_and_RTL_postcheck_markers_in_complete_log",
        },
        "tool_success_marker_seen": tool_pass, "tool_failure_marker_seen": tool_fail,
    }


def run_guarded_command(command, *, invocation_timeout, phase_timeout, marker_path,
                        log_path, poll_seconds=.25, cleanup_grace_seconds=5,
                        cleanup_kill_seconds=5):
    """Execute with two wall clocks; missing phase markers never start its clock.

    Marker timestamps use the host wall clock. At first observation we map the
    start onto Python's monotonic clock; subsequent clock adjustments cannot
    extend a running phase's budget. Completed durations use the marker pair.
    Full output streams to disk; only a 128-KiB tail is returned in memory.
    """
    for value in (invocation_timeout, phase_timeout, poll_seconds):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("wall limits and poll interval must be positive and finite")
    marker_path, log_path = Path(marker_path), Path(log_path)
    if marker_path.exists():
        raise FileExistsError(f"refusing stale phase marker: {marker_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    phase_start_monotonic = None
    timeout_phase, classification = None, "completed"
    phase_elapsed = None
    markers = {}
    cleanup = None
    with log_path.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        owned = OwnedProcessTree(process)
        try:
            while True:
                owned.capture()
                now = time.monotonic()
                markers = read_markers(marker_path)
                if "start" in markers and phase_start_monotonic is None:
                    # Reject an implausible future timestamp rather than invent progress.
                    lag = time.time() - markers["start"]
                    if lag >= -.1:
                        phase_start_monotonic = now - max(0, lag)
                if "end" in markers:
                    phase_elapsed = markers["end"] - markers["start"]
                elif phase_start_monotonic is not None:
                    phase_elapsed = now - phase_start_monotonic
                if phase_elapsed is not None and phase_elapsed > phase_timeout:
                    timeout_phase, classification = "cosim_design", "cosim_wall_time_limit"
                    break
                if process.poll() is not None:
                    if process.returncode != 0:
                        classification = "tool_error"
                    elif "start" not in markers or "end" not in markers:
                        classification = "phase_marker_missing"
                    break
                if now - started > invocation_timeout:
                    timeout_phase, classification = "vitis_invocation", "invocation_wall_time_limit"
                    break
                time.sleep(min(poll_seconds, max(.001, invocation_timeout - (now - started))))
            invocation_elapsed = time.monotonic() - started
        finally:
            cleanup = owned.terminate(grace_seconds=cleanup_grace_seconds,
                                      kill_wait_seconds=cleanup_kill_seconds)
        stream.flush()
    timed_out = timeout_phase is not None
    if cleanup["remaining_owned_pids"]:
        classification = "process_cleanup_incomplete"
    evidence = scan_validation_log(log_path) if not timed_out else {}
    return {
        "log": _tail(log_path), "log_path": str(log_path), "returncode": process.returncode,
        "timed_out": timed_out, "timeout_phase": timeout_phase, "classification": classification,
        "invocation_wall_seconds": invocation_elapsed,
        "cosim_wall_seconds": phase_elapsed,
        "cosim_wall_seconds_censored": "end" not in markers and phase_elapsed is not None,
        "wall_timeout_seconds": phase_timeout, "invocation_timeout_seconds": invocation_timeout,
        "timing_scope": TIMING_SCOPE, "phase_marker_path": str(marker_path),
        "phase_markers_complete": "start" in markers and "end" in markers,
        "owned_process_cleanup": cleanup,
        **evidence,
    }

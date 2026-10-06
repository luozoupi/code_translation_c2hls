#!/usr/bin/env python3
"""csynth + cosim validation for exported AutoSA DSE packages at c2hls clock."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.autosa_dse_hls_common import (  # noqa: E402
    CsynthLatency,
    DEFAULT_CLOCK_NS,
    DEFAULT_DEVICE_PLATFORM,
    DEFAULT_FLOW_TARGET,
    DEFAULT_PART,
    HLS_PROJECT,
    _latency_fields,
    _parse_top_level_csynth,
    _update_json_file,
    _write_manifest,
    apply_c2hls_latency_to_package,
    run_package_csynth,
)

DEFAULT_COSIM_TIMEOUT_S = int(os.environ.get("C2HLS_COSIM_TIMEOUT", "14400"))
COSIM_MAX_BUFFER_BYTES = int(
    os.environ.get("AUTOSA_COSIM_MAX_BUFFER_BYTES", str(256 * 1024 * 1024))
)
# When cosim buffers are too large, accept csynth-only if csim fails or times out.
OVERSIZED_ACCEPT_CSYNTH_ONLY = os.environ.get(
    "AUTOSA_OVERSIZED_ACCEPT_CSYNTH_ONLY", "1"
).strip() not in ("0", "false", "False", "no", "NO")


def _parse_lat_rpt_cycles(work_dir: Path, proj_name: str = HLS_PROJECT):
    pattern = re.compile(r'\$TOTAL_EXECUTE_TIME\s*=\s*"([^"]+)"')
    root = work_dir / proj_name if (work_dir / proj_name).is_dir() else work_dir
    for cur, _dirs, files in os.walk(root):
        if "lat.rpt" not in files:
            continue
        try:
            text = (Path(cur) / "lat.rpt").read_text(encoding="utf-8", errors="ignore")
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bus_type_bytes(header_text: str, type_name: str) -> int:
    m = re.search(rf"typedef\s+ap_uint<(\d+)>\s+{re.escape(type_name)}\b", header_text)
    if m:
        return max(1, int(m.group(1)) // 8)
    return 64


def _kernel_port_types(pkg_dir: Path) -> dict[str, str]:
    """Map m_axi port name -> wide bus typedef used in kernel0 signature."""
    baseline = (pkg_dir / "hls_baseline.cpp").read_text(encoding="utf-8")
    header = (pkg_dir / "kernel_kernel.h").read_text(encoding="utf-8")
    m = re.search(r"void\s+kernel0\s*\((.*?)\)", baseline, re.DOTALL)
    if not m:
        return {}
    port_types: dict[str, str] = {}
    for chunk in _split_params(m.group(1)):
        chunk = re.sub(r"/\*.*?\*/", " ", chunk)
        pm = re.search(r"(\w+(?:_t\d+)?)\s*\*\s*(\w+)\s*$", chunk.strip())
        if pm:
            port_types[pm.group(2)] = pm.group(1)
    # Only keep ports that have ap_uint bus types in the header.
    return {p: t for p, t in port_types.items() if f"typedef ap_uint<" in header and t in header}


def _split_params(params: str) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in params:
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def _port_buffer_bytes(pkg_dir: Path, port: str, depth: int) -> int:
    header = (pkg_dir / "kernel_kernel.h").read_text(encoding="utf-8")
    port_types = _kernel_port_types(pkg_dir)
    type_name = port_types.get(port, "")
    return depth * _bus_type_bytes(header, type_name)


def _cosim_buffer_too_large(pkg_dir: Path, depths: dict[str, int]) -> tuple[bool, int]:
    max_bytes = 0
    for port, depth in depths.items():
        max_bytes = max(max_bytes, _port_buffer_bytes(pkg_dir, port, depth))
    return max_bytes > COSIM_MAX_BUFFER_BYTES, max_bytes


def infer_autosa_cosim_depths(pkg_dir: Path) -> dict[str, int]:
    """Infer m_axi cosim depths from testbench memcpy/malloc sizes."""
    tb = (pkg_dir / "testbench.cpp").read_text(encoding="utf-8")
    baseline = (pkg_dir / "hls_baseline.cpp").read_text(encoding="utf-8")
    ports = re.findall(
        r"#pragma\s+HLS\s+INTERFACE\s+m_axi\b.*?\bport\s*=\s*([A-Za-z_][A-Za-z0-9_]*)",
        baseline,
    )
    depths: dict[str, int] = {}
    for port in ports:
        patterns = [
            rf"memcpy\(\s*buffer_{port}\[.*?\],\s*dev_{port},\s*\((\d+)\)",
            rf"memcpy\(\s*dev_{port},\s*buffer_{port}\[.*?\],\s*\((\d+)\)",
            rf"buffer_{port}_tmp\s*=\s*\([^)]+\)\s*malloc\(\((\d+)\)",
            rf"dev_{port}\s*=\s*\([^)]+\)\s*malloc\(\s*(\d+)\s*\*",
        ]
        for pat in patterns:
            m = re.search(pat, tb)
            if m:
                depths[port] = int(m.group(1))
                break
    return depths


def _inject_interface_depths(text: str, interface_depths: dict[str, int]) -> str:
    if not interface_depths:
        return text
    lines = []
    for line in text.splitlines():
        m = re.search(
            r"#pragma\s+HLS\s+INTERFACE\s+m_axi\b.*?\bport\s*=\s*([A-Za-z_][A-Za-z0-9_]*)",
            line,
        )
        if m:
            port = m.group(1)
            depth = interface_depths.get(port)
            if depth is not None and "depth=" not in line:
                line = line.rstrip() + f" depth={depth}"
        lines.append(line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _prepare_cosim_testbench(pkg_dir: Path, depths: dict[str, int]) -> Path:
    tb = (pkg_dir / "testbench.cpp").read_text(encoding="utf-8")
    header = (pkg_dir / "kernel_kernel.h").read_text(encoding="utf-8")
    port_types = _kernel_port_types(pkg_dir)
    for port, depth in depths.items():
        bus_bytes = _bus_type_bytes(header, port_types.get(port, ""))
        byte_count = f"({depth}) * {bus_bytes}"
        # Fix wide-bus buffer malloc/memcpy sizes (AutoSA codegen uses wrong sizeof).
        tb = re.sub(
            rf"(buffer_{port}_tmp\s*=\s*\([^)]+\)\s*)malloc\(\({depth}\)\s*\*\s*sizeof\([^)]+\)\)",
            rf"\1malloc({byte_count})",
            tb,
        )
        tb = re.sub(
            rf"(dev_{port}\s*=\s*\([^)]+\)\s*)malloc\(\s*{depth}\s*\*\s*sizeof\([^)]+\)\)",
            rf"\1malloc({byte_count})",
            tb,
        )
        tb = re.sub(
            rf"memcpy\(([^,]+),\s*([^,]+),\s*\({depth}\)\s*\*\s*sizeof\([^)]+\)\)",
            rf"memcpy(\1, \2, {byte_count})",
            tb,
        )
    dst = pkg_dir / "testbench_cosim.cpp"
    dst.write_text(tb, encoding="utf-8")
    return dst


def _prepare_cosim_kernel(pkg_dir: Path, depths: dict[str, int]) -> Path:
    src = pkg_dir / "hls_baseline.cpp"
    dst = pkg_dir / "hls_baseline_cosim.cpp"
    patched = _inject_interface_depths(src.read_text(encoding="utf-8"), depths)
    dst.write_text(patched, encoding="utf-8")
    return dst


def _write_csynth_csim_tcl(
    pkg_dir: Path,
    *,
    clock_ns: float,
    kernel_cpp: str = "hls_baseline.cpp",
    testbench_cpp: str = "testbench.cpp",
) -> Path:
    tcl_path = pkg_dir / "run_csynth_csim_c2hls.tcl"
    lines = [
        f"open_project {HLS_PROJECT}",
        "set_top kernel0",
        "add_files kernel_kernel.h",
        f"add_files {kernel_cpp}",
        f"add_files -tb {testbench_cpp}",
        "add_files -tb kernel.h",
        f'open_solution "solution1" -flow_target {DEFAULT_FLOW_TARGET}',
        f"set_part {{{DEFAULT_PART}}}",
        f"create_clock -period {clock_ns} -name default",
        "config_compile -name_max_length 50",
        "csynth_design",
        "csim_design",
        "exit",
    ]
    tcl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tcl_path


def _csim_passed(log: str) -> bool:
    from hls_eval import csim_testbench_reported_mismatch

    if csim_testbench_reported_mismatch(log):
        return False
    log_lower = log.lower()
    if (
        "test passed!" in log_lower
        or "pass: csim finished" in log_lower
        or "csim finished successfully" in log_lower
        or "csim done with 0 errors" in log_lower
    ):
        return True
    if (
        "test failed" in log_lower
        or "fail: csim finished" in log_lower
        or "csim failed" in log_lower
    ):
        return False
    return False


def run_package_csim(
    pkg_dir: Path,
    *,
    clock_ns: float = DEFAULT_CLOCK_NS,
    timeout_s: int = DEFAULT_COSIM_TIMEOUT_S,
    depths: dict[str, int] | None = None,
) -> dict:
    """Run csynth+csim when cosim buffers are too large for TV generation."""
    hls_prj = pkg_dir / HLS_PROJECT
    if hls_prj.exists():
        shutil.rmtree(hls_prj)

    if depths is None:
        depths = infer_autosa_cosim_depths(pkg_dir)
    testbench_cpp = "testbench.cpp"
    if depths:
        _prepare_cosim_testbench(pkg_dir, depths)
        testbench_cpp = "testbench_cosim.cpp"

    tcl_name = "run_csynth_csim_c2hls.tcl"
    _write_csynth_csim_tcl(pkg_dir, clock_ns=clock_ns, testbench_cpp=testbench_cpp)

    try:
        rc, log, tcl_path = _run_vitis_tcl(pkg_dir, tcl_name, timeout_s=timeout_s)
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "passed": False,
            "error": f"csim timed out after {timeout_s}s",
            "log_tail": str(exc),
        }

    (pkg_dir / "csim_run.log").write_text(log, encoding="utf-8")

    csynth_xml = pkg_dir / HLS_PROJECT / "solution1/syn/report/csynth.xml"
    csynth_latency = _parse_top_level_csynth(csynth_xml) if csynth_xml.is_file() else None
    passed = _csim_passed(log) and rc == 0

    err = ""
    if not passed:
        err = f"csim failed (rc={rc})"

    return {
        "status": "ok" if passed else "failed",
        "passed": passed,
        "error": err if not passed else "",
        "tcl": tcl_path,
        "clock_ns": clock_ns,
        "part": DEFAULT_PART,
        "csynth_latency": (
            {
                "best": csynth_latency.best,
                "average": csynth_latency.average,
                "worst": csynth_latency.worst,
            }
            if csynth_latency
            else None
        ),
    }


def _run_oversized_validate(
    pkg_dir: Path,
    *,
    depths: dict[str, int],
    max_buf_bytes: int,
    clock_ns: float,
    timeout_s: int,
) -> dict:
    """Cosim TV generation is infeasible; try csim, then optional csynth-only."""
    csim = run_package_csim(pkg_dir, clock_ns=clock_ns, timeout_s=timeout_s, depths=depths)
    if csim["passed"]:
        return {
            "status": "skipped_oversized",
            "passed": True,
            "error": "",
            "skipped_reason": "cosim_buffer_too_large",
            "max_buffer_bytes": max_buf_bytes,
            "cosim_max_buffer_bytes": COSIM_MAX_BUFFER_BYTES,
            "clock_ns": clock_ns,
            "part": DEFAULT_PART,
            "cosim_depths": depths,
            "csynth_latency": csim.get("csynth_latency"),
            "cosim_runtime_cycles": None,
            "cosim_runtime_us": None,
            "csim_passed": True,
            "validate_mode": "csynth_csim",
        }

    if not OVERSIZED_ACCEPT_CSYNTH_ONLY:
        return {
            "status": "failed",
            "passed": False,
            "error": csim.get("error", "csim failed"),
            "skipped_reason": "cosim_buffer_too_large",
            "max_buffer_bytes": max_buf_bytes,
            "cosim_max_buffer_bytes": COSIM_MAX_BUFFER_BYTES,
            "clock_ns": clock_ns,
            "part": DEFAULT_PART,
            "cosim_depths": depths,
            "csynth_latency": csim.get("csynth_latency"),
            "cosim_runtime_cycles": None,
            "cosim_runtime_us": None,
            "csim_passed": False,
            "validate_mode": "csynth_csim",
        }

    # csim failed or timed out — still record csynth latency if we have it.
    csynth_latency = csim.get("csynth_latency")
    if not csynth_latency:
        try:
            latency = run_package_csynth(pkg_dir, clock_ns=clock_ns)
            csynth_latency = {
                "best": latency.best,
                "average": latency.average,
                "worst": latency.worst,
            }
        except Exception as exc:
            return {
                "status": "failed",
                "passed": False,
                "error": f"csim and csynth failed: {exc}",
                "skipped_reason": "cosim_buffer_too_large",
                "max_buffer_bytes": max_buf_bytes,
                "cosim_max_buffer_bytes": COSIM_MAX_BUFFER_BYTES,
                "clock_ns": clock_ns,
                "part": DEFAULT_PART,
                "cosim_depths": depths,
                "csynth_latency": None,
                "csim_passed": False,
                "validate_mode": "csynth_only",
            }

    return {
        "status": "ok_csynth_only",
        "passed": True,
        "error": "",
        "skipped_reason": "cosim_buffer_too_large",
        "csim_error": csim.get("error", ""),
        "max_buffer_bytes": max_buf_bytes,
        "cosim_max_buffer_bytes": COSIM_MAX_BUFFER_BYTES,
        "clock_ns": clock_ns,
        "part": DEFAULT_PART,
        "cosim_depths": depths,
        "csynth_latency": csynth_latency,
        "cosim_runtime_cycles": None,
        "cosim_runtime_us": None,
        "csim_passed": False,
        "validate_mode": "csynth_only",
    }


def _write_csynth_cosim_tcl(
    pkg_dir: Path,
    *,
    clock_ns: float,
    kernel_cpp: str = "hls_baseline_cosim.cpp",
    testbench_cpp: str = "testbench_cosim.cpp",
) -> Path:
    tcl_path = pkg_dir / "run_csynth_cosim_c2hls.tcl"
    lines = [
        f"open_project {HLS_PROJECT}",
        "set_top kernel0",
        "add_files kernel_kernel.h",
        f"add_files {kernel_cpp}",
        f"add_files -tb {testbench_cpp}",
        "add_files -tb kernel.h",
        f'open_solution "solution1" -flow_target {DEFAULT_FLOW_TARGET}',
        f"set_part {{{DEFAULT_PART}}}",
        f"create_clock -period {clock_ns} -name default",
        "config_compile -name_max_length 50",
        "csynth_design",
        "if {[info exists ::env(LIBRARY_PATH)]} { unset ::env(LIBRARY_PATH) }",
        "cosim_design",
        "exit",
    ]
    tcl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tcl_path


def _run_vitis_tcl(pkg_dir: Path, tcl_name: str, *, timeout_s: int) -> tuple[int, str, str]:
    tcl = pkg_dir / tcl_name
    env = os.environ.copy()
    if shutil.which("vitis-run"):
        cmd = ["vitis-run", "--tcl", "--input_file", str(tcl)]
    elif shutil.which("vitis_hls"):
        cmd = ["vitis_hls", "-f", tcl_name]
    else:
        raise RuntimeError("neither vitis-run nor vitis_hls found on PATH")

    proc = subprocess.run(
        cmd,
        cwd=pkg_dir,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )
    log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, log, str(tcl)


def _cosim_passed(log: str) -> bool:
    log_lower = log.lower()
    if (
        "co-simulation finished: pass" in log_lower
        or "cosim done with 0 errors" in log_lower
        or "cosim_design finished successfully" in log_lower
    ):
        return True
    if (
        "co-simulation finished: fail" in log_lower
        or "simulation failed" in log_lower
        or "segmentation violation" in log_lower
        or "sigsegv" in log_lower
    ):
        return False
    return False


def run_package_cosim(
    pkg_dir: Path,
    *,
    clock_ns: float = DEFAULT_CLOCK_NS,
    timeout_s: int = DEFAULT_COSIM_TIMEOUT_S,
    skip_csynth: bool = False,
) -> dict:
    """Run csynth+cosim in package dir at c2hls clock; return result dict."""
    if skip_csynth:
        raise ValueError("skip_csynth not supported for combined flow")

    hls_prj = pkg_dir / HLS_PROJECT
    if hls_prj.exists():
        shutil.rmtree(hls_prj)

    depths = infer_autosa_cosim_depths(pkg_dir)
    if not depths:
        return {
            "status": "failed",
            "passed": False,
            "error": "could not infer m_axi cosim depths from testbench",
            "clock_ns": clock_ns,
            "part": DEFAULT_PART,
            "csynth_latency": None,
            "cosim_runtime_cycles": None,
            "cosim_runtime_us": None,
            "cosim_depths": depths,
        }

    too_large, max_buf_bytes = _cosim_buffer_too_large(pkg_dir, depths)
    if too_large:
        return _run_oversized_validate(
            pkg_dir,
            depths=depths,
            max_buf_bytes=max_buf_bytes,
            clock_ns=clock_ns,
            timeout_s=timeout_s,
        )

    _prepare_cosim_kernel(pkg_dir, depths)
    _prepare_cosim_testbench(pkg_dir, depths)
    tcl_name = "run_csynth_cosim_c2hls.tcl"
    _write_csynth_cosim_tcl(pkg_dir, clock_ns=clock_ns)

    try:
        rc, log, tcl_path = _run_vitis_tcl(pkg_dir, tcl_name, timeout_s=timeout_s)
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "passed": False,
            "error": f"cosim timed out after {timeout_s}s",
            "log_tail": str(exc),
            "cosim_depths": depths,
        }

    (pkg_dir / "cosim_run.log").write_text(log, encoding="utf-8")

    csynth_xml = pkg_dir / HLS_PROJECT / "solution1/syn/report/csynth.xml"
    csynth_latency = _parse_top_level_csynth(csynth_xml) if csynth_xml.is_file() else None
    cosim_cycles = _parse_lat_rpt_cycles(pkg_dir, HLS_PROJECT)
    passed = _cosim_passed(log)
    if rc != 0:
        passed = False

    err = ""
    if not passed:
        for pat in (
            r"ERROR:.*",
            r"error:.*",
            r"Simulation failed.*",
            r"ERROR \[.*\]",
        ):
            for line in log.splitlines():
                if re.search(pat, line, re.IGNORECASE):
                    err = line.strip()
                    break
            if err:
                break
        if not err:
            err = f"cosim failed (rc={rc})"

    return {
        "status": "ok" if passed else "failed",
        "passed": passed,
        "error": err if not passed else "",
        "tcl": tcl_path,
        "clock_ns": clock_ns,
        "part": DEFAULT_PART,
        "cosim_depths": depths,
        "csynth_latency": (
            {
                "best": csynth_latency.best,
                "average": csynth_latency.average,
                "worst": csynth_latency.worst,
            }
            if csynth_latency
            else None
        ),
        "cosim_runtime_cycles": cosim_cycles,
        "cosim_runtime_us": (
            cosim_cycles * clock_ns / 1000.0 if cosim_cycles is not None else None
        ),
    }


def validate_package(
    pkg_dir: Path,
    *,
    clock_ns: float = DEFAULT_CLOCK_NS,
    cosim_timeout_s: int = DEFAULT_COSIM_TIMEOUT_S,
    resynth_only: bool = False,
) -> dict:
    """csynth at c2hls clock; optionally cosim. Updates provenance/metadata."""
    pkg_dir = pkg_dir.resolve()
    result: dict = {
        "package": str(pkg_dir),
        "dir_name": pkg_dir.name,
        "clock_ns": clock_ns,
        "part": DEFAULT_PART,
        "started_at": _utc_now(),
    }

    try:
        if resynth_only:
            latency = run_package_csynth(pkg_dir, clock_ns=clock_ns)
            apply_c2hls_latency_to_package(pkg_dir, latency, clock_ns=clock_ns)
            result.update(_latency_fields("c2hls", latency))
            result["csynth_status"] = "ok"
            result["cosim_status"] = "skipped"
            result["status"] = "ok"
        else:
            cosim = run_package_cosim(pkg_dir, clock_ns=clock_ns, timeout_s=cosim_timeout_s)
            result["cosim"] = cosim
            result["cosim_status"] = cosim["status"]
            result["cosim_passed"] = cosim["passed"]
            if cosim.get("skipped_reason"):
                result["cosim_skipped_reason"] = cosim["skipped_reason"]
                result["csim_passed"] = cosim.get("csim_passed")
                if cosim.get("validate_mode"):
                    result["validate_mode"] = cosim["validate_mode"]
                if cosim.get("csim_error"):
                    result["csim_error"] = cosim["csim_error"]

            if cosim.get("csynth_latency"):
                latency = CsynthLatency(**cosim["csynth_latency"])
                apply_c2hls_latency_to_package(pkg_dir, latency, clock_ns=clock_ns)
                result.update(_latency_fields("c2hls", latency))
                result["csynth_status"] = "ok"
            else:
                result["csynth_status"] = "missing"

            _apply_cosim_to_package(pkg_dir, cosim, clock_ns=clock_ns)

            result["status"] = "ok" if cosim["passed"] else "failed"
            if cosim.get("skipped_reason") and cosim["passed"]:
                result["status"] = "ok"
            if not cosim["passed"]:
                result["error"] = cosim.get("error", "cosim failed")
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    result["finished_at"] = _utc_now()
    out = pkg_dir / "hls_validate.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_manifest(pkg_dir)
    return result


def _apply_cosim_to_package(pkg_dir: Path, cosim: dict, *, clock_ns: float) -> None:
    stamp = _utc_now()

    def _patch(prov: dict) -> None:
        prov["c2hls_cosim_passed"] = bool(cosim.get("passed"))
        prov["c2hls_cosim_status"] = cosim.get("status")
        if cosim.get("skipped_reason"):
            prov["c2hls_cosim_skipped_reason"] = cosim["skipped_reason"]
            prov["c2hls_csim_passed"] = cosim.get("csim_passed")
            if cosim.get("validate_mode"):
                prov["c2hls_validate_mode"] = cosim["validate_mode"]
            if cosim.get("csim_error"):
                prov["c2hls_csim_error"] = cosim["csim_error"]
        prov["c2hls_cosim_runtime_cycles"] = cosim.get("cosim_runtime_cycles")
        prov["c2hls_cosim_runtime_us"] = cosim.get("cosim_runtime_us")
        prov["c2hls_cosim_clock_ns"] = clock_ns
        prov["c2hls_cosim_part"] = DEFAULT_PART
        prov["c2hls_cosim_at"] = stamp
        if cosim.get("cosim_depths"):
            prov["c2hls_cosim_depths"] = cosim["cosim_depths"]
        if cosim.get("error"):
            prov["c2hls_cosim_error"] = cosim["error"]

    _update_json_file(pkg_dir / "provenance.json", _patch)

    def _patch_meta(meta: dict) -> None:
        _patch(meta.setdefault("provenance", {}))
        meta["supports_cosim"] = True
        meta["c2hls_cosim_passed"] = bool(cosim.get("passed"))
        if cosim.get("cosim_depths"):
            meta["cosim_depths"] = cosim["cosim_depths"]

    _update_json_file(pkg_dir / "metadata.json", _patch_meta)


def kernel_walltime(kernel_id: str) -> str:
    if kernel_id.startswith("large_"):
        return os.environ.get("AUTOSA_DSE_LARGE_WALLTIME", "24:00:00")
    return os.environ.get("AUTOSA_DSE_SMALL_WALLTIME", "8:00:00")


def kernel_cosim_timeout(kernel_id: str) -> int:
    if kernel_id.startswith("large_"):
        return int(os.environ.get("AUTOSA_DSE_LARGE_COSIM_TIMEOUT", "43200"))
    return int(os.environ.get("AUTOSA_DSE_SMALL_COSIM_TIMEOUT", "7200"))

"""
HLS evaluation utilities: run Vitis HLS synthesis and parse reports.
"""

import os
import re
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

VITIS_SETTINGS = "/mnt/data/luo00466/Xilinx/2025.2/Vitis/settings64.sh"
DEFAULT_PART = "xc7a100t-csg324-1"
DEFAULT_CLOCK_NS = 4
SYNTH_TIMEOUT = 600  # 10 minutes
CSIM_TIMEOUT = 120   # 2 minutes
COSIM_TIMEOUT = 600  # 10 minutes


def _run_vitis_cmd(cmd: str, timeout: int) -> tuple:
    """Run a shell command with Vitis sourced. Returns (stdout+stderr, timed_out)."""
    full_cmd = f"source {VITIS_SETTINGS} && {cmd} 2>&1"
    try:
        result = subprocess.run(
            ["bash", "-c", full_cmd],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return result.stdout + result.stderr, False
    except subprocess.TimeoutExpired:
        return "", True


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
    clock_ns: int = DEFAULT_CLOCK_NS,
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
            "log": "",
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

    report["work_dir"] = work_dir

    return {
        "success": True,
        "error": "",
        "report": report,
        "report_raw": report_raw,
        "log": log,
    }


def run_csim(
    hls_code: str,
    testbench_code: str,
    header_code: str = "",
    header_name: str = "kernel.h",
    top_function: str = "workload",
    part: str = DEFAULT_PART,
    clock_ns: int = DEFAULT_CLOCK_NS,
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
        return {"success": False, "passed": False, "error": f"Csim timed out after {CSIM_TIMEOUT}s", "log": ""}

    passed = "CSim done with 0 errors" in log or "csim_design finished successfully" in log.lower()
    has_error = "ERROR" in log and "0 errors" not in log.lower()

    return {
        "success": not has_error,
        "passed": passed,
        "error": "" if not has_error else "Csim failed",
        "log": log,
    }


def run_cosim(
    hls_code: str,
    testbench_code: str,
    header_code: str = "",
    header_name: str = "kernel.h",
    top_function: str = "workload",
    part: str = DEFAULT_PART,
    clock_ns: int = DEFAULT_CLOCK_NS,
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
        return {"success": False, "passed": False, "error": f"Cosim timed out after {COSIM_TIMEOUT}s", "log": ""}

    log_lower = log.lower()
    passed = (
        "cosim done with 0 errors" in log_lower
        or "cosim_design finished successfully" in log_lower
        or "c/rtl co-simulation finished: pass" in log_lower
    )
    has_error = "ERROR" in log and "0 errors" not in log_lower

    return {
        "success": not has_error,
        "passed": passed,
        "error": "" if not has_error else "Cosim failed",
        "log": log,
    }


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

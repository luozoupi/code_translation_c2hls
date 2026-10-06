#!/usr/bin/env python3
"""Shared csynth/cosim helpers for AutoSA DSE exported packages."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PART = "xcu280-fsvh2892-2L-e"
DEFAULT_CLOCK_NS = 3.33
DEFAULT_DEVICE_PLATFORM = "xilinx_u280_gen3x16_xdma_1_202211_1"
DEFAULT_FLOW_TARGET = "vitis"
HLS_PROJECT = "hls_prj"


@dataclass(frozen=True)
class CsynthLatency:
    best: int
    average: int
    worst: int


def _bench_name(kernel_id: str) -> str:
    return f"autosa_{kernel_id}"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_top_level_csynth(xml_path: Path) -> CsynthLatency | None:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None
    perf = root.find("PerformanceEstimates")
    if perf is None:
        return None
    summary = perf.find("SummaryOfOverallLatency")
    if summary is None:
        return None
    worst_el = summary.find("Worst-caseLatency")
    best_el = summary.find("Best-caseLatency")
    avg_el = summary.find("Average-caseLatency")
    if worst_el is None or not (worst_el.text or "").strip():
        return None
    worst = int(worst_el.text)
    best = int(best_el.text) if best_el is not None and (best_el.text or "").strip() else worst
    average = int(avg_el.text) if avg_el is not None and (avg_el.text or "").strip() else worst
    if worst <= 0:
        return None
    return CsynthLatency(best=best, average=average, worst=worst)


def _latency_fields(prefix: str, latency: CsynthLatency) -> dict[str, int]:
    return {
        f"{prefix}_best_cycles": latency.best,
        f"{prefix}_average_cycles": latency.average,
        f"{prefix}_worst_cycles": latency.worst,
    }


def _write_manifest(out_dir: Path) -> None:
    lines = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_file():
            continue
        lines.append(f"{_sha256_file(path)}  {path.name}")
    (out_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_json_file(path: Path, updater) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    updater(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_csynth_tcl(
    pkg_dir: Path,
    *,
    clock_ns: float,
    tcl_name: str,
    project_name: str = HLS_PROJECT,
    kernel_cpp: str = "hls_baseline.cpp",
    testbench_cpp: str | None = None,
) -> Path:
    tcl_path = pkg_dir / tcl_name
    lines = [
        f"open_project {project_name}",
        "set_top kernel0",
        "add_files kernel_kernel.h",
        f"add_files {kernel_cpp}",
    ]
    if testbench_cpp:
        lines.extend(
            [
                f"add_files -tb {testbench_cpp}",
                "add_files -tb kernel.h",
            ]
        )
    lines.extend(
        [
            f'open_solution "solution1" -flow_target {DEFAULT_FLOW_TARGET}',
            f"set_part {{{DEFAULT_PART}}}",
            f"create_clock -period {clock_ns} -name default",
            "config_compile -name_max_length 50",
        ]
    )
    tcl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tcl_path


def emit_autosa_dataset_tcls(
    pkg_dir: Path,
    *,
    clock_ns: float = DEFAULT_CLOCK_NS,
    project_name: str = HLS_PROJECT,
    has_cosim: bool = False,
) -> dict[str, str]:
    """Write dataset_hls*.tcl files for c2hls gold-gate / flash flows."""
    synth = _write_csynth_tcl(
        pkg_dir,
        clock_ns=clock_ns,
        tcl_name="dataset_hls.tcl",
        project_name=project_name,
        kernel_cpp="hls_baseline.cpp",
    )
    # Append csynth + exit to synth tcl (template only has setup)
    synth_text = synth.read_text(encoding="utf-8")
    if "csynth_design" not in synth_text:
        synth.write_text(synth_text + "csynth_design\nexit\n", encoding="utf-8")

    csim = _write_csynth_tcl(
        pkg_dir,
        clock_ns=clock_ns,
        tcl_name="dataset_hls_csim.tcl",
        project_name=project_name,
        kernel_cpp="hls_baseline.cpp",
        testbench_cpp="testbench.cpp",
    )
    csim_text = csim.read_text(encoding="utf-8")
    if "csim_design" not in csim_text:
        csim.write_text(csim_text + "csim_design\nexit\n", encoding="utf-8")

    out = {
        "dataset_hls.tcl": str(synth),
        "dataset_hls_csim.tcl": str(csim),
    }
    if has_cosim:
        cosim = _write_csynth_tcl(
            pkg_dir,
            clock_ns=clock_ns,
            tcl_name="dataset_hls_cosim.tcl",
            project_name=project_name,
            kernel_cpp="hls_baseline_cosim.cpp",
            testbench_cpp="testbench_cosim.cpp",
        )
        cosim_text = cosim.read_text(encoding="utf-8")
        extra = [
            "if {[info exists ::env(LIBRARY_PATH)]} { unset ::env(LIBRARY_PATH) }",
            "csynth_design",
            "cosim_design",
            "exit",
        ]
        if "cosim_design" not in cosim_text:
            cosim.write_text(cosim_text + "\n".join(extra) + "\n", encoding="utf-8")
        out["dataset_hls_cosim.tcl"] = str(cosim)
    return out


def run_package_csynth(
    pkg_dir: Path,
    *,
    clock_ns: float,
    tcl_name: str = "run_synth_resynth.tcl",
    project_name: str = HLS_PROJECT,
) -> CsynthLatency:
    hls_prj = pkg_dir / project_name
    if hls_prj.exists():
        shutil.rmtree(hls_prj)

    tcl = _write_csynth_tcl(
        pkg_dir,
        clock_ns=clock_ns,
        tcl_name=tcl_name,
        project_name=project_name,
    )
    tcl_text = tcl.read_text(encoding="utf-8")
    if "csynth_design" not in tcl_text:
        tcl.write_text(tcl_text + "csynth_design\nexit\n", encoding="utf-8")
    if shutil.which("vitis-run"):
        cmd = ["vitis-run", "--tcl", "--input_file", str(tcl)]
    elif shutil.which("vitis_hls"):
        cmd = ["vitis_hls", "-f", str(tcl.name)]
    else:
        raise RuntimeError("neither vitis-run nor vitis_hls found on PATH")
    subprocess.run(cmd, cwd=pkg_dir, check=True)

    csynth_xml = pkg_dir / project_name / "solution1/syn/report/csynth.xml"
    parsed = _parse_top_level_csynth(csynth_xml)
    if not parsed:
        raise RuntimeError(f"could not parse {csynth_xml}")
    return parsed


def apply_c2hls_latency_to_package(
    pkg_dir: Path,
    latency: CsynthLatency,
    *,
    clock_ns: float = DEFAULT_CLOCK_NS,
) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    latency_fields = _latency_fields("c2hls", latency)

    def _patch_provenance(prov: dict) -> None:
        prov.update(latency_fields)
        prov["c2hls_resynth_clock_ns"] = clock_ns
        prov["c2hls_resynth_part"] = DEFAULT_PART
        prov["c2hls_resynth_at"] = stamp

    _update_json_file(pkg_dir / "provenance.json", _patch_provenance)

    def _patch_metadata(meta: dict) -> None:
        _patch_provenance(meta.setdefault("provenance", {}))
        for variant in meta.get("variants", []):
            variant.update(latency_fields)

    _update_json_file(pkg_dir / "metadata.json", _patch_metadata)
    _write_manifest(pkg_dir)


def iter_export_packages(
    out_root: Path,
    *,
    kernel_ids: list[str] | None = None,
    packages: list[Path] | None = None,
) -> list[Path]:
    if packages:
        return [p.resolve() for p in packages]

    allowed_benches = {_bench_name(k) for k in kernel_ids} if kernel_ids else None
    found: list[Path] = []
    for bench_dir in sorted(out_root.iterdir()):
        if not bench_dir.is_dir():
            continue
        if allowed_benches is not None and bench_dir.name not in allowed_benches:
            continue
        for pkg_dir in sorted(bench_dir.iterdir()):
            if not pkg_dir.is_dir():
                continue
            if not pkg_dir.name.startswith("rank") or "_latency_" not in pkg_dir.name:
                continue
            if (pkg_dir / "hls_baseline.cpp").is_file() and (pkg_dir / "provenance.json").is_file():
                found.append(pkg_dir)
    return found

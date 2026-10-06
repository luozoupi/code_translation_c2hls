#!/usr/bin/env python3
"""Export AutoSA DSE training-csynth winners into c2hls-compatible AutoSA_sources/."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.prepare_autosa_ready import KERNELS as AUTOSA_KERNELS  # noqa: E402
from scripts.strip_autosa_hls import build_plain_from_src_dir, strip_autosa_hls  # noqa: E402

AUTOSA_ROOT = Path(
    __import__("os").environ.get(
        "AUTOSA_ROOT",
        "/scratch/hpc-prf-llmfpga/asa582/projects/AutoSA",
    )
)
DEFAULT_CAMPAIGN = (
    AUTOSA_ROOT / "artifacts/dse/campaigns/20260709_full21_dse"
)
DEFAULT_OUT = REPO / "AutoSA_sources"

KERNEL_ID_TO_BENCH = {kid: f"autosa_{kid}" for kid, _ in AUTOSA_KERNELS}
KERNEL_ID_TO_REL = {kid: rel for kid, rel in AUTOSA_KERNELS}

REQUIRED_SRC = (
    "kernel_kernel.cpp",
    "kernel_kernel_modules.cpp",
    "kernel_kernel.h",
    "kernel_host.cpp",
)

CSYNTH_CLOCK_NS = 5.0  # DSE training csynth clock (ranking latencies measured here)

from scripts.autosa_dse_hls_common import (  # noqa: E402
    CsynthLatency,
    DEFAULT_CLOCK_NS,
    DEFAULT_DEVICE_PLATFORM,
    DEFAULT_FLOW_TARGET,
    DEFAULT_PART,
    _latency_fields,
    _parse_top_level_csynth,
    _update_json_file,
    _write_csynth_tcl,
    _write_manifest,
    apply_c2hls_latency_to_package,
    iter_export_packages,
    run_package_csynth,
)


@dataclass
class DesignRecord:
    kernel_id: str
    bench: str
    latency: CsynthLatency
    src_dir: Path
    output_dir: Path
    csynth_xml: Path
    sa_sizes: str | None
    autosa_cmd: str | None

    @property
    def worst_latency(self) -> int:
        return self.latency.worst

    @property
    def best_latency(self) -> int:
        return self.latency.best

    @property
    def average_latency(self) -> int:
        return self.latency.average


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _bench_name(kernel_id: str) -> str:
    return KERNEL_ID_TO_BENCH.get(kernel_id, f"autosa_{kernel_id}")


def _kernel_h_path(kernel_id: str) -> Path:
    rel = KERNEL_ID_TO_REL.get(kernel_id)
    if not rel:
        raise KeyError(f"unknown kernel_id: {kernel_id}")
    return AUTOSA_ROOT / rel / "kernel.h"


def _src_dir_from_csynth(xml_path: Path) -> Path | None:
    # .../output/hls_prj/solution1/syn/report/csynth.xml -> .../output/src
    try:
        output_dir = xml_path.parents[4]
    except IndexError:
        return None
    src = output_dir / "src"
    return src if src.is_dir() else None


def _read_sa_sizes(src_dir: Path) -> tuple[str | None, str | None]:
    cmd_path = src_dir / "cmd"
    if not cmd_path.is_file():
        return None, None
    text = cmd_path.read_text(encoding="utf-8", errors="replace").strip()
    m = re.search(r"--sa-sizes=(\{[^}]+\})", text)
    sa = m.group(1) if m else None
    return sa, text


def _validate_src_dir(src_dir: Path) -> list[str]:
    missing = [name for name in REQUIRED_SRC if not (src_dir / name).is_file()]
    return missing


def discover_designs(
    campaign_root: Path,
    kernel_ids: list[str] | None = None,
) -> dict[str, list[DesignRecord]]:
    exhaustive = campaign_root / "exhaustive"
    if not exhaustive.is_dir():
        raise FileNotFoundError(f"missing exhaustive/: {exhaustive}")

    allowed = set(kernel_ids) if kernel_ids else None
    by_kernel: dict[str, list[DesignRecord]] = {}

    for kernel_dir in sorted(exhaustive.iterdir()):
        if not kernel_dir.is_dir():
            continue
        kernel_id = kernel_dir.name
        if allowed is not None and kernel_id not in allowed:
            continue

        records: list[DesignRecord] = []
        for xml_path in kernel_dir.rglob("csynth.xml"):
            if "/tmp/optimizer/synth/" not in str(xml_path):
                continue
            parsed = _parse_top_level_csynth(xml_path)
            if not parsed:
                continue
            src_dir = _src_dir_from_csynth(xml_path)
            if src_dir is None:
                continue
            missing = _validate_src_dir(src_dir)
            if missing:
                continue
            sa_sizes, autosa_cmd = _read_sa_sizes(src_dir)
            output_dir = src_dir.parent
            records.append(
                DesignRecord(
                    kernel_id=kernel_id,
                    bench=_bench_name(kernel_id),
                    latency=parsed,
                    src_dir=src_dir,
                    output_dir=output_dir,
                    csynth_xml=xml_path,
                    sa_sizes=sa_sizes,
                    autosa_cmd=autosa_cmd,
                )
            )

        # Deduplicate by output_dir, keep best (lowest) latency per dir
        dedup: dict[str, DesignRecord] = {}
        for rec in records:
            key = str(rec.output_dir.resolve())
            prev = dedup.get(key)
            if prev is None or rec.worst_latency < prev.worst_latency:
                dedup[key] = rec

        ranked = sorted(dedup.values(), key=lambda r: (r.worst_latency, str(r.output_dir)))
        by_kernel[kernel_id] = ranked

    return by_kernel


def _sanitize_testbench(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "/artifacts/dse/campaigns/" in line:
            continue
        lines.append(line)
    out = "\n".join(lines)
    if '#include "kernel_kernel.h"' not in out:
        out = '#include "kernel_kernel.h"\n' + out
    if '#include "kernel.h"' not in out:
        out = out.replace(
            '#include "kernel_kernel.h"',
            '#include "kernel_kernel.h"\n#include "kernel.h"',
            1,
        )
    if not out.endswith("\n"):
        out += "\n"
    return out


def _write_tcl_files(out_dir: Path) -> None:
    # DSE training csynth uses merged kernel_kernel.cpp only (modules are embedded).
    # kernel_kernel_modules.cpp is kept as corpus support but must not be added twice.
    kernel_files = [
        "add_files kernel_kernel.h",
        "add_files hls_baseline.cpp",
    ]
    csim = [
        "open_project hls_prj",
        "set_top kernel0",
        *kernel_files,
        "add_files -tb testbench.cpp",
        f'open_solution "solution1" -flow_target {DEFAULT_FLOW_TARGET}',
        f"set_part {{{DEFAULT_PART}}}",
        f"create_clock -period {DEFAULT_CLOCK_NS} -name default",
        "config_compile -name_max_length 50",
        "csim_design",
        "exit",
    ]
    synth = [
        "open_project hls_prj",
        "set_top kernel0",
        *kernel_files,
        f'open_solution "solution1" -flow_target {DEFAULT_FLOW_TARGET}',
        f"set_part {{{DEFAULT_PART}}}",
        f"create_clock -period {DEFAULT_CLOCK_NS} -name default",
        "config_compile -name_max_length 50",
        "csynth_design",
        "exit",
    ]
    full = [
        "open_project hls_prj",
        "set_top kernel0",
        *kernel_files,
        "add_files -tb testbench.cpp",
        f'open_solution "solution1" -flow_target {DEFAULT_FLOW_TARGET}',
        f"set_part {{{DEFAULT_PART}}}",
        f"create_clock -period {DEFAULT_CLOCK_NS} -name default",
        "config_compile -name_max_length 50",
        "csim_design",
        "csynth_design",
        "exit",
    ]
    (out_dir / "run_csim.tcl").write_text("\n".join(csim) + "\n", encoding="utf-8")
    (out_dir / "run_synth.tcl").write_text("\n".join(synth) + "\n", encoding="utf-8")
    (out_dir / "hls_script.tcl").write_text("\n".join(full) + "\n", encoding="utf-8")


def _write_manifest(out_dir: Path) -> None:
    lines = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_file():
            continue
        lines.append(f"{_sha256_file(path)}  {path.name}")
    (out_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_design(
    rec: DesignRecord,
    rank: int,
    out_bench_dir: Path,
    campaign_stamp: str,
) -> Path:
    dir_name = f"rank{rank}_latency_{rec.worst_latency}"
    out_dir = out_bench_dir / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    src = rec.src_dir
    kernel_cpp = (src / "kernel_kernel.cpp").read_text(encoding="utf-8")
    shutil.copy2(src / "kernel_kernel.cpp", out_dir / "gold_hls_source.cpp")
    shutil.copy2(src / "kernel_kernel.cpp", out_dir / "hls_baseline.cpp")
    shutil.copy2(src / "kernel_kernel.h", out_dir / "kernel_kernel.h")
    shutil.copy2(src / "kernel_kernel_modules.cpp", out_dir / "kernel_kernel_modules.cpp")
    shutil.copy2(_kernel_h_path(rec.kernel_id), out_dir / "kernel.h")

    tb = _sanitize_testbench((src / "kernel_host.cpp").read_text(encoding="utf-8"))
    (out_dir / "testbench.cpp").write_text(tb, encoding="utf-8")

    plain, strip_report = build_plain_from_src_dir(src)
    (out_dir / "plain.cpp").write_text(plain, encoding="utf-8")

    skip_phase_a = bool(strip_report.get("skip_phase_a_recommended", True))

    provenance = {
        "dse_campaign": campaign_stamp,
        "kernel_id": rec.kernel_id,
        "bench": rec.bench,
        "rank": rank,
        "csynth_best_cycles": rec.best_latency,
        "csynth_average_cycles": rec.average_latency,
        "csynth_worst_cycles": rec.worst_latency,
        "csynth_clock_ns": CSYNTH_CLOCK_NS,
        "csynth_part": DEFAULT_PART,
        "csynth_xml": str(rec.csynth_xml),
        "source_src_dir": str(rec.src_dir),
        "source_output_dir": str(rec.output_dir),
        "sa_sizes": rec.sa_sizes,
        "autosa_cmd": rec.autosa_cmd,
        "latency_source": "training_synth",
        "c2hls_target_clock_ns": DEFAULT_CLOCK_NS,
        "c2hls_target_part": DEFAULT_PART,
        "c2hls_target_device_platform": DEFAULT_DEVICE_PLATFORM,
        "latency_note": (
            "csynth_*_cycles are from DSE training csynth at csynth_clock_ns; "
            "c2hls flash/csynth uses c2hls_target_clock_ns on U280 (xcu280)."
        ),
        "validate_phase": False,
        "strip_report": strip_report,
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "benchmark": rec.bench,
        "source_repo": "AutoSA",
        "corpus": "autosa_dse",
        "algorithm_source_path": str((_kernel_h_path(rec.kernel_id).parent / "kernel.c").resolve()),
        "gold_hls_source_path": str((out_dir / "gold_hls_source.cpp").resolve()),
        "gold_hls_source_file": "gold_hls_source.cpp",
        "gold_hls_baseline_file": "hls_baseline.cpp",
        "kernel_file": "plain.cpp",
        "plain_c_file": "plain.cpp",
        "header_file": "kernel.h",
        "testbench_file": "testbench.cpp",
        "baseline_variant": f"{rec.bench}_dse_rank{rank}",
        "translated_hls_top": "kernel0",
        "hls_top": "kernel0",
        "kernel_top": "kernel0",
        # Modules are already embedded in hls_baseline.cpp / gold_hls_source.cpp.
        # Listing kernel_kernel_modules.cpp here makes c2hls stage it as a TB
        # extra and csim then fails with multiple-definition linker errors.
        "support_files": ["kernel_kernel.h"],
        "include_dirs": [],
        "supports_csim": True,
        "supports_cosim": False,
        "cosim_depths": {},
        "target_part": DEFAULT_PART,
        "target_clock_ns": DEFAULT_CLOCK_NS,
        "target_device_platform": DEFAULT_DEVICE_PLATFORM,
        "synth_timeout_s": 14400,
        "csim_timeout_s": 1800,
        "skip_phase_a": skip_phase_a,
        "variants": [
            {
                "name": f"{rec.bench}_dse_rank{rank}",
                "file": "hls_baseline.cpp",
                "source_path": str(rec.src_dir / "kernel_kernel.cpp"),
                "csynth_best_cycles": rec.best_latency,
                "csynth_average_cycles": rec.average_latency,
                "csynth_worst_cycles": rec.worst_latency,
            }
        ],
        "provenance": provenance,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    _write_tcl_files(out_dir)
    _write_manifest(out_dir)

    # Verify csynth latency still matches directory name
    check = _parse_top_level_csynth(rec.csynth_xml)
    if not check or check.worst != rec.worst_latency:
        raise RuntimeError(
            f"latency mismatch for {out_dir}: expected worst={rec.worst_latency}, got {check}"
        )

    return out_dir


def export_campaign(
    campaign_root: Path,
    out_root: Path,
    *,
    top_k: int = 3,
    kernel_ids: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    campaign_stamp = campaign_root.name
    discovered = discover_designs(campaign_root, kernel_ids)

    report: dict = {
        "campaign": campaign_stamp,
        "campaign_root": str(campaign_root),
        "out_root": str(out_root),
        "top_k": top_k,
        "dry_run": dry_run,
        "kernels": {},
        "packages": [],
        "errors": [],
    }

    if dry_run:
        for kernel_id, records in sorted(discovered.items()):
            top = records[:top_k]
            report["kernels"][kernel_id] = {
                "bench": _bench_name(kernel_id),
                "designs_found": len(records),
                "export_count": len(top),
                "top": [
                    {
                        "rank": i + 1,
                        "best_latency": r.best_latency,
                        "average_latency": r.average_latency,
                        "worst_latency": r.worst_latency,
                        "src_dir": str(r.src_dir),
                        "dir_name": f"rank{i + 1}_latency_{r.worst_latency}",
                    }
                    for i, r in enumerate(top)
                ],
            }
        return report

    staging = Path(tempfile.mkdtemp(prefix="autosa_export_", dir=out_root.parent))
    try:
        staging_out = staging / "AutoSA_sources"
        staging_out.mkdir(parents=True, exist_ok=True)

        for kernel_id, records in sorted(discovered.items()):
            top = records[:top_k]
            kernel_info = {
                "bench": _bench_name(kernel_id),
                "designs_found": len(records),
                "export_count": len(top),
                "exported": [],
                "skipped": [],
            }
            if not top:
                kernel_info["skipped"].append("no valid training csynth designs")
                report["kernels"][kernel_id] = kernel_info
                continue

            bench_dir = staging_out / _bench_name(kernel_id)
            bench_dir.mkdir(parents=True, exist_ok=True)

            for rank, rec in enumerate(top, start=1):
                try:
                    pkg = package_design(rec, rank, bench_dir, campaign_stamp)
                    kernel_info["exported"].append(
                        {
                            "rank": rank,
                            "best_latency": rec.best_latency,
                            "average_latency": rec.average_latency,
                            "worst_latency": rec.worst_latency,
                            "path": str(pkg),
                            "dir_name": pkg.name,
                        }
                    )
                    report["packages"].append(str(pkg))
                except Exception as exc:
                    msg = f"{kernel_id} rank{rank}: {exc}"
                    report["errors"].append(msg)
                    kernel_info["skipped"].append(msg)

            report["kernels"][kernel_id] = kernel_info

        report_path = staging_out / "export_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        if out_root.exists():
            backup = out_root.with_name(out_root.name + ".bak")
            if backup.exists():
                shutil.rmtree(backup)
            out_root.rename(backup)
        staging_out.rename(out_root)

        # Fix absolute paths captured during staging.
        for meta_path in out_root.rglob("metadata.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            pkg_dir = meta_path.parent
            meta["gold_hls_source_path"] = str((pkg_dir / "gold_hls_source.cpp").resolve())
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            _write_manifest(pkg_dir)

        for kernel_id, kernel_info in report.get("kernels", {}).items():
            bench = kernel_info.get("bench") or _bench_name(kernel_id)
            for entry in kernel_info.get("exported", []):
                entry["path"] = str(out_root / bench / entry["dir_name"])
        report["packages"] = [
            entry["path"]
            for info in report.get("kernels", {}).values()
            for entry in info.get("exported", [])
        ]
        (out_root / "export_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return report


def _write_csynth_tcl(
    pkg_dir: Path,
    *,
    clock_ns: float,
    tcl_name: str,
    project_name: str = "hls_prj",
) -> Path:
    tcl_path = pkg_dir / tcl_name
    tcl_path.write_text(
        "\n".join(
            [
                f"open_project {project_name}",
                "set_top kernel0",
                "add_files kernel_kernel.h",
                "add_files hls_baseline.cpp",
                f'open_solution "solution1" -flow_target {DEFAULT_FLOW_TARGET}',
                f"set_part {{{DEFAULT_PART}}}",
                f"create_clock -period {clock_ns} -name default",
                "config_compile -name_max_length 50",
                "csynth_design",
                "exit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return tcl_path


def _latency_fields(prefix: str, latency: CsynthLatency) -> dict[str, int]:
    return {
        f"{prefix}_best_cycles": latency.best,
        f"{prefix}_average_cycles": latency.average,
        f"{prefix}_worst_cycles": latency.worst,
    }


def run_package_csynth(
    pkg_dir: Path,
    *,
    clock_ns: float,
    tcl_name: str = "run_synth_resynth.tcl",
    project_name: str = "hls_prj",
) -> CsynthLatency:
    """Run vitis_hls csynth in pkg_dir and return top-level latency cycles."""
    hls_prj = pkg_dir / project_name
    if hls_prj.exists():
        shutil.rmtree(hls_prj)

    tcl = _write_csynth_tcl(
        pkg_dir,
        clock_ns=clock_ns,
        tcl_name=tcl_name,
        project_name=project_name,
    )
    subprocess.run(
        ["vitis_hls", "-f", str(tcl.name)],
        cwd=pkg_dir,
        check=True,
    )
    csynth_xml = pkg_dir / project_name / "solution1/syn/report/csynth.xml"
    parsed = _parse_top_level_csynth(csynth_xml)
    if not parsed:
        raise RuntimeError(f"could not parse {csynth_xml}")
    return parsed


def _update_json_file(path: Path, updater) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    updater(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def apply_c2hls_latency_to_package(
    pkg_dir: Path,
    latency: CsynthLatency,
    *,
    clock_ns: float = DEFAULT_CLOCK_NS,
) -> None:
    """Write c2hls_*_cycles into provenance.json and metadata.json."""
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

    allowed_benches = (
        {_bench_name(k) for k in kernel_ids} if kernel_ids else None
    )
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


def resynth_c2hls_clock(
    out_root: Path,
    *,
    kernel_ids: list[str] | None = None,
    packages: list[Path] | None = None,
    clock_ns: float = DEFAULT_CLOCK_NS,
    dry_run: bool = False,
) -> dict:
    """Re-run csynth at c2hls clock on exported packages and record c2hls_*_cycles."""
    pkg_dirs = iter_export_packages(out_root, kernel_ids=kernel_ids, packages=packages)
    report: dict = {
        "out_root": str(out_root),
        "clock_ns": clock_ns,
        "part": DEFAULT_PART,
        "dry_run": dry_run,
        "packages": [],
        "errors": [],
    }

    for pkg_dir in pkg_dirs:
        entry = {"package": str(pkg_dir), "dir_name": pkg_dir.name}
        if dry_run:
            entry["status"] = "pending"
            report["packages"].append(entry)
            continue
        try:
            latency = run_package_csynth(pkg_dir, clock_ns=clock_ns)
            apply_c2hls_latency_to_package(pkg_dir, latency, clock_ns=clock_ns)
            entry.update(_latency_fields("c2hls", latency))
            entry["status"] = "ok"
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            report["errors"].append(f"{pkg_dir}: {exc}")
        report["packages"].append(entry)

    report_path = out_root / "c2hls_resynth_report.json"
    if not dry_run or pkg_dirs:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def verify_package(pkg_dir: Path, *, clock_ns: float | None = None) -> dict:
    """Run csynth and compare latency against provenance."""
    prov_path = pkg_dir / "provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    expected = {
        "best": int(prov["csynth_best_cycles"]),
        "average": int(prov["csynth_average_cycles"]),
        "worst": int(prov["csynth_worst_cycles"]),
    }
    clock = clock_ns if clock_ns is not None else float(
        prov.get("csynth_clock_ns", prov.get("training_clock_ns", CSYNTH_CLOCK_NS))
    )

    parsed = run_package_csynth(
        pkg_dir,
        clock_ns=clock,
        tcl_name="run_synth_verify.tcl",
    )
    actual = {
        "best": parsed.best,
        "average": parsed.average,
        "worst": parsed.worst,
    }
    return {
        "package": str(pkg_dir),
        "expected_latency": expected,
        "actual_latency": actual,
        "verify_clock_ns": clock,
        "match": actual == expected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--kernels", type=str, default="", help="comma-separated kernel ids")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify",
        type=str,
        default="",
        help="comma-separated package dirs to csynth-verify against DSE training latencies",
    )
    parser.add_argument(
        "--resynth-c2hls-clock",
        action="store_true",
        help=(
            "re-run csynth at c2hls target clock (3.33 ns, U280) on exported packages "
            "and write c2hls_best/average/worst_cycles into provenance + metadata"
        ),
    )
    parser.add_argument(
        "--packages",
        type=str,
        default="",
        help="comma-separated package dirs (for --resynth-c2hls-clock)",
    )
    args = parser.parse_args()

    kernel_ids = [k.strip() for k in args.kernels.split(",") if k.strip()] or None
    package_paths = [Path(p.strip()) for p in args.packages.split(",") if p.strip()] or None

    if args.verify:
        results = []
        for pkg in [p.strip() for p in args.verify.split(",") if p.strip()]:
            results.append(verify_package(Path(pkg)))
        print(json.dumps(results, indent=2))
        return 0 if all(r["match"] for r in results) else 1

    if args.resynth_c2hls_clock:
        report = resynth_c2hls_clock(
            args.out.resolve(),
            kernel_ids=kernel_ids,
            packages=package_paths,
            dry_run=args.dry_run,
        )
        print(json.dumps(report, indent=2))
        if not args.dry_run:
            print(f"\nWrote resynth report: {args.out / 'c2hls_resynth_report.json'}")
        return 1 if report.get("errors") else 0

    report = export_campaign(
        args.campaign.resolve(),
        args.out.resolve(),
        top_k=args.top_k,
        kernel_ids=kernel_ids,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))
    if not args.dry_run:
        print(f"\nWrote export report: {args.out / 'export_report.json'}")
    return 1 if report.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())

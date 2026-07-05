#!/usr/bin/env python3
"""Export full Vitis HLS csynth report trees into a kernel bundle.

For each benchmark, creates two directories alongside the kernel sources:

  <kernel_bundle>/hlsfactory_<bench>/flash_csynth/
  <kernel_bundle>/hlsfactory_<bench>/dataflow_csynth/

Each contains all files from the corresponding work_dir ``syn/report`` tree,
plus ``vitis_hls.log`` (from ``logs/hls_run_tcl.log``) and ``sol1.log`` when present.

Per-module ``*_Pipeline_*_csynth.rpt`` files include loop-level Initiation Interval
(achieved vs target) that the top-level summary table alone does not spell out.

Example::

    python3 scripts/pc2/export_post_flash_dataflow_csynth_bundle.py \\
        --matrix-root artifacts/pc2/flash_all_new_skills_avoids_global_20260623_024548
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hls_feedback import _syn_report_dir
from post_flash_mem_parallel import discover_matrix_cells


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _work_dir_from_report(report: dict[str, Any]) -> Optional[str]:
    wd = report.get("work_dir")
    if isinstance(wd, str) and wd.strip():
        return wd.strip()
    return None


def _flash_synth_report(flash_bundle_root: Path, bench: str) -> tuple[Optional[Path], dict[str, Any]]:
    report_path = flash_bundle_root / bench / "selected" / "synth_report.json"
    if not report_path.is_file():
        return None, {}
    return report_path, _load_json(report_path)


def _dataflow_synth_report(cell_dir: Path, bench: str) -> tuple[Optional[Path], dict[str, Any]]:
    for name in (
        f"{bench}_dataflow_report.json",
        f"{bench}_dataflow_result.json",
    ):
        path = cell_dir / name
        if not path.is_file():
            continue
        data = _load_json(path)
        if name.endswith("_result.json"):
            sr = data.get("synth_report")
            if isinstance(sr, dict):
                return path, sr
        return path, data
    return None, {}


def _synth_work_dir_log_paths(work_dir: str) -> list[tuple[Path, str]]:
    """Return (src, dest_name) pairs for synth logs under a work_dir."""
    root = Path(work_dir)
    out: list[tuple[Path, str]] = []
    tcl_log = root / "logs" / "hls_run_tcl.log"
    if tcl_log.is_file():
        # c2hls records vitis-run output here; same content hls_feedback parses.
        out.append((tcl_log, "vitis_hls.log"))
    sol_log = root / "hls_proj" / "sol1" / "sol1.log"
    if sol_log.is_file():
        out.append((sol_log, "sol1.log"))
    return out


def _copy_synth_artifacts(
    *,
    work_dir: str,
    dest_dir: Path,
    force: bool,
) -> dict[str, Any]:
    report_src = _syn_report_dir(work_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if report_src is None:
        return {
            "work_dir": work_dir,
            "report_dir": None,
            "copied": [],
            "logs_copied": [],
            "missing": True,
            "error": "syn/report directory not found",
        }

    copied: list[str] = []
    src_path = Path(report_src)
    for item in sorted(src_path.iterdir()):
        if not item.is_file():
            continue
        dst = dest_dir / item.name
        if dst.exists() and not force:
            copied.append(item.name)
            continue
        shutil.copy2(item, dst)
        copied.append(item.name)

    logs_copied: list[str] = []
    for src, dest_name in _synth_work_dir_log_paths(work_dir):
        dst = dest_dir / dest_name
        if dst.exists() and not force:
            logs_copied.append(dest_name)
            continue
        shutil.copy2(src, dst)
        logs_copied.append(dest_name)

    return {
        "work_dir": work_dir,
        "report_dir": report_src,
        "copied": copied,
        "logs_copied": logs_copied,
        "missing": False,
        "error": "",
    }


def _remove_legacy_top_level_csynth_dirs(kernel_bundle: Path) -> None:
    """Drop pre-per-bench layout: <bundle>/flash_csynth and dataflow_csynth."""
    for name in ("flash_csynth", "dataflow_csynth"):
        legacy = kernel_bundle / name
        if legacy.is_dir():
            shutil.rmtree(legacy)


def export_csynth_bundle(
    *,
    matrix_root: Path,
    flash_bundle_root: Path,
    kernel_bundle: Path,
    force: bool = False,
    benches: Optional[list[str]] = None,
    remove_legacy_top_level: bool = True,
) -> dict[str, Any]:
    kernel_bundle.mkdir(parents=True, exist_ok=True)
    if remove_legacy_top_level:
        _remove_legacy_top_level_csynth_dirs(kernel_bundle)

    want = set(benches) if benches else None
    cells = discover_matrix_cells(matrix_root)

    flash_rows: list[dict[str, Any]] = []
    dataflow_rows: list[dict[str, Any]] = []

    for cell in cells:
        bench = str(cell.get("bench") or "")
        if not bench.startswith("hlsfactory_"):
            continue
        if want is not None and bench not in want:
            continue

        short = bench.removeprefix("hlsfactory_")
        bench_dir = kernel_bundle / bench
        bench_dir.mkdir(parents=True, exist_ok=True)

        # --- flash-selected ---
        flash_report_path, flash_report = _flash_synth_report(flash_bundle_root, bench)
        flash_wd = _work_dir_from_report(flash_report)
        flash_dest = bench_dir / "flash_csynth"
        if flash_wd:
            flash_copy = _copy_synth_artifacts(
                work_dir=flash_wd,
                dest_dir=flash_dest,
                force=force,
            )
            if flash_report_path and flash_report_path.is_file():
                shutil.copy2(
                    flash_report_path,
                    flash_dest / "synth_report.json",
                )
                flash_copy.setdefault("parsed_report", "synth_report.json")
            flash_copy["dest_dir"] = str(flash_dest.resolve())
            flash_rows.append({
                "bench": bench,
                "short": short,
                **flash_copy,
            })
        else:
            flash_rows.append({
                "bench": bench,
                "short": short,
                "dest_dir": str(flash_dest.resolve()),
                "work_dir": None,
                "report_dir": None,
                "copied": [],
                "missing": True,
                "error": "no work_dir in flash selected synth_report.json",
            })

        # --- dataflow ---
        cell_dir = Path(cell["cell_dir"])
        df_report_path, df_report = _dataflow_synth_report(cell_dir, bench)
        df_wd = _work_dir_from_report(df_report)
        df_dest = bench_dir / "dataflow_csynth"
        if df_wd:
            df_copy = _copy_synth_artifacts(
                work_dir=df_wd,
                dest_dir=df_dest,
                force=force,
            )
            if df_report_path and df_report_path.is_file():
                parsed_name = (
                    "dataflow_report.json"
                    if df_report_path.name.endswith("_dataflow_report.json")
                    else "dataflow_result.json"
                )
                shutil.copy2(df_report_path, df_dest / parsed_name)
                df_copy["parsed_report"] = parsed_name
            df_copy["dest_dir"] = str(df_dest.resolve())
            dataflow_rows.append({
                "bench": bench,
                "short": short,
                **df_copy,
            })
        else:
            dataflow_rows.append({
                "bench": bench,
                "short": short,
                "dest_dir": str(df_dest.resolve()),
                "work_dir": None,
                "report_dir": None,
                "copied": [],
                "missing": True,
                "error": "no work_dir in dataflow synth report",
            })

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_root": str(matrix_root.resolve()),
        "flash_bundle_root": str(flash_bundle_root.resolve()),
        "kernel_bundle": str(kernel_bundle.resolve()),
        "layout": "<kernel_bundle>/hlsfactory_<bench>/{flash_csynth,dataflow_csynth}/",
        "flash": {
            "exported": sum(1 for r in flash_rows if not r.get("missing")),
            "missing": sum(1 for r in flash_rows if r.get("missing")),
            "benches": flash_rows,
        },
        "dataflow": {
            "exported": sum(1 for r in dataflow_rows if not r.get("missing")),
            "missing": sum(1 for r in dataflow_rows if r.get("missing")),
            "benches": dataflow_rows,
        },
    }
    _write_json(kernel_bundle / "csynth_bundle_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-root",
        type=Path,
        required=True,
        help="Flash matrix root (contains hlsfactory_*/devstral2__*)",
    )
    parser.add_argument(
        "--flash-bundle-root",
        type=Path,
        default=None,
        help="flash_selected_bundle/<matrix-name> (default: sibling under artifacts/pc2/flash_selected_bundle)",
    )
    parser.add_argument(
        "--kernel-bundle",
        type=Path,
        default=None,
        help="Kernel bundle dir (default: <matrix-root>/post_flash_dataflow_kernel_bundle)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing report files")
    parser.add_argument("--benches", nargs="*", help="Limit to hlsfactory_* bench names")
    args = parser.parse_args()

    matrix_root = args.matrix_root.resolve()
    flash_bundle_root = (
        args.flash_bundle_root.resolve()
        if args.flash_bundle_root
        else (REPO / "artifacts/pc2/flash_selected_bundle" / matrix_root.name).resolve()
    )
    kernel_bundle = (
        args.kernel_bundle.resolve()
        if args.kernel_bundle
        else (matrix_root / "post_flash_dataflow_kernel_bundle").resolve()
    )

    manifest = export_csynth_bundle(
        matrix_root=matrix_root,
        flash_bundle_root=flash_bundle_root,
        kernel_bundle=kernel_bundle,
        force=args.force,
        benches=args.benches,
    )
    print(
        f"flash_csynth: {manifest['flash']['exported']} exported, "
        f"{manifest['flash']['missing']} missing"
    )
    print(
        f"dataflow_csynth: {manifest['dataflow']['exported']} exported, "
        f"{manifest['dataflow']['missing']} missing"
    )
    print(f"manifest: {kernel_bundle / 'csynth_bundle_manifest.json'}")
    missing_flash = [r for r in manifest["flash"]["benches"] if r.get("missing")]
    missing_df = [r for r in manifest["dataflow"]["benches"] if r.get("missing")]
    unexpected = [
        r["bench"]
        for r in missing_flash + missing_df
        if r.get("bench") != "hlsfactory_doitgen"
    ]
    return 0 if not unexpected else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export top-5 PC2 flash matrix + full-size cosim results to schema-1.0 JSONL.

Joins flash artifact csim/csynth (*_multistep_results.json) with standalone
cosim cells (cosim_result.json) from a flash_cosim manifest run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "misc"))
PC2 = REPO / "artifacts" / "pc2"
BENCHMARKS = REPO / "benchmarks"

from export_schema_jsonl import (  # noqa: E402
    SCHEMA_VERSION,
    TARGET_COSIM,
    TARGET_CSIM,
    TARGET_CSYNTH,
    _build_hls_synth_payload,
    _build_implementation,
    _build_problem,
    _build_run,
    validate_jsonl,
)
from export_pc2_flash_matrix_jsonl import (  # noqa: E402
    SUITE,
    DEFAULT_CLOCK_NS,
    DEFAULT_PART,
    _bench_meta,
    _status_from_block,
    _top_model_payload,
)

DEFAULT_COSIM_RUN = PC2 / "flash_cosim" / "20260626_022917_top5_full_indiv"
DEFAULT_OUTPUT = REPO / "misc" / "devstral2_flash_pc2_top5_full_cosim_schema.jsonl"


def _cosim_status(result: dict[str, Any]) -> str:
    if result.get("status") == "ok" and result.get("passed"):
        return "pass"
    err = (result.get("error") or "").lower()
    if "timed out" in err or "timeout" in err:
        return "timeout"
    return "fail"


def _load_flash_results(cell_dir: Path, bench: str) -> dict[str, Any] | None:
    path = cell_dir / f"{bench}_multistep_results.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _variant_label(provenance: dict[str, Any], artifact_basename: str) -> str:
    variant = (provenance.get("variant") or "").strip()
    if variant:
        return variant
    name = artifact_basename.removeprefix("flash_")
    for token in (
        "all_new_skills_avoids_global",
        "all_new_skills_no_avoids_global",
        "all_skills_avoids_global",
        "all_skills_no_avoids_global",
        "noskills",
    ):
        if token in name:
            return token
    return name


def export_cosim_manifest(cosim_run_root: Path) -> list[dict[str, Any]]:
    manifest_path = cosim_run_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    records: list[dict[str, Any]] = []

    for cell in manifest.get("cells", []):
        cell_id = cell.get("cell_id") or ""
        bench = cell.get("bench") or ""
        cell_dir = Path(cell.get("cell_dir") or "")
        artifact_dir = Path(cell.get("artifact_dir") or "")
        if not cell_id or not bench or not cell_dir.is_dir():
            continue

        cosim_path = cosim_run_root / "cells" / cell_id / "cosim_result.json"
        if not cosim_path.is_file():
            continue
        cosim = json.loads(cosim_path.read_text())

        flash = _load_flash_results(cell_dir, bench)
        meta = _bench_meta(bench)
        top = meta.get("hls_top") or meta.get("kernel_top") or "workload"

        run_meta: dict[str, Any] = {}
        if flash:
            run_meta = flash.get("run") or {}

        part = run_meta.get("part") or DEFAULT_PART
        clock_ns = float(run_meta.get("clock_ns") or DEFAULT_CLOCK_NS)
        model = cell.get("model") or run_meta.get("model") or ""
        artifact_basename = cell.get("artifact_basename") or artifact_dir.name
        origin_version = f"pc2_{artifact_basename}"

        origin_meta = {
            "model": model,
            "model_tag": "devstral2",
            "mode": cell.get("mode") or "flash",
            "skills": "off" if "noskills" in artifact_basename else "on",
            "phase": (flash or {}).get("phase") or "flash",
            "artifact_dir": str(artifact_dir),
            "artifact_setup": cell.get("setup_tag") or "",
            "artifact_stamp": cell.get("artifact_stamp") or "",
            "cell_dir": str(cell_dir),
            "matrix_status": cell.get("source_matrix_status") or "ok",
            "variant": _variant_label(cell, artifact_basename),
            "cosim_run_root": str(cosim_run_root),
            "cosim_cell_id": cell_id,
            "cosim_size_mode": cosim.get("cosim_size_mode"),
        }

        impl_common = dict(
            meta=meta,
            variant_name="final",
            variant_index=0,
            origin_override="c2hls_orchestrator",
            origin_version=origin_version,
            origin_meta=origin_meta,
        )

        synth = (flash or {}).get("final_report") or {}
        if synth:
            synth_status = "pass" if synth.get("latency_cycles") is not None else "fail"
            records.append({
                "schema_version": SCHEMA_VERSION,
                "report_type": "hls_synth",
                "run": _build_run(TARGET_CSYNTH, part, cosim.get("runtime_seconds"), run_meta),
                "problem": {**_build_problem(meta), "suite": SUITE},
                "implementation": _build_implementation(**impl_common),
                "hls_synth": _top_model_payload(synth, synth_status, top, part, clock_ns),
            })

        csim = None
        if flash:
            steps = flash.get("steps") or []
            if steps:
                csim = steps[-1].get("csim")
            if not csim:
                csim = flash.get("csim") or flash.get("baseline_csim")

        csim_status = _status_from_block(csim)
        if csim_status:
            records.append({
                "schema_version": SCHEMA_VERSION,
                "report_type": "sw_run",
                "run": _build_run(TARGET_CSIM, part, cosim.get("runtime_seconds"), run_meta),
                "problem": {**_build_problem(meta), "suite": SUITE},
                "implementation": _build_implementation(**impl_common),
                "sw_run": {
                    "status": csim_status,
                    "error": (csim.get("error") or "")[:300] if csim_status != "pass" else None,
                },
            })

        cosim_status = _cosim_status(cosim)
        records.append({
            "schema_version": SCHEMA_VERSION,
            "report_type": "rtl_sim",
            "run": _build_run(TARGET_COSIM, part, cosim.get("runtime_seconds"), run_meta),
            "problem": {**_build_problem(meta), "suite": SUITE},
            "implementation": _build_implementation(**impl_common),
            "rtl_sim": {
                "status": cosim_status,
                "kernel_runtime_cycles": cosim.get("kernel_runtime_cycles"),
                "kernel_runtime_us": cosim.get("kernel_runtime_us"),
                "kernel_clock_freq_mhz": cosim.get("kernel_clock_freq_mhz"),
                "error": (cosim.get("error") or "")[:300] if cosim_status != "pass" else None,
            },
        })

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cosim-run-root",
        type=Path,
        default=DEFAULT_COSIM_RUN,
        help="flash_cosim run root containing manifest.json + cells/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    records = export_cosim_manifest(args.cosim_run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    validation = validate_jsonl(args.output, verbose=True)
    by_type: dict[str, int] = {}
    rtl_with_cycles = 0
    for rec in records:
        rt = rec.get("report_type", "")
        by_type[rt] = by_type.get(rt, 0) + 1
        if rt == "rtl_sim" and rec.get("rtl_sim", {}).get("kernel_runtime_cycles") is not None:
            rtl_with_cycles += 1

    summary = {
        "output": str(args.output),
        "cosim_run_root": str(args.cosim_run_root),
        "records": len(records),
        "by_report_type": by_type,
        "rtl_sim_with_kernel_runtime_cycles": rtl_with_cycles,
        "validation": validation,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if validation.get("invalid", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

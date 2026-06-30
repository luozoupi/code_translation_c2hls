#!/usr/bin/env python3
"""Export fixed benchmarks_cosim baseline + 5-variant LLM flash + cosim to schema JSONL.

Includes:
  - Fixed baseline gold (csynth/csim/cosim) from full_cosim baseline JSONL
  - Per-variant flash selected: csynth + csim from multistep results
  - Per-variant phase_b translator: csynth from phase_b_report.json
  - Full-size cosim rtl_sim for selected and phase_b kernels
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
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

PC2 = REPO / "artifacts" / "pc2"
BENCHMARKS_COSIM = REPO / "benchmarks_cosim"
BENCHMARKS = REPO / "benchmarks"

from export_schema_jsonl import (  # noqa: E402
    SCHEMA_VERSION,
    TARGET_COSIM,
    TARGET_CSIM,
    TARGET_CSYNTH,
    _build_implementation,
    _build_problem,
    _build_run,
    validate_jsonl,
)
from export_pc2_flash_matrix_jsonl import (  # noqa: E402
    DEFAULT_CLOCK_NS,
    DEFAULT_PART,
    SUITE,
    _status_from_block,
    _top_model_payload,
)
from flash_cosim_lib import classify_cosim_outcome  # noqa: E402

DEFAULT_BASELINE_JSONL = REPO / "misc" / "hlsfactory_baseline_u280_20260616_benchmarks_full_cosim.jsonl"
DEFAULT_OUTPUT = REPO / "misc" / "hlsfactory_fixed_cosim_flash_u280_20260626.jsonl"
DEFAULT_FLASH_STAMP = "20260626_fixed_cosim_flash"
DEFAULT_SELECTED_COSIM = PC2 / "flash_cosim" / "fixed_cosim_flash_20260626"
DEFAULT_PHASE_B_COSIM = PC2 / "flash_cosim" / "fixed_cosim_flash_phase_b_20260626"

VARIANT_ORDER = ("aav_n", "nav_n", "aav_o", "nav_o", "noskills")
VARIANT_LABELS = {
    "aav_n": "All+avoids (new, 90 skills)",
    "nav_n": "No avoids (new, 73 skills)",
    "aav_o": "All+avoids (old, 55 skills)",
    "nav_o": "No avoids (old, 55 skills)",
    "noskills": "Noskills",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def _bench_meta(bench: str) -> dict[str, Any]:
    bench_dir = BENCHMARKS_COSIM / bench
    if not bench_dir.is_dir():
        bench_dir = BENCHMARKS / bench
    if not bench_dir.is_dir():
        alt = bench.replace("-", "_")
        for root in (BENCHMARKS_COSIM, BENCHMARKS):
            for p in root.glob("hlsfactory_*"):
                if p.name.removeprefix("hlsfactory_").replace("-", "_") == alt.removeprefix("hlsfactory_"):
                    bench_dir = p
                    break
    meta = json.loads((bench_dir / "metadata.json").read_text())
    meta = dict(meta)
    meta["group_path"] = [bench_dir.name.removeprefix("hlsfactory_")]
    meta["corpus"] = bench_dir.parent.name
    return meta


def _load_flash_results(cell_dir: Path, bench: str) -> dict[str, Any] | None:
    path = cell_dir / f"{bench}_multistep_results.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _phase_b_csim_from_flash(flash: dict[str, Any] | None) -> dict[str, Any] | None:
    """Flash flow records Phase B csim on generated_step_history step 'baseline'."""
    if not flash:
        return None
    for key in ("generated_step_history", "optimization_history", "steps"):
        for step in flash.get(key) or []:
            if not isinstance(step, dict):
                continue
            if step.get("step_name") != "baseline":
                continue
            csim = step.get("csim")
            if isinstance(csim, dict):
                return csim
    return None


def _flash_csim_from_flash(flash: dict[str, Any] | None) -> dict[str, Any] | None:
    """Selected / flash-step csim (step_name flash, or last steps[] entry)."""
    if not flash:
        return None
    for key in ("generated_step_history", "optimization_history"):
        for step in flash.get(key) or []:
            if isinstance(step, dict) and step.get("step_name") == "flash":
                csim = step.get("csim")
                if isinstance(csim, dict):
                    return csim
    steps = flash.get("steps") or []
    if steps:
        csim = steps[-1].get("csim")
        if isinstance(csim, dict):
            return csim
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _flash_origin_meta(
    *,
    row: dict[str, Any],
    artifact_dir: Path,
    cell_dir: Path,
    kernel_role: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant = row.get("variant") or ""
    meta = {
        "model": row.get("model") or "",
        "model_tag": "devstral2",
        "mode": row.get("mode") or "flash",
        "corpus": row.get("corpus") or "benchmarks_cosim",
        "matrix_family": row.get("matrix_family") or "",
        "variant": variant,
        "variant_label": VARIANT_LABELS.get(variant, variant),
        "skills_json": row.get("skills_json"),
        "record_flow": row.get("record_flow"),
        "kernel_role": kernel_role,
        "artifact_dir": str(artifact_dir),
        "cell_dir": str(cell_dir),
        "matrix_status": row.get("status"),
        "wallclock_s": row.get("wallclock_s"),
    }
    if extra:
        meta.update(extra)
    return meta


def _append_synth_csim(
    records: list[dict[str, Any]],
    *,
    meta: dict[str, Any],
    synth: dict[str, Any],
    csim: dict[str, Any] | None,
    run_meta: dict[str, Any],
    part: str,
    clock_ns: float,
    top: str,
    origin_version: str,
    origin_meta: dict[str, Any],
    variant_name: str,
    variant_index: int,
    runtime_seconds: Any,
) -> None:
    impl_common = dict(
        meta=meta,
        variant_name=variant_name,
        variant_index=variant_index,
        origin_override="c2hls_orchestrator",
        origin_version=origin_version,
        origin_meta=origin_meta,
    )
    if synth:
        synth_status = "pass" if synth.get("latency_cycles") is not None else "fail"
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "report_type": "hls_synth",
                "run": _build_run(TARGET_CSYNTH, part, runtime_seconds, run_meta),
                "problem": {**_build_problem(meta), "suite": SUITE},
                "implementation": _build_implementation(**impl_common),
                "hls_synth": _top_model_payload(synth, synth_status, top, part, clock_ns),
            }
        )
    csim_status = _status_from_block(csim)
    if csim_status:
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "report_type": "sw_run",
                "run": _build_run(TARGET_CSIM, part, runtime_seconds, run_meta),
                "problem": {**_build_problem(meta), "suite": SUITE},
                "implementation": _build_implementation(**impl_common),
                "sw_run": {
                    "status": csim_status,
                    "error": (csim.get("error") or "")[:300] if csim_status != "pass" else None,
                },
            }
        )


def _append_rtl_sim(
    records: list[dict[str, Any]],
    *,
    meta: dict[str, Any],
    cosim: dict[str, Any],
    run_meta: dict[str, Any],
    part: str,
    origin_version: str,
    origin_meta: dict[str, Any],
    variant_name: str,
    variant_index: int,
) -> None:
    cosim_status = classify_cosim_outcome(cosim)
    impl_common = dict(
        meta=meta,
        variant_name=variant_name,
        variant_index=variant_index,
        origin_override="c2hls_orchestrator",
        origin_version=origin_version,
        origin_meta=origin_meta,
    )
    records.append(
        {
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
                "error": (cosim.get("error") or "")[:300] if cosim_status not in ("pass",) else None,
            },
        }
    )


def export_flash_variant(artifact_dir: Path) -> list[dict[str, Any]]:
    matrix_path = artifact_dir / "matrix.json"
    rows = json.loads(matrix_path.read_text())
    origin_version = f"pc2_{artifact_dir.name}"
    records: list[dict[str, Any]] = []

    for row in rows:
        bench = row.get("bench") or ""
        cell_dir = Path(row.get("cell_dir") or "")
        if not bench or not cell_dir.is_dir() or row.get("status") != "ok":
            continue

        flash = _load_flash_results(cell_dir, bench)
        meta = _bench_meta(bench)
        top = meta.get("translated_hls_top") or meta.get("hls_top") or meta.get("kernel_top") or "workload"
        run_meta = (flash or {}).get("run") or {}
        part = run_meta.get("part") or DEFAULT_PART
        clock_ns = float(run_meta.get("clock_ns") or DEFAULT_CLOCK_NS)

        synth = (flash or {}).get("final_report") or {}
        if not synth and isinstance(row.get("summary"), dict):
            synth = row["summary"].get("synth_report") or {}

        csim = _flash_csim_from_flash(flash)

        selected_meta = _flash_origin_meta(
            row=row,
            artifact_dir=artifact_dir,
            cell_dir=cell_dir,
            kernel_role="selected",
            extra={"kernel_file": f"{bench}_selected.cpp"},
        )
        _append_synth_csim(
            records,
            meta=meta,
            synth=synth,
            csim=csim,
            run_meta=run_meta,
            part=part,
            clock_ns=clock_ns,
            top=top,
            origin_version=origin_version,
            origin_meta=selected_meta,
            variant_name="selected",
            variant_index=0,
            runtime_seconds=row.get("wallclock_s"),
        )

        phase_b_report = _load_json(cell_dir / f"{bench}_phase_b_report.json")
        phase_b_csim = _phase_b_csim_from_flash(flash)
        if phase_b_report:
            phase_b_meta = _flash_origin_meta(
                row=row,
                artifact_dir=artifact_dir,
                cell_dir=cell_dir,
                kernel_role="phase_b",
                extra={
                    "kernel_file": f"{bench}_phase_b.cpp",
                    "csim_step_name": "baseline",
                },
            )
            _append_synth_csim(
                records,
                meta=meta,
                synth=phase_b_report,
                csim=phase_b_csim,
                run_meta=run_meta,
                part=part,
                clock_ns=clock_ns,
                top=top,
                origin_version=origin_version,
                origin_meta=phase_b_meta,
                variant_name="phase_b",
                variant_index=1,
                runtime_seconds=row.get("wallclock_s"),
            )

    return records


def export_cosim_manifest(
    cosim_run_root: Path,
    *,
    kernel_role: str,
    variant_index: int,
) -> list[dict[str, Any]]:
    manifest = json.loads((cosim_run_root / "manifest.json").read_text())
    records: list[dict[str, Any]] = []

    for cell in manifest.get("cells", []):
        cell_id = cell.get("cell_id") or ""
        bench = cell.get("bench") or ""
        cell_dir = Path(cell.get("cell_dir") or "")
        artifact_dir = Path(cell.get("artifact_dir") or "")
        if not cell_id or not bench or not cell_dir.is_dir():
            continue

        cosim_path = cosim_run_root / "cells" / cell_id / "cosim_result.json"
        if cosim_path.is_file():
            cosim = json.loads(cosim_path.read_text())
        else:
            cosim = {
                "status": "missing",
                "passed": False,
                "error": "cosim_result.json not produced",
                "runtime_seconds": None,
                "kernel_runtime_cycles": None,
                "kernel_runtime_us": None,
                "kernel_clock_freq_mhz": None,
            }

        flash = _load_flash_results(cell_dir, bench)
        meta = _bench_meta(bench)
        run_meta = (flash or {}).get("run") or {}
        part = run_meta.get("part") or DEFAULT_PART
        artifact_basename = cell.get("artifact_basename") or artifact_dir.name
        origin_version = f"pc2_{artifact_basename}"

        row_stub = {
            "bench": bench,
            "model": cell.get("model") or "",
            "mode": cell.get("mode") or "flash",
            "corpus": "benchmarks_cosim",
            "matrix_family": cell.get("matrix_family") or "",
            "variant": cell.get("variant") or "",
            "skills_json": cell.get("skills_json"),
            "record_flow": True,
            "status": cell.get("source_matrix_status") or "ok",
        }
        origin_meta = _flash_origin_meta(
            row=row_stub,
            artifact_dir=artifact_dir if artifact_dir.is_dir() else artifact_dir,
            cell_dir=cell_dir,
            kernel_role=kernel_role,
            extra={
                "kernel_file": Path(cell.get("final_cpp") or "").name,
                "cosim_run_root": str(cosim_run_root),
                "cosim_cell_id": cell_id,
                "cosim_size_mode": cosim.get("cosim_size_mode"),
            },
        )
        _append_rtl_sim(
            records,
            meta=meta,
            cosim=cosim,
            run_meta=run_meta,
            part=part,
            origin_version=origin_version,
            origin_meta=origin_meta,
            variant_name=kernel_role,
            variant_index=variant_index,
        )

    return records


def export_fixed_cosim_flash_jsonl(
    *,
    baseline_jsonl: Path,
    flash_stamp: str,
    selected_cosim_root: Path,
    phase_b_cosim_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = _read_jsonl(baseline_jsonl)

    flash_dirs = [
        PC2 / f"flash_fixed_cosim_{variant}_{flash_stamp}"
        for variant in VARIANT_ORDER
    ]
    per_variant: dict[str, int] = {}
    for artifact_dir in flash_dirs:
        if not artifact_dir.is_dir():
            raise FileNotFoundError(f"missing flash artifact dir: {artifact_dir}")
        batch = export_flash_variant(artifact_dir)
        per_variant[artifact_dir.name] = len(batch)
        records.extend(batch)

    selected_cosim = export_cosim_manifest(
        selected_cosim_root,
        kernel_role="selected",
        variant_index=0,
    )
    phase_b_cosim = export_cosim_manifest(
        phase_b_cosim_root,
        kernel_role="phase_b",
        variant_index=1,
    )
    records.extend(selected_cosim)
    records.extend(phase_b_cosim)

    meta = {
        "baseline_jsonl": str(baseline_jsonl),
        "flash_stamp": flash_stamp,
        "flash_artifact_dirs": [str(d) for d in flash_dirs],
        "selected_cosim_root": str(selected_cosim_root),
        "phase_b_cosim_root": str(phase_b_cosim_root),
        "flash_records_by_artifact": per_variant,
        "selected_cosim_records": len(selected_cosim),
        "phase_b_cosim_records": len(phase_b_cosim),
        "total_records": len(records),
    }
    return records, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-jsonl", type=Path, default=DEFAULT_BASELINE_JSONL)
    parser.add_argument("--flash-stamp", default=DEFAULT_FLASH_STAMP)
    parser.add_argument("--selected-cosim-root", type=Path, default=DEFAULT_SELECTED_COSIM)
    parser.add_argument("--phase-b-cosim-root", type=Path, default=DEFAULT_PHASE_B_COSIM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records, meta = export_fixed_cosim_flash_jsonl(
        baseline_jsonl=args.baseline_jsonl,
        flash_stamp=args.flash_stamp,
        selected_cosim_root=args.selected_cosim_root,
        phase_b_cosim_root=args.phase_b_cosim_root,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    validation = validate_jsonl(args.output, verbose=True)
    by_type: dict[str, int] = {}
    rtl_by_status: dict[str, int] = {}
    rtl_by_role: dict[str, dict[str, int]] = {}
    for rec in records:
        rt = rec.get("report_type", "")
        by_type[rt] = by_type.get(rt, 0) + 1
        if rt == "rtl_sim":
            st = rec.get("rtl_sim", {}).get("status", "")
            rtl_by_status[st] = rtl_by_status.get(st, 0) + 1
            role = (rec.get("implementation", {}).get("origin_meta") or {}).get("kernel_role")
            if not role and "benchmarks_cosim_baseline" in (
                rec.get("implementation", {}).get("origin_version") or ""
            ):
                role = "baseline_gold"
            role = role or "unknown"
            bucket = rtl_by_role.setdefault(role, {})
            bucket[st] = bucket.get(st, 0) + 1

    summary = {
        **meta,
        "output": str(args.output),
        "by_report_type": by_type,
        "rtl_sim_by_status": rtl_by_status,
        "rtl_sim_by_role": rtl_by_role,
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

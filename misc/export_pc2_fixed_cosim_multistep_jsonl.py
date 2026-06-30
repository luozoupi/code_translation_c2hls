#!/usr/bin/env python3
"""Export fixed benchmarks_cosim baseline + multistep aav_n + cosim to schema JSONL."""

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
from flash_flow_artifacts import MULTISTEP_OPT_STEPS

DEFAULT_BASELINE_JSONL = REPO / "misc" / "hlsfactory_baseline_u280_20260616_benchmarks_full_cosim.jsonl"
DEFAULT_OUTPUT = REPO / "misc" / "hlsfactory_fixed_cosim_multistep_u280.jsonl"

KERNEL_ROLES = ("phase_b", *MULTISTEP_OPT_STEPS, "selected")


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
    meta = json.loads((bench_dir / "metadata.json").read_text())
    meta = dict(meta)
    meta["group_path"] = [bench_dir.name.removeprefix("hlsfactory_")]
    meta["corpus"] = bench_dir.parent.name
    return meta


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _load_multistep_results(cell_dir: Path, bench: str) -> dict[str, Any] | None:
    return _load_json(cell_dir / f"{bench}_multistep_results.json")


def _origin_meta(row: dict[str, Any], cell_dir: Path, kernel_role: str) -> dict[str, Any]:
    return {
        "model": row.get("model") or "",
        "mode": row.get("mode") or "multistep",
        "corpus": row.get("corpus") or "benchmarks_cosim",
        "matrix_family": row.get("matrix_family") or "",
        "variant": row.get("variant") or "aav_n",
        "skills_json": row.get("skills_json"),
        "record_flow": row.get("record_flow"),
        "kernel_role": kernel_role,
        "cell_dir": str(cell_dir),
        "origin_meta": {
            "note": (
                "HLSFactory benchmarks_cosim has gold/baseline only; "
                "per-step vs_ground_truth compares against overall gold, not step-specific GT."
            ),
        },
    }


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
                "error": (cosim.get("error") or "")[:300] if cosim_status != "pass" else None,
            },
        }
    )


def _step_csim(results: dict[str, Any] | None, step_name: str) -> dict[str, Any] | None:
    if not results:
        return None
    for step in results.get("generated_step_history") or results.get("steps") or []:
        if step.get("step_name") == step_name:
            csim = step.get("csim")
            if isinstance(csim, dict):
                return csim
    if step_name == "baseline":
        return results.get("baseline_csim") if isinstance(results.get("baseline_csim"), dict) else None
    return None


def export_multistep_artifact(artifact_dir: Path) -> list[dict[str, Any]]:
    rows = json.loads((artifact_dir / "matrix.json").read_text())
    origin_version = f"pc2_{artifact_dir.name}"
    records: list[dict[str, Any]] = []

    for row in rows:
        bench = row.get("bench") or ""
        cell_dir = Path(row.get("cell_dir") or "")
        if not bench or not cell_dir.is_dir() or row.get("status") != "ok":
            continue

        ms = _load_multistep_results(cell_dir, bench)
        meta = _bench_meta(bench)
        top = meta.get("translated_hls_top") or meta.get("hls_top") or "workload"
        run_meta = (ms or {}).get("run") or {}
        part = run_meta.get("part") or DEFAULT_PART
        clock_ns = float(run_meta.get("clock_ns") or DEFAULT_CLOCK_NS)

        for idx, role in enumerate(KERNEL_ROLES):
            if role == "phase_b":
                synth = _load_json(cell_dir / f"{bench}_phase_b_report.json") or (ms or {}).get("baseline_report") or {}
                csim = _step_csim(ms, "baseline")
            elif role == "selected":
                synth = (ms or {}).get("final_report") or _load_json(cell_dir / f"{bench}_selected_report.json") or {}
                csim = None
                for step in reversed((ms or {}).get("steps") or []):
                    if step.get("success") and isinstance(step.get("csim"), dict):
                        csim = step["csim"]
                        break
            else:
                synth = _load_json(cell_dir / f"{bench}_{role}_report.json") or {}
                csim = _step_csim(ms, role)
            if not synth:
                continue
            role_meta = _origin_meta(row, cell_dir, role)
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
                origin_meta=role_meta,
                variant_name=role,
                variant_index=idx,
                runtime_seconds=row.get("wallclock_s"),
            )
    return records


def export_cosim_manifest(cosim_run_root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((cosim_run_root / "manifest.json").read_text())
    records: list[dict[str, Any]] = []
    role_index = {role: idx for idx, role in enumerate(KERNEL_ROLES)}

    for cell in manifest.get("cells", []):
        cell_id = cell.get("cell_id") or ""
        bench = cell.get("bench") or ""
        cell_dir = Path(cell.get("cell_dir") or "")
        if not cell_id or not bench or not cell_dir.is_dir():
            continue
        cosim_path = cosim_run_root / "cells" / cell_id / "cosim_result.json"
        if not cosim_path.is_file():
            continue
        cosim = json.loads(cosim_path.read_text())
        ms = _load_multistep_results(cell_dir, bench)
        meta = _bench_meta(bench)
        run_meta = (ms or {}).get("run") or {}
        part = run_meta.get("part") or DEFAULT_PART
        role = cell.get("kernel_source") or "selected"
        artifact_basename = cell.get("artifact_basename") or ""
        origin_version = f"pc2_{artifact_basename}"
        row_stub = {
            "bench": bench,
            "model": cell.get("model") or "",
            "mode": cell.get("mode") or "multistep",
            "corpus": "benchmarks_cosim",
            "matrix_family": cell.get("matrix_family") or "",
            "variant": cell.get("variant") or "aav_n",
            "skills_json": cell.get("skills_json"),
            "record_flow": True,
        }
        role_meta = _origin_meta(row_stub, cell_dir, role)
        _append_rtl_sim(
            records,
            meta=meta,
            cosim=cosim,
            run_meta=run_meta,
            part=part,
            origin_version=origin_version,
            origin_meta=role_meta,
            variant_name=role,
            variant_index=role_index.get(role, 99),
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-jsonl", default=str(DEFAULT_BASELINE_JSONL))
    parser.add_argument("--multistep-stamp", required=True, help="Artifact stamp glob suffix")
    parser.add_argument("--variant", default="aav_n", help="Multistep variant key (aav_n, nav_n, …)")
    parser.add_argument("--cosim-root", default="", help="Multistep cosim run root")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    variant = args.variant
    stamp = args.multistep_stamp
    artifact_glob = f"multistep_fixed_cosim_{variant}_{stamp}"
    if not artifact_glob.endswith("_pipelined") and "_pipelined" not in stamp:
        artifact_glob = f"multistep_fixed_cosim_{variant}_{stamp}_pipelined"

    artifact_dirs = sorted(PC2.glob(f"{artifact_glob}"))
    if not artifact_dirs:
        artifact_dirs = sorted(PC2.glob(f"multistep_fixed_cosim_*_{args.multistep_stamp}*"))

    records: list[dict[str, Any]] = []
    baseline_path = Path(args.baseline_jsonl)
    if baseline_path.is_file():
        records.extend(_read_jsonl(baseline_path))

    for artifact_dir in artifact_dirs:
        if (artifact_dir / "matrix.json").is_file():
            records.extend(export_multistep_artifact(artifact_dir))

    if args.cosim_root:
        cosim_root = Path(args.cosim_root)
        if (cosim_root / "manifest.json").is_file():
            records.extend(export_cosim_manifest(cosim_root))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")

    validate_jsonl(out)
    summary = {
        "output": str(out),
        "record_count": len(records),
        "artifact_dirs": [str(p) for p in artifact_dirs],
        "cosim_root": args.cosim_root or None,
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

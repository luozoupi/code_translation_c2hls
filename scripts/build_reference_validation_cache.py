#!/usr/bin/env python3
"""Seed exact-input gold-reference cache entries from prior sweep artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FULL28 = [
    "hlsfactory_2mm", "hlsfactory_3mm", "hlsfactory_atax", "hlsfactory_bicg",
    "hlsfactory_cholesky", "hlsfactory_correlation", "hlsfactory_covariance",
    "hlsfactory_doitgen", "hlsfactory_durbin", "hlsfactory_fdtd_2d",
    "hlsfactory_floyd_warshall", "hlsfactory_gemm", "hlsfactory_gemver",
    "hlsfactory_gesummv", "hlsfactory_gramschmidt", "hlsfactory_heat_3d",
    "hlsfactory_jacobi_1d", "hlsfactory_jacobi_2d", "hlsfactory_lu",
    "hlsfactory_ludcmp", "hlsfactory_mvt", "hlsfactory_nussinov",
    "hlsfactory_seidel_2d", "hlsfactory_symm", "hlsfactory_syr2k",
    "hlsfactory_syrk", "hlsfactory_trisolv", "hlsfactory_trmm",
]


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _latest_input_mtime(bench_dir: Path) -> float:
    mtimes = [
        path.stat().st_mtime
        for path in bench_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return max(mtimes, default=0.0)


def _matching_run(run: dict, *, vitis_version: str, part: str,
                  clock_ns: float, flow_target: str) -> bool:
    try:
        observed_clock = float(run.get("clock_ns"))
    except (TypeError, ValueError):
        return False
    return (
        str(run.get("vitis_version")) == vitis_version
        and run.get("part") == part
        and abs(observed_clock - clock_ns) <= 1e-9
        and run.get("flow_target") == flow_target
    )


def _quality(result_path: Path, validation: dict) -> tuple[int, int, float]:
    cosim_status = (validation.get("cosim") or {}).get("status")
    reference_source = validation.get("reference_source")
    return (
        int(cosim_status == "passed"),
        int(reference_source == "local_vitis"),
        result_path.stat().st_mtime,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks-dir",
        default=str(REPO_ROOT / "benchmarks_external" / "HLSFactory" / "polybench_float_small"),
    )
    parser.add_argument("--results-root", default=str(REPO_ROOT / "results_sweeps"))
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "artifacts" / "reference_validation_cache"),
    )
    parser.add_argument("--benchmarks", default=",".join(FULL28))
    parser.add_argument("--vitis-version", default="2023.2")
    parser.add_argument("--part", default="xcu280-fsvh2892-2L-e")
    parser.add_argument("--clock-ns", type=float, default=3.33)
    parser.add_argument("--flow-target", default="vitis")
    parser.add_argument("--validation-mode", default="trusted_external")
    args = parser.parse_args()

    os.environ["C2HLS_REFERENCE_CACHE_DIR"] = str(Path(args.cache_dir).resolve())
    os.environ["C2HLS_REFERENCE_VALIDATE_MODE"] = args.validation_mode
    os.environ["C2HLS_VITIS_VERSION"] = args.vitis_version
    os.environ["C2HLS_PART"] = args.part
    os.environ["C2HLS_CLOCK_NS"] = str(args.clock_ns)
    os.environ["C2HLS_FLOW_TARGET"] = args.flow_target

    import c2hls

    benchmarks_dir = Path(args.benchmarks_dir).resolve()
    results_root = Path(args.results_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    requested = _split_csv(args.benchmarks)
    candidates: dict[str, tuple[tuple[int, int, float], Path, dict]] = {}
    rejected_newer_inputs: dict[str, int] = {}

    for result_path in results_root.rglob("*_multistep_results.json"):
        result = _read_json(result_path)
        if not result:
            continue
        benchmark = result.get("benchmark")
        if benchmark not in requested:
            continue
        run = result.get("run") or {}
        if not _matching_run(
            run,
            vitis_version=args.vitis_version,
            part=args.part,
            clock_ns=args.clock_ns,
            flow_target=args.flow_target,
        ):
            continue
        validation = result.get("reference_validation") or {}
        bench_dir = benchmarks_dir / benchmark
        if not bench_dir.is_dir():
            continue
        try:
            inputs = c2hls._load_benchmark_inputs(str(bench_dir))
        except (OSError, KeyError, ValueError):
            continue
        if not c2hls._reference_validation_cacheable(inputs, validation):
            continue
        selected_file = validation.get("selected_variant_file") or ""
        current_files = {
            item.get("file") for item in c2hls._ground_truth_candidates(inputs)
        }
        if selected_file and selected_file not in current_files:
            continue
        if result_path.stat().st_mtime < _latest_input_mtime(bench_dir):
            rejected_newer_inputs[benchmark] = rejected_newer_inputs.get(benchmark, 0) + 1
            continue
        score = _quality(result_path, validation)
        if benchmark not in candidates or score > candidates[benchmark][0]:
            candidates[benchmark] = (score, result_path, validation)

    entries = []
    for benchmark in requested:
        selected = candidates.get(benchmark)
        if selected is None:
            continue
        _, result_path, validation = selected
        inputs = c2hls._load_benchmark_inputs(str(benchmarks_dir / benchmark))
        cache_path = c2hls._write_reference_validation_cache(
            inputs,
            validation,
            source_result_json=str(result_path),
        )
        if cache_path is None:
            continue
        entries.append({
            "benchmark": benchmark,
            "cache_path": str(cache_path),
            "source_result_json": str(result_path),
            "synthesis_status": (validation.get("synthesis") or {}).get("status"),
            "csim_status": (validation.get("csim") or {}).get("status"),
            "cosim_status": (validation.get("cosim") or {}).get("status"),
            "reference_source": validation.get("reference_source"),
        })

    cached = {entry["benchmark"] for entry in entries}
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "vitis_version": args.vitis_version,
            "part": args.part,
            "clock_ns": args.clock_ns,
            "flow_target": args.flow_target,
            "validation_mode": args.validation_mode,
        },
        "requested_benchmarks": requested,
        "cached_count": len(entries),
        "cosim_pass_count": sum(entry["cosim_status"] == "passed" for entry in entries),
        "partial_cosim_count": sum(entry["cosim_status"] != "passed" for entry in entries),
        "missing_benchmarks": [name for name in requested if name not in cached],
        "rejected_artifacts_older_than_inputs": rejected_newer_inputs,
        "entries": entries,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(manifest_path),
        "cached_count": manifest["cached_count"],
        "cosim_pass_count": manifest["cosim_pass_count"],
        "partial_cosim_count": manifest["partial_cosim_count"],
        "missing_benchmarks": manifest["missing_benchmarks"],
        "rejected_artifacts_older_than_inputs": rejected_newer_inputs,
    }, indent=2, sort_keys=True))
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())

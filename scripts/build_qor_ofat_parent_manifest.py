#!/usr/bin/env python3
"""Select the lowest-cycle valid saved parent and inventory its QoR knobs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qor_design_space import discover_qor_knobs  # noqa: E402


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cycles(result: dict[str, Any]) -> int | None:
    report = result.get("final_report") or {}
    raw = report.get("latency_cycles_worst") or report.get("latency_cycles")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _valid_parent(result: dict[str, Any]) -> bool:
    report = result.get("final_report") or {}
    resources = ("dsp", "bram", "lut", "ff", "uram")
    return bool(
        result.get("success") is True
        and (result.get("csim") or {}).get("passed") is True
        and result.get("hls_code")
        and _cycles(result) is not None
        and all(report.get(key) is not None for key in resources)
    )


def build(results_root: Path, benchmarks_root: Path) -> dict[str, Any]:
    files = sorted(results_root.glob("*/*_multistep_results.json"))
    selected: dict[str, tuple[int, Path, dict[str, Any]]] = {}
    valid_result_count = 0
    for path in files:
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not _valid_parent(result):
            continue
        valid_result_count += 1
        benchmark = str(result.get("benchmark") or "").strip()
        cycles = _cycles(result)
        if not benchmark or cycles is None:
            continue
        previous = selected.get(benchmark)
        if previous is None or (cycles, str(path)) < (previous[0], str(previous[1])):
            selected[benchmark] = (cycles, path, result)

    parents = []
    for benchmark, (cycles, path, result) in sorted(selected.items()):
        code = str(result.get("hls_code") or "")
        knobs = discover_qor_knobs(code, max_knobs=None)
        counts: dict[str, int] = {}
        for knob in knobs:
            counts[knob.kind] = counts.get(knob.kind, 0) + 1
        report = result.get("final_report") or {}
        bench_dir = benchmarks_root / benchmark
        parents.append({
            "benchmark": benchmark,
            "benchmark_dir": str(bench_dir.resolve()),
            "benchmark_dir_exists": bench_dir.is_dir(),
            "result_path": str(path.resolve()),
            "result_sha256": _file_sha256(path),
            "setup_directory": path.parent.name,
            "cycles": cycles,
            "metrics": {
                key: report.get(key)
                for key in (
                    "latency_cycles",
                    "latency_cycles_worst",
                    "interval",
                    "estimated_clock_period_ns",
                    "slack_ns",
                    "dsp",
                    "bram",
                    "lut",
                    "ff",
                    "uram",
                )
            },
            "knob_count": len(knobs),
            "knob_kind_counts": dict(sorted(counts.items())),
            "knobs": [knob.public() for knob in knobs],
        })

    return {
        "schema_version": "c2hls.qor-ofat-parent-manifest.v1",
        "selection_policy": (
            "lowest exact Vitis worst-case cycles among saved results with "
            "passing CSim and complete CSynth resource evidence"
        ),
        "results_root": str(results_root.resolve()),
        "benchmarks_root": str(benchmarks_root.resolve()),
        "result_file_count": len(files),
        "valid_result_count": valid_result_count,
        "selected_parent_count": len(parents),
        "parents": parents,
    }


def _write_csv(path: Path, parents: list[dict[str, Any]]) -> None:
    fields = [
        "benchmark",
        "cycles",
        "knob_count",
        "knob_kinds",
        "setup_directory",
        "result_path",
        "benchmark_dir",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for parent in parents:
            writer.writerow({
                "benchmark": parent["benchmark"],
                "cycles": parent["cycles"],
                "knob_count": parent["knob_count"],
                "knob_kinds": ";".join(
                    f"{kind}:{count}"
                    for kind, count in parent["knob_kind_counts"].items()
                ),
                "setup_directory": parent["setup_directory"],
                "result_path": parent["result_path"],
                "benchmark_dir": parent["benchmark_dir"],
            })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--benchmarks-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    payload = build(args.results_root, args.benchmarks_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(args.output_csv, payload["parents"])
    print(json.dumps({
        "result_file_count": payload["result_file_count"],
        "valid_result_count": payload["valid_result_count"],
        "selected_parent_count": payload["selected_parent_count"],
        "output_json": str(args.output_json),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

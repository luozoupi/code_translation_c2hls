#!/usr/bin/env python3
"""Freeze validated Phase-B baselines and router rankings for a sweep."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import c2hls
from phase_b_manifest import (
    SCHEMA_VERSION,
    canonical_json_sha256,
    file_sha256,
    text_sha256,
    toolchain_fingerprint,
)
from skill_library import load_frozen_library
from smart_skill_router import route_smart_skills


DEFAULT_BENCHMARKS = (
    "2mm",
    "bicg",
    "cholesky",
    "correlation",
    "fdtd_2d",
    "lu",
    "mvt",
    "seidel_2d",
)


def _phase_b_code(history_path: Path) -> str:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    for event in payload.get("event_history") or []:
        if not isinstance(event, dict) or event.get("role") != "assistant":
            continue
        code = c2hls.extract_cpp_code(str(event.get("content") or ""))
        if code:
            return code
    raise ValueError(f"no Phase-B assistant code found in {history_path}")


def _compact_csim(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keep = {
        "status",
        "supported",
        "ran",
        "success",
        "passed",
        "correctness",
        "golden_output_sha256",
        "error",
        "log_excerpt",
    }
    return {key: value[key] for key in keep if key in value}


def _matrix_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("strategy") == "flash"
                and row.get("skill_mode") == "skillless"
                and row.get("valid_csim_csynth") == "True"
            ):
                rows[str(row["problem"])] = row
    return rows


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    requested = tuple(
        item.strip() for item in args.benchmarks.split(",") if item.strip()
    )
    matrix = _matrix_rows(args.matrix_csv)
    library = load_frozen_library(args.skill_library)
    output_path = args.output.resolve()
    code_dir = output_path.parent / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict[str, Any]] = {}
    observed_toolchains: list[dict[str, Any]] = []
    for problem in requested:
        if problem not in matrix:
            raise KeyError(f"no valid flash/skillless source row for {problem}")
        row = matrix[problem]
        result_path = Path(row["source_result_path"]).resolve()
        result = json.loads(result_path.read_text(encoding="utf-8"))
        benchmark = str(result.get("benchmark") or row["benchmark"])
        history_path = result_path.with_name(f"{benchmark}_history.json")
        code = _phase_b_code(history_path)

        bench_dir = args.benchmarks_dir / benchmark
        inputs = c2hls._load_benchmark_inputs(str(bench_dir))
        if inputs["meta"].get("benchmark") != benchmark:
            raise ValueError(
                f"benchmark identity mismatch for {problem}: "
                f"{inputs['meta'].get('benchmark')} != {benchmark}"
            )

        report = result.get("baseline_report")
        csim = _compact_csim(result.get("baseline_csim"))
        if not isinstance(report, dict) or not report:
            raise ValueError(f"{problem} lacks a baseline CSynth report")
        if csim.get("passed") is not True or csim.get("ran") is not True:
            raise ValueError(f"{problem} lacks an executed passing baseline CSim")

        run = result.get("run") or {}
        observed_toolchains.append(
            toolchain_fingerprint(
                vitis_version=str(run.get("vitis_version") or ""),
                part=str(run.get("part") or ""),
                clock_ns=float(run.get("clock_ns")),
                flow_target=str(run.get("flow_target") or "vitis"),
            )
        )
        route = route_smart_skills(
            library,
            scope="smart_exhaustive_v2",
            step_name="flash",
            current_code=code,
            synth_report=report,
            vitis_version=str(run.get("vitis_version") or ""),
            fpga=str(run.get("part") or ""),
            max_skills=42,
            min_score=-1.0e9,
        )
        ranking = [
            {
                "rank": index + 1,
                **score,
            }
            for index, score in enumerate(route.audit["scores"])
        ]

        code_path = code_dir / f"{benchmark}.cpp"
        code_path.write_text(code, encoding="utf-8")
        relative_code_path = code_path.relative_to(output_path.parent)
        entry = {
            "benchmark": benchmark,
            "problem": problem,
            "input_c_sha256": text_sha256(inputs["c_code"]),
            "header_sha256": text_sha256(inputs["header_code"]),
            "code_path": str(relative_code_path),
            "code_sha256": text_sha256(code),
            "csim": csim,
            "csim_sha256": canonical_json_sha256(csim),
            "csynth_report": report,
            "csynth_report_sha256": canonical_json_sha256(report),
            "router_ranking": ranking,
            "router_ranking_skill_ids": [
                item["skill_id"] for item in ranking
            ],
            "router_ranking_sha256": canonical_json_sha256(ranking),
            "source_artifacts": {
                "result_path": str(result_path),
                "result_sha256": file_sha256(result_path),
                "history_path": str(history_path),
                "history_sha256": file_sha256(history_path),
            },
        }
        entries[benchmark] = entry

    if not observed_toolchains or any(
        item != observed_toolchains[0] for item in observed_toolchains
    ):
        raise ValueError("source Phase-B entries do not share one toolchain")
    expected_toolchain = toolchain_fingerprint(
        vitis_version=args.vitis_version,
        part=args.part,
        clock_ns=args.clock_ns,
        flow_target=args.flow_target,
    )
    if observed_toolchains[0] != expected_toolchain:
        raise ValueError(
            f"source toolchain {observed_toolchains[0]} does not match "
            f"requested toolchain {expected_toolchain}"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "controlled_skill_dose_phase_b_seed",
        "toolchain": expected_toolchain,
        "skill_library": {
            "path": str(args.skill_library.resolve()),
            "sha256": file_sha256(args.skill_library),
            "skill_count": len(library.all()),
        },
        "benchmarks": list(requested),
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-csv", type=Path, required=True)
    parser.add_argument("--benchmarks-dir", type=Path, required=True)
    parser.add_argument("--skill-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
    )
    parser.add_argument("--vitis-version", default="2023.2")
    parser.add_argument("--part", default="xcu280-fsvh2892-2L-e")
    parser.add_argument("--clock-ns", type=float, default=3.33)
    parser.add_argument("--flow-target", default="vitis")
    return parser.parse_args()


if __name__ == "__main__":
    manifest = build_manifest(parse_args())
    print(
        json.dumps(
            {
                "entries": len(manifest["entries"]),
                "toolchain": manifest["toolchain"],
            },
            sort_keys=True,
        )
    )

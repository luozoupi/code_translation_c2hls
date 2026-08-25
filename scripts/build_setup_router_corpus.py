#!/usr/bin/env python3
"""Build a leakage-safe benchmark/setup outcome corpus for setup routing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import c2hls
from setup_router import (
    CORRECTED_VERSION,
    LEGACY_VERSION,
    registry_by_id,
)


SCHEMA_VERSION = "c2hls.setup-router-corpus.v1"
TEST_PROBLEMS = {"durbin", "floyd_warshall", "gemm", "trmm"}
VALIDATION_PROBLEMS = {
    "gramschmidt",
    "atax",
    "jacobi_2d",
    "nussinov",
}
EXCLUDED_PROBLEMS = {"doitgen"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase_b_code(history_path: Path) -> str:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    for event in payload.get("event_history") or []:
        if isinstance(event, dict) and event.get("role") == "assistant":
            code = c2hls.extract_cpp_code(str(event.get("content") or ""))
            if code:
                return code
    return ""


def _source_features(code: str) -> dict[str, float]:
    text = code or ""
    compact = re.sub(r"/\*.*?\*/|//[^\n]*", " ", text, flags=re.S)
    loops = len(re.findall(r"\bfor\s*\(", compact))
    array_refs = len(re.findall(r"\b[A-Za-z_]\w*\s*\[[^\]]+\]", compact))
    names = re.findall(r"\b([A-Za-z_]\w*)\s*\[[^\]]+\]", compact)
    unique_arrays = len(set(names))
    return {
        "source_characters": float(len(text)),
        "source_lines": float(len(text.splitlines())),
        "source_loop_count": float(loops),
        "source_array_reference_count": float(array_refs),
        "source_unique_array_count": float(unique_arrays),
        "source_function_count": float(
            len(
                re.findall(
                    r"\b(?:void|int|float|double)\s+[A-Za-z_]\w*\s*\(",
                    compact,
                )
            )
        ),
        "source_reduction_markers": float(
            len(re.findall(r"\+=|-=|\b(?:sum|acc|accum)\w*\b", compact))
        ),
        "source_neighbor_index_markers": float(
            len(
                re.findall(
                    r"\[[^\]]*[A-Za-z_]\w*\s*[+-]\s*\d+[^\]]*\]",
                    compact,
                )
            )
        ),
        "source_multiply_ops": float(compact.count("*")),
        "source_add_ops": float(compact.count("+")),
        "source_has_multidimensional_arrays": float(
            bool(re.search(r"\[[^\]]+\]\s*\[[^\]]+\]", compact))
        ),
        "source_has_nested_loops": float(loops >= 2),
    }


def _phase_b_features(
    report: dict[str, Any],
    code: str,
    csim: dict[str, Any],
) -> dict[str, float]:
    features: dict[str, float] = {
        "phase_b_code_characters": float(len(code or "")),
        "phase_b_code_lines": float(len((code or "").splitlines())),
        "phase_b_csim_passed": float(
            isinstance(csim, dict)
            and csim.get("ran") is True
            and csim.get("passed") is True
        ),
    }
    for metric in (
        "latency_cycles",
        "interval",
        "bram",
        "dsp",
        "ff",
        "lut",
        "uram",
        "fmax_mhz",
        "estimated_clock_period_ns",
        "requested_clock_period_ns",
        "slack_ns",
    ):
        value = report.get(metric)
        features[f"phase_b_{metric}"] = (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else 0.0
        )
    cycles = features["phase_b_latency_cycles"]
    features["phase_b_log_latency_cycles"] = (
        math.log(cycles) if cycles > 0 else 0.0
    )
    bottlenecks = [
        str(item.get("kind") or "")
        for item in ((report.get("feedback") or {}).get("bottlenecks") or [])
        if isinstance(item, dict) and item.get("kind")
    ]
    counts = Counter(bottlenecks)
    features["phase_b_bottleneck_count"] = float(len(bottlenecks))
    for family, kinds in {
        "pipeline": {
            "ii_target_miss",
            "pipeline_blocked",
            "non_pipelined_hot_loop",
            "interval_exceeds_latency",
            "resource_limited_ii",
        },
        "memory": {
            "global_memory_latency",
            "memory_bandwidth",
            "memory_port_limited",
            "local_memory_port_limited",
            "port_conflict",
            "non_contiguous_access",
        },
        "recurrence": {
            "loop_carried_dep",
            "recurrence_limited_ii",
            "reduction_loop",
            "true_loop_carried_dep",
        },
        "dataflow": {
            "dataflow_blocked",
            "function_stage_serialization",
            "load_compute_serialize",
            "store_compute_serialize",
        },
    }.items():
        features[f"phase_b_{family}_bottlenecks"] = float(
            sum(counts[kind] for kind in kinds)
        )
    return features


def _split(problem: str) -> str:
    if problem in TEST_PROBLEMS:
        return "test"
    if problem in VALIDATION_PROBLEMS:
        return "validation"
    return "train"


def _load_input(
    benchmarks_dir: Path, benchmark: str
) -> tuple[str, str]:
    bench_dir = benchmarks_dir / benchmark
    metadata = json.loads(
        (bench_dir / "metadata.json").read_text(encoding="utf-8")
    )
    plain = (bench_dir / "plain.cpp").read_text(encoding="utf-8")
    header_name = metadata.get("header_file") or "kernel.h"
    header_path = bench_dir / header_name
    header = (
        header_path.read_text(encoding="utf-8")
        if header_path.is_file()
        else ""
    )
    return plain, header


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    legacy = registry_by_id(LEGACY_VERSION)
    corrected = registry_by_id(CORRECTED_VERSION)
    raw_rows = list(
        csv.DictReader(args.matrix_csv.open(newline="", encoding="utf-8"))
    )
    coverage: dict[tuple[str, str], set[str]] = defaultdict(set)
    staged: list[dict[str, Any]] = []
    source_cache: dict[str, dict[str, float]] = {}
    seen: set[tuple[str, str, str, str]] = set()

    for row in raw_rows:
        problem = str(row.get("problem") or "")
        benchmark = str(row.get("benchmark") or "")
        if not problem or problem in EXCLUDED_PROBLEMS:
            continue
        setup_id = (
            f"{LEGACY_VERSION}:{row.get('strategy')}:{row.get('skill_mode')}"
        )
        setup = legacy.get(setup_id)
        if setup is None:
            continue
        result_path = Path(str(row.get("source_result_path") or ""))
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        history_path = result_path.with_name(f"{benchmark}_history.json")
        phase_b_code = _phase_b_code(history_path)
        final_path = result_path.with_name(f"{benchmark}_final.cpp")
        final_hash = (
            _sha256(final_path)
            if final_path.is_file()
            else str(result.get("selected_code_sha256") or "")
        )
        run_fingerprint = str(
            (result.get("run_fingerprint") or {}).get("sha256") or ""
        )
        lineage = f"polybench:{problem}"
        identity = (
            lineage,
            setup.fingerprint,
            run_fingerprint,
            final_hash,
        )
        if identity in seen:
            continue
        seen.add(identity)

        if benchmark not in source_cache:
            plain, _header = _load_input(args.benchmarks_dir, benchmark)
            source_cache[benchmark] = _source_features(plain)
        baseline_report = result.get("baseline_report") or {}
        baseline_csim = result.get("baseline_csim") or {}
        features: dict[str, Any] = {
            **source_cache[benchmark],
            **_phase_b_features(
                baseline_report,
                phase_b_code,
                baseline_csim,
            ),
            "setup_strategy": setup.strategy,
            "setup_skill_scope": setup.skill_scope,
            "setup_behavior_version": setup.behavior_version,
            "setup_prompt_mode": setup.prompt_mode,
            "setup_router_version": float(setup.router_version),
            "setup_separate_candidates": float(
                setup.candidate_policy
                == "separate_skill_directed_candidates"
            ),
            "model_id": str((result.get("run") or {}).get("model") or ""),
        }
        valid = row.get("valid_csim_csynth") == "True"
        cycles = (
            float(row["latency_cycles"])
            if valid and row.get("latency_cycles")
            else None
        )
        coverage[(problem, LEGACY_VERSION)].add(setup_id)
        staged.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_kind": "setup_outcome",
                "benchmark": benchmark,
                "problem": problem,
                "benchmark_lineage": lineage,
                "split": _split(problem),
                "setup": setup.to_record(),
                "features": features,
                "labels": {
                    "valid": valid,
                    "latency_cycles": cycles,
                    "log_latency_cycles": (
                        math.log(cycles) if cycles and cycles > 0 else None
                    ),
                    "setup_rank": None,
                    "is_best_setup": False,
                    "within_5pct_of_best": False,
                    "regret": None,
                },
                "eligibility": {
                    "feasibility_model": True,
                    "ranking_model": False,
                },
                "provenance": {
                    "source_kind": "historical_complete_matrix",
                    "source_result_path": str(result_path.resolve()),
                    "source_result_sha256": _sha256(result_path),
                    "run_fingerprint": run_fingerprint,
                    "phase_b_code_sha256": hashlib.sha256(
                        phase_b_code.encode("utf-8")
                    ).hexdigest(),
                    "final_code_sha256": final_hash,
                    "dedup_key_sha256": hashlib.sha256(
                        "\0".join(identity).encode("utf-8")
                    ).hexdigest(),
                },
            }
        )
        events = (
            (result.get("synthesis_evaluations") or {}).get("events") or []
        )
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            event_code_hash = str(event.get("code_sha256") or "")
            if not event_code_hash:
                continue
            event_identity = (
                lineage,
                setup.fingerprint,
                run_fingerprint,
                event_code_hash,
            )
            if event_identity in seen:
                continue
            seen.add(event_identity)
            event_valid = (
                event.get("correctness_status") == "passed"
                and event.get("synthesis_status") == "passed"
                and event.get("resource_fit") is True
                and event.get("timing_met") is True
                and isinstance(
                    event.get("synthesized_latency_cycles"), int
                )
                and not isinstance(
                    event.get("synthesized_latency_cycles"), bool
                )
                and event.get("synthesized_latency_cycles") > 0
            )
            event_cycles = (
                float(event["synthesized_latency_cycles"])
                if event_valid
                else None
            )
            staged.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_kind": "incomplete_candidate_trace",
                    "benchmark": benchmark,
                    "problem": problem,
                    "benchmark_lineage": lineage,
                    "split": _split(problem),
                    "setup": setup.to_record(),
                    "features": dict(features),
                    "labels": {
                        "valid": event_valid,
                        "latency_cycles": event_cycles,
                        "log_latency_cycles": (
                            math.log(event_cycles)
                            if event_cycles and event_cycles > 0
                            else None
                        ),
                        "setup_rank": None,
                        "is_best_setup": False,
                        "within_5pct_of_best": False,
                        "regret": None,
                    },
                    "eligibility": {
                        "feasibility_model": True,
                        "ranking_model": False,
                    },
                    "provenance": {
                        "source_kind": "historical_incomplete_candidate_trace",
                        "source_result_path": str(result_path.resolve()),
                        "source_result_sha256": _sha256(result_path),
                        "run_fingerprint": run_fingerprint,
                        "phase_b_code_sha256": hashlib.sha256(
                            phase_b_code.encode("utf-8")
                        ).hexdigest(),
                        "final_code_sha256": event_code_hash,
                        "candidate_evaluation_index": event.get(
                            "candidate_evaluation_index", event_index
                        ),
                        "failure_class": event.get("failure_class"),
                        "dedup_key_sha256": hashlib.sha256(
                            "\0".join(event_identity).encode("utf-8")
                        ).hexdigest(),
                    },
                }
            )

    if args.corrected_matrix_records:
        if not args.corrected_phase_b_manifest:
            raise ValueError(
                "--corrected-matrix-records requires "
                "--corrected-phase-b-manifest"
            )
        phase_manifest = json.loads(
            args.corrected_phase_b_manifest.read_text(encoding="utf-8")
        )
        with args.corrected_matrix_records.open(encoding="utf-8") as handle:
            matrix_records = [
                json.loads(line) for line in handle if line.strip()
            ]
        for matrix_record in matrix_records:
            benchmark = str(matrix_record.get("benchmark") or "")
            problem = str(matrix_record.get("problem") or "")
            if (
                not benchmark
                or not problem
                or problem in EXCLUDED_PROBLEMS
            ):
                continue
            phase_entry = (phase_manifest.get("entries") or {}).get(
                benchmark
            )
            if not isinstance(phase_entry, dict):
                raise ValueError(
                    f"corrected Phase-B manifest lacks {benchmark}"
                )
            if benchmark not in source_cache:
                plain, _header = _load_input(
                    args.benchmarks_dir, benchmark
                )
                source_cache[benchmark] = _source_features(plain)
            phase_code_path = (
                args.corrected_phase_b_manifest.parent
                / str(phase_entry["code_path"])
            ).resolve()
            phase_code = phase_code_path.read_text(encoding="utf-8")
            phase_report = phase_entry["csynth_report"]
            phase_csim = phase_entry["csim"]
            for candidate in matrix_record.get("candidates") or []:
                setup_id = str(candidate.get("setup_id") or "")
                setup = corrected.get(setup_id)
                if setup is None:
                    continue
                result_path = Path(
                    str(candidate.get("result_path") or "")
                )
                result = (
                    json.loads(result_path.read_text(encoding="utf-8"))
                    if result_path.is_file()
                    else {}
                )
                run_fingerprint = str(
                    (result.get("run_fingerprint") or {}).get("sha256")
                    or ""
                )
                final_hash = str(
                    candidate.get("code_sha256")
                    or result.get("selected_code_sha256")
                    or ""
                )
                lineage = f"polybench:{problem}"
                identity = (
                    lineage,
                    setup.fingerprint,
                    run_fingerprint,
                    final_hash,
                )
                if identity in seen:
                    continue
                seen.add(identity)
                valid = candidate.get("valid") is True
                cycles = (
                    float(candidate["latency_cycles"])
                    if valid and candidate.get("latency_cycles")
                    else None
                )
                features = {
                    **source_cache[benchmark],
                    **_phase_b_features(
                        phase_report,
                        phase_code,
                        phase_csim,
                    ),
                    "setup_strategy": setup.strategy,
                    "setup_skill_scope": setup.skill_scope,
                    "setup_behavior_version": setup.behavior_version,
                    "setup_prompt_mode": setup.prompt_mode,
                    "setup_router_version": float(
                        setup.router_version
                    ),
                    "setup_separate_candidates": float(
                        setup.candidate_policy
                        == "separate_skill_directed_candidates"
                    ),
                    "model_id": str(
                        (result.get("run") or {}).get("model")
                        or "claude-sonnet-4-6"
                    ),
                }
                coverage[(problem, CORRECTED_VERSION)].add(setup_id)
                staged.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "record_kind": "setup_outcome",
                        "benchmark": benchmark,
                        "problem": problem,
                        "benchmark_lineage": lineage,
                        "split": _split(problem),
                        "setup": setup.to_record(),
                        "features": features,
                        "labels": {
                            "valid": valid,
                            "latency_cycles": cycles,
                            "log_latency_cycles": (
                                math.log(cycles)
                                if cycles and cycles > 0
                                else None
                            ),
                            "setup_rank": None,
                            "is_best_setup": False,
                            "within_5pct_of_best": False,
                            "regret": None,
                        },
                        "eligibility": {
                            "feasibility_model": True,
                            "ranking_model": False,
                        },
                        "provenance": {
                            "source_kind": "corrected_complete_matrix",
                            "source_result_path": str(
                                result_path.resolve()
                            ),
                            "source_result_sha256": (
                                _sha256(result_path)
                                if result_path.is_file()
                                else None
                            ),
                            "run_fingerprint": run_fingerprint,
                            "phase_b_code_sha256": phase_entry[
                                "code_sha256"
                            ],
                            "phase_b_code_path": str(phase_code_path),
                            "final_code_sha256": final_hash,
                            "dedup_key_sha256": hashlib.sha256(
                                "\0".join(identity).encode("utf-8")
                            ).hexdigest(),
                        },
                    }
                )

    expected_by_version = {
        LEGACY_VERSION: set(legacy),
        CORRECTED_VERSION: set(corrected),
    }
    by_problem_version: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for record in staged:
        by_problem_version[
            (
                record["problem"],
                record["setup"]["behavior_version"],
            )
        ].append(record)
    for (problem, behavior_version), records in by_problem_version.items():
        complete = (
            coverage[(problem, behavior_version)]
            == expected_by_version[behavior_version]
        )
        valid_records = [
            record
            for record in records
            if record["record_kind"] == "setup_outcome"
            and record["labels"]["valid"]
        ]
        ordered = sorted(
            valid_records,
            key=lambda record: (
                record["labels"]["latency_cycles"],
                record["setup"]["setup_fingerprint"],
            ),
        )
        best = (
            ordered[0]["labels"]["latency_cycles"] if ordered else None
        )
        for rank, record in enumerate(ordered, start=1):
            cycles = record["labels"]["latency_cycles"]
            regret = cycles / best if best else None
            record["labels"].update(
                {
                    "setup_rank": rank,
                    "is_best_setup": rank == 1,
                    "within_5pct_of_best": bool(
                        regret is not None and regret <= 1.05
                    ),
                    "regret": regret,
                }
            )
        for record in records:
            record["eligibility"]["ranking_model"] = bool(
                record["record_kind"] == "setup_outcome"
                and complete
                and ordered
            )
            record["eligibility"]["complete_crossed_matrix"] = complete
            record["eligibility"]["ranking_label_group"] = (
                f"{record['benchmark_lineage']}:{behavior_version}"
            )

    records = sorted(
        staged,
        key=lambda record: (
            record["split"],
            record["problem"],
            record["setup"]["setup_id"],
        ),
    )
    available = {
        record["problem"]
        for record in records
        if record["labels"]["valid"]
    }
    expected_train = available - TEST_PROBLEMS - VALIDATION_PROBLEMS
    if len(expected_train) != 19:
        raise ValueError(
            f"expected 19 train kernels, found {len(expected_train)}: "
            f"{sorted(expected_train)}"
        )
    if not TEST_PROBLEMS <= available or not VALIDATION_PROBLEMS <= available:
        raise ValueError("one or more fixed validation/test kernels are absent")
    return records


def _write_outputs(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "setup_router_outcomes.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    feature_names = sorted(
        {name for record in records for name in record["features"]}
    )
    csv_path = args.output_dir / "setup_router_outcomes.csv"
    columns = [
        "benchmark",
        "record_kind",
        "problem",
        "benchmark_lineage",
        "split",
        "setup_id",
        "setup_fingerprint",
        *feature_names,
        "valid",
        "latency_cycles",
        "log_latency_cycles",
        "setup_rank",
        "is_best_setup",
        "within_5pct_of_best",
        "regret",
        "ranking_model_eligible",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "benchmark": record["benchmark"],
                    "record_kind": record["record_kind"],
                    "problem": record["problem"],
                    "benchmark_lineage": record["benchmark_lineage"],
                    "split": record["split"],
                    "setup_id": record["setup"]["setup_id"],
                    "setup_fingerprint": record["setup"][
                        "setup_fingerprint"
                    ],
                    **record["features"],
                    **record["labels"],
                    "ranking_model_eligible": record["eligibility"][
                        "ranking_model"
                    ],
                }
            )

    split_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lineage_policy": (
            "PolyBench-derived duplicates remain in one polybench:<problem> "
            "lineage; split membership is by lineage, never by outcome row."
        ),
        "train": sorted(
            {
                record["problem"]
                for record in records
                if record["split"] == "train"
            }
        ),
        "validation": sorted(VALIDATION_PROBLEMS),
        "test": sorted(TEST_PROBLEMS),
        "excluded": sorted(EXCLUDED_PROBLEMS),
        "record_counts": dict(
            Counter(record["split"] for record in records)
        ),
        "feature_names": feature_names,
        "forbidden_router_input_families": [
            "reference metrics",
            "ground-truth metrics",
            "post-candidate code or synthesis measurements",
            "selected winner outcome",
        ],
    }
    split_path = args.output_dir / "split_manifest.json"
    split_path.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hashes = {
        path.name: {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in (jsonl_path, csv_path, split_path)
    }
    artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(records),
        "benchmark_count": len({r["problem"] for r in records}),
        "artifacts": hashes,
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-csv", type=Path, required=True)
    parser.add_argument("--benchmarks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corrected-matrix-records", type=Path)
    parser.add_argument("--corrected-phase-b-manifest", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output_records = build_records(arguments)
    _write_outputs(output_records, arguments)
    print(
        json.dumps(
            {
                "records": len(output_records),
                "benchmarks": len(
                    {record["problem"] for record in output_records}
                ),
                "output_dir": str(arguments.output_dir),
            },
            sort_keys=True,
        )
    )

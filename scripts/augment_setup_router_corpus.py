#!/usr/bin/env python3
"""Append complete external corrected-v2 outcomes to a router corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_setup_router_corpus import (
    _load_input,
    _phase_b_features,
    _sha256,
    _source_features,
)
from setup_router import CORRECTED_VERSION, registry_by_id


SCHEMA_VERSION = "c2hls.setup-router-corpus.v1"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _latest_matrix_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in _load_jsonl(path):
        benchmark = str(record.get("benchmark") or "")
        if benchmark:
            records[benchmark] = record
    return records


def _external_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    phase_manifest = json.loads(
        args.phase_b_manifest.read_text(encoding="utf-8")
    )
    if phase_manifest.get("status") != "complete":
        raise ValueError("external Phase-B manifest is not complete")
    phase_entries = phase_manifest.get("entries") or {}
    matrix = _latest_matrix_records(args.matrix_records)
    registry = registry_by_id(CORRECTED_VERSION)
    expected_setup_ids = set(registry)
    records: list[dict[str, Any]] = []

    for benchmark, phase_entry in sorted(phase_entries.items()):
        if not isinstance(phase_entry, dict):
            continue
        matrix_record = matrix.get(benchmark)
        if not matrix_record or matrix_record.get("success") is not True:
            raise ValueError(
                f"external matrix lacks a successful record for {benchmark}"
            )
        candidates = {
            str(candidate.get("setup_id") or ""): candidate
            for candidate in matrix_record.get("candidates") or []
            if candidate.get("setup_id")
        }
        if set(candidates) != expected_setup_ids:
            raise ValueError(
                f"{benchmark} setup coverage mismatch: missing "
                f"{sorted(expected_setup_ids - set(candidates))}, extra "
                f"{sorted(set(candidates) - expected_setup_ids)}"
            )

        plain, _header = _load_input(args.benchmarks_dir, benchmark)
        source_features = _source_features(plain)
        phase_code_path = (
            args.phase_b_manifest.parent
            / str(phase_entry["code_path"])
        ).resolve()
        phase_code = phase_code_path.read_text(encoding="utf-8")
        phase_features = _phase_b_features(
            phase_entry["csynth_report"],
            phase_code,
            phase_entry["csim"],
        )
        lineage = str(phase_entry["benchmark_lineage"])
        problem = str(phase_entry["problem"])
        split = str(phase_entry["split"])

        benchmark_records = []
        for setup_id in sorted(expected_setup_ids):
            setup = registry[setup_id]
            candidate = candidates[setup_id]
            result_path = Path(
                str(candidate.get("result_path") or "")
            ).resolve()
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
            identity = (
                lineage,
                setup.fingerprint,
                run_fingerprint,
                final_hash,
            )
            valid = candidate.get("valid") is True
            cycles = (
                float(candidate["latency_cycles"])
                if valid and candidate.get("latency_cycles")
                else None
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                "record_kind": "setup_outcome",
                "benchmark": benchmark,
                "problem": problem,
                "benchmark_lineage": lineage,
                "split": split,
                "setup": setup.to_record(),
                "features": {
                    **source_features,
                    **phase_features,
                    "setup_strategy": setup.strategy,
                    "setup_skill_scope": setup.skill_scope,
                    "setup_behavior_version": setup.behavior_version,
                    "setup_prompt_mode": setup.prompt_mode,
                    "setup_router_version": float(setup.router_version),
                    "setup_separate_candidates": float(
                        setup.candidate_policy
                        == "separate_skill_directed_candidates"
                    ),
                    "model_id": str(
                        (result.get("run") or {}).get("model")
                        or "claude-sonnet-4-6"
                    ),
                },
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
                    "ranking_model": True,
                    "complete_crossed_matrix": True,
                    "ranking_label_group": (
                        f"{lineage}:{CORRECTED_VERSION}"
                    ),
                },
                "provenance": {
                    "source_kind": "external_corrected_complete_matrix",
                    "dataset": phase_entry.get("dataset", "MachSuite"),
                    "source_result_path": str(result_path),
                    "source_result_sha256": (
                        _sha256(result_path)
                        if result_path.is_file()
                        else None
                    ),
                    "run_fingerprint": run_fingerprint,
                    "phase_b_code_sha256": phase_entry["code_sha256"],
                    "phase_b_code_path": str(phase_code_path),
                    "final_code_sha256": final_hash,
                    "dedup_key_sha256": hashlib.sha256(
                        "\0".join(identity).encode("utf-8")
                    ).hexdigest(),
                },
            }
            benchmark_records.append(record)

        valid_records = [
            record
            for record in benchmark_records
            if record["labels"]["valid"]
        ]
        if not valid_records:
            raise ValueError(
                f"{benchmark} has no valid corrected-v2 setup outcome"
            )
        ordered = sorted(
            valid_records,
            key=lambda record: (
                record["labels"]["latency_cycles"],
                record["setup"]["setup_fingerprint"],
            ),
        )
        best = float(ordered[0]["labels"]["latency_cycles"])
        for rank, record in enumerate(ordered, start=1):
            cycles = float(record["labels"]["latency_cycles"])
            regret = cycles / best
            record["labels"].update(
                {
                    "setup_rank": rank,
                    "is_best_setup": rank == 1,
                    "within_5pct_of_best": regret <= 1.05,
                    "regret": regret,
                }
            )
        records.extend(benchmark_records)
    return records


def _write_outputs(
    records: list[dict[str, Any]],
    external_records: list[dict[str, Any]],
    args: argparse.Namespace,
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

    split_lineages: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split_lineages[record["split"]].add(
            record["benchmark_lineage"]
        )
    external_lineages: dict[str, set[str]] = defaultdict(set)
    for record in external_records:
        external_lineages[record["split"]].add(
            record["benchmark_lineage"]
        )
    split_path = args.output_dir / "split_manifest.json"
    split_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "lineage_policy": (
                    "Split membership is assigned by benchmark lineage; "
                    "no lineage crosses train, validation, or test."
                ),
                "lineages": {
                    split: sorted(values)
                    for split, values in sorted(split_lineages.items())
                },
                "external_lineages": {
                    split: sorted(values)
                    for split, values in sorted(
                        external_lineages.items()
                    )
                },
                "record_counts": dict(
                    Counter(record["split"] for record in records)
                ),
                "external_record_counts": dict(
                    Counter(
                        record["split"] for record in external_records
                    )
                ),
                "forbidden_router_input_families": [
                    "reference metrics",
                    "ground-truth metrics",
                    "post-candidate code or synthesis measurements",
                    "selected winner outcome",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_path = args.output_dir / "artifact_manifest.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "base_corpus": str(args.base_corpus.resolve()),
                "base_corpus_sha256": _sha256(args.base_corpus),
                "external_matrix_records": str(
                    args.matrix_records.resolve()
                ),
                "external_phase_b_manifest": str(
                    args.phase_b_manifest.resolve()
                ),
                "record_count": len(records),
                "external_record_count": len(external_records),
                "external_benchmark_count": len(
                    {
                        record["benchmark"]
                        for record in external_records
                    }
                ),
                "artifacts": {
                    path.name: {
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in (jsonl_path, csv_path, split_path)
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    base_records = _load_jsonl(args.base_corpus)
    if not any(
        record.get("split") == "train"
        and (record.get("features") or {}).get(
            "setup_behavior_version"
        )
        == CORRECTED_VERSION
        and (record.get("eligibility") or {}).get("ranking_model")
        is True
        for record in base_records
    ):
        raise ValueError(
            "base corpus lacks ranking-eligible corrected-v2 training data"
        )
    external_records = _external_records(args)
    base_lineages = {
        str(record.get("benchmark_lineage") or "")
        for record in base_records
    }
    external_lineages = {
        str(record["benchmark_lineage"])
        for record in external_records
    }
    overlap = base_lineages & external_lineages
    if overlap:
        raise ValueError(
            f"external lineages overlap the base corpus: {sorted(overlap)}"
        )
    lineage_splits: dict[str, set[str]] = defaultdict(set)
    for record in [*base_records, *external_records]:
        lineage_splits[str(record["benchmark_lineage"])].add(
            str(record["split"])
        )
    leaked = {
        lineage: splits
        for lineage, splits in lineage_splits.items()
        if len(splits) > 1
    }
    if leaked:
        raise ValueError(f"lineage split leakage: {leaked}")
    combined = sorted(
        [*base_records, *external_records],
        key=lambda record: (
            record["split"],
            record["benchmark_lineage"],
            record["problem"],
            record["setup"]["setup_id"],
            record["record_kind"],
        ),
    )
    _write_outputs(combined, external_records, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-corpus", type=Path, required=True)
    parser.add_argument("--matrix-records", type=Path, required=True)
    parser.add_argument("--phase-b-manifest", type=Path, required=True)
    parser.add_argument("--benchmarks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

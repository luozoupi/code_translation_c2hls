#!/usr/bin/env python3
"""Export corrected setup-matrix results to schema-1.0 JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import export_schema_jsonl as schema


def export(args: argparse.Namespace) -> dict:
    matrix_records = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_records = []
    missing_results = []
    for benchmark_record in matrix_records:
        benchmark = str(benchmark_record["benchmark"])
        bench_dir = args.benchmarks_dir / benchmark
        for candidate in benchmark_record.get("candidates") or []:
            result_path = Path(str(candidate.get("result_path") or ""))
            if not result_path.is_file():
                missing_results.append(str(result_path))
                continue
            records = schema._records_from_multistep(
                bench_dir,
                result_path,
                args.part,
                args.clock_ns,
            )
            for record in records:
                implementation = record.get("implementation") or {}
                origin_meta = implementation.setdefault("origin_meta", {})
                origin_meta.update(
                    {
                        "setup_id": candidate.get("setup_id"),
                        "setup_fingerprint": candidate.get(
                            "setup_fingerprint"
                        ),
                        "setup_registry_version": "corrected_v2",
                        "tournament_candidate_valid": candidate.get(
                            "valid"
                        ),
                        "tournament_candidate_latency_cycles": candidate.get(
                            "latency_cycles"
                        ),
                        "tournament_winner": (
                            candidate.get("setup_id")
                            == benchmark_record.get("winner_setup_id")
                        ),
                        "reference_blind": True,
                    }
                )
                errors = schema._validate_record(record)
                if errors:
                    raise ValueError(
                        f"{result_path} produced invalid schema record: "
                        f"{errors}"
                    )
                output_records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "schema_records.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(schema._strict_json_dumps(record) + "\n")
    counts = Counter(record["report_type"] for record in output_records)
    manifest = {
        "schema_version": schema.SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_records": str(args.records.resolve()),
        "records": len(output_records),
        "counts_by_report_type": dict(counts),
        "invalid_records": 0,
        "missing_result_paths": missing_results,
        "cosim_records": counts.get("rtl_sim", 0),
        "output_file": output_path.name,
    }
    (args.output_dir / "schema_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "benchmarks_external"
            / "HLSFactory"
            / "polybench_float_small"
        ),
    )
    parser.add_argument("--part", default="xcu280-fsvh2892-2L-e")
    parser.add_argument("--clock-ns", type=float, default=3.33)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(export(parse_args()), sort_keys=True))

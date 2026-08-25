#!/usr/bin/env python3
"""Export skill-dose outcomes to the repository's schema-1.0 JSONL."""

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


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def export(args: argparse.Namespace) -> dict:
    outcomes = _load(args.records)
    output_records = []
    for outcome in outcomes:
        benchmark = str(outcome["benchmark"])
        meta = json.loads(
            (
                args.benchmarks_dir / benchmark / "metadata.json"
            ).read_text(encoding="utf-8")
        )
        count = int(outcome["requested_positive_skill_count"])
        sample = int(outcome["sample_index"])
        policy = str(outcome["prompt_policy"])
        variant_name = f"skill_dose_k{count}_{policy}_sample{sample}"
        origin_meta = {
            "experiment": "skill_dose_pilot_v1",
            "cell_id": outcome["cell_id"],
            "strategy": "flash",
            "sample_index": sample,
            "requested_positive_skill_count": count,
            "requested_positive_skill_ids": outcome.get(
                "requested_positive_skill_ids"
            )
            or [],
            "prompt_policy": policy,
            "skill_telemetry": outcome.get("skill_telemetry") or {},
            "phase_b_code_sha256": outcome.get("phase_b_code_sha256"),
            "router_ranking_sha256": outcome.get(
                "router_ranking_sha256"
            ),
            "reference_blind": True,
            "reference_reporting_only": outcome.get(
                "reference_reporting_only"
            )
            or {},
            "candidate_status": outcome.get("candidate_status"),
        }
        run_meta = {
            "vitis_version": "2023.2",
            "flow_target": "vitis",
            "clock_ns": 3.33,
        }
        implementation = schema._build_implementation(
            meta,
            variant_name=variant_name,
            variant_index=count * 10 + sample,
            origin_override="c2hls_orchestrator",
            origin_version="claude-sonnet-4-6",
            origin_meta=origin_meta,
        )
        status = (
            "pass"
            if outcome.get("valid")
            else "timeout"
            if "timeout" in str(outcome.get("candidate_status") or "").lower()
            else "fail"
        )
        output_records.append(
            {
                "schema_version": schema.SCHEMA_VERSION,
                "report_type": "hls_synth",
                "run": schema._build_run(
                    schema.TARGET_CSYNTH,
                    args.part,
                    outcome.get("elapsed_seconds"),
                    run_meta,
                ),
                "problem": schema._build_problem(meta),
                "implementation": implementation,
                "hls_synth": schema._build_hls_synth_payload(
                    outcome.get("candidate_csynth_report") or {},
                    args.part,
                    args.clock_ns,
                    status=status,
                ),
            }
        )
        csim = outcome.get("candidate_csim") or {}
        if csim.get("ran"):
            output_records.append(
                {
                    "schema_version": schema.SCHEMA_VERSION,
                    "report_type": "sw_run",
                    "run": schema._build_run(
                        schema.TARGET_CSIM,
                        args.part,
                        None,
                        run_meta,
                    ),
                    "problem": schema._build_problem(meta),
                    "implementation": implementation,
                    "sw_run": {
                        "status": (
                            "pass" if csim.get("passed") else "fail"
                        ),
                        "error": (
                            None
                            if csim.get("passed")
                            else str(outcome.get("result_error") or "")[:300]
                        ),
                    },
                }
            )

    invalid = []
    valid_records = []
    for index, record in enumerate(output_records, start=1):
        errors = schema._validate_record(record)
        if errors:
            invalid.append({"record": index, "errors": errors})
        else:
            valid_records.append(record)
    if invalid:
        raise ValueError(f"schema validation failed: {invalid[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "schema_records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in valid_records:
            handle.write(schema._strict_json_dumps(record) + "\n")
    counts = Counter(record["report_type"] for record in valid_records)
    manifest = {
        "schema_version": schema.SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_records": str(args.records.resolve()),
        "records": len(valid_records),
        "counts_by_report_type": dict(counts),
        "invalid_records": 0,
        "cosim_records": 0,
        "output_file": jsonl_path.name,
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

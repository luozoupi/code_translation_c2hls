#!/usr/bin/env python3
"""Verify a predictive cosim-timeout result and its canonical JSONL row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import export_schema_jsonl as schema_export  # noqa: E402


def _predicted_summaries(result: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = [
        result.get("cosim"),
        result.get("baseline_cosim"),
    ]
    for step in result.get("steps") or []:
        if isinstance(step, dict):
            candidates.append(step.get("cosim"))
    summaries = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        policy = candidate.get("cosim_policy") or {}
        if policy.get("classification") == "predicted_timeout":
            summaries.append(candidate)
    return summaries


def verify(result_json: Path, jsonl: Path) -> dict[str, Any]:
    result = json.loads(result_json.read_text())
    summaries = _predicted_summaries(result)
    summary_errors: list[str] = []
    for index, summary in enumerate(summaries):
        if summary.get("status") != "timeout":
            summary_errors.append(f"summary[{index}].status != timeout")
        if summary.get("ran") is not False:
            summary_errors.append(f"summary[{index}].ran is not false")
        if summary.get("skip_reason") != "predicted_longer_than_gold":
            summary_errors.append(
                f"summary[{index}].skip_reason != predicted_longer_than_gold"
            )

    schema_validation = schema_export.validate_jsonl(jsonl)
    matching_records: list[dict[str, Any]] = []
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("report_type") != "rtl_sim":
            continue
        payload = record.get("rtl_sim") or {}
        origin_meta = ((record.get("implementation") or {}).get("origin_meta") or {})
        policy = origin_meta.get("cosim_policy") or {}
        if (
            payload.get("status") == "timeout"
            and policy.get("classification") == "predicted_timeout"
        ):
            matching_records.append(record)

    record_errors: list[str] = []
    for index, record in enumerate(matching_records):
        payload = record["rtl_sim"]
        origin_meta = record["implementation"]["origin_meta"]
        if origin_meta.get("cosim_ran") is not False:
            record_errors.append(f"record[{index}].origin_meta.cosim_ran is not false")
        for field in (
            "kernel_runtime_cycles",
            "kernel_runtime_us",
            "kernel_clock_freq_mhz",
        ):
            if payload.get(field) is not None:
                record_errors.append(f"record[{index}].rtl_sim.{field} is not null")

    errors = []
    if not summaries:
        errors.append("no predictive-timeout summary found")
    if not matching_records:
        errors.append("no canonical predictive-timeout rtl_sim record found")
    if schema_validation.get("invalid"):
        errors.append(
            f"canonical JSONL has {schema_validation.get('invalid')} invalid record(s)"
        )
    errors.extend(summary_errors)
    errors.extend(record_errors)

    return {
        "ok": not errors,
        "result_json": str(result_json),
        "jsonl": str(jsonl),
        "predictive_timeout_summaries": len(summaries),
        "predictive_timeout_rtl_records": len(matching_records),
        "schema_records": schema_validation.get("total"),
        "schema_invalid": schema_validation.get("invalid"),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.result_json, args.jsonl)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

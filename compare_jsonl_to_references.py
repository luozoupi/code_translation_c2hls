#!/usr/bin/env python3
"""Compare schema JSONL records against results/references_philip keys."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_schema_jsonl import validate_jsonl


def _iter_records(path: Path):
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        yield lineno, record


def _key(record: dict) -> tuple:
    variant = record["implementation"]["variant"]
    return (
        tuple(record["problem"]["group_path"]),
        int(variant["index"]),
        variant["name"],
        record["report_type"],
    )


def _payload(record: dict) -> dict:
    return record.get(record["report_type"], {}) or {}


def _load_reference_records(ref_dir: Path) -> dict:
    refs = {}
    for path in sorted(ref_dir.glob("*.jsonl")):
        validation = validate_jsonl(path)
        if validation["invalid"]:
            raise SystemExit(f"reference JSONL failed validation: {path}")
        for _, record in _iter_records(path):
            refs[_key(record)] = record
    return refs


def _status_delta(candidate: dict, reference: dict | None) -> str:
    if reference is None:
        return "missing_ref"
    cand_status = _payload(candidate).get("status")
    ref_status = _payload(reference).get("status")
    return "same" if cand_status == ref_status else f"{ref_status}->{cand_status}"


def _cycle_delta(candidate: dict, reference: dict | None) -> str:
    if reference is None or candidate.get("report_type") != "rtl_sim":
        return "-"
    cand_cycles = _payload(candidate).get("kernel_runtime_cycles")
    ref_cycles = _payload(reference).get("kernel_runtime_cycles")
    if not isinstance(cand_cycles, int) or not isinstance(ref_cycles, int):
        return "-"
    return str(cand_cycles - ref_cycles)


def compare(candidate_paths: list[Path], ref_dir: Path, output: Path) -> dict:
    refs = _load_reference_records(ref_dir)
    lines = [
        "# JSONL Reference Delta",
        "",
        "| file | problem | variant | report_type | candidate_origin | status_delta | cycle_delta |",
        "|---|---|---:|:---:|---|---|---:|",
    ]
    total = 0
    matched = 0
    status_changed = 0
    missing_ref = 0

    for path in candidate_paths:
        validation = validate_jsonl(path)
        if validation["invalid"]:
            raise SystemExit(f"candidate JSONL failed validation: {path}")
        for _, record in _iter_records(path):
            total += 1
            key = _key(record)
            reference = refs.get(key)
            if reference is None:
                missing_ref += 1
            else:
                matched += 1
            delta = _status_delta(record, reference)
            if delta not in {"same", "missing_ref"}:
                status_changed += 1
            problem = "/".join(key[0])
            lines.append(
                f"| {path.name} | {problem} | {key[1]} {key[2]} | {key[3]} | "
                f"{record['implementation'].get('origin')} | {delta} | {_cycle_delta(record, reference)} |"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return {
        "candidate_records": total,
        "matched_reference_records": matched,
        "missing_reference_records": missing_ref,
        "status_changed_records": status_changed,
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="+", type=Path)
    parser.add_argument("--references", type=Path,
                        default=Path("results/references_philip"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/jsonl_reference_deltas.md"))
    args = parser.parse_args()
    summary = compare(args.candidate, args.references, args.output)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

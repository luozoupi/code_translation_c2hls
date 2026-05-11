#!/usr/bin/env python3
"""Build a repaired hw_emu reference candidate from targeted rerun records."""
from __future__ import annotations

import json
import os
from pathlib import Path

from export_schema_jsonl import validate_jsonl

REPO = Path(__file__).resolve().parent
REFERENCE = Path(os.getenv(
    "C2HLS_HWEMU_REFERENCE_JSONL",
    str(REPO / "results" / "references_philip" / "hw_emu_vitis_2023.2__device_xilinx_u280_gen3x16_xdma_1_202211_1.jsonl"),
))
RERUN = Path(os.getenv(
    "C2HLS_HWEMU_RERUN_JSONL",
    str(REPO / "artifacts" / "requested_hwemu_mismatch_rerun.jsonl"),
))
OUT = Path(os.getenv(
    "C2HLS_HWEMU_REFERENCE_CANDIDATE",
    str(REPO / "artifacts" / "hw_emu_reference_candidate_after_mismatch_rerun.jsonl"),
))


def _key(record: dict) -> tuple:
    variant = record["implementation"]["variant"]
    return (
        tuple(record["problem"]["group_path"]),
        int(variant["index"]),
        variant["name"],
        record["report_type"],
    )


def _load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def main() -> int:
    reference_records = _load_records(REFERENCE)
    rerun_records = _load_records(RERUN)
    rerun_by_key = {_key(record): record for record in rerun_records}

    replaced = []
    merged = []
    for record in reference_records:
        key = _key(record)
        replacement = rerun_by_key.get(key)
        if replacement is None:
            merged.append(record)
            continue
        merged.append(replacement)
        replaced.append({
            "problem": "/".join(key[0]),
            "variant_index": key[1],
            "variant_name": key[2],
            "old_status": record.get("rtl_sim", {}).get("status"),
            "new_status": replacement.get("rtl_sim", {}).get("status"),
            "old_cycles": record.get("rtl_sim", {}).get("kernel_runtime_cycles"),
            "new_cycles": replacement.get("rtl_sim", {}).get("kernel_runtime_cycles"),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        for record in merged:
            handle.write(json.dumps(record) + "\n")

    validation = validate_jsonl(OUT)
    print(json.dumps({
        "reference": str(REFERENCE),
        "rerun": str(RERUN),
        "output": str(OUT),
        "replaced": replaced,
        "total": validation["total"],
        "invalid": validation["invalid"],
    }))
    return 1 if validation["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

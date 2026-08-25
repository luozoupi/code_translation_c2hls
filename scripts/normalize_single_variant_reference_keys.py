#!/usr/bin/env python3
"""Normalize schema JSONL envelopes for single-variant benchmark corpora.

External datasets such as HLSFactory usually have one bundled reference
variant, `0 baseline`. Generated multistep records still carry the optimization
step in `implementation.origin_meta.step`, so `implementation.variant` should
remain the benchmark reference key for direct-reference joins.

The helper also fills reviewer-visible Vitis `UserAssignments` fields that can
be rehydrated from the target part/clock without rerunning synthesis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import export_schema_jsonl as schema  # noqa: E402


def _metadata_path(benchmarks_dir: Path, record: dict[str, Any]) -> Path | None:
    problem = record.get("problem") or {}
    group_path = problem.get("group_path") or []
    if not isinstance(group_path, list) or not group_path:
        return None
    candidate = benchmarks_dir / group_path[0] / "metadata.json"
    return candidate if candidate.exists() else None


def _normalize_record(record: dict[str, Any], benchmarks_dir: Path) -> bool:
    meta_path = _metadata_path(benchmarks_dir, record)
    if meta_path is None:
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    variants = meta.get("variants") or []
    if len(variants) != 1:
        return False

    impl = record.setdefault("implementation", {})
    old_variant = dict(impl.get("variant") or {})
    index, name = schema._variant_identity_for_step(meta, old_variant.get("name") or "")
    if old_variant.get("index") == index and old_variant.get("name") == name:
        return False

    origin_meta = impl.setdefault("origin_meta", {})
    if isinstance(origin_meta, dict):
        origin_meta.setdefault("optimization_step", origin_meta.get("step") or old_variant.get("name"))
        origin_meta.setdefault("reference_key_variant_before_fix", old_variant)
    impl["variant"] = {"index": index, "name": name}
    return True


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fill_user_assignments(record: dict[str, Any]) -> bool:
    if record.get("report_type") != "hls_synth":
        return False
    payload = record.get("hls_synth") or {}
    if payload.get("status") != "pass":
        return False
    assignments = payload.get("UserAssignments")
    if not isinstance(assignments, dict):
        return False

    changed = False
    part = assignments.get("Part") or (record.get("run") or {}).get("device")
    if not assignments.get("ProductFamily"):
        family = schema._product_family_for_part(part)
        if family:
            assignments["ProductFamily"] = family
            changed = True
    if not assignments.get("ClockUncertainty"):
        target = _as_float(assignments.get("TargetClockPeriod") or (record.get("run") or {}).get("clock_ns"))
        if target is not None:
            assignments["ClockUncertainty"] = f"{target * 0.27:.2f}"
            changed = True
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmarks-dir", type=Path, required=True)
    args = parser.parse_args()

    variant_changed = 0
    envelope_changed = 0
    total = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1
            if _normalize_record(record, args.benchmarks_dir):
                variant_changed += 1
            if _fill_user_assignments(record):
                envelope_changed += 1
            fout.write(schema._strict_json_dumps(record) + "\n")

    validation = schema.validate_jsonl(args.output)
    print(schema._strict_json_dumps({
        "input": str(args.input),
        "output": str(args.output),
        "records": total,
        "changed": variant_changed + envelope_changed,
        "envelope_changed": envelope_changed,
        "variant_changed": variant_changed,
        "schema_invalid": validation.get("invalid"),
    }, sort_keys=True))
    return 1 if validation.get("invalid") else 0


if __name__ == "__main__":
    raise SystemExit(main())

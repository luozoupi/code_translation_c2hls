#!/usr/bin/env python3
"""Build a revision-style HLSFactory comparison JSONL.

The website groups submitted runs by `implementation.origin_version`, so this
script keeps harness-level ablations there, and uses `implementation.variant`
only for code variants produced within that run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import export_schema_jsonl as schema  # noqa: E402


DEFAULT_BASE = REPO / "artifacts" / "u280__all_setups_per_revision_schema.jsonl"
DEFAULT_OUT = REPO / "artifacts" / "u280__all_requested_setups_per_revision_schema_20260615.jsonl"
U280_PART = "xcu280-fsvh2892-2L-e"
U280_CLOCK_NS = 3.33
U280_CLOCK_MHZ = 1000.0 / U280_CLOCK_NS
MULTISTEP_ORDER = ["tiling", "pipeline", "unroll", "doublebuffer", "coalescing"]


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    origin_version: str
    mode: str
    skills: str
    skills_variant: str


def _website_group(raw: str) -> str:
    return raw.removeprefix("hlsfactory_").replace("_", "-")


def _source_spec(raw: str) -> SourceSpec:
    parts = raw.split(",")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "--add-source must be path,origin_version,mode,skills,skills_variant"
        )
    path, origin_version, mode, skills, skills_variant = [part.strip() for part in parts]
    if mode not in {"flash", "multistep"}:
        raise argparse.ArgumentTypeError("source mode must be flash or multistep")
    if not origin_version:
        raise argparse.ArgumentTypeError("origin_version must be non-empty")
    return SourceSpec(Path(path), origin_version, mode, skills, skills_variant)


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    problem = record.get("problem") or {}
    implementation = record.get("implementation") or {}
    variant = implementation.get("variant") or {}
    return (
        record.get("report_type"),
        problem.get("suite"),
        tuple(problem.get("group_path") or []),
        implementation.get("origin"),
        implementation.get("origin_version"),
        variant.get("index"),
        variant.get("name"),
    )


def _step_name(record: dict[str, Any]) -> str:
    implementation = record.get("implementation") or {}
    meta = implementation.get("origin_meta") or {}
    variant = implementation.get("variant") or {}
    raw = meta.get("step") or meta.get("multistep_step") or variant.get("name") or ""
    raw = str(raw).strip()
    if raw.startswith("step_"):
        raw = raw.split("_", 2)[-1]
    return raw or "implementation"


def _normalize_problem(record: dict[str, Any]) -> None:
    problem = record.setdefault("problem", {})
    group_path = problem.get("group_path") or []
    if problem.get("suite") == "hlsfactory_polybench_float_small":
        return
    if group_path and str(group_path[0]).startswith("hlsfactory_"):
        problem["suite"] = "hlsfactory_polybench_float_small"
        problem["group_path"] = [_website_group(str(group_path[0]))]


def _repair_hls_assignments(record: dict[str, Any]) -> None:
    if record.get("report_type") != "hls_synth":
        return
    payload = record.get("hls_synth")
    if not isinstance(payload, dict):
        return
    assignments = payload.setdefault("UserAssignments", {})
    if not isinstance(assignments, dict):
        return
    part = assignments.get("Part") or (record.get("run") or {}).get("device") or U280_PART
    assignments.setdefault("ProductFamily", schema._product_family_for_part(str(part)))
    if assignments.get("ProductFamily") is None:
        assignments["ProductFamily"] = schema._product_family_for_part(str(part))
    target_clock = assignments.get("TargetClockPeriod") or str(U280_CLOCK_NS)
    try:
        clock_ns = float(str(target_clock).replace(",", "").strip())
    except ValueError:
        clock_ns = U280_CLOCK_NS
    assignments.setdefault("ClockUncertainty", f"{clock_ns * 0.27:.2f}")
    if assignments.get("ClockUncertainty") is None:
        assignments["ClockUncertainty"] = f"{clock_ns * 0.27:.2f}"


def _repair_rtl_timing(record: dict[str, Any]) -> None:
    if record.get("report_type") != "rtl_sim":
        return
    payload = record.get("rtl_sim")
    if not isinstance(payload, dict):
        return
    cycles = payload.get("kernel_runtime_cycles")
    if not isinstance(cycles, int) or cycles <= 0:
        return
    if payload.get("kernel_clock_freq_mhz") is None:
        payload["kernel_clock_freq_mhz"] = U280_CLOCK_MHZ
    if payload.get("kernel_runtime_us") is None:
        payload["kernel_runtime_us"] = cycles / float(payload["kernel_clock_freq_mhz"])


def _retarget_source_record(record: dict[str, Any], source: SourceSpec) -> dict[str, Any]:
    out = schema._json_safe(record)
    _normalize_problem(out)
    implementation = out.setdefault("implementation", {})
    old_origin_version = implementation.get("origin_version")
    old_variant = implementation.get("variant")
    meta = implementation.get("origin_meta")
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("model", old_origin_version or "unknown")
    meta["mode"] = source.mode
    meta["skills"] = source.skills
    meta["skills_variant"] = source.skills_variant
    meta.setdefault("source_jsonl", str(source.path))
    if old_origin_version not in (None, source.origin_version):
        meta.setdefault("origin_version_before_revision_export", old_origin_version)
    if isinstance(old_variant, dict):
        meta.setdefault("variant_before_revision_export", old_variant)

    implementation["origin"] = "c2hls_orchestrator"
    implementation["origin_version"] = source.origin_version
    implementation["origin_meta"] = meta

    if source.mode == "flash":
        meta.setdefault("phase", "phase_b_final")
        implementation["variant"] = {"index": 0, "name": "final"}
    else:
        step = _step_name(out)
        index = MULTISTEP_ORDER.index(step) if step in MULTISTEP_ORDER else len(MULTISTEP_ORDER)
        meta["phase"] = f"multistep_step_{index}_{step}"
        meta["multistep_step"] = step
        meta["multistep_step_index"] = index
        implementation["variant"] = {"index": index, "name": f"step_{index}_{step}"}

    _repair_hls_assignments(out)
    _repair_rtl_timing(out)
    return out


def _normalize_existing_record(record: dict[str, Any]) -> dict[str, Any]:
    out = schema._json_safe(record)
    _normalize_problem(out)
    _repair_hls_assignments(out)
    _repair_rtl_timing(out)
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as handle:
        for line in handle:
            if line.strip():
                records.append(schema._strict_json_loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(schema._strict_json_dumps(record) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-jsonl", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--add-source", action="append", type=_source_spec, default=[])
    args = parser.parse_args()

    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    source_counts: dict[str, int] = {}

    for record in _read_jsonl(args.base_jsonl):
        normalized = _normalize_existing_record(record)
        by_key[_record_key(normalized)] = normalized
    source_counts[str(args.base_jsonl)] = len(by_key)

    for source in args.add_source:
        source_records = _read_jsonl(source.path)
        written = 0
        for record in source_records:
            normalized = _retarget_source_record(record, source)
            by_key[_record_key(normalized)] = normalized
            written += 1
        source_counts[str(source.path)] = written

    records = sorted(
        by_key.values(),
        key=lambda record: (
            (record.get("problem") or {}).get("suite") or "",
            tuple((record.get("problem") or {}).get("group_path") or []),
            (record.get("implementation") or {}).get("origin") or "",
            (record.get("implementation") or {}).get("origin_version") or "",
            ((record.get("implementation") or {}).get("variant") or {}).get("index", 0),
            ((record.get("implementation") or {}).get("variant") or {}).get("name", ""),
            record.get("report_type") or "",
        ),
    )
    _write_jsonl(args.out_jsonl, records)
    validation = schema.validate_jsonl(args.out_jsonl)
    print(schema._strict_json_dumps({
        "out_jsonl": str(args.out_jsonl),
        "records": len(records),
        "schema_invalid": validation.get("invalid"),
        "sources": source_counts,
    }, sort_keys=True))
    return 1 if validation.get("invalid") else 0


if __name__ == "__main__":
    raise SystemExit(main())

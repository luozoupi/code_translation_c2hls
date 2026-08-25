#!/usr/bin/env python3
"""Export HLSFactory multistep results in the website upload shape.

This is intentionally different from the raw multistep trace JSONL:
- one selected/best generated implementation per benchmark;
- website-compatible HLSFactory problem keys;
- no duplicate `(problem, origin, origin_version, variant, report_type)` records;
- harness-level ablations such as skill usage are encoded in `origin_version`,
  not in `implementation.variant`;
- non-finite metadata values sanitized through the canonical strict JSON writer.
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


DEFAULT_STAMP = "hlsfactory_multistep_sonnet46_skill_on_trace_cosim1800_20260613"
DEFAULT_SUMMARY = REPO / "artifacts" / f"agentic_no_streamcluster_{DEFAULT_STAMP}.summary.json"
DEFAULT_COSIM = REPO / "artifacts" / "hlsfactory_multistep_best_cosim10800_final_20260614.referencekeyfix.schema.jsonl"
DEFAULT_BUNDLE_EXAMPLE = REPO / "artifacts" / "u280_all_phases_schema_laptop.jsonl"
DEFAULT_OUT = REPO / "artifacts" / "hlsfactory_multistep_sonnet46_skill_on_website_revstyle_20260615.jsonl"
DEFAULT_COMBINED_OUT = REPO / "artifacts" / "hlsfactory_multistep_sonnet46_skill_on_website_revstyle_combined_20260615.jsonl"
DEFAULT_ORIGIN_VERSION = "630ce11__multistep__skills_on"
DEFAULT_VARIANT_NAME = "final"
U280_CLOCK_MHZ = 1000.0 / 3.33


def _website_group_from_bench(bench: str) -> str:
    raw = bench.removeprefix("hlsfactory_")
    return raw.replace("_", "-")


def _load_bundle_groups(path: Path) -> set[str]:
    groups: set[str] = set()
    if not path.exists():
        return groups
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = schema._strict_json_loads(line)
            problem = record.get("problem") or {}
            if problem.get("suite") != "hlsfactory_polybench_float_small":
                continue
            group_path = problem.get("group_path") or []
            if len(group_path) == 1:
                groups.add(str(group_path[0]))
    return groups


def _load_baseline_records(path: Path, groups: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = schema._strict_json_loads(line)
            problem = record.get("problem") or {}
            variant = (record.get("implementation") or {}).get("variant") or {}
            if problem.get("suite") != "hlsfactory_polybench_float_small":
                continue
            group_path = problem.get("group_path") or []
            if len(group_path) != 1 or str(group_path[0]) not in groups:
                continue
            if variant.get("name") != "baseline":
                continue
            if (record.get("implementation") or {}).get("origin") != "hlsfactory_benchmark":
                continue
            records.append(_complete_rtl_timing(schema._json_safe(record)))
    return records


def _complete_rtl_timing(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("report_type") != "rtl_sim":
        return record
    payload = record.get("rtl_sim") or {}
    cycles = payload.get("kernel_runtime_cycles")
    if payload.get("status") == "pass" and isinstance(cycles, int) and cycles > 0:
        payload.setdefault("kernel_clock_freq_mhz", U280_CLOCK_MHZ)
        if payload.get("kernel_clock_freq_mhz") is None:
            payload["kernel_clock_freq_mhz"] = U280_CLOCK_MHZ
        payload.setdefault("kernel_runtime_us", cycles / U280_CLOCK_MHZ)
        if payload.get("kernel_runtime_us") is None:
            payload["kernel_runtime_us"] = cycles / U280_CLOCK_MHZ
    return record


def _retarget_record(
    record: dict[str, Any],
    *,
    bench: str,
    origin_version: str,
    variant_index: int,
    variant_name: str,
) -> dict[str, Any]:
    out = schema._json_safe(record)
    group = _website_group_from_bench(bench)
    out["problem"] = {
        "suite": "hlsfactory_polybench_float_small",
        "group_path": [group],
    }
    impl = out.setdefault("implementation", {})
    origin_meta = impl.get("origin_meta")
    if not isinstance(origin_meta, dict):
        origin_meta = {}
    previous_variant = impl.get("variant")
    if isinstance(previous_variant, dict):
        origin_meta.setdefault("selected_step_variant_before_export", previous_variant)
    if impl.get("origin_version") not in (None, origin_version):
        origin_meta.setdefault("origin_version_before_export", impl.get("origin_version"))
    impl["origin_version"] = origin_version
    impl["origin_meta"] = origin_meta
    impl["variant"] = {"index": int(variant_index), "name": variant_name}
    return out


def _best_step_records(
    row: dict[str, Any],
    *,
    origin_version: str,
    variant_index: int,
    variant_name: str,
) -> list[dict[str, Any]]:
    bench = row.get("bench") or ""
    cur = row.get("current") or {}
    best = cur.get("best") or {}
    best_step = best.get("step")
    result_path = Path(cur.get("json") or "")
    bench_dir = Path(row.get("bench_dir") or "")
    if not best_step or not result_path.exists() or not bench_dir.exists():
        return []

    out: list[dict[str, Any]] = []
    for record in schema._records_from_multistep(
        bench_dir,
        result_path,
        default_part="xcu280-fsvh2892-2L-e",
        default_clock_ns=3.33,
    ):
        if record.get("report_type") not in {"hls_synth", "sw_run"}:
            continue
        origin_meta = (record.get("implementation") or {}).get("origin_meta") or {}
        if origin_meta.get("step") != best_step:
            continue
        out.append(_retarget_record(
            record,
            bench=bench,
            origin_version=origin_version,
            variant_index=variant_index,
            variant_name=variant_name,
        ))
    return out


def _load_cosim_records(
    path: Path,
    *,
    origin_version: str,
    variant_index: int,
    variant_name: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = schema._strict_json_loads(line)
            group_path = (record.get("problem") or {}).get("group_path") or []
            if not group_path:
                continue
            bench = str(group_path[0])
            records[bench] = _retarget_record(
                record,
                bench=bench,
                origin_version=origin_version,
                variant_index=variant_index,
                variant_name=variant_name,
            )
    return records


def _key(record: dict[str, Any]) -> tuple[Any, ...]:
    problem = record.get("problem") or {}
    implementation = record.get("implementation") or {}
    variant = implementation.get("variant") or {}
    return (
        problem.get("suite"),
        tuple(problem.get("group_path") or []),
        implementation.get("origin"),
        implementation.get("origin_version"),
        variant.get("index"),
        variant.get("name"),
        record.get("report_type"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cosim-jsonl", type=Path, default=DEFAULT_COSIM)
    parser.add_argument("--bundle-example-jsonl", type=Path, default=DEFAULT_BUNDLE_EXAMPLE)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--combined-out-jsonl", type=Path, default=DEFAULT_COMBINED_OUT)
    parser.add_argument("--origin-version", default=DEFAULT_ORIGIN_VERSION)
    parser.add_argument("--variant-index", type=int, default=0)
    parser.add_argument("--variant-name", default=DEFAULT_VARIANT_NAME)
    parser.add_argument("--include-unbundled", action="store_true")
    parser.add_argument("--no-combined", action="store_true")
    args = parser.parse_args()

    summary = json.loads(args.summary_json.read_text())
    bundle_groups = _load_bundle_groups(args.bundle_example_jsonl)
    cosim_by_bench = _load_cosim_records(
        args.cosim_jsonl,
        origin_version=args.origin_version,
        variant_index=args.variant_index,
        variant_name=args.variant_name,
    )

    records: list[dict[str, Any]] = []
    skipped_unbundled: list[str] = []
    skipped_failed: list[str] = []
    for row in summary.get("rows") or []:
        bench = row.get("bench") or ""
        group = _website_group_from_bench(bench)
        if bundle_groups and group not in bundle_groups and not args.include_unbundled:
            skipped_unbundled.append(bench)
            continue
        if not ((row.get("current") or {}).get("success")):
            skipped_failed.append(bench)
            continue
        records.extend(_best_step_records(
            row,
            origin_version=args.origin_version,
            variant_index=args.variant_index,
            variant_name=args.variant_name,
        ))
        cosim = cosim_by_bench.get(bench)
        if cosim:
            records.append(cosim)

    seen: set[tuple[Any, ...]] = set()
    duplicate_keys: list[tuple[Any, ...]] = []
    for record in records:
        key = _key(record)
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)
    if duplicate_keys:
        raise SystemExit(f"duplicate website keys: {duplicate_keys[:10]}")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as handle:
        for record in records:
            handle.write(schema._strict_json_dumps(record) + "\n")

    validation = schema.validate_jsonl(args.out_jsonl)
    combined_records: list[dict[str, Any]] = []
    combined_validation = {"invalid": 0}
    if not args.no_combined:
        generated_groups = {
            (record.get("problem") or {}).get("group_path", [""])[0]
            for record in records
            if (record.get("problem") or {}).get("group_path")
        }
        baseline_records = _load_baseline_records(args.bundle_example_jsonl, {str(group) for group in generated_groups})
        combined_records = [*baseline_records, *records]
        args.combined_out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.combined_out_jsonl.open("w") as handle:
            for record in combined_records:
                handle.write(schema._strict_json_dumps(record) + "\n")
        combined_validation = schema.validate_jsonl(args.combined_out_jsonl)
    print(schema._strict_json_dumps({
        "out_jsonl": str(args.out_jsonl),
        "records": len(records),
        "unique_keys": len(seen),
        "schema_invalid": validation.get("invalid"),
        "combined_out_jsonl": None if args.no_combined else str(args.combined_out_jsonl),
        "combined_records": len(combined_records),
        "combined_schema_invalid": combined_validation.get("invalid"),
        "skipped_unbundled": skipped_unbundled,
        "skipped_failed": skipped_failed,
    }, sort_keys=True))
    return 1 if validation.get("invalid") or combined_validation.get("invalid") else 0


if __name__ == "__main__":
    raise SystemExit(main())

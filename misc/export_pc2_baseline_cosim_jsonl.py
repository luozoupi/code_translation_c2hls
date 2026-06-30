#!/usr/bin/env python3
"""Append baseline cosim rtl_sim records to an existing naive baseline JSONL.

Only pairs csynth/csim from benchmarks/hls_baseline.cpp with cosim from the
matching naive baseline campaign (benchmarks/ corpus). Refuses benchmarks_cosim
unless --allow-fixed-corpus is passed explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pc2"))

from export_schema_jsonl import (  # noqa: E402
    SCHEMA_VERSION,
    TARGET_COSIM,
    validate_jsonl,
)
from flash_cosim_lib import classify_cosim_outcome  # noqa: E402

PC2 = REPO / "artifacts" / "pc2"
DEFAULT_BASELINE_JSONL = REPO / "misc" / "hlsfactory_baseline_u280_20260616_benchmarks.jsonl"
DEFAULT_FIXED_BASELINE_JSONL = REPO / "misc" / "hlsfactory_cosim_baseline_u280_20260616_benchmarks.jsonl"
DEFAULT_NAIVE_COSIM_RUN = PC2 / "baseline_cosim" / "20260626_022917_top5_full_indiv"
DEFAULT_OUTPUT = REPO / "misc" / "hlsfactory_baseline_u280_20260616_benchmarks_naive_cosim.jsonl"
DEFAULT_FIXED_OUTPUT = REPO / "misc" / "hlsfactory_baseline_u280_20260616_benchmarks_full_cosim.jsonl"
FIXED_COSIM_RUN = PC2 / "baseline_cosim" / "20260626_063907_fixed_cosim_benchmark"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def _group_key(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(record.get("problem", {}).get("group_path") or [])


def _bench_group_path(bench: str) -> list[str]:
    short = bench.removeprefix("hlsfactory_").replace("-", "_")
    return [short]


def _cosim_status(result: dict[str, Any]) -> str:
    return classify_cosim_outcome(result)


def _manifest_corpus(manifest: dict[str, Any], cosim_run_root: Path) -> str:
    corpus = (manifest.get("corpus") or "").strip()
    if corpus:
        return corpus
    artifact_dir = str(manifest.get("benchmarks_root") or manifest.get("cells", [{}])[0].get("artifact_dir") or "")
    if "benchmarks_cosim" in artifact_dir:
        return "benchmarks_cosim"
    if "/benchmarks" in artifact_dir and "benchmarks_cosim" not in artifact_dir:
        return "benchmarks"
    if "fixed_cosim_benchmark" in str(cosim_run_root):
        return "benchmarks_cosim"
    if "top5_full_indiv" in str(cosim_run_root):
        return "benchmarks"
    return "unknown"


def _kernel_file_for_corpus(corpus: str) -> str:
    return "hls_baseline_cosim.cpp" if corpus == "benchmarks_cosim" else "hls_baseline.cpp"


def _cosim_origin_version_suffix(corpus: str) -> str:
    if corpus == "benchmarks_cosim":
        return "fixed_cosim"
    if corpus == "benchmarks":
        return "naive_cosim"
    return "cosim"


def _apply_origin_version_suffix(record: dict[str, Any], suffix: str) -> dict[str, Any]:
    rec = json.loads(json.dumps(record))
    impl = rec.setdefault("implementation", {})
    current = str(impl.get("origin_version") or "").strip()
    tagged_suffix = f"_{suffix}"
    if current.endswith(tagged_suffix):
        tagged = current
    elif current:
        tagged = f"{current}{tagged_suffix}"
    else:
        tagged = suffix
    impl["origin_version"] = tagged
    meta = dict(impl.get("origin_meta") or {})
    meta["cosim_export_suffix"] = suffix
    impl["origin_meta"] = meta
    return rec


def _index_baseline_templates(records: list[dict[str, Any]]) -> dict[tuple[str, ...], dict[str, Any]]:
    templates: dict[tuple[str, ...], dict[str, Any]] = {}
    for rec in records:
        if rec.get("report_type") != "hls_synth":
            continue
        key = _group_key(rec)
        if key:
            templates[key] = rec
    return templates


def _rtl_sim_record(
    *,
    template: dict[str, Any],
    cosim: dict[str, Any],
    cosim_run_root: Path,
    cell_id: str,
    bench: str,
    corpus: str,
) -> dict[str, Any]:
    status = _cosim_status(cosim)
    impl = json.loads(json.dumps(template.get("implementation") or {}))
    origin_meta = dict(impl.get("origin_meta") or {})
    origin_meta.update({
        "cosim_run_root": str(cosim_run_root),
        "cosim_cell_id": cell_id,
        "cosim_work_dir": cosim.get("work_dir"),
        "cosim_kernel_file": _kernel_file_for_corpus(corpus),
        "cosim_corpus": corpus,
        "cosim_size_mode": cosim.get("cosim_size_mode"),
        "error": (cosim.get("error") or "")[:300] if status != "pass" else None,
    })
    impl["origin_meta"] = origin_meta

    run = dict(template.get("run") or {})
    run["target"] = TARGET_COSIM
    if cosim.get("runtime_seconds") is not None:
        run["runtime_seconds"] = cosim.get("runtime_seconds")

    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "rtl_sim",
        "run": run,
        "problem": template.get("problem") or {"suite": "hlsfactory", "group_path": _bench_group_path(bench)},
        "implementation": impl,
        "rtl_sim": {
            "status": status,
            "kernel_runtime_cycles": cosim.get("kernel_runtime_cycles"),
            "kernel_runtime_us": cosim.get("kernel_runtime_us"),
            "kernel_clock_freq_mhz": cosim.get("kernel_clock_freq_mhz"),
            "error": (cosim.get("error") or "")[:300] if status != "pass" else None,
        },
    }


def export_baseline_with_cosim(
    baseline_jsonl: Path,
    cosim_run_root: Path,
    *,
    allow_fixed_corpus: bool = False,
    origin_version_suffix: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_records = _read_jsonl(baseline_jsonl)
    templates = _index_baseline_templates(baseline_records)

    manifest_path = cosim_run_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    corpus = _manifest_corpus(manifest, cosim_run_root)

    if corpus == "benchmarks_cosim" and not allow_fixed_corpus:
        raise ValueError(
            f"refusing benchmarks_cosim cosim run {cosim_run_root.name}; "
            "pass --allow-fixed-corpus only when you intend to mix fixed-corpus cosim"
        )

    rtl_records: list[dict[str, Any]] = []
    missing_template: list[str] = []
    missing_cosim: list[str] = []

    for cell in manifest.get("cells", []):
        bench = cell.get("bench") or ""
        cell_id = cell.get("cell_id") or ""
        if not bench or not cell_id:
            continue

        group_path = tuple(_bench_group_path(bench))
        template = templates.get(group_path)
        if not template:
            missing_template.append(bench)
            continue

        cosim_path = cosim_run_root / "cells" / cell_id / "cosim_result.json"
        if not cosim_path.is_file():
            missing_cosim.append(bench)
            continue

        cosim = json.loads(cosim_path.read_text())
        rtl_records.append(
            _rtl_sim_record(
                template=template,
                cosim=cosim,
                cosim_run_root=cosim_run_root,
                cell_id=cell_id,
                bench=bench,
                corpus=corpus,
            )
        )

    suffix = (origin_version_suffix or _cosim_origin_version_suffix(corpus)).strip()
    if not suffix:
        raise ValueError("origin_version_suffix must be non-empty")
    out_records = [
        _apply_origin_version_suffix(rec, suffix)
        for rec in baseline_records + rtl_records
    ]
    meta = {
        "baseline_jsonl": str(baseline_jsonl),
        "cosim_run_root": str(cosim_run_root),
        "cosim_corpus": corpus,
        "origin_version_suffix": suffix,
        "baseline_records": len(baseline_records),
        "rtl_sim_added": len(rtl_records),
        "total_records": len(out_records),
        "missing_template": missing_template,
        "missing_cosim": missing_cosim,
    }
    return out_records, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-jsonl", type=Path, default=DEFAULT_BASELINE_JSONL)
    parser.add_argument(
        "--cosim-run-root",
        type=Path,
        default=DEFAULT_NAIVE_COSIM_RUN,
        help="Naive baseline cosim campaign (benchmarks/hls_baseline.cpp)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-fixed-corpus",
        action="store_true",
        help="Allow benchmarks_cosim cosim (off by default)",
    )
    parser.add_argument(
        "--origin-version-suffix",
        default=None,
        help="Append to implementation.origin_version on all exported records "
        "(default: naive_cosim for benchmarks corpus, fixed_cosim for benchmarks_cosim)",
    )
    args = parser.parse_args()

    records, meta = export_baseline_with_cosim(
        args.baseline_jsonl,
        args.cosim_run_root,
        allow_fixed_corpus=args.allow_fixed_corpus,
        origin_version_suffix=args.origin_version_suffix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    validation = validate_jsonl(args.output, verbose=True)
    rtl_with_cycles = sum(
        1 for r in records
        if r.get("report_type") == "rtl_sim"
        and r.get("rtl_sim", {}).get("kernel_runtime_cycles") is not None
    )
    by_type: dict[str, int] = {}
    rtl_by_status: dict[str, int] = {}
    for rec in records:
        rt = rec.get("report_type", "")
        by_type[rt] = by_type.get(rt, 0) + 1
        if rt == "rtl_sim":
            s = rec.get("rtl_sim", {}).get("status", "")
            rtl_by_status[s] = rtl_by_status.get(s, 0) + 1

    summary = {
        **meta,
        "output": str(args.output),
        "by_report_type": by_type,
        "rtl_sim_by_status": rtl_by_status,
        "rtl_sim_with_kernel_runtime_cycles": rtl_with_cycles,
        "validation": validation,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if validation.get("invalid", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

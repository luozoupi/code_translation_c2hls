#!/usr/bin/env python3
"""Normalize ML4Accel-Dataset design-space rows into metric_points.jsonl.

ML4Accel-Dataset is a multi-vendor FPGA/ASIC design-space dataset. For RL
supervision we ingest two of its richest sources:

  1. `hls_experiments/data/*.csv`  — row-per-design-point with HLS + impl
                                       metrics across ~11 Polybench/CHStone/
                                       MachSuite kernels.
  2. `fpga_ml_dataset/HLS_dataset/hlsyn/designs/*.json`  — pragma-point ->
                                       res_util mappings for 28 kernels,
                                       including synth estimates.

Output schema (one JSON object per JSONL line):
  {
    "source":          "ml4accel/<filename>",
    "kernel":          normalized kernel id (lowercase, stable),
    "benchmark":       optional — set when kernel matches one of our 17,
    "split":           optional — inherited from benchmark fixed splits,
    "design_point_id": unique identifier within the source,
    "pragma_choices":  {pragma name: value} — populated for JSON source,
                       empty for CSV sources where pragma info isn't
                       readily recoverable from the row id,
    "target_part":     Vitis/Vivado part id or null,
    "target_clock_ns": float or null,
    "metrics":         {latency_cycles, clock_estimated_ns, bram, dsp, ff,
                        lut, uram, power_total_mw},
    "device_util_pct": {bram, dsp, ff, lut} against target_part when known
  }

Rows missing critical fields (no kernel, no metrics) are skipped with a
counter increment; --verbose prints per-source skip counts.

Usage:
    python export_ml4accel_points.py
    python export_ml4accel_points.py --output artifacts/rl_corpus
    python export_ml4accel_points.py --dataset-root /path/to/ML4Accel-Dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent

from c2hls_paths import configure_site, ml4accel_repo_root

configure_site()

# Kernel name alias map: raw dataset string -> canonical lowercase form.
# Canonical names align with our 17-benchmark suite where possible.
KERNEL_ALIASES = {
    # design_space.csv names
    "aes256_encrypt_ecb": "aes",
    "Gsm_LPC_Analysis": "gsm",
    "gsm": "gsm",
    "needwun": "nw",
    "spmv": "spmv_crs",
    "stencil": "stencil2D",
    "stencil3d": "stencil3d",
    "bbgemm": "gemm_ncubed",
    "gemm": "gemm",
    "md_kernel": "md_knn",
    "md": "md_knn",
    "ms_mergesort": "sort_merge",
    "ss_sort": "sort_merge",
    "sha_stream": "sha",
    "viterbi": "viterbi",
    "bfs": "bfs",
    "atax": "atax",
    "bicg": "bicg",
    "syr2k": "syr2k",
    "syrk": "syrk",
    "gesummv": "gesummv",
    "ellpack": "ellpack",
    "mvt": "mvt",
    "k2mm": "k2mm",
    "k3mm": "k3mm",
    # hlsyn JSON stems (hyphenated)
    "gemm-ncubed": "gemm_ncubed",
    "gemm-blocked": "gemm",
    "gemm-p": "gemm",
    "gemm-p-large": "gemm",
    "spmv-crs": "spmv_crs",
    "spmv-ellpack": "spmv_crs",
    "stencil-3d": "stencil3d",
    "bicg-large": "bicg",
    "fdtd-2d-large": "fdtd_2d",
    "doitgen-red": "doitgen",
    "symm-opt": "symm",
    "trmm-opt": "trmm",
    "2mm": "k2mm",
    # merged_post_hls_poly_mach.csv prj-prefix fallbacks (best-effort)
    "io": "machsuite_io",  # MachSuite IO kernels — ambiguous, flagged as aux
}

# From our fixed splits — populated at import from the project splits.json
# when present, otherwise hard-coded here.
_FIXED_SPLITS = {
    "val":  {"StreamCluster", "viterbi"},
    "test": {"nw", "spmv_crs"},
}

OUR_BENCHMARKS = {
    "StreamCluster", "aes", "fft", "gemm_ncubed", "hotspot", "kmeans", "knn",
    "lavaMD", "lud", "md_knn", "nw", "pathfinder", "sort_merge", "spmv_crs",
    "srad", "stencil2D", "viterbi",
}


def _canonical_kernel(raw: str) -> str:
    """Map a raw kernel/benchmark name to a canonical lowercase id."""
    if not raw:
        return ""
    return KERNEL_ALIASES.get(raw, raw.strip().lower())


def _split_for(benchmark: str) -> Optional[str]:
    if benchmark in _FIXED_SPLITS["val"]:
        return "val"
    if benchmark in _FIXED_SPLITS["test"]:
        return "test"
    if benchmark in OUR_BENCHMARKS:
        return "train_aux"  # matches user's RL plan: ML4Accel rows never in final test
    return None


def _device_limits(part: str) -> Optional[dict]:
    """Reuse rubric._DEVICE_TABLE for prefix lookup."""
    try:
        import sys as _sys
        _sys.path.insert(0, str(REPO_ROOT))
        from rubric import _DEVICE_TABLE, _device_limits_for_part  # type: ignore
    except Exception:
        return None
    if not part:
        return None
    limits = _device_limits_for_part(part)
    # _device_limits_for_part falls back to xc7a100t when unknown; we want
    # an explicit null for unknown parts so downstream can tell.
    part_lc = part.lower()
    for key in _DEVICE_TABLE:
        if part_lc.startswith(key):
            return limits
    return None


def _util_pct(value, limit) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if limit is None or limit <= 0:
        return None
    return round(100.0 * v / limit, 3)


def _try_int(s) -> Optional[int]:
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _try_float(s) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _nonempty(s) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    return s or None


def _finalize(record: dict, kernel_raw: str, part: Optional[str]) -> dict:
    """Attach benchmark/split/device_util_pct if derivable. Mutates in place."""
    kernel = _canonical_kernel(kernel_raw)
    record["kernel"] = kernel
    if kernel in OUR_BENCHMARKS:
        record["benchmark"] = kernel
        split = _split_for(kernel)
        if split:
            record["split"] = split
    # device_util_pct
    limits = _device_limits(part) if part else None
    if limits:
        metrics = record.get("metrics", {})
        record["device_util_pct"] = {
            "bram": _util_pct(metrics.get("bram"), limits.get("bram")),
            "dsp":  _util_pct(metrics.get("dsp"),  limits.get("dsp")),
            "ff":   _util_pct(metrics.get("ff"),   limits.get("ff")),
            "lut":  _util_pct(metrics.get("lut"),  limits.get("lut")),
        }
    return record


# ─── Ingesters ───────────────────────────────────────────────────────────────

def _ingest_design_space(path: Path) -> Iterable[dict]:
    """design_space.csv — verbose 55-col schema with HLS+impl metrics."""
    source = f"ml4accel/{path.name}"
    with path.open() as f:
        for row in csv.DictReader(f):
            kernel_raw = row.get("name") or row.get("dataset_name") or ""
            part = _nonempty(row.get("part"))
            target_clock = _try_float(row.get("target_clock_period"))
            metrics = {
                "latency_cycles":       _try_int(row.get("hls_synth__latency_worst_cycles")),
                "clock_estimated_ns":   _try_float(row.get("hls_synth__clock_period"))
                                          and _try_float(row.get("hls_synth__clock_period")) * 1e9
                                          if row.get("hls_synth__clock_period") else None,
                "bram":                 _try_int(row.get("hls_synth__resources_bram_used")),
                "dsp":                  _try_int(row.get("hls_synth__resources_dsp_used")),
                "ff":                   _try_int(row.get("hls_synth__resources_ff_used")),
                "lut":                  _try_int(row.get("hls_synth__resources_lut_used")),
                "uram":                 _try_int(row.get("hls_synth__resources_uram_used")),
                "power_total_mw":       _try_float(row.get("impl__power__total_power")),
            }
            # The HLS clock_period comes in seconds in this CSV (e.g. "7.26e-09"
            # means 7.26 ns). Convert to ns for consistency.
            clk_s = _try_float(row.get("hls_synth__clock_period"))
            metrics["clock_estimated_ns"] = round(clk_s * 1e9, 3) if clk_s else None

            if not kernel_raw or all(v is None for v in metrics.values()):
                continue

            record = {
                "source": source,
                "design_point_id": row.get("name_unique") or kernel_raw,
                "pragma_choices": {},  # not recoverable from this CSV
                "target_part": part,
                "target_clock_ns": target_clock,
                "metrics": metrics,
            }
            yield _finalize(record, kernel_raw, part)


def _ingest_merged_post_hls(path: Path) -> Iterable[dict]:
    """merged_post_hls_poly_mach.csv — compact 14-col schema."""
    source = f"ml4accel/{path.name}"
    # No kernel column — the prj id encodes kernel + pragmas together.
    # For normalization, extract the leading token before the first digit.
    import re
    kernel_prefix_re = re.compile(r"^([a-zA-Z_]+)")

    with path.open() as f:
        for row in csv.DictReader(f):
            prj = row.get("prj") or ""
            m = kernel_prefix_re.match(prj)
            kernel_raw = m.group(1) if m else prj
            # Strip trailing "_" if present
            kernel_raw = kernel_raw.rstrip("_")
            if not kernel_raw:
                continue

            metrics = {
                "latency_cycles":     None,   # not in this CSV
                "clock_estimated_ns": _try_float(row.get("clk_estimated(ns)")),
                "bram":               _try_int(row.get("bram")),
                "dsp":                _try_int(row.get("dsp")),
                "ff":                 _try_int(row.get("ff")),
                "lut":                _try_int(row.get("lut")),
                "uram":               None,
                "power_total_mw":     None,
            }
            if all(v is None for v in metrics.values()):
                continue

            record = {
                "source": source,
                "design_point_id": prj,
                "pragma_choices": {},  # encoded in prj but not parsed here
                "target_part": None,   # this CSV doesn't record part
                "target_clock_ns": _try_float(row.get("clk_target(ns)")),
                "metrics": metrics,
            }
            yield _finalize(record, kernel_raw, None)


def _ingest_hlsyn_json(path: Path) -> Iterable[dict]:
    """hlsyn/designs/<kernel>.json — pragma-point -> res_util mapping."""
    source = f"ml4accel/hlsyn/designs/{path.name}"
    kernel_raw = path.stem  # "aes.json" -> "aes"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return

    for design_point_id, point in data.items():
        if not isinstance(point, dict):
            continue
        res_util = point.get("res_util") or {}
        metrics = {
            "latency_cycles":     _try_float(point.get("perf")),  # stored as float
            "clock_estimated_ns": None,
            "bram":               _try_float(res_util.get("total-BRAM")),
            "dsp":                _try_float(res_util.get("total-DSP")),
            "ff":                 _try_float(res_util.get("total-FF")),
            "lut":                _try_float(res_util.get("total-LUT")),
            "uram":               None,
            "power_total_mw":     None,
        }
        if all(v is None or v == 0 for v in metrics.values()):
            continue

        record = {
            "source": source,
            "design_point_id": design_point_id,
            "pragma_choices": dict(point.get("point") or {}),
            "target_part": None,  # hlsyn JSONs don't record part
            "target_clock_ns": None,
            "metrics": metrics,
            "valid": bool(point.get("valid", True)),
        }
        yield _finalize(record, kernel_raw, None)


# ─── Main ────────────────────────────────────────────────────────────────────

def export(dataset_root: Path, output_dir: Path, verbose: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metric_points.jsonl"

    ingesters = [
        (dataset_root / "hls_experiments/data/design_space.csv", _ingest_design_space),
        (dataset_root / "hls_experiments/data/merged_post_hls_poly_mach.csv", _ingest_merged_post_hls),
    ]
    json_dir = dataset_root / "fpga_ml_dataset/HLS_dataset/hlsyn/designs"

    counts_by_source: dict[str, int] = {}
    counts_by_kernel: dict[str, int] = {}
    total = 0

    with out_path.open("w") as out:
        for path, fn in ingesters:
            if not path.exists():
                if verbose:
                    print(f"skip (missing): {path}", file=sys.stderr)
                continue
            for rec in fn(path):
                out.write(json.dumps(rec) + "\n")
                total += 1
                counts_by_source[rec["source"]] = counts_by_source.get(rec["source"], 0) + 1
                counts_by_kernel[rec["kernel"]] = counts_by_kernel.get(rec["kernel"], 0) + 1

        if json_dir.is_dir():
            for jpath in sorted(json_dir.glob("*.json")):
                for rec in _ingest_hlsyn_json(jpath):
                    out.write(json.dumps(rec) + "\n")
                    total += 1
                    counts_by_source[rec["source"]] = counts_by_source.get(rec["source"], 0) + 1
                    counts_by_kernel[rec["kernel"]] = counts_by_kernel.get(rec["kernel"], 0) + 1
        elif verbose:
            print(f"skip (missing): {json_dir}", file=sys.stderr)

    manifest = {
        "version": 1,
        "corpus_type": "metric_points",
        "record_count": total,
        "counts_by_source": counts_by_source,
        "counts_by_kernel": {k: v for k, v in sorted(counts_by_kernel.items(), key=lambda x: -x[1])},
        "benchmarks_in_corpus": sorted(set(counts_by_kernel) & OUR_BENCHMARKS),
        "output_file": out_path.name,
    }
    manifest_path = output_dir / "metric_points_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    default_root = ml4accel_repo_root()
    p.add_argument(
        "--dataset-root",
        default=str(default_root) if default_root is not None else None,
        help="ML4Accel-Dataset root (or set C2HLS_ML4ACCEL_ROOT in local.env)",
    )
    p.add_argument("--output", default=str(REPO_ROOT / "artifacts" / "rl_corpus"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if not args.dataset_root:
        print("Set C2HLS_ML4ACCEL_ROOT in local.env or pass --dataset-root", file=sys.stderr)
        return 2

    manifest = export(Path(args.dataset_root), Path(args.output), verbose=args.verbose)
    print(f"wrote {manifest['record_count']} metric_points records "
          f"across {len(manifest['counts_by_source'])} sources")
    print(f"  overlap with our 17 benchmarks: {manifest['benchmarks_in_corpus']}")
    print(f"  manifest: {args.output}/metric_points_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

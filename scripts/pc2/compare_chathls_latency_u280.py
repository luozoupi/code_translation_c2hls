#!/usr/bin/env python3
"""Compare c2hls campaign ChatHLS-bench latency against a ChatHLS-native U280 baseline.

For each c2hls campaign directory, this walks the 14 overlapping ChatHLS
benches (the PolyBench-style set shared with the ChatHLS U280 baseline run;
``mobilenet``/``transformer`` are only included if the baseline CSV marks
them as a passed optimization), locates the newest result "cell" under
``variants/chathls_aav_n/chathls_<bench>/``, and takes the best available
latency in cycles across:

  * ``chathls_<bench>_selected_report.json``  -> ``latency_cycles``
  * ``chathls_<bench>_flow_manifest.json``    -> ``latency_cycles`` (flash stages)
  * ``chathls_<bench>_latency_opt_result.json`` / ``_report.json``
  * ``chathls_<bench>_dataflow_latency_opt_result.json`` / ``_report.json``
  * ``chathls_<bench>_dataflow_result.json``  -> ``latency_cycles`` (only if ``success``)

Also returns the winning stage name (e.g. ``dataflow_latency_opt``) for labeling.

The ``latency_cycles`` field can appear as a plain number or as a nested
dict (e.g. ``{"phase_b": ..., "flash_opt": ..., "selected": ...}``); both
shapes are handled.

Usage:

    python scripts/pc2/compare_chathls_latency_u280.py \\
      --chat-hls-csv PATH/final_latency_csynth.csv \\
      --campaign NAME=PATH [--campaign NAME=PATH ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

# Default ordering: the 14 PolyBench-style benches overlapping between the
# ChatHLS U280 baseline run and the c2hls campaigns. mobilenet/transformer
# are appended only if the baseline CSV shows them as a passed optimization.
CORE_BENCHES = [
    "atax",
    "bicg",
    "covariance",
    "gemm",
    "gemm_blocked",
    "gemm_ncubed",
    "gesummv",
    "kernel_2mm",
    "kernel_3mm",
    "kernel_symm",
    "kernel_syr2k",
    "kernel_syrk",
    "matmul",
    "mvt",
]
OPTIONAL_BENCHES = ["mobilenet", "transformer"]


def _is_true(value: str) -> bool:
    return str(value).strip().lower() == "true"


def read_u280_baseline(csv_path: Path) -> dict[str, float]:
    """Read ChatHLS U280 baseline cycles keyed by bench.

    Only rows with ``passed_optimization == True`` are kept; ``undef`` (or
    otherwise non-numeric) ``csynth_best_cycles`` values are skipped.
    """
    baseline: dict[str, float] = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bench = (row.get("bench") or "").strip()
            if not bench:
                continue
            if not _is_true(row.get("passed_optimization", "")):
                continue
            raw = (row.get("csynth_best_cycles") or "").strip()
            if not raw or raw.lower() == "undef":
                continue
            try:
                baseline[bench] = float(raw)
            except ValueError:
                continue
    return baseline


def _coerce_latency(value: Any) -> float | None:
    """Robustly coerce a latency_cycles field (int/float/str/nested dict) to float."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    if isinstance(value, dict):
        candidates: list[float] = []
        # Prefer well-known "final" keys before falling back to any numeric leaf.
        for key in ("selected", "best", "value", "cycles", "latency_cycles", "flash_opt"):
            if key in value:
                coerced = _coerce_latency(value[key])
                if coerced is not None:
                    candidates.append(coerced)
        if not candidates:
            for sub in value.values():
                coerced = _coerce_latency(sub)
                if coerced is not None:
                    candidates.append(coerced)
        return min(candidates) if candidates else None
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _newest_cells(bench_dir: Path) -> list[Path]:
    try:
        cells = [p for p in bench_dir.iterdir() if p.is_dir()]
    except OSError:
        return []
    return sorted(cells, key=lambda p: p.stat().st_mtime, reverse=True)


def best_campaign_latency(
    campaign_root: Path, bench: str
) -> tuple[float | None, dict[str, float], str | None]:
    """Best-of latency in cycles for one bench within a c2hls campaign.

    Returns ``(best_cycles_or_None, {source_name: value}, best_stage_or_None)``
    for whichever sources were found in the newest cell that yielded a usable
    value. ``best_stage`` is the source key that achieved the minimum.
    """
    bench_dir_name = f"chathls_{bench}"
    bench_dir = campaign_root / "variants" / "chathls_aav_n" / bench_dir_name
    if not bench_dir.is_dir():
        return None, {}, None

    for cell in _newest_cells(bench_dir):
        prefix = bench_dir_name
        sources: dict[str, float] = {}

        sr = _load_json(cell / f"{prefix}_selected_report.json")
        if sr is not None:
            v = _coerce_latency(sr.get("latency_cycles"))
            if v is not None:
                sources["selected_report"] = v

        fm = _load_json(cell / f"{prefix}_flow_manifest.json")
        if fm is not None:
            v = _coerce_latency(fm.get("latency_cycles"))
            if v is not None:
                sources["flow_manifest"] = v

        for stage_key, res_name, report_name in (
            ("latency_opt", f"{prefix}_latency_opt_result.json", f"{prefix}_latency_opt_report.json"),
            (
                "dataflow_latency_opt",
                f"{prefix}_dataflow_latency_opt_result.json",
                f"{prefix}_dataflow_latency_opt_report.json",
            ),
        ):
            res = _load_json(cell / res_name)
            if res is not None and res.get("success"):
                v = _coerce_latency(res.get("latency_cycles"))
                if v is not None:
                    sources[stage_key] = v
                    continue
            rep = _load_json(cell / report_name)
            if rep is not None:
                v = _coerce_latency(rep.get("latency_cycles"))
                if v is not None:
                    sources[stage_key] = v

        dr = _load_json(cell / f"{prefix}_dataflow_result.json")
        if dr is not None and dr.get("success"):
            v = _coerce_latency(dr.get("latency_cycles"))
            if v is not None:
                # Prefer explicit selected_stage label when dataflow_result was
                # promoted to latency-opt.
                stage = dr.get("selected_stage")
                if stage == "dataflow_latency_opt" and "dataflow_latency_opt" not in sources:
                    sources["dataflow_latency_opt"] = v
                else:
                    sources["dataflow_result"] = v

        if sources:
            best_stage = min(sources.items(), key=lambda kv: kv[1])[0]
            return min(sources.values()), sources, best_stage

    return None, {}, None


def geomean(values: list[float]) -> float | None:
    if not values:
        return None
    log_sum = 0.0
    for v in values:
        if v <= 0:
            return None
        log_sum += math.log(v)
    return math.exp(log_sum / len(values))


def parse_campaign_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--campaign expects NAME=PATH, got: {raw!r}")
    name, path_str = raw.split("=", 1)
    name = name.strip()
    path_str = path_str.strip()
    if not name or not path_str:
        raise argparse.ArgumentTypeError(f"--campaign expects NAME=PATH, got: {raw!r}")
    return name, Path(path_str)


def build_bench_list(baseline: dict[str, float], campaigns: list[tuple[str, Path]]) -> list[str]:
    benches = [b for b in CORE_BENCHES if b in baseline]
    for opt_bench in OPTIONAL_BENCHES:
        if opt_bench in baseline:
            benches.append(opt_bench)
    return benches


def fmt_cycles(v: float | None) -> str:
    return "N/A" if v is None else f"{v:,.0f}"


def fmt_ratio(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.3f}x"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chat-hls-csv", required=True, type=Path, help="Path to ChatHLS U280 final_latency_csynth.csv baseline")
    parser.add_argument(
        "--campaign",
        action="append",
        default=[],
        dest="campaigns",
        metavar="NAME=PATH",
        help="A c2hls campaign root, given as NAME=PATH. Repeatable.",
    )
    args = parser.parse_args(argv)

    if not args.campaigns:
        parser.error("at least one --campaign NAME=PATH is required")

    campaigns: list[tuple[str, Path]] = [parse_campaign_arg(c) for c in args.campaigns]

    baseline = read_u280_baseline(args.chat_hls_csv)
    if not baseline:
        print(f"No passed_optimization rows found in {args.chat_hls_csv}", file=sys.stderr)
        return 1

    benches = build_bench_list(baseline, campaigns)

    # bench -> {campaign_name: best_cycles}
    results: dict[str, dict[str, float | None]] = {b: {} for b in benches}
    for name, root in campaigns:
        if not root.is_dir():
            print(f"warning: campaign path does not exist: {root}", file=sys.stderr)
        for bench in benches:
            best, _sources, _stage = best_campaign_latency(root, bench)
            results[bench][name] = best

    campaign_names = [name for name, _ in campaigns]

    # --- Table ---
    header = ["bench", "U280_cycles"]
    for name in campaign_names:
        header.append(f"{name}_cycles")
        header.append(f"{name}_ratio")
    col_widths = [max(len(h), 12) for h in header]

    rows: list[list[str]] = []
    for bench in benches:
        u280 = baseline[bench]
        row = [bench, fmt_cycles(u280)]
        for name in campaign_names:
            v = results[bench][name]
            ratio = (v / u280) if (v is not None and u280) else None
            row.append(fmt_cycles(v))
            row.append(fmt_ratio(ratio))
        rows.append(row)

    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def print_row(cells: list[str]) -> None:
        print("  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells)))

    print("=" * 100)
    print(f"ChatHLS U280 baseline: {args.chat_hls_csv}")
    print(f"Benches compared: {len(benches)}  ({', '.join(benches)})")
    print("=" * 100)
    print_row(header)
    print_row(["-" * w for w in col_widths])
    for row in rows:
        print_row(row)

    # --- Geomean ranking ---
    print()
    print("Geomean(campaign_cycles / U280_cycles) over benches with data in both:")
    geomeans: list[tuple[str, float | None, int]] = []
    for name in campaign_names:
        ratios = []
        for bench in benches:
            v = results[bench][name]
            u280 = baseline[bench]
            if v is not None and u280:
                ratios.append(v / u280)
        gm = geomean(ratios)
        geomeans.append((name, gm, len(ratios)))

    for name, gm, n in geomeans:
        gm_str = "N/A" if gm is None else f"{gm:.4f}"
        print(f"  {name:<20s} geomean_ratio={gm_str:<10s} (n={n}/{len(benches)})")

    print()
    print("Ranking (best/lowest geomean ratio vs U280 first; U280 itself = 1.0000 by definition):")
    ranked = sorted(
        [("U280", 1.0, len(benches))] + [(n, gm, cnt) for n, gm, cnt in geomeans if gm is not None],
        key=lambda t: t[1],
    )
    for rank, (name, gm, cnt) in enumerate(ranked, start=1):
        print(f"  {rank}. {name:<20s} geomean_ratio={gm:.4f}  (n={cnt})")

    # --- Win lists vs U280 ---
    print()
    print("Per-campaign wins vs U280 (campaign_cycles < U280_cycles):")
    for name in campaign_names:
        wins = []
        for bench in benches:
            v = results[bench][name]
            u280 = baseline[bench]
            if v is not None and u280 and v < u280:
                wins.append(bench)
        print(f"  {name}: {len(wins)}/{len(benches)} wins -> {', '.join(wins) if wins else '(none)'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

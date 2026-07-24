#!/usr/bin/env python3
"""Compare c2hls port campaigns vs ChatHLS hybrid U280 for prefixed benches.

Reads ChatHLS session CSVs keyed by prefixed bench names (``hlsfactory_atax``,
``machsuite_gemm_ncubed``, …) and extracts best c2hls flash latency (and
LUT/DSP when available) from campaign artifacts:

  * ``flash_selected/<bench>/selected/synth_report.json``
  * ``variants/*/<bench>/<cell>/{bench}_selected_report.json``
  * ``variants/*/<bench>/<cell>/{bench}_flow_manifest.json``
  * ``variants/*/<bench>/<cell>/{bench}_dataflow_result.json`` (success only)

Usage::

    ./.venv/bin/python scripts/pc2/compare_c2hls_chathls_port_u280.py \\
      --chathls-latency-csv PATH/final_latency_csynth.csv \\
      --chathls-resources-csv PATH/final_resources_csynth.csv \\
      --c2hls-machsuite-campaign PATH \\
      --c2hls-hlsfactory-campaign PATH \\
      --out docs/pc2/2026-07-18-hlsfactory-machsuite-deepseek-dual-track.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BENCH_LIST = REPO / "scripts" / "pc2" / "c2hls_port_46_benches.txt"


@dataclass(frozen=True)
class BenchMetrics:
    latency: float | None = None
    lut: int | None = None
    dsp: int | None = None
    source: str | None = None


@dataclass(frozen=True)
class ChatHLSRow:
    latency: float | None
    passed_optimization: bool
    lut: int | None = None
    dsp: int | None = None


def _is_true(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _parse_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raw = str(value).strip()
    if not raw or raw.lower() in {"undef", "na", "n/a"}:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _parse_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw or raw.lower() in {"undef", "na", "n/a"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _coerce_latency(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return _parse_float(value)
    if isinstance(value, dict):
        candidates: list[float] = []
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
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_chathls_latency_csv(csv_path: Path | None) -> dict[str, ChatHLSRow]:
    rows: dict[str, ChatHLSRow] = {}
    if csv_path is None or not csv_path.is_file():
        return rows
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bench = (row.get("bench") or "").strip()
            if not bench:
                continue
            passed = _is_true(row.get("passed_optimization", ""))
            latency = _parse_float(row.get("csynth_best_cycles")) if passed else None
            rows[bench] = ChatHLSRow(latency=latency, passed_optimization=passed)
    return rows


def read_chathls_resources_csv(csv_path: Path | None) -> dict[str, tuple[int | None, int | None]]:
    resources: dict[str, tuple[int | None, int | None]] = {}
    if csv_path is None or not csv_path.is_file():
        return resources
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bench = (row.get("bench") or "").strip()
            if not bench:
                continue
            resources[bench] = (_parse_int(row.get("LUT")), _parse_int(row.get("DSP")))
    return resources


def merge_chathls_rows(
    latency_rows: dict[str, ChatHLSRow],
    resources: dict[str, tuple[int | None, int | None]],
) -> dict[str, ChatHLSRow]:
    benches = sorted(set(latency_rows) | set(resources))
    merged: dict[str, ChatHLSRow] = {}
    for bench in benches:
        lat = latency_rows.get(bench)
        lut, dsp = resources.get(bench, (None, None))
        if lat is None:
            merged[bench] = ChatHLSRow(latency=None, passed_optimization=False, lut=lut, dsp=dsp)
        else:
            merged[bench] = ChatHLSRow(
                latency=lat.latency,
                passed_optimization=lat.passed_optimization,
                lut=lut if lut is not None else lat.lut,
                dsp=dsp if dsp is not None else lat.dsp,
            )
    return merged


def _metrics_from_report(data: dict[str, Any], source: str) -> BenchMetrics | None:
    latency = _coerce_latency(data.get("latency_cycles"))
    lut = _parse_int(data.get("lut") or data.get("LUT"))
    dsp = _parse_int(data.get("dsp") or data.get("DSP"))
    if latency is None and lut is None and dsp is None:
        return None
    return BenchMetrics(latency=latency, lut=lut, dsp=dsp, source=source)


def _newest_cells(bench_dir: Path) -> list[Path]:
    try:
        cells = [p for p in bench_dir.iterdir() if p.is_dir()]
    except OSError:
        return []
    return sorted(cells, key=lambda p: p.stat().st_mtime, reverse=True)


def _find_bench_dir(campaign_root: Path, bench: str) -> Path | None:
    variants = campaign_root / "variants"
    if not variants.is_dir():
        return None
    for variant_dir in sorted(variants.iterdir()):
        if not variant_dir.is_dir():
            continue
        bench_dir = variant_dir / bench
        if bench_dir.is_dir():
            return bench_dir
    return None


def _flash_selected_report(campaign_root: Path, bench: str) -> BenchMetrics | None:
    for base in (
        campaign_root / "flash_selected",
        campaign_root,
    ):
        report = base / bench / "selected" / "synth_report.json"
        data = _load_json(report)
        if data is not None:
            metrics = _metrics_from_report(data, "flash_selected/synth_report")
            if metrics is not None:
                return metrics
    return None


def best_c2hls_metrics(campaign_root: Path, bench: str) -> BenchMetrics:
    """Best latency (and resources from the same source) for one prefixed bench."""
    if not campaign_root.is_dir():
        return BenchMetrics()

    flash = _flash_selected_report(campaign_root, bench)
    if flash is not None and flash.latency is not None:
        return flash

    bench_dir = _find_bench_dir(campaign_root, bench)
    if bench_dir is None:
        return flash or BenchMetrics()

    for cell in _newest_cells(bench_dir):
        sources: dict[str, BenchMetrics] = {}

        sr = _load_json(cell / f"{bench}_selected_report.json")
        if sr is not None:
            m = _metrics_from_report(sr, "selected_report")
            if m is not None and m.latency is not None:
                sources["selected_report"] = m

        fm = _load_json(cell / f"{bench}_flow_manifest.json")
        if fm is not None:
            m = _metrics_from_report(fm, "flow_manifest")
            if m is not None and m.latency is not None:
                sources["flow_manifest"] = m

        dr = _load_json(cell / f"{bench}_dataflow_result.json")
        if dr is not None and dr.get("success"):
            m = _metrics_from_report(dr, "dataflow_result")
            if m is not None and m.latency is not None:
                sources["dataflow_result"] = m

        if sources:
            best = min(sources.values(), key=lambda m: m.latency or math.inf)
            if flash is not None:
                if flash.latency is not None and (
                    best.latency is None or flash.latency <= best.latency
                ):
                    return flash
            return best

    return flash or BenchMetrics()


def discover_campaign_benches(campaign_root: Path, prefix: str) -> list[str]:
    found: set[str] = set()
    if not campaign_root.is_dir():
        return []

    flash_selected = campaign_root / "flash_selected"
    if flash_selected.is_dir():
        for entry in flash_selected.iterdir():
            if entry.is_dir() and entry.name.startswith(prefix):
                found.add(entry.name)

    variants = campaign_root / "variants"
    if variants.is_dir():
        for variant_dir in variants.iterdir():
            if not variant_dir.is_dir():
                continue
            for entry in variant_dir.iterdir():
                if entry.is_dir() and entry.name.startswith(prefix):
                    found.add(entry.name)
    return sorted(found)


def load_default_benches(prefix: str) -> list[str]:
    if not DEFAULT_BENCH_LIST.is_file():
        return []
    benches: list[str] = []
    for line in DEFAULT_BENCH_LIST.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name.startswith(prefix):
            benches.append(name)
    return benches


def build_bench_list(
    prefix: str,
    chathls: dict[str, ChatHLSRow],
    campaign_roots: list[Path],
) -> list[str]:
    benches: set[str] = set(load_default_benches(prefix))
    benches.update(name for name in chathls if name.startswith(prefix))
    for root in campaign_roots:
        benches.update(discover_campaign_benches(root, prefix))
    return sorted(benches)


def fmt_cycles(value: float | None) -> str:
    return "N/A" if value is None else f"{value:,.0f}"


def fmt_int(value: int | None) -> str:
    return "N/A" if value is None else f"{value:,}"


def fmt_ratio(c2hls: float | None, chathls: float | None) -> str:
    if c2hls is None or chathls is None or chathls <= 0:
        return "N/A"
    return f"{c2hls / chathls:.3f}×"


def geomean(values: list[float]) -> float | None:
    if not values:
        return None
    log_sum = 0.0
    for v in values:
        if v <= 0:
            return None
        log_sum += math.log(v)
    return math.exp(log_sum / len(values))


def _section_table_rows(
    benches: list[str],
    chathls: dict[str, ChatHLSRow],
    c2hls: dict[str, BenchMetrics],
) -> tuple[list[list[str]], list[float]]:
    rows: list[list[str]] = []
    ratios: list[float] = []
    for bench in benches:
        ch = chathls.get(bench, ChatHLSRow(latency=None, passed_optimization=False))
        c2 = c2hls.get(bench, BenchMetrics())
        ratio_val: float | None = None
        if ch.latency is not None and c2.latency is not None and ch.latency > 0:
            ratio_val = c2.latency / ch.latency
            ratios.append(ratio_val)
        ch_lat = fmt_cycles(ch.latency) if ch.passed_optimization else (
            fmt_cycles(ch.latency) if ch.latency is not None else "N/A"
        )
        if not ch.passed_optimization and ch.latency is not None:
            ch_lat = f"{ch_lat}†"
        rows.append([
            bench,
            ch_lat,
            fmt_cycles(c2.latency),
            fmt_ratio(c2.latency, ch.latency if ch.passed_optimization else None),
            fmt_int(ch.lut),
            fmt_int(ch.dsp),
            fmt_int(c2.lut),
            fmt_int(c2.dsp),
        ])
    return rows, ratios


def render_markdown(
    *,
    chathls_latency_csv: Path | None,
    chathls_resources_csv: Path | None,
    machsuite_campaign: Path | None,
    hlsfactory_campaign: Path | None,
    chathls: dict[str, ChatHLSRow],
    hlsfactory_benches: list[str],
    machsuite_benches: list[str],
    hlsfactory_c2hls: dict[str, BenchMetrics],
    machsuite_c2hls: dict[str, BenchMetrics],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# HLSFactory + MachSuite U280: c2hls vs ChatHLS (prefixed port)",
        "",
        f"_Generated {now}_",
        "",
        "## Inputs",
        "",
        "| Source | Path |",
        "| --- | --- |",
        f"| ChatHLS latency CSV | `{chathls_latency_csv or 'N/A'}` |",
        f"| ChatHLS resources CSV | `{chathls_resources_csv or 'N/A'}` |",
        f"| c2hls HLSFactory campaign | `{hlsfactory_campaign or 'N/A'}` |",
        f"| c2hls MachSuite campaign | `{machsuite_campaign or 'N/A'}` |",
        "",
        "Latency = csynth best/selected cycles. Ratio = c2hls ÷ ChatHLS (**<1 means c2hls faster**).",
        "Geomeans use benches where both sides have numeric latency and ChatHLS passed optimization.",
        "",
    ]

    sections = [
        ("HLSFactory (28 benches)", hlsfactory_benches, hlsfactory_c2hls),
        ("MachSuite (18 benches)", machsuite_benches, machsuite_c2hls),
    ]
    for title, benches, c2hls_map in sections:
        rows, ratios = _section_table_rows(benches, chathls, c2hls_map)
        paired = sum(
            1
            for bench in benches
            if (
                (ch := chathls.get(bench)) is not None
                and ch.passed_optimization
                and ch.latency is not None
                and (c2hls_map.get(bench) or BenchMetrics()).latency is not None
            )
        )
        gm = geomean(ratios)
        gm_str = "N/A" if gm is None else f"{gm:.3f}×"

        lines.extend([
            f"## {title}",
            "",
            f"Paired benches: {paired}/{len(benches)}. Geomean ratio: {gm_str}.",
            "",
            "| Bench | ChatHLS cycles | c2hls cycles | Ratio | ChatHLS LUT | ChatHLS DSP | c2hls LUT | c2hls DSP |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("† ChatHLS row did not pass optimization (latency shown for reference only).")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chathls-latency-csv", type=Path, default=None, help="ChatHLS final_latency_csynth.csv")
    parser.add_argument("--chathls-resources-csv", type=Path, default=None, help="ChatHLS final_resources_csynth.csv")
    parser.add_argument("--c2hls-machsuite-campaign", type=Path, default=None, help="c2hls MachSuite campaign root")
    parser.add_argument("--c2hls-hlsfactory-campaign", type=Path, default=None, help="c2hls HLSFactory campaign root")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs/pc2/2026-07-18-hlsfactory-machsuite-deepseek-dual-track.md",
        help="Markdown output path",
    )
    args = parser.parse_args(argv)

    for label, path in (
        ("ChatHLS latency CSV", args.chathls_latency_csv),
        ("ChatHLS resources CSV", args.chathls_resources_csv),
        ("c2hls MachSuite campaign", args.c2hls_machsuite_campaign),
        ("c2hls HLSFactory campaign", args.c2hls_hlsfactory_campaign),
    ):
        if path is not None and not path.exists():
            print(f"warning: {label} not found: {path}", file=sys.stderr)

    latency_rows = read_chathls_latency_csv(args.chathls_latency_csv)
    resource_rows = read_chathls_resources_csv(args.chathls_resources_csv)
    chathls = merge_chathls_rows(latency_rows, resource_rows)

    hlsfactory_benches = build_bench_list(
        "hlsfactory_",
        chathls,
        [p for p in [args.c2hls_hlsfactory_campaign] if p is not None],
    )
    machsuite_benches = build_bench_list(
        "machsuite_",
        chathls,
        [p for p in [args.c2hls_machsuite_campaign] if p is not None],
    )

    hlsfactory_c2hls = {
        bench: best_c2hls_metrics(args.c2hls_hlsfactory_campaign or Path(), bench)
        for bench in hlsfactory_benches
    }
    machsuite_c2hls = {
        bench: best_c2hls_metrics(args.c2hls_machsuite_campaign or Path(), bench)
        for bench in machsuite_benches
    }

    md = render_markdown(
        chathls_latency_csv=args.chathls_latency_csv,
        chathls_resources_csv=args.chathls_resources_csv,
        machsuite_campaign=args.c2hls_machsuite_campaign,
        hlsfactory_campaign=args.c2hls_hlsfactory_campaign,
        chathls=chathls,
        hlsfactory_benches=hlsfactory_benches,
        machsuite_benches=machsuite_benches,
        hlsfactory_c2hls=hlsfactory_c2hls,
        machsuite_c2hls=machsuite_c2hls,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

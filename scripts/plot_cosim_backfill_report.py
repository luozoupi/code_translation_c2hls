#!/usr/bin/env python3
"""Render static paper figures from a frozen co-simulation report snapshot.

No synthesis estimates substitute for RTL measurements. Cohorts keep their own
denominators, and unsuccessful or unfinished validations never become zero-cycle
bars. This module only reads the report and writes presentation artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, MaxNLocator, PercentFormatter


STATUSES = (
    "pass", "fail", "timeout", "running", "pending", "interrupted",
    "execution_error", "evidence_error",
)
STATUS_LABELS = {
    "pass": "RTL pass", "fail": "Validation failed", "timeout": "Timed out",
    "running": "Running", "pending": "Pending", "interrupted": "Interrupted",
    "execution_error": "Execution error", "evidence_error": "Evidence error",
}
COLORS = {
    "pass": "#009E73", "fail": "#D55E00", "timeout": "#E69F00",
    "running": "#56B4E9", "pending": "#E6E6E6", "interrupted": "#CC79A7",
    "execution_error": "#595959", "evidence_error": "#777777",
}
HATCHES = {
    "pass": "", "fail": "xx", "timeout": "///", "running": "...",
    "pending": "", "interrupted": "\\\\", "execution_error": "++",
    "evidence_error": "oo",
}
MARKERS = {
    "pass": "", "fail": "F", "timeout": "T", "running": "R",
    "pending": "P", "interrupted": "I", "execution_error": "E",
    "evidence_error": "E",
}
ARMS = ("review_off", "review_on")
ARM_NAMES = {"review_off": "Reviewer off", "review_on": "Reviewer on"}
ARM_COLORS = {"review_off": "#0072B2", "review_on": "#E69F00"}
ARM_HATCHES = {"review_off": "", "review_on": "///"}
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.55,
    "grid.linewidth": 0.4, "grid.alpha": 0.3, "hatch.linewidth": 0.55,
    "savefig.facecolor": "white",
}


def _benchmark(value: str) -> str:
    return str(value).removeprefix("hlsfactory_").replace("_", " ")


def _status(value: Any) -> str:
    return str(value) if value in STATUSES else "evidence_error"


def _cycles(row: dict[str, Any]) -> float | None:
    """Accept only a positive RTL measurement from a passing validation."""
    value = row.get("rtl_cycles")
    if row.get("status") != "pass" or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def _row_status(row: dict[str, Any]) -> str:
    status = _status(row.get("status"))
    return "evidence_error" if status == "pass" and _cycles(row) is None else status


def _status_handles(statuses: list[str]) -> list[Patch]:
    return [Patch(facecolor=COLORS[s], edgecolor="#555555", linewidth=0.35,
                  hatch=HATCHES[s], label=STATUS_LABELS[s]) for s in statuses]


def _arm_handles() -> list[Patch]:
    return [Patch(facecolor=ARM_COLORS[a], edgecolor="#333333", linewidth=0.4,
                  hatch=ARM_HATCHES[a], label=ARM_NAMES[a]) for a in ARMS]


def _finish_axis(ax: Any) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)


def _stamp(fig: Any, data: dict[str, Any]) -> None:
    captured = str(data.get("captured_at_local", "unspecified"))
    try:
        moment = datetime.fromisoformat(captured)
        if moment.tzinfo is not None:
            captured = moment.astimezone(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M %Z")
    except ValueError:
        pass
    fig.text(0.99, 0.009, f"Snapshot: {captured}",
             ha="right", va="bottom", fontsize=5.5, color="#666666")


def _overview(data: dict[str, Any]) -> Any:
    cohorts = data.get("cohorts", [])
    height = max(3.2, 0.67 * len(cohorts) + 1.60)
    fig, ax = plt.subplots(figsize=(7.2, height))
    fig.subplots_adjust(left=0.29, right=0.91, top=0.84, bottom=0.31)
    active = [s for s in STATUSES if any(c.get("counts", {}).get(s, 0) for c in cohorts)]
    labels = []
    for y, cohort in enumerate(cohorts):
        total = int(cohort["n_unique"])
        counts = {s: int(cohort.get("counts", {}).get(s, 0)) for s in STATUSES}
        if sum(counts.values()) != total:
            raise ValueError(f"Cohort {cohort['id']} counts do not sum to n_unique")
        left = 0.0
        for status in STATUSES:
            count = counts[status]
            width = 100 * count / total if total else 0
            if count:
                ax.barh(y, width, left=left, height=0.60, color=COLORS[status],
                        hatch=HATCHES[status], edgecolor="#555555", linewidth=0.45)
                if width >= 5:
                    ax.text(left + width / 2, y, str(count), va="center", ha="center",
                            fontsize=7, color="white" if status in {"execution_error", "evidence_error"} else "black",
                            bbox={"facecolor": COLORS[status], "edgecolor": "none", "pad": 0.2})
                else:
                    ax.annotate(str(count), (left + width / 2, y - 0.3),
                                xytext=(0, 3), textcoords="offset points", ha="center",
                                va="bottom", fontsize=6)
                left += width
        labels.append(str(cohort["label"]))
        ax.text(101.5, y, f"n={total}", va="center", fontsize=7)
        if cohort.get("id") == "diagnostic_6h":
            ax.axhline(y - 0.49, linewidth=0.7, linestyle=(0, (4, 3)), color="#888888")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_xlabel("Share of unique validations within each cohort")
    _finish_axis(ax)
    ax.set_title("Co-simulation outcomes", pad=12)
    fig.legend(handles=_status_handles(active), loc="lower center",
               bbox_to_anchor=(0.53, 0.07), ncol=min(4, max(1, len(active))), frameon=False)
    fig.text(0.5, 0.044, "Each row has its own denominator; diagnostic reruns are a selected subset.",
             ha="center", fontsize=7)
    _stamp(fig, data)
    return fig


def _by_benchmark(data: dict[str, Any]) -> Any:
    rows = data.get("validations", [])
    names = sorted({r["benchmark"] for r in rows if r.get("cohort") in {"historical_1h", "replay_3h"}})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, max(5.4, len(names) * 0.235 + 1.6)))
    fig.subplots_adjust(left=0.125, right=0.925, bottom=0.17, top=0.91, wspace=0.62)
    active = [s for s in STATUSES if any(_status(r.get("status")) == s for r in rows
                                       if r.get("cohort") in {"historical_1h", "replay_3h"})]
    cohorts = {c["id"]: c for c in data.get("cohorts", [])}
    for ax, cohort_id, title in zip(axes, ("historical_1h", "replay_3h"),
                                    ("Original backfill · 1 h limit", "Timeout replay · 3 h limit")):
        selected = [r for r in rows if r.get("cohort") == cohort_id]
        grouped = {name: Counter(_status(r.get("status")) for r in selected if r["benchmark"] == name)
                   for name in names}
        max_total = max((sum(c.values()) for c in grouped.values()), default=1)
        for y, name in enumerate(names):
            left = 0
            for status in STATUSES:
                count = grouped[name][status]
                if count:
                    ax.barh(y, count, left=left, height=0.68, color=COLORS[status],
                            hatch=HATCHES[status], edgecolor="#555555", linewidth=0.35)
                    if count >= max_total * 0.05:
                        ax.text(left + count / 2, y, str(count), va="center", ha="center", fontsize=6,
                                bbox={"facecolor": COLORS[status], "edgecolor": "none", "pad": 0.1})
                    left += count
            ax.text(max_total * 1.045, y, str(left) if left else "—", va="center", fontsize=6.5)
        ax.text(max_total * 1.045, -1, "n", ha="left", fontsize=7)
        ax.set_yticks(range(len(names)), [_benchmark(n) for n in names])
        ax.set_ylim(len(names) - 0.5, -0.65)
        ax.set_xlim(0, max(1, max_total))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.set_xlabel("Unique validations")
        ax.set_title(f"{title}\n(n={cohorts.get(cohort_id, {}).get('n_unique', len(selected))})", pad=14)
        _finish_axis(ax)
    fig.legend(handles=_status_handles(active), loc="lower center", bbox_to_anchor=(0.52, 0.065),
               ncol=min(4, max(1, len(active))), frameon=False)
    fig.text(0.5, 0.040, "The replay contains only original timeouts; it is not an independent benchmark sample.",
             ha="center", fontsize=6.5)
    _stamp(fig, data)
    return fig


def _reviewer_cycles(data: dict[str, Any], strategy: str) -> Any:
    rows = [r for r in data.get("reviewer_rows", []) if r.get("strategy") == strategy]
    names = sorted({r["benchmark"] for r in rows})
    lookup = {(r["benchmark"], r["arm"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError(f"Duplicate benchmark/arm in {strategy} reviewer rows")
    midpoint = (len(names) + 1) // 2
    chunks = [names[:midpoint], names[midpoint:]]
    measured = [value for row in rows if (value := _cycles(row)) is not None]
    xmax = 10 ** max(1, math.ceil(math.log10(max(measured, default=10))))
    fig = plt.figure(figsize=(7.2, max(5.5, midpoint * 0.285 + 1.5)))
    outer = fig.add_gridspec(1, 2, left=0.125, right=0.985, bottom=0.23, top=0.85, wspace=0.62)
    for panel_index, chunk in enumerate(chunks):
        inner = outer[panel_index].subgridspec(1, 2, width_ratios=[1, 0.13], wspace=0.02)
        ax = fig.add_subplot(inner[0])
        gutter = fig.add_subplot(inner[1], sharey=ax)
        for y, name in enumerate(chunk):
            for arm_index, arm in enumerate(ARMS):
                row = lookup.get((name, arm), {"status": "pending"})
                yy = y + (-0.18 if arm_index == 0 else 0.18)
                value = _cycles(row)
                if value is not None:
                    ax.barh(yy, value, height=0.29, color=ARM_COLORS[arm],
                            hatch=ARM_HATCHES[arm], edgecolor="#333333", linewidth=0.4)
                else:
                    gutter.text(0.45, yy, MARKERS[_row_status(row)], ha="center", va="center",
                                fontsize=7.5, fontweight="bold", color="#333333")
            ax.axhline(y + 0.5, color="#DDDDDD", linewidth=0.35)
        ax.set_xscale("log")
        ax.set_xlim(1, xmax)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.minorticks_off()
        ax.set_yticks(range(len(chunk)), [_benchmark(n) for n in chunk])
        ax.set_ylim(max(len(chunk), 1) - 0.5, -0.7)
        ax.set_xlabel("RTL cycles (log scale)")
        _finish_axis(ax)
        gutter.set_xlim(0, 1)
        gutter.axis("off")
        gutter.text(0.45, 1.015, "Status", transform=gutter.transAxes, ha="center", fontsize=6)
    label = "Flash" if strategy == "flash" else "Multistep"
    passed = sum(_cycles(r) is not None for r in rows)
    fig.suptitle(f"{label}: co-simulation-measured latency, reviewer ablation", y=0.965, fontsize=10)
    fig.text(0.5, 0.917, f"Repeat 1 only · {passed}/{len(rows)} configurations have measured passing RTL",
             ha="center", fontsize=7.5)
    fig.legend(handles=_arm_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.13), ncol=2, frameon=False)
    fig.text(0.5, 0.099, "Each benchmark: reviewer off above, on below. Status letters replace missing bars.",
             ha="center", fontsize=6.7)
    fig.text(0.5, 0.071, "T timeout   F validation failed   R running   P pending   I interrupted   E evidence/execution error",
             ha="center", fontsize=6.1)
    fig.text(0.5, 0.043, "Passing measurements only; no synthesis estimates or zero-valued substitutes.",
             ha="center", fontsize=6.5)
    _stamp(fig, data)
    return fig


def _reviewer_speedup(data: dict[str, Any]) -> Any:
    pairs = data.get("reviewer_pairs", [])
    names = sorted({r["benchmark"] for r in pairs})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, max(5.4, len(names) * 0.235 + 1.7)))
    fig.subplots_adjust(left=0.13, right=0.945, top=0.90, bottom=0.15, wspace=0.65)
    good = [float(r["speedup"]) for r in pairs if r.get("status") == "paired_pass"
            and isinstance(r.get("speedup"), (float, int)) and math.isfinite(r["speedup"]) and r["speedup"] > 0]
    log_span = max([abs(math.log10(v)) for v in good] + [0.35])
    # Whole decades leave room for endpoint labels and remain readable at the
    # 7.2-inch export width, even when the observed ratios span several decades.
    extent = max(1, math.ceil(log_span)) + 0.15
    lo, hi = 10 ** (-extent), 10 ** extent
    for ax, strategy, label in zip(axes, ("flash", "dynamic"), ("Flash", "Multistep")):
        lookup = {r["benchmark"]: r for r in pairs if r.get("strategy") == strategy}
        count = 0
        for y, name in enumerate(names):
            row = lookup.get(name, {})
            value = row.get("speedup")
            valid = row.get("status") == "paired_pass" and isinstance(value, (float, int))
            valid = valid and math.isfinite(value) and value > 0
            if valid:
                count += 1
                color = "#009E73" if value >= 1 else "#D55E00"
                hatch = "" if value >= 1 else "xx"
                ax.barh(y, abs(value - 1), left=min(1, value), height=0.66,
                        color=color, edgecolor="#333333", linewidth=0.4, hatch=hatch)
                ax.plot(value, y, "|", color="#333333", markersize=6, markeredgewidth=0.75)
                text = f"{value:.2f}×" if 0.01 <= value < 100 else f"{value:.1e}×"
                ax.text(value, y, " " + text if value >= 1 else text + " ", va="center",
                        ha="left" if value >= 1 else "right", fontsize=5.8)
            else:
                off = MARKERS[_status(row.get("off_status", "pending"))] or "✓"
                on = MARKERS[_status(row.get("on_status", "pending"))] or "✓"
                ax.text(1.025, y, f"{off}/{on}", transform=ax.get_yaxis_transform(),
                        va="center", fontsize=6)
        ax.axvline(1, color="#444444", linewidth=0.7)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}×"))
        ax.minorticks_off()
        ax.set_yticks(range(len(names)), [_benchmark(n) for n in names])
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_xlabel("RTL speedup: reviewer off / on")
        ax.set_title(f"{label}: {count}/{len(names)} complete pairs", pad=13)
        _finish_axis(ax)
    fig.text(0.5, 0.085, "Only pairs with two passing RTL measurements. Above 1× favors reviewer on.",
             ha="center", fontsize=7)
    fig.text(0.5, 0.054, "Right gutter: off/on status (✓ pass; T timeout; F failed; R running; P pending; I interrupted; E error).",
             ha="center", fontsize=6.1)
    fig.text(0.5, 0.030, "Repeat 1; incomplete pairs are excluded, not assigned neutral speedup.", ha="center", fontsize=6.5)
    _stamp(fig, data)
    return fig


def render_report(data: dict[str, Any], output: Path) -> list[str]:
    """Export individual PDF/SVG/PNG figures and a combined vector PDF.

    ``output`` is an artifact directory. The returned paths are absolute strings.
    All plotting decisions derive from this one supplied snapshot.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    factories = [
        ("cosim_cohort_outcomes", lambda: _overview(data)),
        ("cosim_benchmark_outcomes", lambda: _by_benchmark(data)),
        ("reviewer_flash_rtl_cycles", lambda: _reviewer_cycles(data, "flash")),
        ("reviewer_multistep_rtl_cycles", lambda: _reviewer_cycles(data, "dynamic")),
    ]
    if data.get("reviewer_pairs"):
        factories.append(("reviewer_paired_rtl_speedup", lambda: _reviewer_speedup(data)))
    files = []
    combined = output / "cosim_backfill_figures.pdf"
    with plt.rc_context(STYLE), PdfPages(combined, metadata={
        "Title": "Co-simulation backfill evidence", "Author": "",
        "Subject": f"Frozen report snapshot: {data.get('captured_at_local', 'unspecified')}",
    }) as pdf:
        for name, factory in factories:
            fig = factory()
            try:
                for extension in ("pdf", "svg", "png"):
                    path = output / f"{name}.{extension}"
                    fig.savefig(path, dpi=300, bbox_inches=None)
                    files.append(str(path))
                pdf.savefig(fig)
            finally:
                plt.close(fig)
    files.append(str(combined))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Frozen report JSON")
    parser.add_argument("--output", type=Path, help="Figure directory; default: report directory")
    args = parser.parse_args()
    with args.report.open(encoding="utf-8") as handle:
        data = json.load(handle)
    files = render_report(data, args.output or args.report.parent)
    print(json.dumps({"figures": files}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write a markdown summary table for a multistep fixed-cosim pipelined stamp."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys_path = REPO / "scripts" / "pc2"
import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(sys_path))

from export_multistep_csynth_speedup_csv import collect, load_bench_jsonl  # noqa: E402
from multistep_fixed_cosim_lib import VARIANTS  # noqa: E402


def _fmt_cycles(val: int | float | None) -> str:
    if val is None:
        return "—"
    try:
        return f"{int(val):,}"
    except (TypeError, ValueError):
        return "—"


def _fmt_speedup(val: float | None) -> str:
    if val is None or val <= 0:
        return "—"
    return f"{val:.2f}×"


def _geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def write_summary_md(
    *,
    stamp: str,
    variant: str,
    out_path: Path,
    baseline_jsonl: Path,
    artifact_root: Path | None = None,
) -> dict:
    bench_jsonl = load_bench_jsonl(baseline_jsonl)
    per_bench, _progression, aggregate = collect(stamp, bench_jsonl, variant=variant)
    variant_label = VARIANTS.get(variant).label if variant in VARIANTS else variant
    ok_rows = [r for r in per_bench if r.get("status") == "ok"]
    fail_rows = [r for r in per_bench if r.get("status") != "ok"]
    sp_pb = [r["speedup_vs_phase_b"] for r in ok_rows if r.get("speedup_vs_phase_b")]
    sp_gt = [r["speedup_vs_gt"] for r in ok_rows if r.get("speedup_vs_gt")]
    g_pb = _geomean(sp_pb)
    g_gt = _geomean(sp_gt)

    if artifact_root is None:
        artifact_root = REPO / "artifacts" / "pc2" / f"multistep_fixed_cosim_{variant}_{stamp}"
    manifest_path = artifact_root / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    lines = [
        f"# Multistep fixed-cosim summary ({variant})",
        "",
        f"- **Variant:** {variant} — {variant_label}",
        f"- **Stamp:** `{stamp}`",
        f"- **Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **Artifact root:** `{artifact_root}`",
        f"- **Model:** {manifest.get('model', '—')}",
        f"- **Benches OK / fail:** {len(ok_rows)} / {len(fail_rows)}",
        "",
        "## Aggregate csynth speedup (selected kernel)",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Gmean vs phase_b | {_fmt_speedup(g_pb)} |",
        f"| Gmean vs GT synth | {_fmt_speedup(g_gt)} |",
        "",
        "## Per-bench results",
        "",
        "| Bench | phase_b | selected | vs phase_b | vs GT | selected step |",
        "|-------|--------:|---------:|-----------:|------:|---------------|",
    ]
    for row in sorted(per_bench, key=lambda r: r.get("bench") or ""):
        if row.get("status") != "ok":
            lines.append(
                f"| {row.get('bench', '?')} | — | — | **FAIL** | — | {row.get('error', '')[:40]} |"
            )
            continue
        lines.append(
            "| {bench} | {pb} | {sel} | {spb} | {sgt} | {step} |".format(
                bench=row.get("bench", "?"),
                pb=_fmt_cycles(row.get("phase_b_cycles")),
                sel=_fmt_cycles(row.get("selected_cycles")),
                spb=_fmt_speedup(row.get("speedup_vs_phase_b")),
                sgt=_fmt_speedup(row.get("speedup_vs_gt")),
                step=row.get("selected_from") or "—",
            )
        )

    if fail_rows:
        lines.extend(["", "## Failures", ""])
        for row in fail_rows:
            lines.append(f"- **{row.get('bench')}:** {row.get('error') or 'unknown error'}")

    if aggregate:
        agg = aggregate[0]
        lines.extend(
            [
                "",
                "## Export notes",
                "",
                "- JSONL includes baseline + per-step csynth/csim records (cosim RTL sim rows when cosim run exists).",
                f"- CSV analysis: `artifacts/pc2/analysis/{stamp}/`",
                "",
                "```json",
                json.dumps(agg, indent=2),
                "```",
            ]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "out_path": str(out_path),
        "ok": len(ok_rows),
        "fail": len(fail_rows),
        "gmean_speedup_vs_phase_b": g_pb,
        "gmean_speedup_vs_gt": g_gt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--variant", default="aav_n")
    parser.add_argument(
        "--baseline-jsonl",
        default=str(REPO / "misc/hlsfactory_baseline_u280_20260616_benchmarks_full_cosim.jsonl"),
    )
    parser.add_argument("--output", default="", help="Default: artifacts/pc2/analysis/<stamp>/summary.md")
    args = parser.parse_args()

    out = (
        Path(args.output)
        if args.output
        else REPO / "artifacts/pc2/analysis" / args.stamp / "summary.md"
    )
    summary = write_summary_md(
        stamp=args.stamp,
        variant=args.variant,
        out_path=out,
        baseline_jsonl=Path(args.baseline_jsonl),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

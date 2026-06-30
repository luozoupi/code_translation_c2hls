#!/usr/bin/env python3
"""Per-variant csynth/cosim tables + interactive HTML chart from profile CSV."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VARIANTS = ("aav_n", "aav_o", "nav_n", "nav_o", "noskills")


def load_profile(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def pivot_variant(rows: list[dict], variant: str) -> list[dict]:
    benches = sorted({r["bench"] for r in rows if r["variant"] == variant})
    out: list[dict] = []
    by_key = {(r["bench"], r["phase"]): r for r in rows if r["variant"] == variant}
    for bench in benches:
        pb = by_key.get((bench, "phase_b"), {})
        fl = by_key.get((bench, "flash"), {})
        out.append(
            {
                "bench": bench,
                "phase_b_csynth_s": _f(pb.get("csynth_final_s")),
                "flash_csynth_s": _f(fl.get("csynth_final_s")),
                "phase_b_cosim_s": _f(pb.get("cosim_s")),
                "flash_cosim_s": _f(fl.get("cosim_s")),
                "phase_b_cosim_status": pb.get("cosim_status") or "",
                "flash_cosim_status": fl.get("cosim_status") or "",
                "phase_b_attempts": pb.get("csynth_attempts") or "",
                "flash_attempts": fl.get("csynth_attempts") or "",
            }
        )
    return out


def write_variant_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def variant_summary(rows: list[dict], variant: str) -> dict:
    subset = [r for r in rows if r["variant"] == variant]
    def collect(phase: str, field: str) -> list[float]:
        return [v for r in subset if r["phase"] == phase and (v := _f(r.get(field))) is not None]

    for phase in ("phase_b", "flash"):
        cs = collect(phase, "csynth_final_s")
        co = collect(phase, "cosim_s")
    pb_cs, fl_cs = collect("phase_b", "csynth_final_s"), collect("flash", "csynth_final_s")
    pb_co, fl_co = collect("phase_b", "cosim_s"), collect("flash", "cosim_s")
    return {
        "variant": variant,
        "phase_b_csynth_total_h": sum(pb_cs) / 3600 if pb_cs else 0,
        "flash_csynth_total_h": sum(fl_cs) / 3600 if fl_cs else 0,
        "phase_b_cosim_total_h": sum(pb_co) / 3600 if pb_co else 0,
        "flash_cosim_total_h": sum(fl_co) / 3600 if fl_co else 0,
        "phase_b_cosim_median_min": statistics.median(pb_co) / 60 if pb_co else 0,
        "flash_cosim_median_min": statistics.median(fl_co) / 60 if fl_co else 0,
    }


def build_chart_html(profile: list[dict], out_path: Path) -> None:
    charts = {}
    for variant in VARIANTS:
        piv = pivot_variant(profile, variant)
        charts[variant] = {
            "labels": [r["bench"] for r in piv],
            "phase_b_csynth": [r["phase_b_csynth_s"] or 0 for r in piv],
            "flash_csynth": [r["flash_csynth_s"] or 0 for r in piv],
            "phase_b_cosim": [(r["phase_b_cosim_s"] or 0) / 60 for r in piv],
            "flash_cosim": [(r["flash_cosim_s"] or 0) / 60 for r in piv],
        }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Csynth vs Cosim by bench</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background: #0f1117; color: #e8eaed; }}
    h1 {{ font-size: 1.25rem; margin-bottom: 0.25rem; }}
    p {{ color: #9aa0a6; font-size: 0.9rem; max-width: 52rem; }}
    select {{ margin: 1rem 0; padding: 0.4rem 0.6rem; font-size: 1rem; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 1.5rem; max-width: 1400px; }}
    .panel {{ background: #1a1d27; border-radius: 8px; padding: 1rem; }}
    canvas {{ max-height: 520px; }}
  </style>
</head>
<body>
  <h1>Csynth vs Cosim by bench — flash fixed-cosim 20260628</h1>
  <p>Csynth: pipelined LLM+Vitis final run (seconds). Cosim: post-batch full-size (minutes). Log scale on cosim panel.</p>
  <label for="variant">Variant </label>
  <select id="variant">{"".join(f'<option value="{v}">{v}</option>' for v in VARIANTS)}</select>
  <div class="grid">
    <div class="panel"><h2>Csynth time (seconds)</h2><canvas id="csynth"></canvas></div>
    <div class="panel"><h2>Cosim time (minutes, log y)</h2><canvas id="cosim"></canvas></div>
  </div>
  <script>
    const DATA = {json.dumps(charts)};
    let csynthChart, cosimChart;

    function render(variant) {{
      const d = DATA[variant];
      const opts = (logY) => ({{
        responsive: true,
        plugins: {{ legend: {{ position: 'top' }} }},
        scales: {{
          x: {{ ticks: {{ maxRotation: 90, minRotation: 45, color: '#9aa0a6' }}, grid: {{ color: '#2a2f3a' }} }},
          y: {{
            type: logY ? 'logarithmic' : 'linear',
            min: logY ? 0.1 : 0,
            ticks: {{ color: '#9aa0a6' }},
            grid: {{ color: '#2a2f3a' }},
            title: {{ display: true, text: logY ? 'minutes' : 'seconds', color: '#9aa0a6' }}
          }}
        }}
      }});
      if (csynthChart) csynthChart.destroy();
      if (cosimChart) cosimChart.destroy();
      csynthChart = new Chart(document.getElementById('csynth'), {{
        type: 'bar',
        data: {{
          labels: d.labels,
          datasets: [
            {{ label: 'phase_b csynth', data: d.phase_b_csynth, backgroundColor: '#4e79a7' }},
            {{ label: 'flash csynth', data: d.flash_csynth, backgroundColor: '#f28e2b' }},
          ]
        }},
        options: opts(false)
      }});
      cosimChart = new Chart(document.getElementById('cosim'), {{
        type: 'bar',
        data: {{
          labels: d.labels,
          datasets: [
            {{ label: 'phase_b cosim', data: d.phase_b_cosim, backgroundColor: '#59a14f' }},
            {{ label: 'flash cosim', data: d.flash_cosim, backgroundColor: '#e15759' }},
          ]
        }},
        options: opts(true)
      }});
    }}
    document.getElementById('variant').addEventListener('change', (e) => render(e.target.value));
    render('aav_n');
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def write_markdown_tables(profile: list[dict], out_path: Path) -> None:
    lines = [
        "# Csynth vs Cosim per variant (20260628_fixed_cosim_flash_r2_pipelined)",
        "",
        "Csynth = pipelined final report. Cosim = post-batch full-size. Times: csynth seconds, cosim minutes.",
        "",
    ]
    for variant in VARIANTS:
        piv = pivot_variant(profile, variant)
        summ = variant_summary(profile, variant)
        lines += [
            f"## {variant}",
            "",
            f"Totals: phase_b csynth {summ['phase_b_csynth_total_h']:.2f}h | flash csynth {summ['flash_csynth_total_h']:.2f}h | "
            f"phase_b cosim {summ['phase_b_cosim_total_h']:.1f}h | flash cosim {summ['flash_cosim_total_h']:.1f}h",
            "",
            "| bench | pb csynth | fl csynth | pb cosim | fl cosim | pb status | fl status |",
            "|-------|----------:|----------:|---------:|---------:|-----------|-----------|",
        ]
        for r in piv:
            def fmt_s(v: float | None) -> str:
                return f"{v:.0f}s" if v is not None else "—"
            def fmt_m(v: float | None) -> str:
                if v is None:
                    return "—"
                if v >= 3600:
                    return f"{v/3600:.1f}h"
                return f"{v/60:.0f}m"
            lines.append(
                f"| {r['bench']} | {fmt_s(r['phase_b_csynth_s'])} | {fmt_s(r['flash_csynth_s'])} | "
                f"{fmt_m(r['phase_b_cosim_s'])} | {fmt_m(r['flash_cosim_s'])} | "
                f"{r['phase_b_cosim_status'] or '—'} | {r['flash_cosim_status'] or '—'} |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=REPO / "artifacts/pc2/analysis/20260628_fixed_cosim_flash_r2_pipelined/csynth_cosim_time_profile.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "artifacts/pc2/analysis/20260628_fixed_cosim_flash_r2_pipelined",
    )
    args = parser.parse_args()
    profile = load_profile(args.profile_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for variant in VARIANTS:
        piv = pivot_variant(profile, variant)
        write_variant_csv(piv, args.out_dir / f"csynth_cosim_by_bench_{variant}.csv")

    build_chart_html(profile, args.out_dir / "csynth_cosim_by_bench_chart.html")
    write_markdown_tables(profile, args.out_dir / "csynth_cosim_by_variant.md")

    summaries = [variant_summary(profile, v) for v in VARIANTS]
    write_variant_csv(summaries, args.out_dir / "csynth_cosim_variant_totals.csv")

    print(f"Wrote charts and tables under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

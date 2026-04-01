#!/usr/bin/env python3
"""
HTML Report Generator for C-to-HLS Translation Results.

Reads rubric JSON output and per-benchmark result files to produce a
self-contained interactive HTML report with charts and tables.

Usage:
    python report.py --results results --output report.html
    python report.py --results results --multistep --output report_multistep.html
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from html import escape
from pathlib import Path


def _run_rubric(results_dir: str, multistep: bool = False) -> list[dict]:
    """Run rubric.py --json and return parsed output."""
    cmd = [sys.executable, "rubric.py", "--results", results_dir, "--json"]
    if multistep:
        cmd.append("--multistep")
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    return json.loads(out)


def _load_bench_results(results_dir: str, multistep: bool = False) -> dict[str, dict]:
    """Load per-benchmark result JSON files."""
    rd = Path(results_dir)
    bench_data = {}
    pattern = "*_multistep_results.json" if multistep else "*_results.json"
    for d in sorted(rd.iterdir()):
        if not d.is_dir():
            continue
        candidates = sorted(d.glob(pattern))
        if not candidates and multistep:
            candidates = sorted(d.glob("*_results.json"))
        for f in candidates:
            if not multistep and f.name.endswith("_multistep_results.json"):
                continue
            data = json.loads(f.read_text())
            bench_data[d.name] = data
            break
    return bench_data


def _grade_color(grade: str) -> str:
    return {"A": "#22c55e", "B": "#3b82f6", "C": "#eab308",
            "D": "#f97316", "F": "#ef4444"}.get(grade, "#6b7280")


def _grade_bg(grade: str) -> str:
    return {"A": "#dcfce7", "B": "#dbeafe", "C": "#fef9c3",
            "D": "#ffedd5", "F": "#fee2e2"}.get(grade, "#f3f4f6")


def _fmt(v, decimals=1) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _pct_bar(value: float, max_val: float = 100.0, color: str = "#3b82f6") -> str:
    w = min(value / max_val * 100, 100) if max_val else 0
    return (f'<div style="background:#e5e7eb;border-radius:4px;height:18px;width:100%;position:relative">'
            f'<div style="background:{color};border-radius:4px;height:100%;width:{w:.1f}%"></div>'
            f'<span style="position:absolute;left:4px;top:0;font-size:11px;line-height:18px">{_fmt(value)}%</span>'
            f'</div>')


def _status_chip(status) -> str:
    if status is True or status == "passed":
        return '<span class="chip chip-pass">PASS</span>'
    if status is False or status == "failed":
        return '<span class="chip chip-fail">FAIL</span>'
    if status == "not_run":
        return '<span class="chip chip-na">SKIP</span>'
    if status == "not_supported":
        return '<span class="chip chip-na">N/A</span>'
    return '<span class="chip chip-na">-</span>'


def generate_html(rubric_data: list[dict], bench_results: dict[str, dict],
                  multistep: bool = False) -> str:
    """Generate the full HTML report string."""

    # ── Aggregate stats ─────────────────────────────────────────────
    n = len(rubric_data)
    scores = [b["composite"] for b in rubric_data]
    grades = [b["grade"] for b in rubric_data]
    synth_ok = sum(1 for b in rubric_data if b["synthesis_rate"] == 100)
    avg_score = sum(scores) / n if n else 0
    synth_only_scores = [b["composite"] for b in rubric_data if b["synthesis_rate"] == 100]
    avg_synth = sum(synth_only_scores) / len(synth_only_scores) if synth_only_scores else 0

    grade_dist = {g: grades.count(g) for g in ["A", "B", "C", "D", "F"]}

    # Detect model from bench results
    model_name = "Unknown"
    for br in bench_results.values():
        if "model" in br:
            model_name = br["model"]
            break

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode = "Multi-Step" if multistep else "Single-Shot"

    # ── HTML start ──────────────────────────────────────────────────
    h = []
    h.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>C-to-HLS Translation Report</title>
<style>
:root {{ --bg: #f8fafc; --card: #fff; --border: #e2e8f0; --text: #1e293b; --muted: #64748b; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--text); line-height:1.5; }}
.container {{ max-width:1200px; margin:0 auto; padding:24px 16px; }}
h1 {{ font-size:24px; margin-bottom:4px; }}
h2 {{ font-size:18px; margin:32px 0 12px; border-bottom:2px solid var(--border); padding-bottom:6px; }}
h3 {{ font-size:15px; margin:16px 0 8px; }}
.subtitle {{ color:var(--muted); font-size:14px; margin-bottom:24px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:24px; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; text-align:center; }}
.card .value {{ font-size:28px; font-weight:700; }}
.card .label {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; }}
table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--border); border-radius:8px; overflow:hidden; margin-bottom:16px; font-size:13px; }}
th {{ background:#f1f5f9; font-weight:600; text-align:left; padding:8px 10px; border-bottom:2px solid var(--border); white-space:nowrap; }}
td {{ padding:6px 10px; border-bottom:1px solid var(--border); }}
tr:last-child td {{ border-bottom:none; }}
tr:hover td {{ background:#f8fafc; }}
.grade {{ display:inline-block; width:28px; height:28px; line-height:28px; text-align:center; border-radius:6px; font-weight:700; font-size:14px; color:#fff; }}
.bar-cell {{ min-width:120px; }}
.metric-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:16px; }}
.metric-card {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:16px; }}
.chip {{ display:inline-block; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600; }}
.chip-pass {{ background:#dcfce7; color:#166534; }}
.chip-fail {{ background:#fee2e2; color:#991b1b; }}
.chip-na {{ background:#f3f4f6; color:#6b7280; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin:8px 0; }}
.legend span::before {{ content:''; display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:4px; vertical-align:middle; }}
details {{ margin-bottom:8px; }}
details summary {{ cursor:pointer; font-weight:600; padding:8px 0; }}
.resource-row {{ display:grid; grid-template-columns:70px 1fr 60px 60px 60px; gap:4px; align-items:center; font-size:12px; padding:2px 0; }}
footer {{ text-align:center; color:var(--muted); font-size:12px; margin-top:40px; padding:16px 0; border-top:1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">
<h1>C-to-HLS Translation Report</h1>
<div class="subtitle">{mode} mode &middot; Generated {now}</div>
""")

    # ── Summary cards ───────────────────────────────────────────────
    def _overall_grade(s):
        if s >= 90: return "A"
        if s >= 75: return "B"
        if s >= 60: return "C"
        if s >= 40: return "D"
        return "F"

    og = _overall_grade(avg_score)
    h.append(f"""
<div class="cards">
  <div class="card"><div class="value" style="color:{_grade_color(og)}">{avg_score:.1f}</div><div class="label">Overall Score</div></div>
  <div class="card"><div class="value">{synth_ok}/{n}</div><div class="label">Synthesized</div></div>
  <div class="card"><div class="value">{avg_synth:.1f}</div><div class="label">Avg (synth only)</div></div>
  <div class="card">
    <div class="value" style="font-size:20px">
      {"".join(f'<span class="grade" style="background:{_grade_color(g)};margin:1px">{grade_dist[g]}</span>' for g in ["A","B","C","D","F"] if grade_dist[g])}
    </div>
    <div class="label">Grade Distribution</div>
  </div>
</div>
""")

    # ── Scoreboard table ────────────────────────────────────────────
    h.append("""<h2>Benchmark Scoreboard</h2>
<table>
<tr>
  <th>Benchmark</th><th>Grade</th><th>Score</th><th></th>
  <th>Synth</th><th>Csim</th><th>Cosim</th>
  <th>Latency</th><th>Fmax</th><th>Resources</th><th>ADP</th><th>Feasibility</th>
</tr>
""")
    for b in sorted(rubric_data, key=lambda x: -x["composite"]):
        name = b["benchmark"]
        g = b["grade"]
        sc = b["composite"]
        bar_color = _grade_color(g)
        syn_chip = _status_chip(b["synthesis_rate"] == 100)
        csim_chip = _status_chip(b["csim_rate"] == 100)
        cosim_chip = _status_chip(b["cosim_rate"] == 100)

        h.append(f"""<tr>
  <td><strong>{escape(name)}</strong></td>
  <td><span class="grade" style="background:{_grade_color(g)}">{g}</span></td>
  <td><strong>{sc:.1f}</strong></td>
  <td class="bar-cell">{_pct_bar(sc, 100, bar_color)}</td>
  <td>{syn_chip}</td>
  <td>{csim_chip}</td>
  <td>{cosim_chip}</td>
  <td>{_fmt(b["avg_latency"])}</td>
  <td>{_fmt(b["avg_fmax"])}</td>
  <td>{_fmt(b["avg_resources"])}</td>
  <td>{_fmt(b["avg_adp"])}</td>
  <td>{_fmt(b["avg_feasibility"])}</td>
</tr>
""")
    h.append("</table>")

    # ── Resource comparison chart (CSS bar chart) ───────────────────
    h.append("<h2>Resource Usage: Generated vs Ground Truth</h2>")
    h.append('<div class="legend">'
             '<span style="--c:#3b82f6">&#9632; Generated</span>'
             '<span style="--c:#94a3b8">&#9632; Ground Truth</span>'
             '<span style="--c:#ef4444">&#9632; Device Limit</span>'
             '</div>')

    DEVICE = {"bram": 270, "dsp": 240, "ff": 126800, "lut": 63400}

    for b in sorted(rubric_data, key=lambda x: -x["composite"]):
        name = b["benchmark"]
        if b["synthesis_rate"] < 100:
            continue
        step = b["steps"][0] if b["steps"] else {}
        br = bench_results.get(name, {})
        comp = br.get("comparison", {})
        gen_report = br.get("synth_report") or comp.get("generated_report", {})
        gt_report = br.get("ground_truth_report") or comp.get("ground_truth_report", {})

        h.append(f'<details><summary>{escape(name)} '
                 f'<span class="grade" style="background:{_grade_color(b["grade"])};font-size:11px">{b["grade"]}</span></summary>')
        for res in ["bram", "dsp", "ff", "lut"]:
            gen_v = float(gen_report.get(res, 0) or 0)
            gt_v = float(gt_report.get(res, 0) or 0)
            limit = DEVICE[res]
            max_v = max(gen_v, gt_v, limit) * 1.1 or 1

            gen_pct = gen_v / max_v * 100
            gt_pct = gt_v / max_v * 100
            lim_pct = limit / max_v * 100

            ratio = step.get("resource_ratios", {}).get(res, None)
            ratio_str = f" ({ratio:.2f}x)" if ratio is not None else ""

            h.append(f'<div class="resource-row">'
                     f'<div><strong>{res.upper()}</strong></div>'
                     f'<div>'
                     f'<div style="background:#dbeafe;border-radius:3px;height:10px;width:{gen_pct:.1f}%;margin-bottom:2px" title="Gen: {gen_v:.0f}"></div>'
                     f'<div style="background:#cbd5e1;border-radius:3px;height:10px;width:{gt_pct:.1f}%" title="GT: {gt_v:.0f}"></div>'
                     f'</div>'
                     f'<div style="font-size:11px">{gen_v:.0f}</div>'
                     f'<div style="font-size:11px">{gt_v:.0f}</div>'
                     f'<div style="font-size:11px;color:var(--muted)">{ratio_str}</div>'
                     f'</div>')
        h.append("</details>")

    # ── Latency & Fmax comparison ───────────────────────────────────
    h.append('<h2>Performance Comparison</h2>')
    h.append('<div class="metric-grid">')

    # Latency card (ns — accounts for both cycles and clock frequency)
    h.append('<div class="metric-card"><h3>Latency (ns)</h3><table>'
             '<tr><th>Benchmark</th><th>Generated</th><th>GT</th><th>Ratio</th><th>Speedup</th></tr>')
    for b in sorted(rubric_data, key=lambda x: -x["composite"]):
        if b["synthesis_rate"] < 100 or not b["steps"]:
            continue
        s = b["steps"][0]
        gen_lat = s.get("latency_ns") or s.get("latency_cycles")
        gt_lat = s.get("gt_latency_ns") or s.get("gt_latency_cycles")
        ratio = s.get("latency_ratio")
        if gen_lat and gt_lat:
            speedup = gt_lat / gen_lat if gen_lat else 0
            color = "#22c55e" if ratio and ratio <= 1 else "#ef4444"
            h.append(f'<tr><td>{escape(b["benchmark"])}</td>'
                     f'<td>{gen_lat:,.0f}</td><td>{gt_lat:,.0f}</td>'
                     f'<td style="color:{color}">{_fmt(ratio, 2)}x</td>'
                     f'<td>{_fmt(speedup, 2)}x</td></tr>')
    h.append('</table></div>')

    # Fmax card
    h.append('<div class="metric-card"><h3>Clock Frequency (MHz)</h3><table>'
             '<tr><th>Benchmark</th><th>Generated</th><th>GT</th><th>Ratio</th><th>Slack (ns)</th></tr>')
    for b in sorted(rubric_data, key=lambda x: -x["composite"]):
        if b["synthesis_rate"] < 100 or not b["steps"]:
            continue
        s = b["steps"][0]
        gen_f = s.get("fmax_mhz")
        gt_f = s.get("gt_fmax_mhz")
        ratio = s.get("fmax_ratio")
        slack = s.get("timing_slack_ns")
        if gen_f and gt_f:
            color = "#22c55e" if slack and slack >= 0 else "#ef4444"
            h.append(f'<tr><td>{escape(b["benchmark"])}</td>'
                     f'<td>{_fmt(gen_f)}</td><td>{_fmt(gt_f)}</td>'
                     f'<td>{_fmt(ratio, 2)}x</td>'
                     f'<td style="color:{color}">{_fmt(slack)}</td></tr>')
    h.append('</table></div>')
    h.append('</div>')  # metric-grid

    # ── Device utilization heatmap ──────────────────────────────────
    h.append('<h2>Device Utilization (% of Artix-7 100T)</h2>')
    h.append('<table><tr><th>Benchmark</th><th>BRAM %</th><th>DSP %</th><th>FF %</th><th>LUT %</th><th>Feasible</th></tr>')

    for b in sorted(rubric_data, key=lambda x: x["benchmark"]):
        if b["synthesis_rate"] < 100 or not b["steps"]:
            continue
        s = b["steps"][0]
        util = s.get("device_util_pct", {})

        def _util_cell(pct):
            if pct is None:
                return '<td>-</td>'
            if pct > 100:
                return f'<td style="background:#fee2e2;color:#991b1b;font-weight:700">{pct:.1f}%</td>'
            elif pct > 70:
                return f'<td style="background:#fef9c3">{pct:.1f}%</td>'
            else:
                return f'<td>{pct:.1f}%</td>'

        feas = s.get("feasibility_score", 0)
        feas_chip = ('<span class="chip chip-pass">YES</span>' if feas >= 100
                     else f'<span class="chip chip-fail">{feas:.0f}%</span>' if feas < 80
                     else f'<span class="chip" style="background:#fef9c3;color:#92400e">{feas:.0f}%</span>')

        h.append(f'<tr><td><strong>{escape(b["benchmark"])}</strong></td>'
                 f'{_util_cell(util.get("bram"))}'
                 f'{_util_cell(util.get("dsp"))}'
                 f'{_util_cell(util.get("ff"))}'
                 f'{_util_cell(util.get("lut"))}'
                 f'<td>{feas_chip}</td></tr>')
    h.append('</table>')

    # ── Per-benchmark detail (collapsible) ──────────────────────────
    h.append('<h2>Detailed Benchmark Results</h2>')

    for b in sorted(rubric_data, key=lambda x: -x["composite"]):
        name = b["benchmark"]
        g = b["grade"]
        br = bench_results.get(name, {})
        comp = br.get("comparison", {})

        h.append(f'<details><summary><span class="grade" style="background:{_grade_color(g)}">{g}</span> '
                 f'<strong>{escape(name)}</strong> &mdash; {b["composite"]:.1f}/100</summary>')

        # Comparison table
        gen_r = br.get("synth_report") or comp.get("generated_report", {})
        gt_r = br.get("ground_truth_report") or comp.get("ground_truth_report", {})
        comp_ratios = comp.get("comparison", {})

        if gen_r or gt_r:
            h.append('<table><tr><th>Metric</th><th>Generated</th><th>Ground Truth</th><th>Ratio</th></tr>')
            metrics = [
                ("Latency (ns)", "latency_ns", False),
                ("BRAM", "bram", False),
                ("DSP", "dsp", False),
                ("FF", "ff", False),
                ("LUT", "lut", False),
                ("Fmax (MHz)", "fmax_mhz", True),
            ]
            for label, key, higher_better in metrics:
                gv = gen_r.get(key, "-")
                gtv = gt_r.get(key, "-")
                cr = comp_ratios.get(key, {})
                ratio = cr.get("ratio")
                if ratio is not None:
                    good = ratio >= 1.0 if higher_better else ratio <= 1.0
                    color = "#22c55e" if good else "#ef4444"
                    ratio_str = f'<span style="color:{color}">{ratio:.3f}</span>'
                else:
                    ratio_str = "-"
                h.append(f'<tr><td>{label}</td><td>{_fmt_val(gv)}</td><td>{_fmt_val(gtv)}</td><td>{ratio_str}</td></tr>')
            h.append('</table>')

        gt_variant = br.get("ground_truth_variant") or {}
        gt_workflow = br.get("ground_truth_workflow") or ((br.get("reference_validation") or {}).get("workflow")) or []
        if gt_variant or gt_workflow:
            sel_name = gt_variant.get("name") or gt_variant.get("file") or "-"
            sel_step = gt_variant.get("step") or "-"
            sel_reason = gt_variant.get("selection_reason") or ""
            h.append(f'<div style="margin:10px 0 6px;font-size:12px;color:var(--muted)"><strong>Selected GT:</strong> {escape(str(sel_name))} &middot; step={escape(str(sel_step))}' + (f' &middot; {escape(str(sel_reason))}' if sel_reason else '') + '</div>')

        if gt_workflow:
            h.append('<h3 style="margin-top:12px;font-size:13px">Ground Truth Optimization Workflow</h3>')
            h.append('<table><tr><th>Stage</th><th>Selected</th><th>Synth</th><th>Csim</th><th>Cosim</th><th>Latency (ns)</th><th>Fmax</th><th>BRAM</th><th>DSP</th><th>FF</th><th>LUT</th></tr>')
            for stage in gt_workflow:
                report = stage.get("report", {})
                synth_status = (stage.get("synthesis") or {}).get("status", "failed")
                csim_status = (stage.get("csim") or {}).get("status", "not_run")
                cosim_status = (stage.get("cosim") or {}).get("status", "not_run")
                selected_chip = '<span class="chip chip-pass">YES</span>' if stage.get("selected") else '-'
                h.append(
                    f'<tr><td>{escape(stage.get("step_name", "-"))}</td>'
                    f'<td>{selected_chip}</td>'
                    f'<td>{_status_chip(synth_status)}</td>'
                    f'<td>{_status_chip(csim_status)}</td>'
                    f'<td>{_status_chip(cosim_status)}</td>'
                    f'<td>{_fmt_val(report.get("latency_ns"))}</td>'
                    f'<td>{_fmt_val(report.get("fmax_mhz"))}</td>'
                    f'<td>{_fmt_val(report.get("bram"))}</td>'
                    f'<td>{_fmt_val(report.get("dsp"))}</td>'
                    f'<td>{_fmt_val(report.get("ff"))}</td>'
                    f'<td>{_fmt_val(report.get("lut"))}</td></tr>'
                )
            h.append('</table>')

        generated_history = br.get("generated_step_history") or br.get("optimization_history") or br.get("steps") or []
        if multistep and generated_history:
            h.append('<h3 style="margin-top:12px;font-size:13px">Generated Optimization Progress</h3>')
            h.append('<table><tr><th>Stage</th><th>Result</th><th>Latency (ns)</th><th>Fmax</th><th>BRAM</th><th>DSP</th><th>FF</th><th>LUT</th></tr>')
            for stage in generated_history:
                report = stage.get("report", {})
                h.append(
                    f'<tr><td>{escape(stage.get("step_name", stage.get("step", "-")))}</td>'
                    f'<td>{_status_chip(bool(stage.get("success", False)))}</td>'
                    f'<td>{_fmt_val(report.get("latency_ns"))}</td>'
                    f'<td>{_fmt_val(report.get("fmax_mhz"))}</td>'
                    f'<td>{_fmt_val(report.get("bram"))}</td>'
                    f'<td>{_fmt_val(report.get("dsp"))}</td>'
                    f'<td>{_fmt_val(report.get("ff"))}</td>'
                    f'<td>{_fmt_val(report.get("lut"))}</td></tr>'
                )
            h.append('</table>')

        # Rubric score breakdown
        if b["steps"]:
            s = b["steps"][0]
            h.append('<table><tr><th>Rubric Metric</th><th>Score</th><th>Weight</th><th>Weighted</th></tr>')
            weights = {"Synthesis": 5, "C-Sim": 10, "Co-Sim": 10, "Latency": 23,
                       "Fmax": 10, "Resources": 17, "ADP": 15, "Feasibility": 10}
            raw_scores = {
                "Synthesis": 100.0 if s.get("synthesised") else 0.0,
                "C-Sim": s.get("csim_score", 0),
                "Co-Sim": s.get("cosim_score", 0),
                "Latency": s.get("latency_score", 0),
                "Fmax": s.get("fmax_score", 0),
                "Resources": s.get("resource_score", 0),
                "ADP": s.get("adp_score", 0),
                "Feasibility": s.get("feasibility_score", 0),
            }
            for metric, w in weights.items():
                raw = raw_scores.get(metric, 0)
                weighted = raw * w / 100
                h.append(f'<tr><td>{metric}</td><td>{raw:.1f}</td><td>{w}%</td><td>{weighted:.1f}</td></tr>')
            h.append('</table>')

        # Turn history (optimization progress across repair attempts)
        turn_hist = br.get("turn_history", [])
        if turn_hist:
            h.append('<h3 style="margin-top:12px;font-size:13px">Synthesis Turn History</h3>')
            h.append('<table><tr><th>Turn</th><th>Result</th><th>Latency (ns)</th><th>Fmax</th><th>BRAM</th><th>DSP</th><th>FF</th><th>LUT</th></tr>')
            for t in turn_hist:
                rpt = t.get("report", {})
                status = '<span class="chip chip-pass">PASS</span>' if t.get("success") else '<span class="chip chip-fail">FAIL</span>'
                h.append(f'<tr><td>{t.get("turn", "?")}</td><td>{status}</td>'
                         f'<td>{_fmt_val(rpt.get("latency_ns"))}</td>'
                         f'<td>{_fmt_val(rpt.get("fmax_mhz"))}</td>'
                         f'<td>{_fmt_val(rpt.get("bram"))}</td>'
                         f'<td>{_fmt_val(rpt.get("dsp"))}</td>'
                         f'<td>{_fmt_val(rpt.get("ff"))}</td>'
                         f'<td>{_fmt_val(rpt.get("lut"))}</td></tr>')
            h.append('</table>')

        h.append('</details>')

    # ── Footer ──────────────────────────────────────────────────────
    h.append(f"""
<footer>
  C-to-HLS Translation Pipeline &middot; Report generated {now}<br>
  Target: xc7a100t-csg324-1 (Artix-7 100T) &middot; Clock: 4 ns (250 MHz)
</footer>
</div>
</body>
</html>""")

    return "\n".join(h)


def _fmt_val(v) -> str:
    """Format a value for display in tables."""
    if v is None or v == "-":
        return "-"
    try:
        fv = float(v)
        if fv == int(fv) and abs(fv) < 1e12:
            return f"{int(fv):,}"
        return f"{fv:,.2f}"
    except (ValueError, TypeError):
        return str(v)


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from C-to-HLS results")
    parser.add_argument("--results", default="results", help="Path to results directory")
    parser.add_argument("--output", default="report.html", help="Output HTML file path")
    parser.add_argument("--multistep", action="store_true", help="Score multistep results")
    args = parser.parse_args()

    print(f"Loading rubric scores from {args.results}/ ...")
    rubric_data = _run_rubric(args.results, args.multistep)
    print(f"  {len(rubric_data)} benchmarks scored")

    print("Loading per-benchmark results ...")
    bench_results = _load_bench_results(args.results, args.multistep)
    print(f"  {len(bench_results)} result files loaded")

    print(f"Generating HTML report ...")
    html = generate_html(rubric_data, bench_results, args.multistep)

    out_path = Path(args.output)
    out_path.write_text(html)
    print(f"Report written to {out_path} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()

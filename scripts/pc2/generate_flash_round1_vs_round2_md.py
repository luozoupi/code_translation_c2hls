#!/usr/bin/env python3
"""Generate round-1 vs round-2 flash comparison markdown."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PC2 = REPO / "artifacts" / "pc2"
OUT = PC2 / "flash_comparison_round1_vs_round2_20260621.md"

R2_STAMP = "20260621_075846"

PAIRS: list[tuple[str, str, str, str]] = [
    # label, round1 dir, round2 dir, short key
    ("Noskills (legacy)", "flash_noskills_20260620_004507", f"flash_noskills_{R2_STAMP}", "nosk_leg"),
    ("Bn 2+2 (legacy)", "flash_skills_20260620_004507", f"flash_skills_{R2_STAMP}", "bn22_leg"),
    ("All+avoids (legacy)", "flash_all_skills_avoids_global_20260620_113247", f"flash_all_skills_avoids_global_{R2_STAMP}", "aav_leg"),
    ("No avoids (legacy)", "flash_all_skills_no_avoids_global_20260620_113247", f"flash_all_skills_no_avoids_global_{R2_STAMP}", "nav_leg"),
    ("Noskills (new)", "flash_noskills_new_20260621_020847", f"flash_noskills_new_{R2_STAMP}", "nosk_new"),
    ("Bn 2+2 (new)", "flash_bn_skills_new_2_2_20260621_020847", f"flash_bn_skills_new_2_2_{R2_STAMP}", "bn22_new"),
    ("Bn 4+2 (new)", "flash_bn_skills_new_4_2_20260621_020847", f"flash_bn_skills_new_4_2_{R2_STAMP}", "bn42_new"),
    ("Bn 6+2 (new)", "flash_bn_skills_new_6_2_20260621_020847", f"flash_bn_skills_new_6_2_{R2_STAMP}", "bn62_new"),
    ("All+avoids (new)", "flash_all_new_skills_avoids_global_20260621_020847", f"flash_all_new_skills_avoids_global_{R2_STAMP}", "aav_new"),
    ("No avoids (new)", "flash_all_new_skills_no_avoids_global_20260621_020847", f"flash_all_new_skills_no_avoids_global_{R2_STAMP}", "nav_new"),
]

STYLE = """<style>
table.flash-cmp { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.85em; }
table.flash-cmp th, table.flash-cmp td { border: 1px solid #ccc; padding: 4px 8px; white-space: nowrap; }
table.flash-cmp th { background: #f5f5f5; font-weight: 600; }
table.flash-cmp td:first-child, table.flash-cmp th:first-child { text-align: left !important; }
table.flash-cmp .fail { color: #c00; font-weight: 600; }
table.flash-meta { border-collapse: collapse; font-size: 0.9em; }
table.flash-meta th, table.flash-meta td { border: 1px solid #ccc; padding: 4px 10px; }
table.flash-meta th { background: #f5f5f5; text-align: left; width: 220px; }
</style>"""


def load_matrix(name: str) -> dict[str, dict[str, Any]]:
    return {r["bench"]: r for r in json.loads((PC2 / name / "matrix.json").read_text())}


def lat(row: dict[str, Any]) -> int | None:
    if row.get("status") != "ok":
        return None
    v = ((row.get("summary") or {}).get("synth_report") or {}).get("latency_cycles")
    return int(v) if v is not None else None


def gt_lat(row: dict[str, Any]) -> int | None:
    vgt = (row.get("summary") or {}).get("vs_ground_truth") or {}
    lc = vgt.get("latency_cycles") or {}
    if lc.get("ground_truth") is not None:
        return int(lc["ground_truth"])
    br = (row.get("summary") or {}).get("baseline_report") or {}
    v = br.get("latency_cycles")
    return int(v) if v is not None else None


def gt_ratio(row: dict[str, Any]) -> float | None:
    vgt = (row.get("summary") or {}).get("vs_ground_truth") or {}
    lc = vgt.get("latency_cycles") or {}
    if lc.get("ratio") is not None:
        return float(lc["ratio"])
    g, s = gt_lat(row), lat(row)
    if g and s:
        return s / g
    return None


def vgt_gen(row: dict[str, Any]) -> int | None:
    lc = ((row.get("summary") or {}).get("vs_ground_truth") or {}).get("latency_cycles") or {}
    if lc.get("generated") is not None:
        return int(lc["generated"])
    return lat(row)


def fmt_cycles(v: int | float | None, *, fail: bool = False) -> str:
    if fail:
        return '<td class="fail" style="text-align:right">FAIL</td>'
    if v is None:
        return '<td style="text-align:right">—</td>'
    return f'<td style="text-align:right">{int(v):,}</td>'


def fmt_ratio(v: float | None) -> str:
    if v is None:
        return '<td style="text-align:right">—</td>'
    return f'<td style="text-align:right">{v:.3f}</td>'


def fmt_pct(ch: float | None) -> str:
    if ch is None:
        return '<td style="text-align:right">—</td>'
    return f'<td style="text-align:right">{ch * 100:+.1f}%</td>'


def bench_short(bench: str) -> str:
    return bench.removeprefix("hlsfactory_")


def geo_ratios(m: dict[str, dict[str, Any]]) -> float | None:
    rs = [x for x in (gt_ratio(r) for r in m.values() if r["status"] == "ok") if x and x > 0]
    return math.exp(sum(math.log(x) for x in rs) / len(rs)) if rs else None


def h2h_r2_vs_r1(r1: dict[str, dict[str, Any]], r2: dict[str, dict[str, Any]], benches: list[str]) -> tuple[int, int, int, int, int]:
    r2_w = r1_w = tie = skip = big = 0
    for bench in benches:
        if bench == "hlsfactory_doitgen":
            skip += 1
            continue
        l1, l2 = lat(r1[bench]), lat(r2[bench])
        if l1 is None or l2 is None:
            skip += 1
            continue
        if l1 and abs((l2 - l1) / l1) > 0.5:
            big += 1
        if l2 < l1 * 0.999:
            r2_w += 1
        elif l1 < l2 * 0.999:
            r1_w += 1
        else:
            tie += 1
    return r2_w, r1_w, tie, skip, big


def stability(m1: dict[str, dict[str, Any]], m2: dict[str, dict[str, Any]], benches: list[str]) -> tuple[int, int, int]:
    within_1 = within_50 = far = 0
    for bench in benches:
        if bench == "hlsfactory_doitgen":
            continue
        l1, l2 = lat(m1[bench]), lat(m2[bench])
        if l1 is None or l2 is None:
            continue
        rel = abs(l2 - l1) / max(l1, 1)
        if rel < 0.01:
            within_1 += 1
        elif rel < 0.5:
            within_50 += 1
        else:
            far += 1
    return within_1, within_50, far


def table_open(cols: list[tuple[str, str, str]]) -> str:
    width = max(7, int(90 / len(cols)))
    colgroup = "\n".join(f'  <col style="width:{width}%">' for _ in cols)
    headers = "\n".join(f'  <th style="text-align:{a}">{h}</th>' for _, h, a in cols)
    return (
        '<table class="flash-cmp">\n<colgroup>\n'
        f"{colgroup}\n</colgroup>\n<thead><tr>\n{headers}\n</tr></thead>\n<tbody>"
    )


def table_close() -> str:
    return "</tbody></table>"


def main() -> None:
    data_r1 = {k: load_matrix(d1) for _, d1, _, k in PAIRS}
    data_r2 = {k: load_matrix(d2) for _, _, d2, k in PAIRS}
    benches = sorted(next(iter(data_r1.values())).keys())

    lines: list[str] = [
        "# Flash HLSFactory — Round 1 vs Round 2",
        "",
        STYLE,
        "",
        '<table class="flash-meta">',
        "<thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>",
        "<tr><td>Round 1 legacy stamp (noskills / bn)</td><td><code>20260620_004507</code></td></tr>",
        "<tr><td>Round 1 legacy stamp (global)</td><td><code>20260620_113247</code></td></tr>",
        "<tr><td>Round 1 new skills stamp</td><td><code>20260621_020847</code></td></tr>",
        f"<tr><td>Round 2 stamp (all 10 modes)</td><td><code>{R2_STAMP}</code></td></tr>",
        "<tr><td>Metric</td><td>Final flash-step synthesis latency (cycles), lower is better</td></tr>",
        "<tr><td>Success</td><td>27/28 per mode per round (<code>doitgen</code> fails gold-ref gate)</td></tr>",
        "<tr><td>Round 2 settings</td><td>watch 60s, compute walltime 12h, auto-stop on success</td></tr>",
        "</tbody></table>",
        "",
        "## Summary — paired modes (Round 2 vs Round 1)",
        "",
        table_open([
            ("m", "Mode", "left"),
            ("r1", "Round 1 artifact", "left"),
            ("r2", "Round 2 artifact", "left"),
            ("ok", "OK", "right"),
            ("geo1", "R1 geo/GT", "right"),
            ("geo2", "R2 geo/GT", "right"),
            ("w2", "R2 wins", "right"),
            ("w1", "R1 wins", "right"),
            ("t", "Ties", "right"),
            ("big", "&gt;50% Δ", "right"),
        ]),
    ]

    tot_r2 = tot_r1 = tot_tie = tot_big = 0
    for label, d1, d2, key in PAIRS:
        r1, r2 = data_r1[key], data_r2[key]
        ok1 = sum(1 for r in r1.values() if r["status"] == "ok")
        ok2 = sum(1 for r in r2.values() if r["status"] == "ok")
        g1, g2 = geo_ratios(r1), geo_ratios(r2)
        w2, w1, tie, _, big = h2h_r2_vs_r1(r1, r2, benches)
        tot_r2 += w2
        tot_r1 += w1
        tot_tie += tie
        tot_big += big
        lines.append(
            f'<tr><td style="text-align:left">{label}</td>'
            f'<td style="text-align:left"><code>{d1}</code></td>'
            f'<td style="text-align:left"><code>{d2}</code></td>'
            f'<td style="text-align:right">{ok1}/{ok2}</td>'
            f'<td style="text-align:right">{g1:.4f}</td>'
            f'<td style="text-align:right">{g2:.4f}</td>'
            f'<td style="text-align:right">{w2}</td>'
            f'<td style="text-align:right">{w1}</td>'
            f'<td style="text-align:right">{tie}</td>'
            f'<td style="text-align:right">{big}</td></tr>'
        )
    lines.append(
        f'<tr><td style="text-align:left"><strong>Total</strong></td>'
        f'<td colspan="5"></td>'
        f'<td style="text-align:right"><strong>{tot_r2}</strong></td>'
        f'<td style="text-align:right"><strong>{tot_r1}</strong></td>'
        f'<td style="text-align:right"><strong>{tot_tie}</strong></td>'
        f'<td style="text-align:right"><strong>{tot_big}</strong></td></tr>'
    )
    lines.append(table_close())

    # Stability
    lines += ["", "## Latency stability (same mode, two runs)", ""]
    lines.append(
        table_open([
            ("m", "Mode", "left"),
            ("a", "Within 1%", "right"),
            ("b", "Within 50%", "right"),
            ("c", "&gt;50% apart", "right"),
        ])
    )
    for label, _, _, key in PAIRS:
        w1, w50, far = stability(data_r1[key], data_r2[key], benches)
        lines.append(
            f'<tr><td style="text-align:left">{label}</td>'
            f'<td style="text-align:right">{w1}/27</td>'
            f'<td style="text-align:right">{w50}/27</td>'
            f'<td style="text-align:right">{far}/27</td></tr>'
        )
    lines.append(table_close())

    # Per-pair latency + GT ratio sections
    for label, d1, d2, key in PAIRS:
        r1, r2 = data_r1[key], data_r2[key]
        lines += ["", f"## {label} — latency (cycles)", ""]
        lines.append(
            table_open([
                ("b", "Benchmark", "left"),
                ("gt", "Ground truth", "right"),
                ("r1", "Round 1", "right"),
                ("r2", "Round 2", "right"),
                ("pct", "Δ R2 vs R1", "right"),
                ("w", "Better", "right"),
            ])
        )
        for bench in benches:
            short = bench_short(bench)
            row1, row2 = r1[bench], r2[bench]
            fail = row1["status"] != "ok" and row2["status"] != "ok"
            gt = gt_lat(row1) or gt_lat(row2)
            l1, l2 = lat(row1), lat(row2)
            cells = [f'<td style="text-align:left"><code>{short}</code></td>', fmt_cycles(gt)]
            if row1["status"] != "ok":
                cells.append(fmt_cycles(None, fail=True))
            else:
                cells.append(fmt_cycles(l1))
            if row2["status"] != "ok":
                cells.append(fmt_cycles(None, fail=True))
            else:
                cells.append(fmt_cycles(l2))
            if l1 is not None and l2 is not None and l1 > 0:
                ch = (l2 - l1) / l1
                cells.append(fmt_pct(ch))
                if l2 < l1 * 0.999:
                    who = "R2"
                elif l1 < l2 * 0.999:
                    who = "R1"
                else:
                    who = "tie"
                cells.append(f'<td style="text-align:right"><strong>{who}</strong></td>')
            else:
                cells.append(fmt_pct(None))
                cells.append('<td style="text-align:right">—</td>')
            lines.append("<tr>" + "".join(cells) + "</tr>")
        lines.append(table_close())

        lines += ["", f"### {label} — ground-truth latency ratio (synth / GT)", ""]
        lines.append(
            table_open([
                ("b", "Benchmark", "left"),
                ("r1", "Round 1", "right"),
                ("r2", "Round 2", "right"),
            ])
        )
        for bench in benches:
            short = bench_short(bench)
            cells = [f'<td style="text-align:left"><code>{short}</code></td>']
            for m in (r1, r2):
                row = m[bench]
                if row["status"] != "ok":
                    cells.append('<td class="fail" style="text-align:right">FAIL</td>')
                else:
                    cells.append(fmt_ratio(gt_ratio(row)))
            lines.append("<tr>" + "".join(cells) + "</tr>")
        lines.append(table_close())

    # Combined legacy latency table
    lines += ["", "## Combined latency — legacy modes (Round 1 vs Round 2)", ""]
    leg_keys = ["nosk_leg", "bn22_leg", "aav_leg", "nav_leg"]
    leg_labels = {k: lab for lab, _, _, k in PAIRS if k in leg_keys}
    lines.append(
        table_open(
            [("b", "Benchmark", "left"), ("gt", "GT", "right")]
            + [(f"{k}_r1", f"{leg_labels[k]} R1", "right") for k in leg_keys]
            + [(f"{k}_r2", f"{leg_labels[k]} R2", "right") for k in leg_keys]
        )
    )
    for bench in benches:
        gt = gt_lat(data_r1["nosk_leg"][bench]) or gt_lat(data_r2["nosk_leg"][bench])
        cells = [f'<td style="text-align:left"><code>{bench_short(bench)}</code></td>', fmt_cycles(gt)]
        for k in leg_keys:
            row = data_r1[k][bench]
            cells.append(fmt_cycles(None, fail=row["status"] != "ok") if row["status"] != "ok" else fmt_cycles(lat(row)))
        for k in leg_keys:
            row = data_r2[k][bench]
            cells.append(fmt_cycles(None, fail=row["status"] != "ok") if row["status"] != "ok" else fmt_cycles(lat(row)))
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append(table_close())

    # Combined new latency table
    lines += ["", "## Combined latency — new skills modes (Round 1 vs Round 2)", ""]
    new_keys = ["nosk_new", "bn22_new", "bn42_new", "bn62_new", "aav_new", "nav_new"]
    new_labels = {k: lab.replace(" (new)", "") for lab, _, _, k in PAIRS if k in new_keys}
    lines.append(
        table_open(
            [("b", "Benchmark", "left"), ("gt", "GT", "right")]
            + [(f"{k}_r1", f"{new_labels[k]} R1", "right") for k in new_keys]
            + [(f"{k}_r2", f"{new_labels[k]} R2", "right") for k in new_keys]
        )
    )
    for bench in benches:
        gt = gt_lat(data_r1["nosk_new"][bench]) or gt_lat(data_r2["nosk_new"][bench])
        cells = [f'<td style="text-align:left"><code>{bench_short(bench)}</code></td>', fmt_cycles(gt)]
        for k in new_keys:
            row = data_r1[k][bench]
            cells.append(fmt_cycles(None, fail=row["status"] != "ok") if row["status"] != "ok" else fmt_cycles(lat(row)))
        for k in new_keys:
            row = data_r2[k][bench]
            cells.append(fmt_cycles(None, fail=row["status"] != "ok") if row["status"] != "ok" else fmt_cycles(lat(row)))
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append(table_close())

    # Large swings
    lines += ["", "## Large swings (&gt;2× Round 2 / Round 1)", ""]
    lines.append(
        table_open([
            ("m", "Mode", "left"),
            ("b", "Benchmark", "left"),
            ("r1", "Round 1", "right"),
            ("r2", "Round 2", "right"),
            ("x", "R2/R1", "right"),
        ])
    )
    any_swing = False
    for label, _, _, key in PAIRS:
        r1, r2 = data_r1[key], data_r2[key]
        swings: list[tuple[float, str, int, int, float]] = []
        for bench in benches:
            if bench == "hlsfactory_doitgen":
                continue
            l1, l2 = lat(r1[bench]), lat(r2[bench])
            if l1 is None or l2 is None or l1 == 0:
                continue
            ratio = l2 / l1
            if ratio > 2 or ratio < 0.5:
                swings.append((max(ratio, 1 / ratio), bench_short(bench), l1, l2, ratio))
        for _, short, l1, l2, ratio in sorted(swings, key=lambda x: -x[0]):
            any_swing = True
            lines.append(
                f'<tr><td style="text-align:left">{label}</td>'
                f'<td style="text-align:left"><code>{short}</code></td>'
                f'<td style="text-align:right">{l1:,}</td>'
                f'<td style="text-align:right">{l2:,}</td>'
                f'<td style="text-align:right">{ratio:.2f}×</td></tr>'
            )
    if not any_swing:
        lines.append('<tr><td colspan="5" style="text-align:left">None</td></tr>')
    lines.append(table_close())

    lines += [
        "",
        "## Conclusions",
        "",
        "1. **Success is stable:** every mode scores **27/28** in both rounds (`doitgen` fails consistently).",
        f"2. **Head-to-head latency:** Round 2 wins **{tot_r2}** bench comparisons, Round 1 wins **{tot_r1}**, ties **{tot_tie}** — roughly even overall.",
        f"3. **High variance:** **{tot_big}** of 270 bench×mode pairs differ by **&gt;50%** between rounds on the same configuration. Only a handful of benches are within **1%** across repeats.",
        "4. **vs ground truth:** geo-mean rankings shift between rounds (e.g. No avoids legacy was best in R1; No avoids **new** is best in R2). All modes remain far below GT on most benches.",
        "5. **Interpretation:** Round-to-round differences are dominated by **LLM sampling noise**, not by skills-file changes (each paired comparison holds skills constant). Use **multiple runs** or aggregation before drawing strong conclusions about skills.",
        "",
        "See also: `artifacts/pc2/flash_comparison_20260621.md` (Round 1 legacy vs new), `artifacts/pc2/flash_comparison_20260620.md` (Round 1 legacy only).",
        "",
    ]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT} ({len(lines)} sections/lines)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate presentation-ready HTML-styled markdown for all flash synthesis tests."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PC2 = REPO / "artifacts" / "pc2"
OUT = PC2 / f"flash_presentation_summary_{date.today().strftime('%Y%m%d')}.md"

STYLE = """<style>
table.flash-cmp { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.85em; }
table.flash-cmp th, table.flash-cmp td { border: 1px solid #ccc; padding: 4px 8px; white-space: nowrap; }
table.flash-cmp th { background: #f5f5f5; font-weight: 600; }
table.flash-cmp td:first-child, table.flash-cmp th:first-child { text-align: left !important; }
table.flash-cmp .fail { color: #c00; font-weight: 600; }
table.flash-cmp .best { background: #e8f5e9; font-weight: 600; }
table.flash-meta { border-collapse: collapse; font-size: 0.9em; margin-bottom: 1em; }
table.flash-meta th, table.flash-meta td { border: 1px solid #ccc; padding: 4px 10px; }
table.flash-meta th { background: #f5f5f5; text-align: left; width: 240px; }
table.flash-rec { border-collapse: collapse; width: 100%; font-size: 0.95em; }
table.flash-rec th, table.flash-rec td { border: 1px solid #ccc; padding: 8px 12px; vertical-align: top; }
table.flash-rec th { background: #e3f2fd; text-align: left; }
</style>"""

MODES: list[tuple[str, str, str, str]] = [
    ("Noskills (old)", "flash_noskills_20260620_004507", "nosk_o", "legacy"),
    ("Bn 2+2 (old)", "flash_skills_20260620_004507", "bn22_o", "legacy"),
    ("All+avoids (old)", "flash_all_skills_avoids_global_20260620_113247", "aav_o", "legacy"),
    ("No avoids (old)", "flash_all_skills_no_avoids_global_20260620_113247", "nav_o", "legacy"),
    ("Noskills (new)", "flash_noskills_new_20260621_020847", "nosk_n", "new"),
    ("Bn 2+2 (new)", "flash_bn_skills_new_2_2_20260621_020847", "bn22_n", "new"),
    ("Bn 4+2 (new)", "flash_bn_skills_new_4_2_20260621_020847", "bn42_n", "new"),
    ("Bn 6+2 (new)", "flash_bn_skills_new_6_2_20260621_020847", "bn62_n", "new"),
    ("All+avoids (new)", "flash_all_new_skills_avoids_global_20260621_020847", "aav_n", "new"),
    ("No avoids (new)", "flash_all_new_skills_no_avoids_global_20260621_020847", "nav_n", "new"),
]

CURATED_STAMP = "20260621_104044"
CURATED_VARIANTS = (
    ("noskills", "Noskills"),
    ("all_avoids_json", "All+avoids json_only"),
    ("all_avoids_llm", "All+avoids json+LLM"),
    ("no_avoids_json", "No avoids json_only"),
    ("no_avoids_llm", "No avoids json+LLM"),
)
CURATED_FOCUSES = ("bottleneck", "warnings", "combined")

BASELINE_3WAY = [
    ("Noskills (new, r3)", "flash_noskills_new_20260622_215520"),
    ("All+avoids (new, r3)", "flash_all_new_skills_avoids_global_20260622_215520"),
    ("No avoids (new, r3)", "flash_all_new_skills_no_avoids_global_20260622_215520"),
]


def load_matrix(dirname: str) -> dict[str, dict[str, Any]]:
    path = PC2 / dirname / "matrix.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return {r["bench"]: r for r in json.loads(path.read_text())}


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


def geo_mean(vals: list[float]) -> float:
    vals = [v for v in vals if v and v > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def bench_short(bench: str) -> str:
    return bench.removeprefix("hlsfactory_")


def h2h(
    data: dict[str, dict[str, dict[str, Any]]],
    key_a: str,
    key_b: str,
    benches: list[str],
) -> tuple[int, int, int]:
    aw = bw = tie = 0
    for bench in benches:
        if bench.endswith("doitgen"):
            continue
        la, lb = lat(data[key_a][bench]), lat(data[key_b][bench])
        if la is None or lb is None:
            continue
        if lb < la * 0.999:
            bw += 1
        elif la < lb * 0.999:
            aw += 1
        else:
            tie += 1
    return aw, bw, tie


def best_wins(
    data: dict[str, dict[str, dict[str, Any]]],
    keys: list[str],
    benches: list[str],
) -> dict[str, float]:
    wins = {k: 0.0 for k in keys}
    for bench in benches:
        if bench.endswith("doitgen"):
            continue
        best: int | None = None
        winners: list[str] = []
        for k in keys:
            l = lat(data[k][bench])
            if l is None:
                continue
            if best is None or l < best:
                best = l
                winners = [k]
            elif best is not None and abs(l - best) / max(best, 1) < 0.001:
                winners.append(k)
        if winners:
            share = 1.0 / len(winners)
            for w in winners:
                wins[w] += share
    return wins


def table_open(cols: list[tuple[str, str, str]]) -> str:
    width = max(8, int(88 / max(len(cols), 1)))
    colgroup = "\n".join(f'  <col style="width:{width}%">' for _ in cols)
    headers = "\n".join(
        f'  <th style="text-align:{align}">{html}</th>' for _, html, align in cols
    )
    return (
        '<table class="flash-cmp">\n<colgroup>\n'
        f"{colgroup}\n</colgroup>\n<thead><tr>\n{headers}\n</tr></thead>\n<tbody>"
    )


def table_close() -> str:
    return "</tbody></table>"


def fmt_cycles(v: int | float | None, *, fail: bool = False) -> str:
    if fail:
        return '<td class="fail" style="text-align:right">FAIL</td>'
    if v is None:
        return '<td style="text-align:right">—</td>'
    return f'<td style="text-align:right">{int(v):,}</td>'


def fmt_ratio(v: float | None) -> str:
    if v is None:
        return '<td style="text-align:right">—</td>'
    return f'<td style="text-align:right">{v:.4f}</td>'


def mode_stats(data: dict, short: str, benches: list[str]) -> dict[str, Any]:
    m = data[short]
    ok = sum(1 for r in m.values() if r["status"] == "ok")
    ratios = [gt_ratio(r) for r in m.values() if r["status"] == "ok"]
    ratios = [x for x in ratios if x is not None and x > 0]
    faster = sum(1 for x in ratios if x < 0.999)
    slower = sum(1 for x in ratios if x > 1.001)
    tie = len(ratios) - faster - slower
    return {
        "ok": ok,
        "geo": geo_mean(ratios),
        "faster": faster,
        "slower": slower,
        "tie": tie,
        "fail": 28 - ok,
    }


def curated_dir(variant: str, focus: str) -> str:
    return f"flash_curated_{variant}_{focus}_{CURATED_STAMP}"


def main() -> None:
    labels = {short: label for label, _, short, _ in MODES}
    data = {short: load_matrix(d) for _, d, short, _ in MODES}
    benches = sorted(next(iter(data.values())).keys())
    keys = [short for _, _, short, _ in MODES]
    wins = best_wins(data, keys, benches)

    curated_data: dict[str, dict[str, dict[str, Any]]] = {}
    curated_labels: dict[str, str] = {}
    for variant, vlabel in CURATED_VARIANTS:
        for focus in CURATED_FOCUSES:
            dirname = curated_dir(variant, focus)
            key = f"{variant}_{focus}"
            curated_labels[key] = f"{vlabel} ({focus})"
            curated_data[key] = load_matrix(dirname)
    curated_keys = list(curated_data.keys())
    curated_wins = best_wins(curated_data, curated_keys, benches)

    # baseline 3way if present
    b3_data: dict[str, dict[str, dict[str, Any]]] = {}
    for label, dirname in BASELINE_3WAY:
        p = PC2 / dirname / "matrix.json"
        if p.exists():
            b3_data[label] = load_matrix(dirname)

    lines: list[str] = [
        "# Flash Synthesis — Presentation Summary (All Tests, Excl. Cosim)",
        "",
        STYLE,
        "",
        '<table class="flash-meta">',
        "<thead><tr><th>Field</th><th>Value</th></tr></thead>",
        "<tbody>",
        f"<tr><td>Generated</td><td><code>{date.today().isoformat()}</code></td></tr>",
        "<tr><td>Benchmarks</td><td>28 <code>hlsfactory_*</code> Polybench kernels</td></tr>",
        "<tr><td>Model</td><td><code>mistralai/Devstral-2-123B-Instruct-2512</code></td></tr>",
        "<tr><td>Mode</td><td>Flash (single-shot LLM + csim + csynth)</td></tr>",
        "<tr><td>Metric</td><td>Final flash-step <strong>synthesis latency</strong> (cycles); lower is better</td></tr>",
        "<tr><td>vs GT ratio</td><td>generated_latency / ground_truth_latency; geo-mean across 27 OK benches</td></tr>",
        "<tr><td>Success</td><td>27/28 per run; <code>doitgen</code> fails gold-reference gate everywhere</td></tr>",
        "<tr><td>Legacy stamp</td><td><code>20260620_004507</code> (noskills/bn2+2), <code>20260620_113247</code> (global skills)</td></tr>",
        "<tr><td>New skills (73)</td><td><code>skills_ii_target_miss_solutions_added(73skills).json</code> — main matrix stamp <code>20260621_020847</code></td></tr>",
        f"<tr><td>Curated matrix stamp</td><td><code>{CURATED_STAMP}</code> — 15 runs (5 variants × 3 curation waves); uses <code>(73skills).json</code></td></tr>",
        "<tr><td>New skills (90)</td><td><code>skills_ii_target_miss_solutions_added(90skills).json</code> — all+avoids (new) best stamp <code>20260623_024548</code></td></tr>",
        "<tr><td>Baseline 3-way stamp (optional)</td><td><code>20260622_215520</code> — 85-skill intermediate (no frozen file)</td></tr>",
        "<tr><td>Deterministic runs</td><td>10 modes (4 legacy + 6 new)</td></tr>",
        "<tr><td>Curated runs</td><td>15 modes</td></tr>",
        "<tr><td>Excluded</td><td>Cosim / cosim-repair (separate experiment axis)</td></tr>",
        "</tbody></table>",
        "",
        "## Executive summary — what to present",
        "",
        '<table class="flash-rec">',
        "<thead><tr><th>Rank</th><th>Recommended mode</th><th>Key numbers</th><th>When to use in slides</th></tr></thead>",
        "<tbody>",
        '<tr><td><strong>1</strong></td><td><strong>No avoids (old)</strong><br><code>flash_all_skills_no_avoids_global_20260620_113247</code></td>'
        f'<td>Best-latency wins <strong>{wins["nav_o"]:.1f}/27</strong>; geo-mean lat/GT <strong>{mode_stats(data, "nav_o", benches)["geo"]:.4f}</strong>; '
        f'vs noskills (old) <strong>18–9</strong>; never slower than GT</td>'
        "<td>Best overall flash synthesis latency; clearest skills win story on old 55-skill library</td></tr>",
        '<tr><td><strong>2</strong></td><td><strong>All+avoids (new)</strong><br><code>flash_all_new_skills_avoids_global_20260621_020847</code></td>'
        f'<td>Best-latency wins <strong>{wins["aav_n"]:.1f}/27</strong>; geo-mean <strong>{mode_stats(data, "aav_n", benches)["geo"]:.4f}</strong>; '
        f'vs noskills (new) <strong>15–11</strong></td>'
        "<td>Best mode on the new 73-skill library</td></tr>",
        '<tr><td><strong>3</strong></td><td><strong>Noskills (old)</strong> baseline<br><code>flash_noskills_20260620_004507</code></td>'
        f'<td>Geo-mean <strong>{mode_stats(data, "nosk_o", benches)["geo"]:.4f}</strong>; wins <strong>{wins["nosk_o"]:.1f}/27</strong>; '
        "0 benches slower than GT</td>"
        "<td>Strong no-skills baseline for “skills vs no skills” comparison</td></tr>",
        '<tr><td>4</td><td><strong>No avoids (new)</strong></td>'
        f'<td>Geo-mean <strong>{mode_stats(data, "nav_n", benches)["geo"]:.4f}</strong>; paired new beats old <strong>14–9</strong></td>'
        "<td>New library helps when avoid-tier is dropped</td></tr>",
        '<tr><td>5</td><td><strong>Best curated:</strong> No avoids json_only (bottleneck)</td>'
        f'<td>Wins <strong>{curated_wins["no_avoids_json_bottleneck"]:.1f}/27</strong>; geo-mean <strong>{mode_stats(curated_data, "no_avoids_json_bottleneck", benches)["geo"]:.4f}</strong></td>'
        "<td>LLM curation highlight — does not beat deterministic no-avoids (old) on geo-mean</td></tr>",
        '<tr><td>Avoid</td><td>All+avoids (old), Bn 6+2 (new)</td>'
        f'<td>Geo-means <strong>{mode_stats(data, "aav_o", benches)["geo"]:.4f}</strong> / <strong>{mode_stats(data, "bn62_n", benches)["geo"]:.4f}</strong>; '
        f'wins <strong>{wins["bn62_n"]:.1f}/27</strong> for bn6+2</td>'
        "<td>Weak modes — do not lead with these</td></tr>",
        "</tbody></table>",
        "",
        "## 1. All deterministic modes — ranked by geo-mean lat/GT",
        "",
        table_open([
            ("rank", "Rank", "right"),
            ("mode", "Mode", "left"),
            ("family", "Family", "left"),
            ("root", "Artifact root", "left"),
            ("ok", "OK", "right"),
            ("wins", "Best-latency wins", "right"),
            ("geo", "Geo-mean lat/GT", "right"),
            ("faster", "Faster than GT", "right"),
            ("slower", "Slower than GT", "right"),
            ("tie", "Tie ~1.0", "right"),
        ]),
    ]

    ranked = []
    for label, dirname, short, family in MODES:
        st = mode_stats(data, short, benches)
        ranked.append((st["geo"], label, dirname, short, family, st))
    ranked.sort(key=lambda x: x[0])

    for i, (_, label, dirname, short, family, st) in enumerate(ranked, 1):
        cls = ' class="best"' if short == "nav_o" else ""
        lines.append(
            f"<tr><td style=\"text-align:right\">{i}</td>"
            f'<td style="text-align:left"{cls}>{label}</td>'
            f'<td style="text-align:left">{family}</td>'
            f'<td style="text-align:left"><code>{dirname}</code></td>'
            f'<td style="text-align:right">{st["ok"]}/28</td>'
            f'<td style="text-align:right">{wins[short]:.1f}/27</td>'
            f'<td style="text-align:right"{cls}>{st["geo"]:.4f}</td>'
            f'<td style="text-align:right">{st["faster"]}</td>'
            f'<td style="text-align:right">{st["slower"]}</td>'
            f'<td style="text-align:right">{st["tie"]}</td></tr>'
        )
    lines.append(table_close())

    # vs GT full table
    lines += ["", "## 2. vs ground truth — all deterministic modes", ""]
    lines.append(
        table_open([
            ("mode", "Mode", "left"),
            ("faster", "Faster than GT", "right"),
            ("slower", "Slower than GT", "right"),
            ("tie", "Tie (~1.0)", "right"),
            ("geo", "Geo-mean ratio", "right"),
            ("fail", "Bench fail", "right"),
        ])
    )
    for label, _, short, _ in MODES:
        st = mode_stats(data, short, benches)
        lines.append(
            f'<tr><td style="text-align:left">{label}</td>'
            f'<td style="text-align:right">{st["faster"]}</td>'
            f'<td style="text-align:right">{st["slower"]}</td>'
            f'<td style="text-align:right">{st["tie"]}</td>'
            f'<td style="text-align:right">{st["geo"]:.4f}</td>'
            f'<td style="text-align:right">{st["fail"]}</td></tr>'
        )
    lines.append(table_close())

    # slower than GT detail
    lines += ["", "### Benches slower than GT (ratio &gt; 1.001)", ""]
    lines.append(
        table_open([
            ("mode", "Mode", "left"),
            ("bench", "Benchmark", "left"),
            ("ratio", "Ratio", "right"),
            ("gen", "Generated cycles", "right"),
            ("gt", "GT cycles", "right"),
        ])
    )
    any_slow = False
    for label, _, short, _ in MODES:
        for bench, r in sorted(data[short].items()):
            if r["status"] != "ok":
                continue
            ratio = gt_ratio(r)
            if ratio is None or ratio <= 1.001:
                continue
            any_slow = True
            lines.append(
                f'<tr><td style="text-align:left">{label}</td>'
                f'<td style="text-align:left"><code>{bench_short(bench)}</code></td>'
                f'<td style="text-align:right">{ratio:.3f}</td>'
                f'<td style="text-align:right">{lat(r):,}</td>'
                f'<td style="text-align:right">{gt_lat(r):,}</td></tr>'
            )
    if not any_slow:
        lines.append('<tr><td colspan="5">None</td></tr>')
    lines.append(table_close())

    # h2h vs noskills old - ALL modes except nosk_o
    lines += ["", "## 3. Head-to-head vs Noskills (old) — latency wins (of 27 benches)", ""]
    lines.append(
        table_open([
            ("mode", "Mode", "left"),
            ("mw", "Mode wins", "right"),
            ("nw", "Noskills (old) wins", "right"),
            ("tie", "Ties", "right"),
        ])
    )
    for label, _, short, _ in MODES:
        if short == "nosk_o":
            continue
        mw, nw, tie = h2h(data, short, "nosk_o", benches)
        lines.append(
            f'<tr><td style="text-align:left">{label}</td>'
            f'<td style="text-align:right">{mw}</td>'
            f'<td style="text-align:right">{nw}</td>'
            f'<td style="text-align:right">{tie}</td></tr>'
        )
    lines.append(table_close())

    # h2h vs noskills new
    lines += ["", "## 4. Head-to-head vs Noskills (new) — latency wins (of 27 benches)", ""]
    lines.append(
        table_open([
            ("mode", "Mode", "left"),
            ("mw", "Mode wins", "right"),
            ("nw", "Noskills (new) wins", "right"),
            ("tie", "Ties", "right"),
        ])
    )
    for label, _, short, _ in MODES:
        if short == "nosk_n":
            continue
        mw, nw, tie = h2h(data, short, "nosk_n", benches)
        lines.append(
            f'<tr><td style="text-align:left">{label}</td>'
            f'<td style="text-align:right">{mw}</td>'
            f'<td style="text-align:right">{nw}</td>'
            f'<td style="text-align:right">{tie}</td></tr>'
        )
    lines.append(table_close())

    # paired old vs new
    lines += ["", "## 5. Paired old → new (same skill policy)", ""]
    pairs = [
        ("nosk_n", "nosk_o", "Noskills"),
        ("bn22_n", "bn22_o", "Bn 2+2"),
        ("aav_n", "aav_o", "All+avoids"),
        ("nav_n", "nav_o", "No avoids"),
    ]
    lines.append(
        table_open([
            ("pair", "Policy", "left"),
            ("new", "New wins", "right"),
            ("old", "Old wins", "right"),
            ("tie", "Ties", "right"),
            ("verdict", "Verdict", "left"),
        ])
    )
    for new_k, old_k, name in pairs:
        old_w, new_w, tie = h2h(data, old_k, new_k, benches)
        if new_w > old_w:
            verdict = "New library better"
        elif old_w > new_w:
            verdict = "Old library better"
        else:
            verdict = "Even"
        lines.append(
            f'<tr><td style="text-align:left">{name}</td>'
            f'<td style="text-align:right">{new_w}</td>'
            f'<td style="text-align:right">{old_w}</td>'
            f'<td style="text-align:right">{tie}</td>'
            f'<td style="text-align:left"><strong>{verdict}</strong></td></tr>'
        )
    lines.append(table_close())

    # BN sweep
    lines += ["", "## 6. New BN skill-count sweep", ""]
    lines.append(
        table_open([
            ("cmp", "Comparison", "left"),
            ("second", "Second wins", "right"),
            ("first", "First wins", "right"),
            ("tie", "Ties", "right"),
        ])
    )
    for a, b, an, bn in [
        ("bn22_n", "bn42_n", "Bn 2+2", "Bn 4+2"),
        ("bn22_n", "bn62_n", "Bn 2+2", "Bn 6+2"),
        ("bn42_n", "bn62_n", "Bn 4+2", "Bn 6+2"),
    ]:
        aw, bw, tie = h2h(data, a, b, benches)
        lines.append(
            f'<tr><td style="text-align:left">{bn} vs {an}</td>'
            f'<td style="text-align:right">{bw}</td>'
            f'<td style="text-align:right">{aw}</td>'
            f'<td style="text-align:right">{tie}</td></tr>'
        )
    lines.append(table_close())

    # curated summary
    lines += ["", "## 7. LLM-curated skills matrix (15 runs)", ""]
    lines.append(
        table_open([
            ("mode", "Mode", "left"),
            ("root", "Artifact root", "left"),
            ("ok", "OK", "right"),
            ("wins", "Best-latency wins", "right"),
            ("geo", "Geo-mean lat/GT", "right"),
        ])
    )
    curated_ranked = []
    for variant, _ in CURATED_VARIANTS:
        for focus in CURATED_FOCUSES:
            key = f"{variant}_{focus}"
            dirname = curated_dir(variant, focus)
            st = mode_stats(curated_data, key, benches)
            curated_ranked.append((st["geo"], curated_labels[key], dirname, key, st))
    curated_ranked.sort(key=lambda x: x[0])
    for _, label, dirname, key, st in curated_ranked:
        lines.append(
            f'<tr><td style="text-align:left">{label}</td>'
            f'<td style="text-align:left"><code>{dirname}</code></td>'
            f'<td style="text-align:right">{st["ok"]}/28</td>'
            f'<td style="text-align:right">{curated_wins[key]:.1f}/27</td>'
            f'<td style="text-align:right">{st["geo"]:.4f}</td></tr>'
        )
    lines.append(table_close())

    # curated h2h vs noskills old and new (bottleneck wave only)
    lines += ["", "## 8. Curated bottleneck wave vs noskills baselines", ""]
    lines.append(
        table_open([
            ("mode", "Curated mode (bottleneck)", "left"),
            ("vs", "Baseline", "left"),
            ("cw", "Curated wins", "right"),
            ("bw", "Baseline wins", "right"),
            ("tie", "Ties", "right"),
        ])
    )
    bn_keys = [k for k in curated_keys if k.endswith("_bottleneck")]
    for ck in bn_keys:
        for base_short, base_label in [("nosk_o", "Noskills (old)"), ("nosk_n", "Noskills (new)")]:
            merged = {**curated_data, base_short: data[base_short]}
            base_w, cur_w, tie = h2h(merged, base_short, ck, benches)
            lines.append(
                f'<tr><td style="text-align:left">{curated_labels[ck]}</td>'
                f'<td style="text-align:left">{base_label}</td>'
                f'<td style="text-align:right">{cur_w}</td>'
                f'<td style="text-align:right">{base_w}</td>'
                f'<td style="text-align:right">{tie}</td></tr>'
            )
    lines.append(table_close())

    # baseline 3way if available
    if b3_data:
        lines += ["", "## 9. Baseline 3-way re-run (85-skill lib, stamp `20260622_215520`)", ""]
        lines.append(
            table_open([
                ("mode", "Mode", "left"),
                ("ok", "OK", "right"),
                ("geo", "Geo-mean lat/GT", "right"),
                ("faster", "Faster than GT", "right"),
                ("slower", "Slower than GT", "right"),
            ])
        )
        for label, dirname in BASELINE_3WAY:
            if label not in b3_data:
                continue
            short = label
            tmp = {short: b3_data[label]}
            st = mode_stats(tmp, short, benches)
            lines.append(
                f'<tr><td style="text-align:left">{label}</td>'
                f'<td style="text-align:right">{st["ok"]}/28</td>'
                f'<td style="text-align:right">{st["geo"]:.4f}</td>'
                f'<td style="text-align:right">{st["faster"]}</td>'
                f'<td style="text-align:right">{st["slower"]}</td></tr>'
            )
        lines.append(table_close())

        lines += ["", "### r3 vs r2 (20260621_020847) — paired new modes", ""]
        r2_map = {
            "Noskills (new, r3)": "nosk_n",
            "All+avoids (new, r3)": "aav_n",
            "No avoids (new, r3)": "nav_n",
        }
        lines.append(
            table_open([
                ("mode", "Mode", "left"),
                ("r3", "r3 wins", "right"),
                ("r2", "r2 wins", "right"),
                ("tie", "Ties", "right"),
            ])
        )
        for label, r2_short in r2_map.items():
            r3 = b3_data[label]
            tmp_r3 = {"r3": r3}
            tmp_r2 = {"r2": data[r2_short]}
            merged = {"r3": r3, "r2": data[r2_short]}
            r2w, r3w, tie = h2h(merged, "r2", "r3", benches)
            lines.append(
                f'<tr><td style="text-align:left">{label}</td>'
                f'<td style="text-align:right">{r3w}</td>'
                f'<td style="text-align:right">{r2w}</td>'
                f'<td style="text-align:right">{tie}</td></tr>'
            )
        lines.append(table_close())

    # full GT ratio table
    lines += ["", "## 10. Ground-truth latency ratio per benchmark (all deterministic modes)", ""]
    lines.append(
        table_open(
            [("b", "Benchmark", "left")]
            + [(short, labels[short], "right") for _, _, short, _ in MODES]
        )
    )
    for bench in benches:
        cells = [f'<td style="text-align:left"><code>{bench_short(bench)}</code></td>']
        for _, _, short, _ in MODES:
            r = data[short][bench]
            if r["status"] != "ok":
                cells.append('<td class="fail" style="text-align:right">FAIL</td>')
            else:
                cells.append(fmt_ratio(gt_ratio(r)))
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append(table_close())

    # full latency legacy
    lines += ["", "## 11. Latency (cycles) — legacy modes", ""]
    legacy_keys = [short for _, _, short, fam in MODES if fam == "legacy"]
    legacy_labels = {short: labels[short] for short in legacy_keys}
    lines.append(
        table_open(
            [("b", "Benchmark", "left"), ("gt", "Ground truth", "right")]
            + [(k, legacy_labels[k], "right") for k in legacy_keys]
        )
    )
    for bench in benches:
        cells = [f'<td style="text-align:left"><code>{bench_short(bench)}</code></td>']
        gt = gt_lat(data["nosk_o"][bench]) or gt_lat(data["nav_o"][bench])
        cells.append(fmt_cycles(gt))
        for k in legacy_keys:
            r = data[k][bench]
            cells.append(fmt_cycles(lat(r), fail=r["status"] != "ok"))
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append(table_close())

    # full latency new
    lines += ["", "## 12. Latency (cycles) — new skills pack modes", ""]
    new_keys = [short for _, _, short, fam in MODES if fam == "new"]
    new_labels = {short: labels[short] for short in new_keys}
    lines.append(
        table_open(
            [("b", "Benchmark", "left"), ("gt", "Ground truth", "right")]
            + [(k, new_labels[k], "right") for k in new_keys]
        )
    )
    for bench in benches:
        cells = [f'<td style="text-align:left"><code>{bench_short(bench)}</code></td>']
        gt = gt_lat(data["nosk_n"][bench]) or gt_lat(data["nav_n"][bench])
        cells.append(fmt_cycles(gt))
        for k in new_keys:
            r = data[k][bench]
            cells.append(fmt_cycles(lat(r), fail=r["status"] != "ok"))
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append(table_close())

    # narrative conclusions
    total_det = len(MODES)
    lines += [
        "",
        "## 13. Narrative bullets (for slides)",
        "",
        "1. **All flash modes beat ground-truth HLS latency on average** — geo-mean ratios 0.034–0.171 across deterministic modes (~3–17% of GT).",
        f'2. **Champion:** No avoids (old) — **{wins["nav_o"]:.1f}/27** best-latency wins, geo-mean **{mode_stats(data, "nav_o", benches)["geo"]:.4f}**, beats noskills (old) **18–9**.',
        f'3. **Best new-library mode:** All+avoids (new) — **{wins["aav_n"]:.1f}/27** wins, geo-mean **{mode_stats(data, "aav_n", benches)["geo"]:.4f}**, beats noskills (new) **15–11**.',
        f'4. **Noskills baselines:** old geo **{mode_stats(data, "nosk_o", benches)["geo"]:.4f}** (0 slower-than-GT); new geo **{mode_stats(data, "nosk_n", benches)["geo"]:.4f}** (2 slower-than-GT).',
        "5. **Skill policy > skill count:** no-avoids (positive only) beats all+avoids; BN 6+2 is worst among new BN sweeps.",
        "6. **New 73-skill library:** helps no-avoids (14–9 vs old) and noskills (18–7); **hurts** all+avoids (8–15 vs old).",
        f'7. **LLM curation:** best curated run is No avoids json_only (bottleneck) with **{curated_wins["no_avoids_json_bottleneck"]:.1f}/27** wins but geo-mean **{mode_stats(curated_data, "no_avoids_json_bottleneck", benches)["geo"]:.4f}** — does not displace No avoids (old).',
        "8. **Cosim is a separate axis** (not in this report): single-shot repair fixed 5/24 failures; 10-loop repair 6/24.",
        "",
        "## 14. Related reports",
        "",
        "- `artifacts/pc2/flash_comparison_20260621.md` — full deterministic comparison",
        f"- `artifacts/pc2/flash_comparison_curated_{CURATED_STAMP[:8]}.md` — curated matrix detail",
        "- `artifacts/pc2/flash_cosim_comparison_20260622.md` — cosim pass/fail (separate metric)",
        "",
        f"_Generated by `scripts/pc2/generate_flash_presentation_summary_md.py` on {date.today().isoformat()}_",
        "",
    ]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()

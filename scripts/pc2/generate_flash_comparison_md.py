#!/usr/bin/env python3
"""Generate flash comparison markdown from matrix.json artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PC2 = REPO / "artifacts" / "pc2"
OUT = PC2 / "flash_comparison_20260621.md"

MODES: list[tuple[str, str, str, str]] = [
    # label, artifact_dir, short_key, family
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
    sr = (row.get("summary") or {}).get("synth_report") or {}
    v = sr.get("latency_cycles")
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


def bench_short(bench: str) -> str:
    return bench.removeprefix("hlsfactory_")


def h2h(
    data: dict[str, dict[str, dict[str, Any]]],
    key_a: str,
    key_b: str,
    benches: list[str],
) -> tuple[int, int, int, int]:
    """Return wins for b, wins for a, ties, skipped."""
    b_w = a_w = tie = skip = 0
    for bench in benches:
        if bench == "hlsfactory_doitgen":
            skip += 1
            continue
        la, lb = lat(data[key_a][bench]), lat(data[key_b][bench])
        if la is None or lb is None:
            skip += 1
            continue
        if lb < la * 0.999:
            b_w += 1
        elif la < lb * 0.999:
            a_w += 1
        else:
            tie += 1
    return b_w, a_w, tie, skip


def best_wins(
    data: dict[str, dict[str, dict[str, Any]]],
    keys: list[str],
    benches: list[str],
) -> dict[str, float]:
    wins = {k: 0.0 for k in keys}
    for bench in benches:
        if bench == "hlsfactory_doitgen":
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


def winner_tag(keys: list[str], labels: dict[str, str], lats: dict[str, int | None]) -> str:
    ok = {k: lats[k] for k in keys if lats.get(k) is not None}
    if not ok:
        return "—"
    best = min(ok.values())
    winners = [labels[k] for k, v in ok.items() if abs(v - best) / max(best, 1) < 0.001]
    return "<strong>" + "+".join(sorted(set(winners))) + "</strong>"


def table_open(cols: list[tuple[str, str]]) -> str:
    width = max(8, int(88 / len(cols)))
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


def main() -> None:
    labels = {short: label for label, _, short, _ in MODES}
    dirs = {short: d for _, d, short, _ in MODES}
    data = {short: load_matrix(d) for _, d, short, _ in MODES}
    benches = sorted(next(iter(data.values())).keys())
    keys = [short for _, _, short, _ in MODES]

    wins = best_wins(data, keys, benches)

    lines: list[str] = [
        "# Flash HLSFactory Results — Legacy vs New Skills (Full Comparison)",
        "",
        STYLE,
        "",
        '<table class="flash-meta">',
        "<thead><tr><th>Field</th><th>Value</th></tr></thead>",
        "<tbody>",
        "<tr><td>Legacy stamp (noskills / bn 2+2)</td><td><code>20260620_004507</code></td></tr>",
        "<tr><td>Legacy stamp (global skills)</td><td><code>20260620_113247</code></td></tr>",
        "<tr><td>New skills stamp</td><td><code>20260621_020847</code></td></tr>",
        "<tr><td>Legacy skills file</td><td><code>skills/skills.json</code> (55 skills)</td></tr>",
        "<tr><td>New skills file</td><td><code>skills_ii_target_miss_solutions_added(73skills).json</code> / <code>(90skills).json</code></td></tr>",
        "<tr><td>Metric</td><td>Final flash-step synthesis latency (cycles), lower is better</td></tr>",
        "<tr><td>Success</td><td>27/28 per mode (<code>doitgen</code> fails gold-ref gate)</td></tr>",
        "<tr><td>Model</td><td><code>mistralai/Devstral-2-123B-Instruct-2512</code></td></tr>",
        "</tbody></table>",
        "",
        "## Summary — all modes",
        "",
        table_open([
            ("mode", "Mode", "left"),
            ("root", "Artifact root", "left"),
            ("ok", "OK", "right"),
            ("wins", "Best latency", "right"),
            ("geo", "Geo-mean lat/GT", "right"),
        ]),
    ]

    for label, dirname, short, family in MODES:
        m = data[short]
        ok = sum(1 for r in m.values() if r["status"] == "ok")
        ratios = [x for x in (gt_ratio(r) for r in m.values() if r["status"] == "ok") if x is not None and x > 0]
        geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else float("nan")
        lines.append(
            f'<tr><td style="text-align:left">{label}</td>'
            f'<td style="text-align:left"><code>{dirname}</code></td>'
            f'<td style="text-align:right">{ok}/28</td>'
            f'<td style="text-align:right">{wins[short]:.1f}/27</td>'
            f'<td style="text-align:right">{geo:.4f}</td></tr>'
        )
    lines.append(table_close())

    # GT section
    lines += ["", "## vs ground truth (latency ratio = synth / GT, lower is better)", ""]
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
        m = data[short]
        faster = slower = tie = fail = 0
        ratios: list[float] = []
        for r in m.values():
            if r["status"] != "ok":
                fail += 1
                continue
            ratio = gt_ratio(r)
            if ratio is None:
                continue
            ratios.append(ratio)
            if ratio < 0.999:
                faster += 1
            elif ratio > 1.001:
                slower += 1
            else:
                tie += 1
        ratios = [x for x in ratios if x > 0]
        geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else float("nan")
        lines.append(
            f'<tr><td style="text-align:left">{label}</td>'
            f'<td style="text-align:right">{faster}</td>'
            f'<td style="text-align:right">{slower}</td>'
            f'<td style="text-align:right">{tie}</td>'
            f'<td style="text-align:right">{geo:.4f}</td>'
            f'<td style="text-align:right">{fail}</td></tr>'
        )
    lines.append(table_close())

    # Slower than GT detail
    lines += ["", "### Benches slower than GT (ratio &gt; 1.001)", ""]
    lines.append(
        table_open([
            ("mode", "Mode", "left"),
            ("bench", "Benchmark", "left"),
            ("ratio", "Ratio", "right"),
            ("gen", "Generated", "right"),
            ("gt", "Ground truth", "right"),
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
        lines.append('<tr><td colspan="5" style="text-align:left">None</td></tr>')
    lines.append(table_close())

    # Legacy h2h vs noskills old
    lines += ["", "## Legacy head-to-head vs noskills (old)", ""]
    lines.append(
        table_open([
            ("opp", "Opponent", "left"),
            ("ow", "Opponent wins", "right"),
            ("nw", "Noskills wins", "right"),
            ("tie", "Ties", "right"),
        ])
    )
    for label, _, short, _ in MODES:
        if short == "nosk_o":
            continue
        if MODES[[m[2] for m in MODES].index(short)][3] != "legacy":
            continue
        ow, nw, tie, _ = h2h(data, "nosk_o", short, benches)
        lines.append(
            f'<tr><td style="text-align:left">{label}</td>'
            f'<td style="text-align:right">{ow}</td>'
            f'<td style="text-align:right">{nw}</td>'
            f'<td style="text-align:right">{tie}</td></tr>'
        )
    lines.append(table_close())

    # New h2h vs noskills new
    lines += ["", "## New head-to-head vs noskills (new)", ""]
    lines.append(
        table_open([
            ("opp", "Opponent", "left"),
            ("ow", "Opponent wins", "right"),
            ("nw", "Noskills wins", "right"),
            ("tie", "Ties", "right"),
        ])
    )
    for label, _, short, _ in MODES:
        if short == "nosk_n" or MODES[[m[2] for m in MODES].index(short)][3] != "new":
            continue
        ow, nw, tie, _ = h2h(data, "nosk_n", short, benches)
        lines.append(
            f'<tr><td style="text-align:left">{label}</td>'
            f'<td style="text-align:right">{ow}</td>'
            f'<td style="text-align:right">{nw}</td>'
            f'<td style="text-align:right">{tie}</td></tr>'
        )
    lines.append(table_close())

    # Paired new vs old
    lines += ["", "## Paired new vs corresponding legacy", ""]
    pairs = [
        ("nosk_n", "nosk_o", "Noskills"),
        ("bn22_n", "bn22_o", "Bn 2+2"),
        ("aav_n", "aav_o", "All+avoids"),
        ("nav_n", "nav_o", "No avoids"),
    ]
    lines.append(
        table_open([
            ("pair", "Pair", "left"),
            ("new", "New wins", "right"),
            ("old", "Old wins", "right"),
            ("tie", "Ties", "right"),
        ])
    )
    for new_k, old_k, name in pairs:
        old_w, new_w, tie, _ = h2h(data, old_k, new_k, benches)
        lines.append(
            f'<tr><td style="text-align:left">{name}</td>'
            f'<td style="text-align:right">{new_w}</td>'
            f'<td style="text-align:right">{old_w}</td>'
            f'<td style="text-align:right">{tie}</td></tr>'
        )
    lines.append(table_close())

    # BN sweep
    lines += ["", "## New BN skill-count sweep", ""]
    lines.append(
        table_open([
            ("cmp", "Comparison", "left"),
            ("b", "Second wins", "right"),
            ("a", "First wins", "right"),
            ("tie", "Ties", "right"),
        ])
    )
    for a, b, an, bn in [
        ("bn22_n", "bn42_n", "Bn 2+2", "Bn 4+2"),
        ("bn22_n", "bn62_n", "Bn 2+2", "Bn 6+2"),
        ("bn42_n", "bn62_n", "Bn 4+2", "Bn 6+2"),
    ]:
        aw, bw, tie, _ = h2h(data, a, b, benches)
        lines.append(
            f'<tr><td style="text-align:left">{bn} vs {an}</td>'
            f'<td style="text-align:right">{bw}</td>'
            f'<td style="text-align:right">{aw}</td>'
            f'<td style="text-align:right">{tie}</td></tr>'
        )
    lines.append(table_close())

    # Paired meaningful diffs >5%
    lines += ["", "## Paired latency changes &gt; 5% (new vs old)", ""]
    for new_k, old_k, name in pairs:
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            table_open([
                ("bench", "Benchmark", "left"),
                ("old", "Old cycles", "right"),
                ("new", "New cycles", "right"),
                ("pct", "Change", "right"),
                ("who", "Better", "right"),
            ])
        )
        diffs: list[tuple[float, str, int, int, float]] = []
        for bench in benches:
            if bench == "hlsfactory_doitgen":
                continue
            lo, ln = lat(data[old_k][bench]), lat(data[new_k][bench])
            if lo is None or ln is None:
                continue
            ch = (ln - lo) / lo
            if abs(ch) > 0.05:
                diffs.append((abs(ch), bench, lo, ln, ch))
        if not diffs:
            lines.append('<tr><td colspan="5" style="text-align:left">No changes &gt; 5%</td></tr>')
        else:
            for _, bench, lo, ln, ch in sorted(diffs, key=lambda x: -x[0]):
                who = "new" if ch < 0 else "old"
                lines.append(
                    f'<tr><td style="text-align:left"><code>{bench_short(bench)}</code></td>'
                    f'<td style="text-align:right">{lo:,}</td>'
                    f'<td style="text-align:right">{ln:,}</td>'
                    f'<td style="text-align:right">{ch * 100:+.1f}%</td>'
                    f'<td style="text-align:right"><strong>{who}</strong></td></tr>'
                )
        lines.append(table_close())
        lines.append("")

    # Full latency table - legacy block
    lines += ["", "## Latency (cycles) — legacy", ""]
    legacy_keys = [short for _, _, short, fam in MODES if fam == "legacy"]
    legacy_labels = {short: label.split(" (")[0] for label, _, short, _ in MODES if short in legacy_keys}
    lines.append(
        table_open([
            ("b", "Benchmark", "left"),
            ("gt", "Ground truth", "right"),
        ]
        + [(k, legacy_labels[k], "right") for k in legacy_keys]
        + [("w", "Winner", "right")])
    )
    for bench in benches:
        short_name = bench_short(bench)
        fail = data["nosk_o"][bench]["status"] != "ok" and all(
            data[k][bench]["status"] != "ok" for k in legacy_keys
        )
        gt = gt_lat(data["nosk_o"][bench]) or gt_lat(data["bn22_o"][bench])
        row_lats = {k: lat(data[k][bench]) for k in legacy_keys}
        cells = [f'<td style="text-align:left"><code>{short_name}</code></td>']
        cells.append(fmt_cycles(gt).replace("<td", '<td'))
        for k in legacy_keys:
            r = data[k][bench]
            if r["status"] != "ok":
                cells.append(fmt_cycles(None, fail=True))
            else:
                cells.append(fmt_cycles(row_lats[k]))
        tag = winner_tag(legacy_keys, legacy_labels, row_lats)
        cells.append(f'<td style="text-align:right">{tag}</td>')
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append(table_close())

    # Full latency table - new block
    lines += ["", "## Latency (cycles) — new skills pack", ""]
    new_keys = [short for _, _, short, fam in MODES if fam == "new"]
    new_labels = {short: label.split(" (")[0].replace("Bn ", "bn") for label, _, short, _ in MODES if short in new_keys}
    lines.append(
        table_open([
            ("b", "Benchmark", "left"),
            ("gt", "Ground truth", "right"),
        ]
        + [(k, new_labels[k], "right") for k in new_keys]
        + [("w", "Winner", "right")])
    )
    for bench in benches:
        short_name = bench_short(bench)
        gt = gt_lat(data["nosk_n"][bench]) or gt_lat(data["bn22_n"][bench])
        row_lats = {k: lat(data[k][bench]) for k in new_keys}
        cells = [f'<td style="text-align:left"><code>{short_name}</code></td>']
        cells.append(fmt_cycles(gt))
        for k in new_keys:
            r = data[k][bench]
            if r["status"] != "ok":
                cells.append(fmt_cycles(None, fail=True))
            else:
                cells.append(fmt_cycles(row_lats[k]))
        tag = winner_tag(new_keys, new_labels, row_lats)
        cells.append(f'<td style="text-align:right">{tag}</td>')
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append(table_close())

    # GT ratio table
    lines += ["", "## Ground-truth latency ratio per benchmark (synth / GT)", ""]
    lines.append(
        table_open(
            [("b", "Benchmark", "left")]
            + [(short, labels[short].split(" (")[0], "right") for _, _, short, _ in MODES]
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

    # Conclusions
    lines += [
        "",
        "## Conclusions",
        "",
        "1. **Success:** Every mode completes **27/28** benches; `doitgen` fails the gold-reference gate in all runs.",
        "2. **vs ground truth:** All modes achieve dramatically lower latency than GT (geo-mean ratios ≈ 0.03–0.15). A few new runs are marginally above GT (ratio ≈ 1.01–1.05) on isolated benches.",
        "3. **Best single-mode latency wins (of 27):** No avoids (old) leads (**8.1**), followed by All+avoids (new) (**4.1**). Bn 6+2 (new) is weakest (**0.2**).",
        "4. **New vs old (paired):** All+avoids **improves** with the new skills pack (15–8). Noskills and No avoids **regress** vs legacy (new wins 7 and 9). Bn 2+2 is roughly even (12–13).",
        "5. **Among new modes:** Bn 4+2 and No avoids beat Noskills (new) most often (16 wins each). Bn 6+2 loses to Noskills (new) 16–11.",
        "6. **BN skill count:** More positive bottleneck skills do not help monotonically — 4+2 ≈ 2+2, while 6+2 is clearly worse.",
        "",
        "See also: `artifacts/pc2/flash_comparison_20260620.md` (legacy-only run from 2026-06-20).",
        "",
    ]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()

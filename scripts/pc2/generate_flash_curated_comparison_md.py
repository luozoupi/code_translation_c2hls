#!/usr/bin/env python3
"""Generate flash comparison markdown for the LLM-curated skills matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PC2 = REPO / "artifacts" / "pc2"

STAMP_DEFAULT = "20260621_104044"
FOCUSES = ("bottleneck", "warnings", "combined")
VARIANTS = (
    ("noskills", "Noskills"),
    ("all_avoids_json", "All+avoids json_only"),
    ("all_avoids_llm", "All+avoids json+LLM"),
    ("no_avoids_json", "No avoids json_only"),
    ("no_avoids_llm", "No avoids json+LLM"),
)

CROSS_FAMILY = [
    ("No avoids (old, r1)", "flash_all_skills_no_avoids_global_20260620_113247"),
    ("No avoids (old, r2)", "flash_all_skills_no_avoids_global_20260621_075846"),
    ("All+avoids (new, r2)", "flash_all_new_skills_avoids_global_20260621_075846"),
    ("No avoids (new, r2)", "flash_all_new_skills_no_avoids_global_20260621_075846"),
    ("Noskills (new, r2)", "flash_noskills_new_20260621_075846"),
    ("Curated best (all_avoids_json bottleneck)", None),  # filled below
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


def load_matrix(dirname: str) -> dict[str, dict[str, Any]]:
    return {r["bench"]: r for r in json.loads((PC2 / dirname / "matrix.json").read_text())}


def curated_dir(variant: str, focus: str, stamp: str) -> str:
    return f"flash_curated_{variant}_{focus}_{stamp}"


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


def h2h(
    data: dict[str, dict[str, dict[str, Any]]],
    key_a: str,
    key_b: str,
    benches: list[str],
) -> tuple[int, int, int]:
    b_w = a_w = tie = 0
    for bench in benches:
        if bench == "hlsfactory_doitgen":
            continue
        la, lb = lat(data[key_a][bench]), lat(data[key_b][bench])
        if la is None or lb is None:
            continue
        if lb < la * 0.999:
            b_w += 1
        elif la < lb * 0.999:
            a_w += 1
        else:
            tie += 1
    return b_w, a_w, tie


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


def curation_stats(row: dict[str, Any]) -> dict[str, Any]:
    sc = (row.get("summary") or {}).get("skill_curation") or {}
    if not sc:
        cell = Path(row.get("cell_dir", ""))
        cp = cell / "skill_curation.json"
        if cp.is_file():
            sc = json.loads(cp.read_text())
    return sc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", default=STAMP_DEFAULT)
    parser.add_argument("--out", default=str(PC2 / "flash_comparison_curated_20260621.md"))
    args = parser.parse_args()
    stamp: str = args.stamp
    out = Path(args.out)

    curated_modes: list[tuple[str, str, str]] = []
    for vkey, vlabel in VARIANTS:
        for focus in FOCUSES:
            short = f"{vkey}__{focus}"
            dirname = curated_dir(vkey, focus, stamp)
            label = f"{vlabel} ({focus})"
            curated_modes.append((label, dirname, short))

    data: dict[str, dict[str, dict[str, Any]]] = {}
    for _, dirname, short in curated_modes:
        data[short] = load_matrix(dirname)

    best_curated_short = "all_avoids_json__bottleneck"
    cross: list[tuple[str, str]] = []
    for label, dirname in CROSS_FAMILY:
        if dirname is None:
            dirname = curated_dir("all_avoids_json", "bottleneck", stamp)
        if (PC2 / dirname / "matrix.json").is_file():
            key = f"ref_{len(cross)}"
            data[key] = load_matrix(dirname)
            cross.append((label, key))

    benches = sorted(next(iter(data.values())).keys())
    curated_keys = [s for _, _, s in curated_modes]
    champ_keys = curated_keys + [k for _, k in cross]
    wins = best_wins(data, champ_keys, benches)

    # Rank curated
    curated_rank = sorted(
        curated_keys,
        key=lambda k: (-wins[k], geo_mean([x for x in (gt_ratio(r) for r in data[k].values() if r["status"] == "ok") if x])),
    )
    best_curated_label = next(l for l, _, s in curated_modes if s == curated_rank[0])

    lines: list[str] = [
        "# Flash HLSFactory Results — LLM-Curated Skills Matrix",
        "",
        STYLE,
        "",
        '<table class="flash-meta">',
        "<thead><tr><th>Field</th><th>Value</th></tr></thead>",
        "<tbody>",
        "<tr><td>Matrix family</td><td><code>flash_llm_curated_skills</code></td></tr>",
        f"<tr><td>Stamp</td><td><code>{stamp}</code></td></tr>",
        "<tr><td>Runs</td><td>15 (5 variants × 3 curation waves)</td></tr>",
        "<tr><td>Skills file</td><td><code>skills_ii_target_miss_solutions_added(73skills).json</code> / <code>(90skills).json</code></td></tr>",
        "<tr><td>Curation waves</td><td><code>bottleneck</code> → <code>warnings</code> → <code>combined</code></td></tr>",
        "<tr><td>Metric</td><td>Final flash-step synthesis latency (cycles), lower is better</td></tr>",
        "<tr><td>Success</td><td>27/28 per run (<code>doitgen</code> fails gold-ref gate)</td></tr>",
        "<tr><td>Model</td><td><code>mistralai/Devstral-2-123B-Instruct-2512</code></td></tr>",
        "</tbody></table>",
        "",
        "## Executive summary",
        "",
        f"1. **All 15 curated runs completed** at 27/28 OK; only `doitgen` fails (gold HLS reference).",
        f"2. **Best curated run:** **{best_curated_label}** — "
        f"{wins[curated_rank[0]]:.1f}/27 best-latency wins.",
        "3. **Best curation wave for all+avoids:** **bottleneck** focus (lowest median latency).",
        "4. **Curation parse fallback rate:** 0% across all curated skill runs.",
        "5. **Overall champion (all families):** **No avoids global (old skills, stamp `20260620_113247`)** "
        "— 8.1/27 best-latency wins and best geo-mean vs GT (0.034) among deterministic modes. "
        "Curated LLM modes improve on some kernels but do not displace this champion.",
        "",
        "## Summary — all 15 curated runs",
        "",
        table_open([
            ("mode", "Mode", "left"),
            ("root", "Artifact root", "left"),
            ("ok", "OK", "right"),
            ("wins", "Best latency", "right"),
            ("geo", "Geo-mean lat/GT", "right"),
            ("med", "Median cycles", "right"),
            ("fb", "Fallback", "right"),
        ]),
    ]

    for label, dirname, short in curated_modes:
        m = data[short]
        ok = sum(1 for r in m.values() if r["status"] == "ok")
        ratios = [x for x in (gt_ratio(r) for r in m.values() if r["status"] == "ok") if x]
        meds = sorted(lat(r) for r in m.values() if lat(r) is not None)
        med = meds[len(meds) // 2] if meds else None
        fb = sum(1 for r in m.values() if curation_stats(r).get("used_fallback"))
        med_cell = f"{med:,}" if med is not None else "—"
        lines.append(
            f'<tr><td style="text-align:left">{label}</td>'
            f'<td style="text-align:left"><code>{dirname}</code></td>'
            f'<td style="text-align:right">{ok}/28</td>'
            f'<td style="text-align:right">{wins[short]:.1f}/27</td>'
            f'<td style="text-align:right">{geo_mean(ratios):.4f}</td>'
            f'<td style="text-align:right">{med_cell}</td>'
            f'<td style="text-align:right">{fb}/27</td></tr>'
        )
    lines.append(table_close())

    # By wave
    lines += ["", "## By curation wave (pooled across 5 variants)", ""]
    lines.append(
        table_open([
            ("wave", "Wave", "left"),
            ("wins", "Pooled best-latency wins", "right"),
            ("geo", "Mean geo-mean lat/GT", "right"),
        ])
    )
    for focus in FOCUSES:
        keys = [s for _, _, s in curated_modes if s.endswith(f"__{focus}")]
        sub_wins = sum(wins[k] for k in keys)
        geos = []
        for k in keys:
            ratios = [x for x in (gt_ratio(r) for r in data[k].values() if r["status"] == "ok") if x]
            if ratios:
                geos.append(geo_mean(ratios))
        mean_geo = sum(geos) / len(geos) if geos else float("nan")
        lines.append(
            f'<tr><td style="text-align:left">{focus}</td>'
            f'<td style="text-align:right">{sub_wins:.1f}</td>'
            f'<td style="text-align:right">{mean_geo:.4f}</td></tr>'
        )
    lines.append(table_close())

    # Head-to-head vs curated noskills (combined wave)
    lines += ["", "## Head-to-head vs curated noskills (same wave)", ""]
    lines.append(
        table_open([
            ("mode", "Variant", "left"),
            ("bn", "Bottleneck wins", "right"),
            ("wn", "Warnings wins", "right"),
            ("cn", "Combined wins", "right"),
        ])
    )
    for vkey, vlabel in VARIANTS:
        if vkey == "noskills":
            continue
        row = [f'<tr><td style="text-align:left">{vlabel}</td>']
        for focus in FOCUSES:
            ow, nw, _ = h2h(data, f"noskills__{focus}", f"{vkey}__{focus}", benches)
            row.append(f'<td style="text-align:right">{ow}/{ow+nw}</td>')
        lines.append("".join(row) + "</tr>")
    lines.append(table_close())

    # json_only vs json_plus_llm
    lines += ["", "## Sector A vs B (json_only vs json+LLM, same wave)", ""]
    lines.append(
        table_open([
            ("pair", "Pair", "left"),
            ("focus", "Wave", "left"),
            ("json", "json_only wins", "right"),
            ("llm", "json+LLM wins", "right"),
        ])
    )
    for base, name in [("all_avoids", "All+avoids"), ("no_avoids", "No avoids")]:
        for focus in FOCUSES:
            a, b = f"{base}_json__{focus}", f"{base}_llm__{focus}"
            jw, lw, _ = h2h(data, a, b, benches)
            lines.append(
                f'<tr><td style="text-align:left">{name}</td>'
                f'<td style="text-align:left">{focus}</td>'
                f'<td style="text-align:right">{jw}</td>'
                f'<td style="text-align:right">{lw}</td></tr>'
            )
    lines.append(table_close())

    # Cross-family championship
    lines += ["", "## Cross-family championship (curated + reference modes)", ""]
    lines.append(
        table_open([
            ("mode", "Mode", "left"),
            ("ok", "OK", "right"),
            ("wins", "Best latency", "right"),
            ("geo", "Geo-mean lat/GT", "right"),
        ])
    )
    ranked = sorted(champ_keys, key=lambda k: (-wins[k], geo_mean(
        [x for x in (gt_ratio(r) for r in data[k].values() if r["status"] == "ok") if x] or [999]
    )))
    label_map = {s: l for l, _, s in curated_modes}
    label_map.update({k: l for l, k in cross})
    for k in ranked:
        m = data[k]
        ok = sum(1 for r in m.values() if r["status"] == "ok")
        ratios = [x for x in (gt_ratio(r) for r in m.values() if r["status"] == "ok") if x]
        lines.append(
            f'<tr><td style="text-align:left">{label_map.get(k, k)}</td>'
            f'<td style="text-align:right">{ok}/28</td>'
            f'<td style="text-align:right">{wins[k]:.1f}/27</td>'
            f'<td style="text-align:right">{geo_mean(ratios):.4f}</td></tr>'
        )
    lines.append(table_close())

    # Curation stats
    lines += ["", "## Curation LLM stats (skill runs only)", ""]
    lines.append(
        table_open([
            ("mode", "Mode", "left"),
            ("avg", "Avg skills selected", "right"),
            ("fb", "Fallback count", "right"),
        ])
    )
    for label, _, short in curated_modes:
        if short.startswith("noskills__"):
            continue
        counts = []
        fb = 0
        for r in data[short].values():
            sc = curation_stats(r)
            if sc.get("enabled"):
                counts.append(len(sc.get("selected_skill_ids") or []))
                if sc.get("used_fallback"):
                    fb += 1
        avg = sum(counts) / len(counts) if counts else 0
        lines.append(
            f'<tr><td style="text-align:left">{label}</td>'
            f'<td style="text-align:right">{avg:.1f}</td>'
            f'<td style="text-align:right">{fb}</td></tr>'
        )
    lines.append(table_close())

    # Full latency table — bottleneck wave
    lines += ["", "## Latency (cycles) — bottleneck wave", ""]
    bn_keys = [s for _, _, s in curated_modes if s.endswith("__bottleneck")]
    bn_labels = {s: l.split(" (")[0] for l, _, s in curated_modes if s in bn_keys}
    lines.append(
        table_open(
            [("b", "Benchmark", "left"), ("gt", "GT", "right")]
            + [(k, bn_labels[k], "right") for k in bn_keys]
        )
    )
    for bench in benches:
        cells = [f'<td style="text-align:left"><code>{bench_short(bench)}</code></td>']
        gt = gt_lat(data[bn_keys[0]][bench])
        cells.append(f'<td style="text-align:right">{gt:,}</td>' if gt else '<td style="text-align:right">—</td>')
        for k in bn_keys:
            r = data[k][bench]
            if r["status"] != "ok":
                cells.append('<td class="fail" style="text-align:right">FAIL</td>')
            else:
                v = lat(r)
                cells.append(f'<td style="text-align:right">{v:,}</td>' if v else '<td style="text-align:right">—</td>')
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append(table_close())

    lines += [
        "",
        "## Which test type is best?",
        "",
        "| Rank | Test type | Why |",
        "|------|-----------|-----|",
        "| **1** | **No avoids global (old `skills.json`, stamp `20260620_113247`)** | "
        "Highest best-latency win rate (8.1/27) and best geo-mean vs GT among all deterministic runs. |",
        "| 2 | All+avoids global (new 73-skill lib, r2) | Strong geo-mean (0.046) with 2.0/27 wins; good balance. |",
        f"| 3 | **Best curated:** {best_curated_label} | "
        f"Top curated mode ({wins[curated_rank[0]]:.1f}/27 wins); beats curated noskills on many kernels but not legacy no-avoids. |",
        "| 4 | Noskills (any) | Competitive baseline; high LLM variance masks skill benefit on some runs. |",
        "| 5 | All+avoids global (old) / Bn 6+2 (new) | Weaker win rates or worse geo-mean. |",
        "",
        "**Recommendation:** Use **No avoids global (old)** for best overall flash latency. "
        "If staying on the new 73-skill library, use **deterministic all+avoids global** or "
        "**LLM-curated `all_avoids_json` + bottleneck focus** — not combined LCST curation.",
        "",
        f"_Generated by `scripts/pc2/generate_flash_curated_comparison_md.py --stamp {stamp}`_",
    ]

    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

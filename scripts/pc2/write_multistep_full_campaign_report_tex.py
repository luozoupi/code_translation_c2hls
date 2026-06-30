#!/usr/bin/env python3
"""Generate LaTeX report for full 28-bench multistep campaigns."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "artifacts/pc2/analysis"
DEFAULT_OUT = REPO / "artifacts/pc2/multistep_full_campaign_20260630.tex"

VARIANTS = {
    "nav_n": {
        "label": "No avoids (90 skills)",
        "stamp": "20260630_fixed_cosim_multistep_nav_n_pipelined",
        "skills": "90 (all injected, no avoids)",
        "artifact": REPO / "artifacts/pc2/multistep_fixed_cosim_nav_n_20260630_fixed_cosim_multistep_nav_n_pipelined",
        "jsonl": "misc/hlsfactory_fixed_cosim_multistep_nav_n_u280_20260630.jsonl",
    },
    "noskills": {
        "label": "Noskills",
        "stamp": "20260630_fixed_cosim_multistep_noskills_pipelined",
        "skills": "0 (skills disabled)",
        "artifact": REPO / "artifacts/pc2/multistep_fixed_cosim_noskills_20260630_fixed_cosim_multistep_noskills_pipelined",
        "jsonl": "misc/hlsfactory_fixed_cosim_multistep_noskills_u280_20260630.jsonl",
    },
}


def load_csv(variant: str) -> list[dict[str, str]]:
    path = ANALYSIS / VARIANTS[variant]["stamp"] / "csynth_speedup_per_bench.csv"
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_agg(variant: str) -> dict[str, str]:
    path = ANALYSIS / VARIANTS[variant]["stamp"] / "csynth_speedup_aggregate.csv"
    with path.open(encoding="utf-8") as f:
        return next(csv.DictReader(f))


def fmt_cycles(value: str | None) -> str:
    if not value:
        return "---"
    try:
        x = float(value)
    except ValueError:
        return "---"
    if x <= 0:
        return "0"
    if x >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if x >= 1_000:
        return f"{x / 1_000:.1f}k"
    return str(int(x))


def fmt_spd(value: str | None) -> str:
    if not value:
        return "---"
    try:
        x = float(value)
    except ValueError:
        return "---"
    if x >= 100:
        return f"{x:.0f}$\\times$"
    if x >= 10:
        return f"{x:.1f}$\\times$"
    return f"{x:.2f}$\\times$"


def tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def short_err(err: str) -> str:
    if not err:
        return ""
    line = err.strip().split("\n")[0]
    if len(line) > 80:
        line = line[:77] + "..."
    return tex_escape(line)


def bench_names(rows_a: dict[str, dict], rows_b: dict[str, dict]) -> list[str]:
    names = sorted(set(rows_a) | set(rows_b), key=lambda b: b.replace("hlsfactory_", ""))
    return names


def generate(out_path: Path = DEFAULT_OUT) -> Path:
    nav = {r["bench"]: r for r in load_csv("nav_n")}
    nos = {r["bench"]: r for r in load_csv("noskills")}
    agg_nav = load_agg("nav_n")
    agg_nos = load_agg("noskills")
    benches = bench_names(nav, nos)

    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("\\documentclass[11pt]{article}")
    w("\\usepackage[margin=18mm]{geometry}")
    w("\\usepackage{booktabs, longtable, array, multirow}")
    w("\\usepackage{amsmath}")
    w("\\usepackage{hyperref}")
    w("\\usepackage{ragged2e}")
    w("\\newcolumntype{L}[1]{>{\\RaggedRight\\arraybackslash}p{#1}}")
    w("")
    w("\\title{Multistep Fixed-Cosim Campaign --- Full 28-Bench Corpus}")
    w("\\author{C2HLS / PC2}")
    w(f"\\date{{{datetime.now(timezone.utc).strftime('%B %Y')}}}")
    w("")
    w("\\begin{document}")
    w("\\maketitle")
    w("")
    w("\\section*{Executive summary}")
    w("")
    w("Full-corpus multistep pipelined runs on \\textbf{28} HLSFactory \\texttt{benchmarks\\_cosim} kernels")
    w("(Xilinx \\texttt{xcu280-fsvh2892-2L-e}, 4\\,ns target clock). Workflow per benchmark:")
    w("Phase~A (C$\\rightarrow$HLS reference validation), Phase~B (initial HLS translate + csynth/csim),")
    w("then five sequential single-technique optimization steps:")
    w("\\emph{coalescing}, \\emph{tiling}, \\emph{pipeline}, \\emph{unroll}, \\emph{doublebuffer}.")
    w("\\textbf{RTL cosim was not executed} (csynth + csim only).")
    w("LLM: \\texttt{mistralai/Devstral-2-123B-Instruct-2512}.")
    w("Pipelined runner with 4 synth workers; GPU session + compute node.")
    w("")
    w("\\begin{itemize}")
    w("  \\item \\textbf{nav\\_n}: 90-skill library, all skills injected, \\emph{no avoids}.")
    w("  \\item \\textbf{noskills}: skills disabled (LLM-only multistep baseline).")
    w("  \\item \\textbf{aav\\_n} (90 skills + avoids): full corpus submitted separately; results pending at report time.")
    w("\\end{itemize}")
    w("")
    w("Both completed campaigns reported here: \\textbf{26/28 OK}, \\textbf{2 failures} each.")
    w("Selected result = lowest-latency successful step among phase\\_b and the five optimization steps.")
    w("")
    w("\\section{Aggregate csynth speedup (26 OK benches)}")
    w("")
    w("\\begin{table}[h]")
    w("\\centering")
    w("\\caption{Geometric and median speedup of \\emph{selected} kernel vs phase\\_b and ground-truth synthesis latency.}")
    w("\\begin{tabular}{@{}lrrrrr@{}}")
    w("\\toprule")
    w("\\textbf{Variant} & \\textbf{OK} & \\textbf{Fail} & \\textbf{Gmean vs phase\\_b} & \\textbf{Gmean vs GT} & \\textbf{Median vs GT} \\\\")
    w("\\midrule")
    w(
        f"nav\\_n & {agg_nav['ok']} & {agg_nav['fail']} & "
        f"{float(agg_nav['gmean_speedup_vs_phase_b']):.2f}$\\times$ & "
        f"{float(agg_nav['gmean_speedup_vs_gt']):.2f}$\\times$ & "
        f"{float(agg_nav['median_speedup_vs_gt']):.2f}$\\times$ \\\\"
    )
    w(
        f"noskills & {agg_nos['ok']} & {agg_nos['fail']} & "
        f"{float(agg_nos['gmean_speedup_vs_phase_b']):.2f}$\\times$ & "
        f"{float(agg_nos['gmean_speedup_vs_gt']):.2f}$\\times$ & "
        f"{float(agg_nos['median_speedup_vs_gt']):.2f}$\\times$ \\\\"
    )
    w("\\bottomrule")
    w("\\end{tabular}")
    w("\\end{table}")
    w("")
    w("\\section{Campaign configuration}")
    w("")
    w("\\begin{table}[h]")
    w("\\centering")
    w("\\small")
    w("\\begin{tabular}{@{}llll@{}}")
    w("\\toprule")
    w("\\textbf{Variant} & \\textbf{Stamp} & \\textbf{Skills} & \\textbf{JSONL export} \\\\")
    w("\\midrule")
    for key in ("nav_n", "noskills"):
        meta = VARIANTS[key]
        w(
            f"{tex_escape(key)} & \\texttt{{{tex_escape(meta['stamp'])}}} & "
            f"{tex_escape(meta['skills'])} & \\texttt{{{tex_escape(meta['jsonl'])}}} \\\\"
        )
    w("\\bottomrule")
    w("\\end{tabular}")
    w("\\end{table}")
    w("")
    w("\\section{Per-benchmark results}")
    w("")
    w("\\footnotesize")
    w("\\setlength{\\tabcolsep}{3pt}")
    w("\\renewcommand{\\arraystretch}{1.12}")
    w("\\begin{longtable}{@{}l cc cc cc L{1.6cm} L{1.6cm}@{}}")
    w("\\caption{Per-bench phase\\_b and selected csynth latency (cycles) with speedup vs phase\\_b.}\\\\")
    w("\\toprule")
    w("\\textbf{Bench} & \\multicolumn{2}{c}{\\textbf{phase\\_b}} & \\multicolumn{2}{c}{\\textbf{selected}} & \\multicolumn{2}{c}{\\textbf{spd vs ph.B}} & \\textbf{nav step} & \\textbf{nos step} \\\\")
    w(" & nav & nos & nav & nos & nav & nos & & \\\\")
    w("\\midrule")
    w("\\endfirsthead")
    w("\\toprule")
    w("\\textbf{Bench} & \\multicolumn{2}{c}{\\textbf{phase\\_b}} & \\multicolumn{2}{c}{\\textbf{selected}} & \\multicolumn{2}{c}{\\textbf{spd vs ph.B}} & \\textbf{nav step} & \\textbf{nos step} \\\\")
    w(" & nav & nos & nav & nos & nav & nos & & \\\\")
    w("\\midrule")
    w("\\endhead")
    w("\\midrule")
    w("\\multicolumn{9}{r}{\\emph{continued on next page}}\\\\")
    w("\\midrule")
    w("\\endfoot")
    w("\\bottomrule")
    w("\\endlastfoot")

    def cell(row: dict[str, str], field: str, *, spd: bool = False) -> str:
        if row.get("status") == "fail":
            return "\\textbf{FAIL}"
        return fmt_spd(row.get(field)) if spd else fmt_cycles(row.get(field))

    for bench in benches:
        bn = bench.replace("hlsfactory_", "")
        rn, rs = nav.get(bench, {}), nos.get(bench, {})
        w(
            f"{bn} & {cell(rn, 'phase_b_cycles')} & {cell(rs, 'phase_b_cycles')} & "
            f"{cell(rn, 'selected_cycles')} & {cell(rs, 'selected_cycles')} & "
            f"{cell(rn, 'speedup_vs_phase_b', spd=True)} & {cell(rs, 'speedup_vs_phase_b', spd=True)} & "
            f"{tex_escape(rn.get('selected_from') or ('fail' if rn.get('status') == 'fail' else '---'))} & "
            f"{tex_escape(rs.get('selected_from') or ('fail' if rs.get('status') == 'fail' else '---'))} \\\\"
        )

    w("\\end{longtable}")
    w("\\normalsize")
    w("")
    w("\\section{Per-step latency (nav\\_n)}")
    w("")
    w("\\footnotesize")
    w("\\begin{longtable}{@{}lrrrrrr@{}}")
    w("\\caption{nav\\_n: csynth latency per optimization step (cycles).}\\\\")
    w("\\toprule")
    w("\\textbf{Bench} & \\textbf{GT} & \\textbf{ph.B} & \\textbf{coal.} & \\textbf{tile} & \\textbf{pipe} & \\textbf{unroll} & \\textbf{dblbuf} \\\\")
    w("\\midrule")
    w("\\endfirsthead")
    w("\\toprule")
    w("\\textbf{Bench} & \\textbf{GT} & \\textbf{ph.B} & \\textbf{coal.} & \\textbf{tile} & \\textbf{pipe} & \\textbf{unroll} & \\textbf{dblbuf} \\\\")
    w("\\midrule")
    w("\\endhead")
    w("\\bottomrule")
    w("\\endlastfoot")
    for bench in benches:
        r = nav.get(bench, {})
        if r.get("status") == "fail":
            w(f"{bench.replace('hlsfactory_', '')} & \\multicolumn{{7}}{{l}}{{\\textbf{{FAIL}}: {short_err(r.get('error', ''))}}} \\\\")
            continue
        w(
            f"{bench.replace('hlsfactory_', '')} & {fmt_cycles(r.get('gt_cycles'))} & "
            f"{fmt_cycles(r.get('phase_b_cycles'))} & {fmt_cycles(r.get('coalescing_cycles'))} & "
            f"{fmt_cycles(r.get('tiling_cycles'))} & {fmt_cycles(r.get('pipeline_cycles'))} & "
            f"{fmt_cycles(r.get('unroll_cycles'))} & {fmt_cycles(r.get('doublebuffer_cycles'))} \\\\"
        )
    w("\\end{longtable}")
    w("\\normalsize")
    w("")
    w("\\section{Per-step latency (noskills)}")
    w("")
    w("\\footnotesize")
    w("\\begin{longtable}{@{}lrrrrrr@{}}")
    w("\\caption{noskills: csynth latency per optimization step (cycles).}\\\\")
    w("\\toprule")
    w("\\textbf{Bench} & \\textbf{GT} & \\textbf{ph.B} & \\textbf{coal.} & \\textbf{tile} & \\textbf{pipe} & \\textbf{unroll} & \\textbf{dblbuf} \\\\")
    w("\\midrule")
    w("\\endfirsthead")
    w("\\toprule")
    w("\\textbf{Bench} & \\textbf{GT} & \\textbf{ph.B} & \\textbf{coal.} & \\textbf{tile} & \\textbf{pipe} & \\textbf{unroll} & \\textbf{dblbuf} \\\\")
    w("\\midrule")
    w("\\endhead")
    w("\\bottomrule")
    w("\\endlastfoot")
    for bench in benches:
        r = nos.get(bench, {})
        if r.get("status") == "fail":
            w(f"{bench.replace('hlsfactory_', '')} & \\multicolumn{{7}}{{l}}{{\\textbf{{FAIL}}: {short_err(r.get('error', ''))}}} \\\\")
            continue
        w(
            f"{bench.replace('hlsfactory_', '')} & {fmt_cycles(r.get('gt_cycles'))} & "
            f"{fmt_cycles(r.get('phase_b_cycles'))} & {fmt_cycles(r.get('coalescing_cycles'))} & "
            f"{fmt_cycles(r.get('tiling_cycles'))} & {fmt_cycles(r.get('pipeline_cycles'))} & "
            f"{fmt_cycles(r.get('unroll_cycles'))} & {fmt_cycles(r.get('doublebuffer_cycles'))} \\\\"
        )
    w("\\end{longtable}")
    w("\\normalsize")
    w("")
    w("\\section{Failures}")
    w("")
    w("\\begin{table}[h]")
    w("\\centering")
    w("\\small")
    w("\\begin{tabular}{@{}llL{7cm}@{}}")
    w("\\toprule")
    w("\\textbf{Variant} & \\textbf{Bench} & \\textbf{Error (truncated)} \\\\")
    w("\\midrule")
    for bench in benches:
        for key, rows in (("nav_n", nav), ("noskills", nos)):
            row = rows.get(bench, {})
            if row.get("status") == "fail":
                w(
                    f"{tex_escape(key)} & {tex_escape(bench.replace('hlsfactory_', ''))} & "
                    f"{short_err(row.get('error', '')) or 'see orchestrator log'} \\\\"
                )
    w("\\bottomrule")
    w("\\end{tabular}")
    w("\\end{table}")
    w("")
    w("\\section{Head-to-head notes}")
    w("")
    nav_wins: list[str] = []
    nos_wins: list[str] = []
    nav_only_ok: list[str] = []
    nos_only_ok: list[str] = []
    for bench in benches:
        rn, rs = nav.get(bench, {}), nos.get(bench, {})
        bn = bench.replace("hlsfactory_", "")
        if rn.get("status") == "fail" and rs.get("status") == "ok":
            nav_only_ok.append(bn)
        if rs.get("status") == "fail" and rn.get("status") == "ok":
            nos_only_ok.append(bn)
        if rn.get("status") != "ok" or rs.get("status") != "ok":
            continue
        try:
            sp_n = float(rn.get("speedup_vs_phase_b") or 0)
            sp_s = float(rs.get("speedup_vs_phase_b") or 0)
        except ValueError:
            continue
        if sp_n > 0 and sp_s > 0:
            ratio = sp_n / sp_s
            if ratio >= 5:
                nav_wins.append(bn)
            elif ratio <= 0.2:
                nos_wins.append(bn)
    w("\\begin{itemize}")
    w("  \\item \\textbf{nav\\_n $\\gg$ noskills} (selected speedup vs phase\\_b $\\geq 5\\times$ higher): "
      + ", ".join(f"\\texttt{{{b}}}" for b in nav_wins) + ".")
    w("  \\item \\textbf{noskills $\\gg$ nav\\_n}: "
      + ", ".join(f"\\texttt{{{b}}}" for b in nos_wins) + ".")
    w("  \\item \\textbf{nav\\_n failed / noskills OK}: "
      + ", ".join(f"\\texttt{{{b}}}" for b in nav_only_ok) + ".")
    w("  \\item \\textbf{noskills failed / nav\\_n OK}: "
      + ", ".join(f"\\texttt{{{b}}}" for b in nos_only_ok) + ".")
    w("\\end{itemize}")
    w("")
    w("\\section{Data exports and analysis CSVs}")
    w("")
    w("\\begin{itemize}")
    for key in ("nav_n", "noskills"):
        stamp = VARIANTS[key]["stamp"]
        w(f"  \\item \\textbf{{{tex_escape(key)}}}: \\texttt{{{tex_escape(VARIANTS[key]['jsonl'])}}}")
        w(f"  \\item CSV: \\texttt{{artifacts/pc2/analysis/{stamp}/csynth\\_speedup\\_per\\_bench.csv}}")
        w(f"  \\item CSV: \\texttt{{artifacts/pc2/analysis/{stamp}/csynth\\_speedup\\_step\\_progression.csv}}")
        w(f"  \\item Summary MD: \\texttt{{artifacts/pc2/analysis/{stamp}/summary.md}}")
    w("\\end{itemize}")
    w("")
    w("\\section{Artifact roots}")
    w("")
    w("\\begin{verbatim}")
    for key in ("nav_n", "noskills"):
        w(str(VARIANTS[key]["artifact"]))
    w("\\end{verbatim}")
    w("")
    w("\\end{document}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = generate()
    print(f"wrote {path}")

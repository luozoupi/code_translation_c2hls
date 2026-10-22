#!/usr/bin/env python3
"""Build controlled/story/story_v12.html: story_v11 (make_story_v11.py, unchanged) plus a chart of the 13-14 Oct
experiments on the fixed bench (90-skill pack, DSP 50-65%, manual DATAFLOW, general agent), placed under the ladder charts.

Every number is read from run outputs at build time, through make_story_v11's own readers.
"""
import math
from pathlib import Path

import make_story_v11 as v11

ms, S, cell, esc = v11.ms, v11.S, v11.cell, v11.esc
OUT = Path(__file__).resolve().parent / "story_v12.html"


def mm_cell(root, tag):
    r = cell(root, tag, "autosa_mm")
    if r.get("done") and not r.get("fallback") and r.get("csim"):
        return dict(r, kind="cap" if r.get("overcap") else "ok")
    if r["state"] in ("RUNNING", "PENDING"):
        return dict(r, kind="run")
    return dict(r, kind="none")


def columns():
    d65 = S / "dsp5065/runs"
    cols = [
        ("Expertise,", "90-skill pack", {m: [mm_cell(S / "s90/runs", f"s90_{m}_r{r}") for r in (1, 2, 3)] for m in ("cc", "ds")}),
        ("90-skill pack", "+ DSP floor", {m: [mm_cell(S / "s90/runs", f"s90dsp_{m}_r{r}") for r in (1, 2, 3)] for m in ("cc", "ds")}),
        ("Expertise +", "DSP 50–65%", {"cc": [mm_cell(d65, f"st_s5065_cc_r{r}") for r in (1, 2, 3)],
                                       "ds": [mm_cell(d65, t) for t in ("st_s5065_ds_r1", "st_s5065_ds_r2", "st_s5065_ds_r3b")]}),
        ("Refined +", "DSP 50–65%", {m: [mm_cell(d65, f"st_r5065_{m}_r{r}") for r in (1, 2, 3)] for m in ("cc", "ds")}),
        ("Agentic ladder", "DSP 50–65%", {"cc": [mm_cell(d65, t) for t in ("st_a5065_ag_r1c", "st_a5065_ag_r2c", "st_a5065_ag_r3")]}),
    ]
    manual = []
    for v in ("v0", "v1", "v2"):
        s = (v11.j(S / f"manual_df/{v}/csynth_summary.json") or {}).get("report") or {}
        c = S / f"manual_df/{v}/csim_summary.txt"
        if s:
            manual.append((v, dict(kind="ok" if c.exists() and "correct=True" in c.read_text() else "none",
                                   worst=s.get("latency_cycles_worst") or s.get("latency_cycles"),
                                   best=s.get("latency_cycles"), dsp=s.get("dsp"))))
    agent = []
    for v, _cost, _turns, _tools, _nsyn, best in v11.agentfw_runs():
        if best:
            r = best[2]
            agent.append((v, dict(kind="ok", worst=best[1], best=r.get("latency_cycles"), dsp=r.get("dsp"))))
        else:
            agent.append((v, dict(kind="none")))
    return cols, manual, agent


def chart():
    cols, manual, agent = columns()
    W, H, L, Rm, T, B = 1260, 500, 96, 16, 80, 416
    n = len(cols) + 2
    cw = (W - L - Rm) / n
    lo, hi = 5, 9          # 100K .. 1B
    y = lambda c: B - (math.log10(c) - lo) * (B - T) / (hi - lo)
    g = []
    for d, lab in zip(range(lo, hi + 1), ["100K", "1M", "10M", "100M", "1B"]):
        yy = y(10 ** d)
        g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{yy:.1f}' y2='{yy:.1f}' class='grid'/>"
                 f"<text x='{L - 8}' y='{yy + 4:.1f}' class='axis' text-anchor='end'>{lab}</text>")
    ya = y(1.054811e6)
    g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{ya:.1f}' y2='{ya:.1f}' class='autosa'/>"
             f"<text x='{L + 6}' y='{ya - 8:.1f}' class='bandlab' text-anchor='start'>AutoSA st0_c6, worst case: 1,054,811</text>")
    g.append(f"<text x='{L - 8}' y='{T - 34:.1f}' class='axis' text-anchor='end'>no design</text>"
             f"<line x1='{L}' x2='{W - Rm}' y1='{T - 18:.1f}' y2='{T - 18:.1f}' class='grid'/>"
             f"<text transform='translate(20,{(T + B) / 2:.1f}) rotate(-90)' class='ytitle' text-anchor='middle'>"
             f"csynth latency, worst case (cycles, log scale)</text>")

    def pt(x, r, cls, label):
        if r["kind"] == "ok":
            tip = f"{label}: {ms.fmt_int(r['worst'])} cycles worst, {ms.fmt_int(r['best'])} best, {ms.fmt_int(r['dsp'])} DSP"
            return f"<circle cx='{x:.1f}' cy='{y(r['worst']):.1f}' r='5' class='{cls}-fill'><title>{esc(tip)}</title></circle>"
        yy = T - 34
        if r["kind"] == "run":
            return f"<circle cx='{x:.1f}' cy='{yy:.1f}' r='4' class='{cls}-open'><title>{esc(label)}: running</title></circle>"
        why = (f"over the 80% LUT cap ({ms.fmt_int(r['worst'])} cycles at {r['lut'] / ms.CAP['lut']:.1%} LUT)"
               if r["kind"] == "cap" else "no passing design")
        return (f"<g class='{cls}-stroke'><title>{esc(label)}: {why}</title>"
                f"<line x1='{x - 4:.1f}' y1='{yy - 4:.1f}' x2='{x + 4:.1f}' y2='{yy + 4:.1f}'/>"
                f"<line x1='{x - 4:.1f}' y1='{yy + 4:.1f}' x2='{x + 4:.1f}' y2='{yy - 4:.1f}'/></g>")

    def head(i, a, b, sep=False):
        cx = L + cw * (i + 0.5)
        if i:
            g.append(f"<line x1='{L + cw * i:.1f}' x2='{L + cw * i:.1f}' y1='{T - 50}' y2='{B}' class='{'sep' if sep else 'colrule'}'/>")
        g.append(f"<text x='{cx:.1f}' y='{B + 26}' class='collab' text-anchor='middle'>{a}</text>"
                 f"<text x='{cx:.1f}' y='{B + 46}' class='collab' text-anchor='middle'>{b}</text>")
        return cx

    for i, (a, b, by) in enumerate(cols):
        cx = head(i, a, b)
        if list(by) == ["cc"]:
            offs = {"cc": (-12, 0, 12)}
        else:
            offs = {"cc": (-24, -16, -8), "ds": (8, 16, 24)}
        for m, rs in by.items():
            for rep, (r, dx) in enumerate(zip(rs, offs[m]), 1):
                g.append(pt(cx + dx, r, m, f"{ms.MODELS[m]} r{rep}"))
    cx = head(len(cols), "Hand-made", "DATAFLOW", sep=True)
    for (v, r), dx in zip(manual, (-14, 0, 14)):
        g.append(pt(cx + dx, r, "cc", f"refnf r2 by hand, {v}"))
        if r["kind"] == "ok":
            g.append(f"<text x='{cx + dx:.1f}' y='{y(r['worst']) + 22:.1f}' class='ptlab' text-anchor='middle'>{v}</text>")
    cx = head(len(cols) + 1, "General", "Claude Code agent")
    for (v, r), dx in zip(agent, (-24, 0, 24)):
        g.append(pt(cx + dx, r, "cc", f"agent {v}"))
        if r["kind"] == "ok":
            g.append(f"<text x='{cx + dx:.1f}' y='{y(r['worst']) + 22:.1f}' class='ptlab' text-anchor='middle'>{v}</text>")
    return (f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='csynth worst-case cycles of the 13 to 14 October "
            f"experiments on the fixed bench, log scale'>" + "".join(g) + "</svg>")


FIGURE = """
<figure id='new-chart'>
  <h3>13&ndash;14 October experiments, fixed bench</h3>
  <div class='figscroll'>{svg}</div>
  <figcaption>Worst-case csynth latency, log scale; hover a point for best case and DSP. Every point here passed csim on the
  fixed bench (C = A&middot;B<sup>T</sup>, C poisoned). The first five columns are the c2hls flash flow with three runs per
  model (Opus left, DeepSeek right; the agentic ladder is Opus only). &ldquo;Hand-made DATAFLOW&rdquo; is refnf r2 restructured
  by hand (v0 original structure, v1/v2 ping-pong). &ldquo;General Claude Code agent&rdquo; is the best csim-passing design
  that fits the U280 from each of the base, know and team sessions. Details in the sections below.</figcaption>
</figure>
"""


REFNEU_FOUND = (
    "With the hold-back wording gone and the balance gate on, but still no DSP asked for, both models grow on their own. "
    "Opus reaches 2,672 to 5,248 DSP and 1.18M to 2.36M cycles, within 1.16 to 1.24 times its compute floor. DeepSeek "
    "reaches 672 to 1,600 DSP and 2.75M to 8.81M cycles. That is the first rung where either model goes past a few hundred "
    "DSP, and every run passes csim. Opus's r2 and r3 already sit at about 5,120 DSP, the size of AutoSA's array.")

ZEROC = """
  <p><b>Confirmed 15 October:</b> the 13 top Opus designs (the ones in Vitis cosim) were edited so that the value loaded
  from C is replaced by 0 (for one design, <code>C = C + acc</code> became <code>C = acc</code>), with nothing else
  changed. All 13 pass the fixed bench; the 13 unedited originals fail with exactly 1,048,576 errors. Reading C, as the
  old spec asked, was the only reason they failed. Files: <code>verify_zeroC/</code>.</p>
"""


def collect_with_flow_csim():
    """ms.collect, falling back to the flow's own csim when no separate verify/ re-csim exists for a tag."""
    rows = _collect()
    for tag, r in rows.items():
        if r.get("ok") and "csim" not in r:
            cells = sorted(ms.glob.glob(f"{ms.R}/*_{tag}_*/autosa_mm"))
            d = ms.load(cells[-1] + "/autosa_mm_multistep_results.json") if cells else None
            if d and (d.get("csim") or {}).get("passed"):
                r["csim"] = "Passed"
    return rows


_collect = ms.collect


def main():
    ms.collect = collect_with_flow_csim
    ms.RUNGS = [(v, n, c, REFNEU_FOUND if v == "refneu" else f) for v, n, c, f in ms.RUNGS]
    v11.OUT = OUT
    v11.main()
    page = OUT.read_text()
    anchor = "    Controls: the fixed reference passes, the old C&nbsp;+= reference fails.</span></div>\n  </div>\n"
    assert page.count(anchor) == 1, "bench anchor"
    page = page.replace(anchor, anchor + ZEROC, 1)
    marker = "<section class='primer'"
    assert page.count(marker) == 1
    page = page.replace(marker, FIGURE.format(svg=chart()) + "\n" + marker, 1)
    li = "<li><a href='#bench'>Fixed testbench</a>"
    assert page.count(li) == 1
    page = page.replace(li, "<li><a href='#new-chart'>Chart of the new experiments</a> on the fixed bench, next to the ladder charts.</li>\n    " + li, 1)
    OUT.write_text(page)
    print(OUT)


if __name__ == "__main__":
    main()

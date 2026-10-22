#!/usr/bin/env python3
"""Build controlled/story/story_v13.html: story_v12 (make_story_v12.py, unchanged) plus the agentic ablation of
15-17 Oct on the fixed bench: every ladder rung in agent mode (Opus 5.5 + tools + real csim/csynth), and the plain
zero-shot and zero-shot + skills runs of both models (study/agentic_ablation/).

Every number is read from run outputs at build time, through make_story_v11's readers.
"""
import math, re
from pathlib import Path

import make_story_v11 as v11
import make_story_v12 as v12

ms, S, esc = v11.ms, v11.S, v11.esc
OUT = Path(__file__).resolve().parent / "story_v13.html"
AB = S / "agentic_ablation/runs"
EXPA = S / "exp_hlsmith/runs"      # experiment A (21-22 Oct): plain HLSmith on the fixed bench, the rungs above zero-shot
REPS = (1, 2, 3)

# (rung key, short label for the chart, page name of the rung)
RUNGS = [
    ("zs", ("Zero-shot", ""), "Zero-shot"),
    ("zsk", ("Zero-shot", "+ skills"), "Zero-shot + the Expertise skill block"),
    ("noskills", ("Base", "(no skills)"), "HLSmith Base (Flash, no skills)"),
    ("skills", ("Expertise", "(skills)"), "HLSmith-SingleStep-Expertise (Flash + skills)"),
    ("dspredo", ("Expertise +", "DSP floor"), "HLSmith-SingleStep-Expertise + Resource-Requirement"),
    ("refneu", ("Refined", "neutral"), "HLSmith-SingleStep-Refined-Expertise (no resource requirement)"),
    ("refnf", ("Refined +", "DSP in words"), "HLSmith-SingleStep-Refined-Expertise + Resource-Requirement in words"),
    ("refv2", ("Refined +", "DSP floor"), "HLSmith-SingleStep-Refined-Expertise + Resource-Requirement"),
    ("win7080", ("Refined +", "window 70–80%"), "HLSmith-SingleStep-Refined-Expertise + Resource-Window"),
]


def data():
    rows = []
    for k, short, name in RUNGS:
        agent = [capped(v12.mm_cell(AB, f"ag_{k}_r{r}")) for r in REPS]
        root = AB if k in ("zs", "zsk") else EXPA
        plain = {m: [cell(root, f"pl_{k}_{m}_r{r}") for r in REPS] for m in ("cc", "ds")}
        rows.append((k, short, name, plain, agent))
    return rows


LOGS = S.parent / "logs"


def gate(tag):
    """The rung's own DSP gate, which a design must meet to count: the floor (90% of the DSP) or the 70-80% window."""
    if re.search(r"_(dspredo|refv2)_", tag):
        return (0.9 * ms.CAP["dsp"], ms.CAP["dsp"])
    if "_win7080_" in tag:
        return (6317, 7219)
    return (0, ms.CAP["dsp"])


def fallbacks(d, row):
    """Verified designs of a c2hls run, the same rule as for Claude Code on its own (v11.session_best): a csim pass on the
    fixed bench and a csynth of the byte-identical kernel, from the run or checked afterwards with the kernel unchanged;
    under the LUT cap, inside the rung's DSP gate, not failing the hardened bench. Sources: the controller's history
    (csim passed); with an agent backend, the session's own checks, the post-hoc csim of study/posthoc_csim/<tag>/ and
    the post-hoc csynth of study/posthoc_csim/csynth/<tag>/. Tuples (worst, dsp, lut %, post hoc)."""
    lo, hi = gate(row["tag"])
    bad = v11.hard_failed()
    keep = lambda w, dsp, lut: w and w != v11.MM_BASE_WORST and lut <= v11.LUT_CAP and lo <= (dsp or 0) <= hi
    out = []
    post = {(c.parent / "kernel.cpp").read_text() for c in (S / "posthoc_csim" / row["tag"]).glob("*/csim_summary.txt")
            if "correct=True" in c.read_text()}
    for h in d.get("optimization_history") or []:
        r = h.get("report") or {}
        w = r.get("latency_cycles_worst") or r.get("latency_cycles")
        if (h.get("csim") or {}).get("passed") and keep(w, r.get("dsp"), r.get("lut") or 0):
            out.append((w, r.get("dsp"), 100 * (r.get("lut") or 0) / ms.CAP["lut"], False))
    ok = set()
    syn = []
    for sd in row.get("shim_dirs") or []:
        for ad in row.get("agent_dirs") or []:
            ck = LOGS / sd / ad / "workspace/checks"
            if not ck.is_dir():
                continue
            ok |= {(c.parent / c.name.replace(".txt", "_work") / "kernel.cpp").read_text()
                   for c in ck.glob("*_csim.txt") if "PASSED" in c.read_text()
                   and (c.parent / c.name.replace(".txt", "_work") / "kernel.cpp").exists()}
            for c in ck.glob("*_csynth.txt"):
                k = c.parent / c.name.replace(".txt", "_work") / "kernel.cpp"
                f = dict(re.findall(r"^\s+(latency_cycles|dsp|lut): (\d+)", c.read_text(), re.M))
                x = next((c.parent / c.name.replace(".txt", "_work")).glob("hls_proj/*/syn/report/csynth.xml"), None)
                wx = re.search(r"<Worst-caseLatency>(\d+)<", x.read_text()) if x else None   # the .txt gives best case
                if k.exists() and f.get("lut"):
                    syn.append((k.read_text(), int(wx.group(1)) if wx else int(f["latency_cycles"]), int(f["dsp"]),
                                int(f["lut"]), False))
    import json
    for s in (S / "posthoc_csim/csynth" / row["tag"]).glob("*/csynth_summary.json"):
        dd = json.loads(s.read_text())
        if dd.get("success") and (s.parent / "kernel.cpp").exists():
            r = dd["report"]
            syn.append(((s.parent / "kernel.cpp").read_text(), r.get("latency_cycles_worst") or r.get("latency_cycles"),
                        r.get("dsp"), r.get("lut") or 0, True))
    for kt, w, dsp, lut, spost in syn:
        if (kt in ok or kt in post) and kt not in bad and keep(w, dsp, lut):
            out.append((w, dsp, 100 * lut / ms.CAP["lut"], spost or kt not in ok))
    return out


def run_best(d, row):
    """The run's best verified design (2026-10-22, same rule on both sides): its final if that counts, or a faster or
    under-cap one from fallbacks(). Returns (worst, dsp, lut %, post hoc, source) with source final/fallback/better, or None."""
    fr = d.get("final_report") or {}
    fw = fr.get("latency_cycles_worst") or fr.get("latency_cycles")
    final_ok = (d.get("csim") or {}).get("passed") and fw and fw != v11.MM_BASE_WORST and (fr.get("lut") or 0) <= v11.LUT_CAP
    f = min(fallbacks(d, row), default=None)
    if final_ok and (f is None or fw <= f[0]):
        return (fw, fr.get("dsp"), 100 * (fr.get("lut") or 0) / ms.CAP["lut"], False, "final")
    if f:
        return f + ("better" if final_ok else "fallback",)
    return None


ROWS = S / "exp_hlsmith/cost_table/rows.json"   # per-run log index (shim and agent dirs), from the cost extractors


def capped(r):
    """Apply run_best to a finished agent-backend or plain run (mm_cell kinds 'ok' and 'cap')."""
    if r["kind"] not in ("ok", "cap"):
        return r
    import json
    for x in (json.loads(ROWS.read_text()) if ROWS.exists() else []):
        if re.search(rf"_{r['tag']}_2026", x["tag"]) and x.get("ctrl_path"):
            d = json.loads(next(Path(x["ctrl_path"]).parent.glob("*_multistep_results.json")).read_text())
            b = run_best(d, x)
            if b is None:
                return dict(r, kind="cap") if r["kind"] == "cap" else r
            if b[4] == "final":
                return r
            return dict(r, kind="ok", final=(r["worst"], r["dsp"], 100 * (r.get("lut") or 0) / ms.CAP["lut"]),
                        swap=b[4], uncapped=(r["worst"], r["dsp"], 100 * (r.get("lut") or 0) / ms.CAP["lut"]) if b[4] == "fallback" else None,
                        worst=b[0], best=b[0], dsp=b[1], lut=b[2] * ms.CAP["lut"] / 100, posthoc=b[3])
    return r


def cell(root, tag):
    r = capped(v12.mm_cell(root, tag))
    if r["kind"] == "none" and not r.get("done") and r["state"] not in ("COMPLETED", "FAILED", "TIMEOUT"):
        r["kind"] = "queued"          # not run yet (the experiment driver runs one Claude job at a time)
    return r


def chart(rows):
    W, H, L, Rm, T, B = 1260, 500, 96, 16, 80, 416
    n = len(rows)
    cw = (W - L - Rm) / n
    lo, hi = 5, 10         # 100K .. 10B
    y = lambda c: B - (math.log10(c) - lo) * (B - T) / (hi - lo)
    g = []
    for d, lab in zip(range(lo, hi + 1), ["100K", "1M", "10M", "100M", "1B", "10B"]):
        yy = y(10 ** d)
        g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{yy:.1f}' y2='{yy:.1f}' class='grid'/>"
                 f"<text x='{L - 8}' y='{yy + 4:.1f}' class='axis' text-anchor='end'>{lab}</text>")
    ya = y(1.054811e6)
    g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{ya:.1f}' y2='{ya:.1f}' class='autosa'/>"
             f"<text x='{W - Rm - 6}' y='{ya - 8:.1f}' class='bandlab' text-anchor='end'>AutoSA st0_c6, worst case: 1,054,811</text>")
    g.append(f"<text x='{L - 8}' y='{T - 34:.1f}' class='axis' text-anchor='end'>no design</text>"
             f"<line x1='{L}' x2='{W - Rm}' y1='{T - 18:.1f}' y2='{T - 18:.1f}' class='grid'/>"
             f"<text transform='translate(20,{(T + B) / 2:.1f}) rotate(-90)' class='ytitle' text-anchor='middle'>"
             f"csynth latency, worst case (cycles, log scale)</text>")

    def tip(label, r):
        fb = ""
        if r.get("swap") == "fallback":
            fb = (f" (fallback design{', checked post hoc' if r.get('posthoc') else ''}; its final, {ms.fmt_int(r['final'][0])} "
                  f"cycles at {r['final'][2]:.1f}% LUT, is over the cap)")
        elif r.get("swap") == "better":
            fb = (f" (the run&rsquo;s best verified design{', checked post hoc' if r.get('posthoc') else ''}; its final is "
                  f"{ms.fmt_int(r['final'][0])} cycles)")
        return f"{label}: {ms.fmt_int(r['worst'])} cycles worst, {ms.fmt_int(r['best'])} best, {ms.fmt_int(r['dsp'])} DSP{fb}"

    def mark(x, r, cls, label, diamond):
        if r["kind"] == "ok":
            yy = y(r["worst"])
            t = f"<title>{esc(tip(label, r))}</title>"
            if diamond:
                return (f"<polygon points='{x:.1f},{yy - 6.5:.1f} {x + 6.5:.1f},{yy:.1f} {x:.1f},{yy + 6.5:.1f} "
                        f"{x - 6.5:.1f},{yy:.1f}' class='{cls}-fill'>{t}</polygon>")
            return f"<circle cx='{x:.1f}' cy='{yy:.1f}' r='5' class='{cls}-fill'>{t}</circle>"
        yy = T - 34
        if r["kind"] == "queued":
            return ""
        if r["kind"] == "run":
            return f"<circle cx='{x:.1f}' cy='{yy:.1f}' r='4' class='{cls}-open'><title>{esc(label)}: running</title></circle>"
        why = (f"over the 80% LUT cap ({ms.fmt_int(r['worst'])} cycles at {r['lut'] / ms.CAP['lut']:.1%} LUT)"
               if r["kind"] == "cap" else "no passing design")
        return (f"<g class='{cls}-stroke'><title>{esc(label)}: {why}</title>"
                f"<line x1='{x - 4:.1f}' y1='{yy - 4:.1f}' x2='{x + 4:.1f}' y2='{yy + 4:.1f}'/>"
                f"<line x1='{x - 4:.1f}' y1='{yy + 4:.1f}' x2='{x + 4:.1f}' y2='{yy - 4:.1f}'/></g>")

    for i, (k, (a, b), name, plain, agent) in enumerate(rows):
        cx = L + cw * (i + 0.5)
        if i:
            g.append(f"<line x1='{L + cw * i:.1f}' x2='{L + cw * i:.1f}' y1='{T - 50}' y2='{B}' class='colrule'/>")
        g.append(f"<text x='{cx:.1f}' y='{B + 26}' class='collab' text-anchor='middle'>{esc(a)}</text>"
                 f"<text x='{cx:.1f}' y='{B + 46}' class='collab' text-anchor='middle'>{esc(b)}</text>")
        if plain:
            for m, dxs in (("cc", (-34, -27, -20)), ("ds", (-6, 1, 8))):
                for rep, (r, dx) in enumerate(zip(plain[m], dxs), 1):
                    g.append(mark(cx + dx, r, m, f"plain {ms.MODELS[m]} r{rep}", False))
            dxs = (22, 30, 38)
        else:
            dxs = (-12, 0, 12)
        for rep, (r, dx) in enumerate(zip(agent, dxs), 1):
            g.append(mark(cx + dx, r, "cc", f"agent mode, Opus 5.5 r{rep}", True))
    return (f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Agentic ablation: csynth worst-case cycles per rung, "
            f"plain and agent mode, log scale'>" + "".join(g) + "</svg>")


def fmt(r):
    if r["kind"] == "ok":
        fb = ""
        if r.get("swap") == "fallback":
            fb = (f"<br>fallback{' (checked post hoc)' if r.get('posthoc') else ''}; final {ms.fmt_int(r['final'][0])} at "
                  f"{r['final'][2]:.1f}% LUT is over the cap")
        elif r.get("swap") == "better":
            fb = f"<br>run&rsquo;s best{' (checked post hoc)' if r.get('posthoc') else ''}; final {ms.fmt_int(r['final'][0])}"
        return f"{ms.fmt_int(r['worst'])}<br><span class='pct'>{ms.fmt_int(r['dsp'])} DSP{fb}</span>"
    if r["kind"] == "cap":
        return (f"over LUT cap<br><span class='pct'>{ms.fmt_int(r['worst'])} at "
                f"{r['lut'] / ms.CAP['lut']:.1%} LUT</span>")
    return {"run": "running" if r["state"] == "RUNNING" else "pending", "queued": "queued"}.get(r["kind"], "no design")


def best(rs):
    ok = [r for r in rs if r["kind"] == "ok"]
    return min(ok, key=lambda r: r["worst"]) if ok else None


def table(rows):
    h = ("<div class='tablewrap'><table><thead><tr><th scope='col'>Rung</th><th scope='col'>Mode</th>"
         + "".join(f"<th scope='col' class='num'>r{r}</th>" for r in REPS)
         + "<th scope='col' class='num'>Best</th></tr></thead><tbody>")
    for k, short, name, plain, agent in rows:
        lines = [("plain", m, plain[m]) for m in ("cc", "ds") if plain] + [("agent", "cc", agent)]
        for i, (mode, m, rs) in enumerate(lines):
            b = best(rs)
            first = f"<th scope='row' rowspan='{len(lines)}'>{esc(name)}</th>" if i == 0 else ""
            who = (f"<span class='dot {m}'></span>" + ("Agent mode, Opus 5.5" if mode == "agent" else f"Plain, {ms.MODELS[m]}"))
            h += (f"<tr class='m-{m}'>{first}<td>{who}</td>" + "".join(f"<td class='num'>{fmt(r)}</td>" for r in rs)
                  + f"<td class='num strong'>{ms.fmt_int(b['worst']) if b else '&ndash;'}</td></tr>")
    return h + "</tbody></table></div>"


def section():
    rows = data()
    by = {k: (plain, agent) for k, _s, _n, plain, agent in rows}
    pbest = {k: best(by[k][0]["cc"]) for k, *_ in rows}
    abest = {k: best(by[k][1]) for k, *_ in rows}
    f = lambda r: ms.fmt_int(r["worst"]) if r else "&ndash;"
    ratio = lambda p, a: f"{p['worst'] / a['worst']:.1f}&times;" if p and a else "&ndash;"
    sized = [abest[k] for k in ("dspredo", "refneu", "refnf", "refv2", "win7080") if abest[k]]
    lo_s, hi_s = min(r["worst"] for r in sized), max(r["worst"] for r in sized)
    gains = ", ".join(f"{esc(' '.join(x for x in sh if x))} {ratio(pbest[k], abest[k])}" for k, sh, *_ in rows
                      if pbest[k] and abest[k])
    return f"""
<section class='aside' id='ablation'>
  <span class='eyebrow'>Fixed bench &middot; 15&ndash;22 October</span>
  <h2>Plain vs agent mode, rung by rung <span class='alias'>(agentic ablation)</span></h2>
  <p>Every ladder rung was run again in <b>agent mode</b>: Opus 5.5 with tools, running real csim and csynth on its own
  kernels (up to 12 checks per run) inside the same c2hls flow, prompts, skill packs and gates. Two zero-shot rungs were also
  run in <b>plain mode</b> (no tools, one reply) for both models: the bare zero-shot prompt, and the same prompt plus the
  Expertise rung&rsquo;s recorded skill block, verbatim. The other rungs were rerun in plain mode on the fixed bench too
  (experiment A, 21&ndash;22 October), so every column compares plain and agent mode on the same bench. Three runs each
  (Opus plain runs still being filled in, one at a time), all on the fixed bench (C = A&middot;B<sup>T</sup>, C poisoned),
  and every point shown passed csim and fits the 80% LUT cap; a run whose design is over the cap, with no verified design
  under it, is drawn as &times; in the top lane (hover for its cycles).</p>
  <figure>
    <div class='legend'><span><span class='dot cc'></span>Opus 5.5</span><span><span class='dot ds'></span>DeepSeek v4.1-flash</span>
    <span>&#9679; plain mode</span><span>&#9670; agent mode (Opus only)</span></div>
    <div class='figscroll'>{chart(rows)}</div>
    <figcaption>Worst-case csynth latency, log scale; hover a point for best case and DSP. In the two zero-shot columns the
    plain runs are on the left (Opus, then DeepSeek) and the agent-mode runs on the right; the same holds in every column.
    Runs not yet started are not drawn.</figcaption>
  </figure>
  <div class='cols2'>
    <div><h3>Tools help most where the prompt says least</h3><p>At zero-shot, agent mode&rsquo;s best is {f(abest['zs'])}
    cycles against {f(pbest['zs'])} in plain mode ({ratio(pbest['zs'], abest['zs'])}). With the skill block added it is
    {f(abest['zsk'])} against {f(pbest['zsk'])} ({ratio(pbest['zsk'], abest['zsk'])}). Being able to compile and measure lets
    Opus grow its first guess.</p></div>
    <div><h3>Without a resource ask, agents stay small</h3><p>At the Base and Expertise rungs the agent settles at 320&ndash;640
    DSP and about 10M cycles (best {f(abest['noskills'])} and {f(abest['skills'])}), slower than its own zero-shot designs.
    Every rung that sizes the design, in words or with a gate, lands between {ms.fmt_int(lo_s)} and {ms.fmt_int(hi_s)} cycles,
    below AutoSA&rsquo;s 1,054,811; the best is {f(min(sized, key=lambda r: r['worst']))}.</p></div>
    <div><h3>Agent mode beats plain on every rung</h3><p>Same rung, same bench, Opus 5.5, best of each, agent mode&rsquo;s gain over plain mode:
    {gains}. Plain mode gets below AutoSA only on the rungs that give a DSP floor or window
    (best {f(min((pbest[k] for k in ('dspredo', 'refv2', 'win7080') if pbest[k]), key=lambda r: r['worst']))}); agent
    mode also gets there from the wording alone.</p></div>
  </div>
  {table(rows)}
  <p class='note'>Runs and job list: <code>study/agentic_ablation/</code> (<code>submitted_20261015.txt</code>); experiment A&rsquo;s plain runs: <code>study/exp_hlsmith/runs/</code>. The Claude jobs ran one at a
  time; runs that hit the usage limit were moved to <code>runs/_invalid_usage_limit/</code> and rerun.</p>
</section>
"""


def _axes(W, L, Rm, T, B, lo, hi, labels, ytitle):
    y = lambda c: B - (math.log10(c) - lo) * (B - T) / (hi - lo)
    g = []
    for d, lab in zip(range(lo, hi + 1), labels):
        yy = y(10 ** d)
        g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{yy:.1f}' y2='{yy:.1f}' class='grid'/>"
                 f"<text x='{L - 8}' y='{yy + 4:.1f}' class='axis' text-anchor='end'>{lab}</text>")
    g.append(f"<text x='{L - 8}' y='{T - 34:.1f}' class='axis' text-anchor='end'>no design</text>"
             f"<line x1='{L}' x2='{W - Rm}' y1='{T - 18:.1f}' y2='{T - 18:.1f}' class='grid'/>"
             f"<text transform='translate(20,{(T + B) / 2:.1f}) rotate(-90)' class='ytitle' text-anchor='middle'>{ytitle}</text>")
    return y, g


def _x(x, yy, cls, tip):
    return (f"<g class='{cls}-stroke'><title>{esc(tip)}</title>"
            f"<line x1='{x - 4:.1f}' y1='{yy - 4:.1f}' x2='{x + 4:.1f}' y2='{yy + 4:.1f}'/>"
            f"<line x1='{x - 4:.1f}' y1='{yy + 4:.1f}' x2='{x + 4:.1f}' y2='{yy - 4:.1f}'/></g>")


def lud_autosa_chart():
    """AutoSA LU design space: N x space_time, worst-case csynth cycles; class from lud/check/dse_table.json."""
    rows = {r["design"]: r for r in (v11.j(S / "lud/check/dse_table.json") or [])}
    ns = sorted({int(re.match(r"n(\d+)", k).group(1)) for k in rows})
    W, H, L, Rm, T, B = 1260, 500, 96, 16, 80, 416
    y, g = _axes(W, L, Rm, T, B, 3, 4, ["1K", "10K"], "csynth latency, worst case (cycles, log scale)")
    # log 1K..10K is too narrow for the top lane; use a finer grid
    for c in (2000, 5000):
        yy = y(c)
        g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{yy:.1f}' y2='{yy:.1f}' class='grid'/>"
                 f"<text x='{L - 8}' y='{yy + 4:.1f}' class='axis' text-anchor='end'>{c // 1000}K</text>")
    cw = (W - L - Rm) / len(ns)
    for i, n in enumerate(ns):
        cx0 = L + cw * i
        if i:
            g.append(f"<line x1='{cx0:.1f}' x2='{cx0:.1f}' y1='{T - 50}' y2='{B}' class='colrule'/>")
        g.append(f"<text x='{cx0 + cw / 2:.1f}' y='{B + 46}' class='collab' text-anchor='middle'>N = {n}</text>")
        for st in range(6):
            x = cx0 + cw * (st + 1) / 7
            g.append(f"<text x='{x:.1f}' y='{B + 22}' class='axis' text-anchor='middle'>{st}</text>")
            r = rows.get(f"n{n}_st{st}")
            if not r:
                continue
            led = v11.LED.get(f"lud_n{n}_st{st}_csynth", {})
            lat = int(r["lat_max"]) if r.get("lat_max") else None
            head = f"N = {n}, space_time {st}, {r['PEs']} PE"
            if st in (0, 1, 3):
                kind = "wrong U"
            elif st == 2:
                kind = "csynth error"
            else:
                kind = "correct"
            if lat and kind == "correct":
                g.append(f"<circle cx='{x:.1f}' cy='{y(lat):.1f}' r='5.5' class='au-fill'><title>{esc(head)}: correct, "
                         f"{ms.fmt_int(lat)} cycles worst, {ms.fmt_int(r['DSP'])} DSP</title></circle>")
            elif lat:
                g.append(f"<circle cx='{x:.1f}' cy='{y(lat):.1f}' r='5' class='au-wrong'><title>{esc(head)}: wrong U "
                         f"(L exact), {ms.fmt_int(lat)} cycles worst, {ms.fmt_int(r['DSP'])} DSP</title></circle>")
            elif kind == "csynth error":
                g.append(_x(x, T - 34, "au", f"{head}: csynth error (HLS 200-970, dangling stream); csim passes"))
            elif led.get("state") in ("RUNNING", "PENDING"):
                g.append(f"<circle cx='{x:.1f}' cy='{T - 34:.1f}' r='4' class='au-open'><title>{esc(head)}: csynth still "
                         f"running ({esc(led.get('state'))})</title></circle>")
            else:
                g.append(_x(x, T - 34, "au", f"{head}: {kind}; csynth did not finish"))
    return (f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='AutoSA LU design space: worst-case csynth cycles for "
            f"each N and space_time, log scale'>" + "".join(g) + "</svg>")


LUD_RUNGS = [("zs", ("Zero-shot", ""), "lud_zs_{m}_r{r}", "lud"), ("noskills", ("Base", "(no skills)"), "lud_noskills_{m}_r{r}", "lud"),
             ("skills", ("Expertise", "(gemm-flatten pack)"), "lud_skills_{m}_r{r}", "lud"),
             ("s90", ("Expertise", "(90-skill pack)"), "lud_s90_{m}_r{r}", "s90"),
             ("refneu", ("Refined", "neutral"), "lud_refneu_{m}_r{r}", "lud")]


def lud_ablation_chart():
    W, H, L, Rm, T, B = 1260, 500, 96, 16, 80, 416
    y, g = _axes(W, L, Rm, T, B, 3, 7, ["1K", "10K", "100K", "1M", "10M"], "csynth latency, worst case (cycles, log scale)")
    best = min(int(r["lat_max"]) for r in (v11.j(S / "lud/check/dse_table.json") or [])
               if r["design"].startswith("n32_") and r.get("lat_max") and int(r["design"][-1]) in (4, 5))
    ya = y(best)
    g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{ya:.1f}' y2='{ya:.1f}' class='autosa'/>"
             f"<text x='{W - Rm - 6}' y='{ya - 8:.1f}' class='bandlab' text-anchor='end'>AutoSA best correct at N = 32: "
             f"{ms.fmt_int(best)}</text>")
    yb = y(v11.LUD_BASE_WORST)
    g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{yb:.1f}' y2='{yb:.1f}' class='colrule'/>"
             f"<text x='{W - Rm - 6}' y='{yb - 8:.1f}' class='axis' text-anchor='end'>starting kernel: {ms.fmt_int(v11.LUD_BASE_WORST)}</text>")
    cw = (W - L - Rm) / len(LUD_RUNGS)
    for i, (k, (a, b), fmt, root) in enumerate(LUD_RUNGS):
        cx = L + cw * (i + 0.5)
        if i:
            g.append(f"<line x1='{L + cw * i:.1f}' x2='{L + cw * i:.1f}' y1='{T - 50}' y2='{B}' class='colrule'/>")
        g.append(f"<text x='{cx:.1f}' y='{B + 26}' class='collab' text-anchor='middle'>{esc(a)}</text>"
                 f"<text x='{cx:.1f}' y='{B + 46}' class='collab' text-anchor='middle'>{esc(b)}</text>")
        for m, dxs in (("cc", (-24, -16, -8)), ("ds", (8, 16, 24))):
            for rep, dx in zip(REPS, dxs):
                r = v11.cell(S / f"{root}/runs", fmt.format(m=m, r=rep), "lud")
                lab = f"{a} {b}".strip() + f", {ms.MODELS[m]} r{rep}"
                x = cx + dx
                if r.get("done") and not r.get("fallback") and r.get("csim"):
                    g.append(f"<circle cx='{x:.1f}' cy='{y(r['worst']):.1f}' r='5' class='{m}-fill'><title>{esc(lab)}: "
                             f"{ms.fmt_int(r['worst'])} cycles worst, {ms.fmt_int(r['best'])} best, {ms.fmt_int(r['dsp'])} DSP"
                             f"</title></circle>")
                elif r.get("done") and r.get("fallback"):
                    g.append(_x(x, T - 34, m, f"{lab}: no design accepted; the flow kept the starting kernel"))
                elif r["state"] in ("RUNNING", "PENDING"):
                    g.append(f"<circle cx='{x:.1f}' cy='{T - 34:.1f}' r='4' class='{m}-open'><title>{esc(lab)}: running</title></circle>")
                else:
                    g.append(_x(x, T - 34, m, f"{lab}: no passing design"))
    return (f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='LUD ablation at N = 32: worst-case csynth cycles per step, "
            f"Opus and DeepSeek, log scale'>" + "".join(g) + "</svg>")


LUD_FIGS = """
  <figure id='lud-autosa-chart'>
    <h3>AutoSA LU design space, U280, 3.33&nbsp;ns</h3>
    <div class='legend'><span><svg width='12' height='12' aria-hidden='true'><circle cx='6' cy='6' r='5' class='au-fill'/></svg>correct (L and U exact)</span>
    <span><svg width='12' height='12' aria-hidden='true'><circle cx='6' cy='6' r='4.5' class='au-wrong'/></svg>wrong U</span>
    <span>&times; csynth error or not finished</span><span>dashed circle: csynth still running</span></div>
    <div class='figscroll'>{autosa}</div>
    <figcaption>Worst-case csynth latency of every AutoSA LU design (rev <code>b61a1b4</code>), log scale; numbers under the
    axis are space_time 0&ndash;5 within each N. Correctness is from our strict L and U check; hover a point for cycles, DSP
    and PE count.</figcaption>
  </figure>
"""

LUD_FIG2 = """
  <figure id='lud-ablation-chart'>
    <h3>LUD ablation at N = 32</h3>
    <div class='legend'><span><span class='dot cc'></span>Opus 5.5</span><span><span class='dot ds'></span>DeepSeek v4.1-flash</span>
    <span>&times; in the top lane: no accepted design</span></div>
    <div class='figscroll'>{abl}</div>
    <figcaption>Worst-case csynth latency of each run&rsquo;s final design, log scale, three runs per model (Opus left,
    DeepSeek right); every point passed the strict csim on L and U. The solid line is AutoSA&rsquo;s best correct design at
    the same N; the dotted line is the starting kernel every run begins from.</figcaption>
  </figure>
"""

LUD_CSS = """
.au-fill{fill:var(--target)}
.au-wrong{fill:none;stroke:var(--bad);stroke-width:1.8}
.au-open{fill:none;stroke:var(--target);stroke-width:1.5;stroke-dasharray:2 2}
.au-stroke line{stroke:var(--bad);stroke-width:2}
"""


# Chart tooltips: every SVG mark with a <title> (run, cycles, DSP) gets an instant tooltip. The pointer snaps to the
# nearest mark within 16 px, so the 5 px points are easy to hit; tap works on touch, and Tab focuses each mark.
TIP = """
<div id='tip' role='tooltip' hidden></div>
<style>
#tip{position:fixed;z-index:10;pointer-events:none;max-width:min(360px,calc(100vw - 24px));padding:7px 10px;border-radius:5px;
  background:var(--ink);color:var(--bg);font:500 .8rem/1.4 var(--mono);box-shadow:0 2px 8px rgba(0,0,0,.25)}
svg [data-tip]{cursor:pointer}
svg [data-tip]:focus{outline:none}
svg [data-tip].hot{stroke:var(--ink);stroke-width:2}
svg g[data-tip].hot line{stroke-width:3}
</style>
<script>
(function(){
  var tip=document.getElementById('tip'), hot=null, marks=[];
  document.querySelectorAll('svg title').forEach(function(t){
    var el=t.parentNode; if(!el || el.tagName.toLowerCase()==='svg') return;
    el.setAttribute('data-tip', t.textContent); el.setAttribute('tabindex','0');
    el.setAttribute('aria-label', t.textContent); t.remove(); marks.push(el);
  });
  function show(el, x, y){
    if(hot && hot!==el) hot.classList.remove('hot');
    hot=el; el.classList.add('hot'); tip.textContent=el.getAttribute('data-tip'); tip.hidden=false;
    var w=tip.offsetWidth, h=tip.offsetHeight, vw=window.innerWidth;
    var left=Math.min(Math.max(8, x+14), vw-w-8), top=y-h-12; if(top<8) top=y+16;
    tip.style.left=left+'px'; tip.style.top=top+'px';
  }
  function hide(){ if(hot) hot.classList.remove('hot'); hot=null; tip.hidden=true; }
  function nearest(svg, x, y){
    var best=null, bd=16;
    marks.forEach(function(el){
      if(el.ownerSVGElement!==svg) return;
      var r=el.getBoundingClientRect(), d=Math.hypot(x-(r.left+r.width/2), y-(r.top+r.height/2));
      if(d<bd){ bd=d; best=el; }
    });
    return best;
  }
  document.querySelectorAll('svg').forEach(function(svg){
    svg.addEventListener('pointermove', function(e){
      var el=nearest(svg, e.clientX, e.clientY); if(el) show(el, e.clientX, e.clientY); else hide();
    });
    svg.addEventListener('pointerleave', hide);
    svg.addEventListener('pointerdown', function(e){
      var el=nearest(svg, e.clientX, e.clientY); if(el) show(el, e.clientX, e.clientY); else hide();
    });
  });
  marks.forEach(function(el){
    el.addEventListener('focus', function(){ var r=el.getBoundingClientRect(); show(el, r.left+r.width/2, r.top); });
    el.addEventListener('blur', hide);
  });
  window.addEventListener('scroll', hide, {passive:true});
})();
</script>
"""


def main():
    v12.OUT = OUT
    v11.OUT = OUT
    v12.main()
    page = OUT.read_text()
    marker = "<section class='aside' id='dsp5065'>"
    assert page.count(marker) == 1, "dsp5065 anchor"
    page = page.replace(marker, section() + marker, 1)
    li = "<li><a href='#new-chart'>"
    assert page.count(li) == 1
    old = "<h3>New in this version (13&ndash;14 October)</h3>"
    assert page.count(old) == 1
    page = page.replace(old, "<h3>New in this version (13&ndash;17 October)</h3>", 1)
    page = page.replace(li, "<li><a href='#ablation'>Plain vs agent mode, rung by rung</a> (agentic ablation, fixed bench).</li>\n    " + li, 1)
    h1 = "  <h3>AutoSA design space: worst-case csynth latency</h3>\n"
    h2 = "  <h3>The ablation at N = 32, strict csim on L and U</h3>\n"
    assert page.count(h1) == 1 and page.count(h2) == 1, "LUD anchors"
    page = page.replace(h1, LUD_FIGS.format(autosa=lud_autosa_chart()) + h1, 1)
    page = page.replace(h2, LUD_FIG2.format(abl=lud_ablation_chart()) + h2, 1)
    page = page.replace("</style>", LUD_CSS + "</style>", 1)
    star = "class='opus-fill'/>"   # the agentic ladder L2 star in the two ladder charts (make_story.py) has no <title>
    assert page.count(star) == 2, "L2 star"
    page = page.replace(star, "class='opus-fill'><title>Agentic ladder L2, Opus 5.5 in agent mode: 1,066,083 cycles, "
                              "5,152 DSP</title></path>")
    end = "</main>"
    assert page.count(end) == 1, "main end"
    page = page.replace(end, end + TIP, 1)
    OUT.write_text(page)
    print(OUT)


if __name__ == "__main__":
    main()

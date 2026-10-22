#!/usr/bin/env python3
"""Build controlled/story/story_v14.html: story_v13 (make_story_v13.py, unchanged) plus a cost section: what one run of
each flow costs on mm 1024^3 (LLM calls, Vitis checks, dollars, wall time) next to the design it produced.

Run data comes from study/exp_hlsmith/cost_table/rows.json (the log extractors in that directory: shim logs, controller
call logs, transcripts) and, for the HLSmith-knowledge free-agent sessions of experiments C and D, from
study/exp_hlsmith/cost_table/exp_cd.json (read off the agentfw workspaces after they ended).
"""
import json, math, re, statistics
from pathlib import Path

import make_story_v13 as v13

ms, S, esc = v13.ms, v13.S, v13.esc
OUT = Path(__file__).resolve().parent / "story_v14.html"
CT = S / "exp_hlsmith/cost_table"
RUNGS = [k for k, *_ in v13.RUNGS]
NAME = {k: n for k, _s, n in v13.RUNGS}
SHORT = {k: " ".join(x for x in s if x) for k, s, _n in v13.RUNGS}

# Free-agent sessions of 15-19 Oct (agentfw section). rows.json has their usage but not their designs; these are the
# csim-passing designs from the agentfw page. team: no design inside its session; its drafts were checked after it
# (exp_post/: v3, csim'd in session, synthesized afterwards at 527,571; v3_ti4, synthesized in session, csim'd afterwards
# at 4,195,795). Both passed the hardened bench (study/posthoc_csim/hard.csv).
# All designs are counted under the page's 80% LUT cap (v13.v11.LUT_CAP); base's fastest (404,451 at 87% LUT) is over it.
FREE = {"mm1024_base_r1": ("Free agent, base tools",),          # designs: v11.session_best (free_best below)
        "mm1024_know_r1": ("Free agent + team knowledge",),
        "mm1024_team_r1": ("Free agent + knowledge + subagents",)}
FREE_CHECKS = {"mm1024_base_r1": 14, "mm1024_know_r1": 22, "mm1024_team_r1": 17}
# base and know hit the session limit after about 2.2 h and were resumed after the reset: active time, not wall time
FREE_ACTIVE = {"mm1024_base_r1": 2.18 * 3600, "mm1024_know_r1": 5.81 * 3600}


WINDOW = (6317, 7219)    # experiment C/D DSP window (70-80% of the U280)


def free_best(tag, window=None, in_session_only=False):
    """worst/dsp/lut_pct/note/uncapped of a Claude Code session under the page's rule (v11.session_best). Matched-budget
    runs (C/D budget rows) count only what the session itself finished: the budget is the condition being compared."""
    sb = v13.v11.session_best(S / "agentfw/runs" / tag, window, in_session_only)
    b = sb["best"]
    if not b:
        return dict(worst=None, dsp=None, note=None, uncapped=None)
    lut = 100 * (b[2].get("lut") or 0) / ms.CAP["lut"]
    return dict(worst=b[1], dsp=b[2].get("dsp"), lut_pct=lut, note="post-session" if b[3] else None,
                uncapped=list(sb["fastest"]) if sb["fastest"] else None)


def capped(base, row):
    """Apply v13.run_best (the run's best verified design, same rule as Claude Code on its own) to a c2hls run."""
    ctrl_path = row.get("ctrl_path")
    if not (ctrl_path and base.get("worst")):
        return base
    for f in Path(ctrl_path).parent.glob("*_multistep_results.json"):
        d = json.loads(f.read_text())
        fr = d.get("final_report") or {}
        b = v13.run_best(d, row)
        lut = 100 * (fr.get("lut") or 0) / ms.CAP["lut"]
        if b is None:
            base = dict(base, worst=None, **({"uncapped": (base["worst"], base["dsp"], lut)} if lut > 80 else {}))
        elif b[4] != "final":
            base = dict(base, worst=b[0], dsp=b[1], lut_pct=b[2],
                        note=("fallback" if b[4] == "fallback" else "run&rsquo;s best, final " + ms.fmt_int(base["worst"]))
                             + (", checked post hoc" if b[3] else ""),
                        **({"uncapped": (base["worst"], base["dsp"], lut)} if b[4] == "fallback" else {}))
    return base


def runs():
    """One dict per Opus mm1024 run: flow, rung, calls, checks, cost, wall, worst, dsp, tag, bench."""
    out = []
    for r in json.loads((CT / "rows.json").read_text()):
        if r.get("excluded") or (r["family"] != "F3" and r.get("model") != "claude-opus-5-5"):
            continue
        t, fam = r["tag"], r["family"]
        base = capped(dict(tag=t, cost=r.get("cost_usd_shim"), wall=r.get("wallclock_s"), worst=r.get("lat_worst"),
                           dsp=r.get("dsp")), r)
        if fam == "F1" and r["mode"] == "agentic-HLSmith":
            m = re.search(r"_ag_(\w+?)_r(\d)_", t)
            out.append(dict(base, flow="agent", rung=m.group(1), rep=m.group(2), calls=r.get("agent_llm_calls"),
                            checks=(r.get("checks_csim") or 0) + (r.get("checks_csynth") or 0), bench="fixed"))
        elif fam == "F1":
            m = re.search(r"_pl_(\w+?)_cc_r(\d)_", t)
            out.append(dict(base, flow="plain", rung=m.group(1), rep=m.group(2), calls=r.get("ctrl_calls"), checks=0,
                            bench="fixed"))
        elif fam == "F2-runs_controlled" and "_ctl_cc_" in t:
            m = re.search(r"_ctl_cc_(\w+?)_r(\d)_", t)
            if m.group(1) in ("zs",):       # the fixed-bench zero-shot runs above replace these
                continue
            out.append(dict(base, flow="plain", rung=m.group(1), rep=m.group(2), calls=r.get("ctrl_calls"), checks=0,
                            bench="old"))
        elif fam == "F3" and t in FREE:
            name = FREE[t][0]
            out.append(dict(tag=t, flow="free", rung=t, rep="1", calls=r.get("llm_calls"), checks=FREE_CHECKS[t],
                            cost=r.get("cost_usd"), wall=FREE_ACTIVE.get(t, r.get("wallclock_s")), bench="fixed",
                            name=name, **free_best(t)))
    # experiment A: plain HLSmith rerun on the fixed bench; where a rung has those runs they replace the old-bench ones
    pa = CT / "rows_expA.json"
    expa = []
    for r in (json.loads(pa.read_text()) if pa.exists() else []):
        m = re.search(r"_pl_(\w+?)_cc_r(\d)_", r["tag"])
        if m and r.get("model") == "claude-opus-5-5" and r.get("cost_usd_shim") is not None:
            expa.append(capped(dict(tag=r["tag"], flow="plain", rung=m.group(1), rep=m.group(2), calls=r.get("ctrl_calls"),
                                    checks=0, cost=r.get("cost_usd_shim"), wall=r.get("wallclock_s"),
                                    worst=r.get("lat_worst") if r.get("csim_ms") == "passed" else None, dsp=r.get("dsp"),
                                    bench="fixed"), r))
    done = {r["rung"] for r in expa}
    out = [r for r in out if not (r["flow"] == "plain" and r["bench"] == "old" and r["rung"] in done)] + expa
    p = CT / "exp_cd.json"
    for r in (json.loads(p.read_text()) if p.exists() else []):
        out.append(dict(r, flow="hlsm" if r["model"] == "cc" else "hlsm_ds", bench="fixed",
                        **free_best(r["tag"], WINDOW, in_session_only=bool(r.get("budget")))))
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def money(x):
    return "&ndash;" if x is None else (f"${x:.2f}" if x < 100 else f"${x:,.0f}")


def dur(s):
    if s is None:
        return "&ndash;"
    return f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


def num(x):
    return "&ndash;" if x is None else ms.fmt_int(round(x))


def best(rs):
    ok = [r for r in rs if r.get("worst")]
    return min(ok, key=lambda r: r["worst"]) if ok else None


def pct(x):
    return f"{x:.1f}%" if 79 < x < 81 else f"{x:.0f}%"     # one decimal near the 80% cap, where rounding misleads


def bestcell(rs):
    b = best(rs)
    unc = [r["uncapped"] for r in rs if r.get("uncapped") and (not b or r["uncapped"][0] < b["worst"])]
    over = f"<br>over the cap: {ms.fmt_int(min(unc)[0])} at {min(unc)[2]:.1f}% LUT" if unc else ""
    if not b:
        return f"<td class='num'>no design<span class='pct'>{over}</span></td>"
    tail = f" &middot; {b['note']}" if b.get("note") else ""
    if b.get("lut_pct"):
        tail += f" &middot; {pct(b['lut_pct'])} LUT"
    tail += over
    return f"<td class='num strong'>{ms.fmt_int(b['worst'])}<br><span class='pct'>{ms.fmt_int(b['dsp'])} DSP{tail}</span></td>"


def row(first, label, rs, n_label=None):
    calls = med([r["calls"] for r in rs])
    checks = med([r["checks"] for r in rs])
    return (f"<tr{first}><td>{label}</td><td class='num'>{n_label or len(rs)}</td><td class='num'>{num(calls)}</td>"
            f"<td class='num'>{num(checks) if checks else 'none'}</td><td class='num'>{money(med([r['cost'] for r in rs]))}</td>"
            f"<td class='num'>{dur(med([r['wall'] for r in rs]))}</td>{bestcell(rs)}</tr>")


def table(rs):
    h = ("<div class='tablewrap'><table><thead><tr><th scope='col'>Rung</th><th scope='col' class='num'>Runs</th>"
         "<th scope='col' class='num'>LLM calls</th><th scope='col' class='num'>Vitis checks by the LLM</th>"
         "<th scope='col' class='num'>Cost per run</th><th scope='col' class='num'>Wall time per run</th>"
         "<th scope='col' class='num'>Best design, worst-case cycles</th></tr></thead><tbody>")
    groups = [("plain", "Plain HLSmith: one reply, no tools (the flow runs csim and csynth)"),
              ("agent", "Agentic HLSmith: the same flow, the LLM step in agent mode")]
    for flow, title in groups:
        h += f"<tr class='grp'><th scope='rowgroup' colspan='7'>{title}</th></tr>"
        for k in RUNGS:
            g = [r for r in rs if r["flow"] == flow and r["rung"] == k]
            if g:
                old = " <span class='sub'>old bench</span>" if g[0]["bench"] == "old" else ""
                h += row("", f"{esc(SHORT[k])}{old}", g)
    h += "<tr class='grp'><th scope='rowgroup' colspan='7'>Claude Code&rsquo;s own agent mode: no HLSmith flow</th></tr>"
    for t, (name, *_x) in FREE.items():
        g = [r for r in rs if r["tag"] == t]
        if g:
            h += row("", esc(name), g)
    for flow, name in (("hlsm", "Free agent + HLSmith&rsquo;s skills and DSP window"),):
        for bud, lab in ((True, "12 checks, 3 h"), (False, "unbounded, 10 h cap")):
            g = [r for r in rs if r["flow"] == flow and bool(r.get("budget")) == bud]
            if g:
                h += row("", f"{name} <span class='sub'>{lab}</span>", g)
    return h + "</tbody></table></div>"


def chart(rs):
    W, H, L, Rm, T, B = 1260, 480, 96, 16, 30, 400
    xlo, xhi = -1.5, 1.5                  # $0.03 .. $30
    ylo, yhi = 5, 8                       # 100K .. 100M
    x = lambda c: L + (math.log10(c) - xlo) * (W - Rm - L) / (xhi - xlo)
    y = lambda c: B - (math.log10(c) - ylo) * (B - T) / (yhi - ylo)
    g = []
    for d, lab in zip(range(ylo, yhi + 1), ["100K", "1M", "10M", "100M"]):
        yy = y(10 ** d)
        g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{yy:.1f}' y2='{yy:.1f}' class='grid'/>"
                 f"<text x='{L - 8}' y='{yy + 4:.1f}' class='axis' text-anchor='end'>{lab}</text>")
    for c, lab in ((0.03, "$0.03"), (0.1, "$0.10"), (0.3, "$0.30"), (1, "$1"), (3, "$3"), (10, "$10"), (30, "$30")):
        xx = x(c)
        g.append(f"<line x1='{xx:.1f}' x2='{xx:.1f}' y1='{T}' y2='{B}' class='grid'/>"
                 f"<text x='{xx:.1f}' y='{B + 22}' class='axis' text-anchor='middle'>{lab}</text>")
    ya = y(1.054811e6)
    g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{ya:.1f}' y2='{ya:.1f}' class='autosa'/>"
             f"<text x='{L + 8}' y='{ya - 8:.1f}' class='bandlab'>AutoSA st0_c6, worst case: 1,054,811</text>"
             f"<text transform='translate(20,{(T + B) / 2:.1f}) rotate(-90)' class='ytitle' text-anchor='middle'>"
             f"csynth latency, worst case (cycles, log scale)</text>"
             f"<text x='{(L + W - Rm) / 2:.1f}' y='{B + 50}' class='ytitle' text-anchor='middle'>LLM cost of the run "
             f"(USD, log scale)</text>")
    for r in rs:
        if not (r.get("cost") and r.get("worst")):     # no design, or none under the LUT cap
            continue
        cx, cy = x(min(max(r["cost"], 0.035), 28)), y(min(max(r["worst"], 1.1e5), 9e7))
        what = {"plain": "Plain HLSmith", "agent": "Agentic HLSmith", "free": r.get("name", "Free agent"),
                "hlsm": "Free agent + HLSmith knowledge"}.get(r["flow"], r["flow"])
        rung = f", {SHORT[r['rung']]}" if r["rung"] in SHORT else ""
        tip = (f"{what}{rung} r{r['rep']}: ${r['cost']:.2f}, {ms.fmt_int(r['worst'])} cycles worst, "
               f"{ms.fmt_int(r['dsp'])} DSP, {dur(r['wall']).replace('&ndash;', '?')}, {num(r['calls'])} LLM calls"
               + (" (old bench)" if r["bench"] == "old" else "") + (f" ({r['note']})" if r.get("note") else "")
               + (f", {pct(r['lut_pct'])} LUT" if r.get("lut_pct") else "")
               + (f" (fastest design {ms.fmt_int(r['uncapped'][0])} at {r['uncapped'][2]:.1f}% LUT, over the cap)"
                  if r.get("uncapped") else ""))
        t = f"<title>{esc(tip)}</title>"
        if r["flow"] == "plain":
            g.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='5' class='cost-plain'>{t}</circle>")
        elif r["flow"] == "agent":
            g.append(f"<polygon points='{cx:.1f},{cy - 6.5:.1f} {cx + 6.5:.1f},{cy:.1f} {cx:.1f},{cy + 6.5:.1f} "
                     f"{cx - 6.5:.1f},{cy:.1f}' class='opus-fill'>{t}</polygon>")
        else:
            cls = "cost-free" if r["flow"] == "free" else "cost-hlsm"
            g.append(f"<rect x='{cx - 5.5:.1f}' y='{cy - 5.5:.1f}' width='11' height='11' class='{cls}'>{t}</rect>")
    return (f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Cost per run against the worst-case csynth cycles of "
            f"its design, mm 1024 cubed, Opus 5.5, log scales'>" + "".join(g) + "</svg>")


COST_CSS = """
.cost-plain{fill:none;stroke:var(--opus);stroke-width:1.8}
.cost-free{fill:var(--target)}
.cost-hlsm{fill:none;stroke:var(--target);stroke-width:2}
.mk{display:inline-block;width:10px;height:10px;margin-right:6px;vertical-align:-1px}
.mk.plain{border:1.8px solid var(--opus);border-radius:50%}
.mk.agent{background:var(--opus);transform:rotate(45deg) scale(.85)}
.mk.free{background:var(--target)}
.mk.hlsm{border:2px solid var(--target)}
"""


def section():
    rs = runs()
    by = lambda f: [r for r in rs if r["flow"] == f]
    sized = ("dspredo", "refneu", "refnf", "refv2", "win7080")
    ag = [r for r in by("agent") if r["rung"] in sized]
    pl = [r for r in by("plain") if r["rung"] in sized]
    fr = [r for r in by("free") if r["tag"] != "mm1024_team_r1"]
    rng = lambda xs: f"{money(min(xs))}&ndash;{money(max(xs))}"
    med_by = lambda g, k: [med([r[k] for r in g if r["rung"] == q]) for q in sized if any(r["rung"] == q for r in g)]
    ag_cost, pl_cost = med_by(ag, "cost"), med_by(pl, "cost")
    ag_wall = med_by(ag, "wall")
    hl = by("hlsm")
    hb = [r for r in hl if r.get("budget")]
    agw = best([r for r in by("agent") if r["rung"] == "win7080"])
    agwin = [r for r in by("agent") if r["rung"] == "win7080" and r.get("worst")]
    hl_note = (f"With HLSmith&rsquo;s 138 skills and DSP window but none of its flow, on the same budget as one agentic "
               f"HLSmith agent step (12 checks, 1 h each), Claude Code&rsquo;s best designs within 80% LUT were "
               f"{' and '.join(ms.fmt_int(r['worst']) for r in sorted(hb, key=lambda r: r['worst']))} cycles, against "
               f"{ms.fmt_int(min(r['worst'] for r in agwin))}&ndash;{ms.fmt_int(max(r['worst'] for r in agwin))} for agentic "
               f"HLSmith on the Refined + window rung: <b>a tie on quality</b>. Its fastest designs "
               f"({ms.fmt_int(min(r['uncapped'][0] for r in hb))}&ndash;{ms.fmt_int(max(r['uncapped'][0] for r in hb))} cycles) "
               f"are over the LUT cap. It costs "
               f"{rng([r['cost'] for r in hb])} a run against {rng([r['cost'] for r in agwin])}, with 10&ndash;12 Vitis "
               f"checks against 2&ndash;4."
               if hb and agw else
               "Experiment C (the free agent with HLSmith&rsquo;s skills and DSP window, on a matched 12-check, 3 h budget) "
               "is not in this table yet.")
    return f"""
<section class='aside' id='cost'>
  <span class='eyebrow'>mm 1024&sup3; &middot; Opus 5.5 &middot; 13&ndash;22 October</span>
  <h2>What one run costs</h2>
  <p>Every Opus run on the matrix multiply, priced from its own call logs: the shim&rsquo;s per-call usage for the c2hls
  runs, and the session transcript for the free agents. Cost is the Claude API list price of the tokens used. Vitis time
  is not in it, but wall time is. <b>Vitis checks by the LLM</b> counts the csim and csynth jobs the model started itself;
  in plain mode it starts none, and the flow runs its own csim and csynth on the reply.</p>
  <figure>
    <div class='legend'><span><span class='mk plain'></span>plain HLSmith</span><span><span class='mk agent'></span>agentic
    HLSmith</span><span><span class='mk free'></span>Claude Code agent mode, no HLSmith flow</span>{"<span><span class='mk hlsm'></span>the same, with HLSmith&rsquo;s skills and DSP window</span>" if hl else ""}</div>
    <div class='figscroll'>{chart(rs)}</div>
    <figcaption>One point per run, both axes log scale; hover a point for the run, its design and its wall time. Runs that
    produced no design are left out (the table counts them). Lower and further left is better.</figcaption>
  </figure>
  <div class='cols2'>
    <div><h3>The agentic flow costs about what a plain run does</h3><p>On the five rungs that size the design, agentic
    HLSmith&rsquo;s median run costs {rng(ag_cost)} and takes {dur(min(ag_wall))}&ndash;{dur(max(ag_wall))}; plain HLSmith on
    the same rungs costs {rng(pl_cost)}. The agent step adds 4&ndash;16 LLM calls and 2&ndash;6 Vitis checks to one run.</p></div>
    <div><h3>Claude Code&rsquo;s own agent mode: 3&ndash;7&times; the cost, the same designs</h3><p>The two free-agent sessions that finished a
    design cost {rng([r['cost'] for r in fr])} each and ran {dur(min(r['wall'] for r in fr))}&ndash;{dur(max(r['wall'] for r in fr))},
    with 81&ndash;153 LLM calls and 14&ndash;22 Vitis checks, and their designs
    ({', '.join(f"{ms.fmt_int(r['worst'])} cycles at {pct(r['lut_pct'])} LUT" for r in fr)}) ignore the DSP window. {hl_note}</p></div>
  </div>
  {table(rs)}
  <p class='note'>Medians over the runs in each row. Free-agent times are active time: two of those sessions waited out a
  usage limit, which is not counted. The best design is the lowest worst-case csynth latency among runs
  whose design passed csim and fits the 80% LUT cap (see the top of the page); a faster design over the cap is shown
  under it in small type. Every design in the table also passes the hardened bench (signed and wide-range inputs, new seed,
  that no run ever saw). All rows are on the fixed bench; the plain rows above zero-shot are experiment A&rsquo;s runs, one or
  two per rung so far. The unbounded HLSmith-knowledge run (r1) used 21 checks in 1.5 h. DeepSeek runs
  are left out because they are not priced here. Claude Code driven by DeepSeek v4.1-flash with the same knowledge
  (experiment D) produced one in-window design in three budgeted sessions (948,196 cycles). The 292,030-cycle design of
  an unbounded session is not a valid result: it quantizes the inputs to 18-bit fixed point, which the original bench&rsquo;s
  [0, 1) inputs let pass, and it fails the hardened bench. Sources: <code>study/exp_hlsmith/cost_table/</code> (<code>rows.json</code>, <code>rows_expA.json</code>, <code>exp_cd.json</code>) and <code>study/bench_hard/verify_hard_b.csv</code>.</p>
</section>
"""

# 2026-10-22: one name for the c2hls flow whose LLM calls are answered by a Claude Code session with tools:
# "HLSmith with an agent backend" (it was "agent mode" / "agentic HLSmith"; HLSmith's own paper uses "agentic" for its
# role-based design). Claude Code without the flow is "Claude Code on its own". Applied to the built page, every section.
RENAME = [
    ("Agentic HLSmith: the same flow, the LLM step in agent mode",
     "HLSmith with an agent backend: the same flow, each LLM call answered by a Claude Code session with tools"),
    ("Every ladder rung was run again in <b>agent mode</b>: Opus 5.5 with tools, running real csim and csynth on its own\n"
     "  kernels (up to 12 checks per run) inside the same c2hls flow, prompts, skill packs and gates.",
     "Every ladder rung was run again as <b>HLSmith with an agent backend</b>: the same c2hls flow, prompts, skill packs and\n"
     "  gates, but each generation and repair call is answered by one Claude Code session (Opus 5.5 with tools; no roles, no\n"
     "  subagents) that decides for itself when to compile, csim and csynth its kernel, up to 12 checks per call, before it\n"
     "  replies. HLSmith&rsquo;s controller then checks the reply as usual, sends its repair prompts, and keeps the best design."),
    ("Plain vs agent mode, rung by rung", "Plain vs agent backend, rung by rung"),
    ("so every column compares plain and agent mode on the same bench", "so every column compares plain and agent backend on the same bench"),
    ("&#9670; agent mode (Opus only)", "&#9670; agent backend (Opus only)"),
    ("plain and agent mode, log scale", "plain and agent backend, log scale"),
    ("At zero-shot, agent mode&rsquo;s best", "At zero-shot, the agent backend&rsquo;s best"),
    ("<h3>Agent mode beats plain on every rung</h3>", "<h3>The agent backend beats plain on every rung</h3>"),
    ("best of each, agent mode&rsquo;s gain over plain mode:", "best of each, the agent backend&rsquo;s gain over plain:"),
    ("agent\n    mode also gets there", "the agent\n    backend also gets there"),
    ("Agent mode, Opus 5.5", "Agent backend, Opus 5.5"),
    ("agent mode, Opus 5.5", "agent backend, Opus 5.5"),
    ("the agentic ladder is Opus only", "the agent-backend column is Opus only"),
    ("HLSmith-Agentic-Refined-Expertise + Resource-Window <span class='alias'>(Agentic ladder L2",
     "HLSmith-Refined-Expertise + Resource-Window, agent backend <span class='alias'>(agent-backend ladder L2"),
    ("ran the same c2hls flow in <b>agent mode</b>", "ran the same c2hls flow with an <b>agent backend</b>"),
    ("<h3>Agent mode</h3>", "<h3>The agent backend</h3>"),
    ("Agentic ladder", "Agent-backend ladder"),
    ("in agent mode", "with an agent backend"),
    ("Agent mode, refined skills", "Agent backend, refined skills"),
    ("Three agent runs", "Three agent-backend runs"),
    ("c2hls agent mode (tools", "c2hls with an agent backend (tools"),
    ("Agentic HLSmith, ", "Agent backend, "),
    ("one agentic HLSmith agent step", "one agent-backend call"),
    ("for agentic HLSmith on the", "for HLSmith with an agent backend on the"),
    ("<span class='mk agent'></span>agentic\n    HLSmith", "<span class='mk agent'></span>HLSmith with an\n    agent backend"),
    ("The agentic flow costs about what a plain run does", "The agent backend costs about what a plain run does"),
    ("agentic\n    HLSmith&rsquo;s median run", "the agent\n    backend&rsquo;s median run"),
    ("The agent step adds", "The agent backend adds"),
    ("Claude Code agent mode, no HLSmith flow", "Claude Code on its own, no HLSmith flow"),
    ("Claude Code&rsquo;s own agent mode", "Claude Code on its own"),
    ("(agentic ablation", "(agent-backend ablation"),
    ("(23 ladder + 3 agentic)", "(23 ladder + 3 agent-backend)"),
    ("The agentic L2 design", "The agent-backend L2 design"),
]


# 2026-10-22: the 80% LUT cap applies to every design on the page (v13.v11.LUT_CAP), stated once, in the header.
HEADER = [
    ("A free-running Claude Code agent with real Vitis feedback went further, to 404K cycles.",
     "A free-running Claude Code agent with real Vitis feedback went further, to 468K cycles."),
    ("<li><a href='#agentfw'>General Claude Code agent</a>: 404K cycles on mm.</li>",
     "<li><a href='#agentfw'>General Claude Code agent</a>: 468K cycles on mm.</li>"),
    ("DeepSeek v4.1-flash is called through its API.</p>",
     "DeepSeek v4.1-flash is called through its API.</p>\n  <p class='meta'><b>LUT cap.</b> A design counts on this page only "
     "if it uses at most 80% of the U280&rsquo;s 1,303,680 LUTs; fuller designs are unlikely to route at 300&nbsp;MHz. No flow "
     "or agent was told about the cap: it is applied afterwards, to every flow alike, in every chart, table and number.</p>\n"
     "  <p class='meta'><b>Which design a run counts.</b> One rule for every flow: a run counts its fastest design that the run "
     "csim&rsquo;d on the fixed bench, synthesized as the byte-identical kernel, fits the LUT cap and the rung&rsquo;s DSP gate, "
     "and passes the hardened bench. With an agent backend this includes the checks the agent session ran itself, so a "
     "run can count a design other than HLSmith&rsquo;s final (marked fallback, or run&rsquo;s best). Kernels a run only "
     "synthesized, or only csim&rsquo;d, were given the missing check afterwards, unchanged, on both sides "
     "(<code>study/posthoc_csim/</code>); designs that count because of it are marked post hoc or post-session. The "
     "matched-budget runs of experiments C and D count only what finished inside their budget.</p>"),
]


def rename(page):
    for a, b in RENAME:
        n = page.count(a)
        assert n, f"rename: {a[:50]!r} not found"
        page = page.replace(a, b)
    return page


def main():
    v13.OUT = OUT
    v13.v12.OUT = OUT
    v13.v11.OUT = OUT
    v13.main()
    page = OUT.read_text()
    marker = "<section class='aside' id='effort'>"
    assert page.count(marker) == 1, "effort anchor"
    page = page.replace(marker, section() + marker, 1)
    li = "<li><a href='#ablation'>"
    assert page.count(li) == 1, "new-list anchor"
    page = page.replace(li, "<li><a href='#cost'>What one run costs</a> (LLM calls, Vitis checks, dollars and wall time "
                            "per run, by flow).</li>\n    " + li, 1)
    page = page.replace("<h3>New in this version (13&ndash;17 October)</h3>", "<h3>New in this version (13&ndash;21 October)</h3>", 1)
    page = page.replace("</style>", COST_CSS + "</style>", 1)
    page = rename(page)
    for a, b in HEADER:
        assert page.count(a) == 1, f"header: {a[:50]!r}"
        page = page.replace(a, b)
    OUT.write_text(page)
    print(OUT)


if __name__ == "__main__":
    main()

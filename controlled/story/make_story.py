#!/usr/bin/env python3
"""Build controlled/story/story.html: the controlled Opus-vs-DeepSeek flash ladder as a story page.

Reads runs_controlled/ (csynth reports, flash attempts, Slurm state) and verify/ (strict csim) at build time,
so rerunning this script after new jobs finish fills in the pending rungs.
"""
import glob, html, json, math, re, subprocess, time
from pathlib import Path

CC = Path("/scratch/asa582/workspaces/c2hls_claude_code")
R = CC / "runs_controlled"
OUT = Path(__file__).resolve().parent / "story.html"
CAP = {"dsp": 9024, "lut": 1303680, "bram": 4032, "uram": 960, "ff": 2607360}   # xcu280 device totals
MACS = 1024 ** 3
MODELS = {"cc": "Opus 5.5", "ds": "DeepSeek v4.1-flash"}

# Failure notes (from the flash attempt logs and the csynth work-dir analysis).
NOTES = {
    "ctl_ds_refv2_r1": "Rejected 3 times by the 90% fill gate: DSP 1,272, 1,164, 1,312",
    "ctl_ds_win7080_r2": "Try 1 below the window (1,120 DSP); tries 2 and 3 in the window (6,888 and 6,560 DSP) "
                         "but memory-starved at 3.0x and 3.7x the floor",
    "ctl_cc_dspredo_r3": "2nd design (1,664-MAC grid, no DATAFLOW, 167K instructions) never finished csynth; timed out at 6 h",
}

# Structure notes checked by hand where the automatic reading is not enough.
STRUCT_OVERRIDE = {
    "ctl_cc_refv2_r2": {"df": "Load phase only (3 parallel loaders)", "pp": "No"},
    "ctl_cc_refv2_r3": {"pp": "Implicit (PIPO arrays between stages)"},
    "ctl_cc_skills_r1": {"pp": "Explicit (B0/B1 tiles, one loop)"},
}


def structure(tag, cpp, work_dir):
    """DATAFLOW from Vitis's own log; streams and ping-pong buffers from the source."""
    src = Path(cpp).read_text()
    code = re.sub(r"//.*", "", re.sub(r"/\*.*?\*/", "", src, flags=re.S))
    procs = []
    log = Path(work_dir or "") / "logs" / "hls_run_tcl.log"
    if work_dir and log.exists():
        procs = re.findall(r"Applying dataflow to function '([^']+)'.*?extracted (\d+) process", log.read_text())
    df = f"Yes ({sum(int(n) for _, n in procs)} stages)" if procs else "No"
    nstream = len(re.findall(r"hls::stream<[^;&()]*>\s+\w+\s*[;(\[]", code))
    st = f"Yes ({nstream})" if nstream else "No"
    two = re.findall(r"\b(\w+)\s*\[\s*2\s*\]\s*\[", code)
    pairs = {m[:-1] for m in re.findall(r"\b([A-Za-z_]+[01])\s*\[", code)}
    pairs = {b for b in pairs if re.search(rf"\b{b}0\s*\[", code) and re.search(rf"\b{b}1\s*\[", code)}
    if two:
        pp = f"Explicit ({two[0]}[2])"
    elif pairs:
        pp = f"Explicit ({sorted(pairs)[0]}0/{sorted(pairs)[0]}1)"
    elif procs and not nstream:
        pp = "Implicit (PIPO)"
    else:
        pp = "No"
    out = {"df": df, "pp": pp, "st": st}
    out.update(STRUCT_OVERRIDE.get(tag, {}))
    return out

BREAKDOWN = {e["tag"]: e for e in json.loads((Path(__file__).resolve().parent / "bottleneck" / "breakdown.json").read_text())} \
    if (Path(__file__).resolve().parent / "bottleneck" / "breakdown.json").exists() else {}


def cycles_cell(tag):
    """Load/compute/store bar plus the bottleneck sentence, from bottleneck/breakdown.json."""
    e = BREAKDOWN.get(tag)
    if not e:
        return "<td class='where'><span class='none'>not analysed yet</span></td>"
    ph, tot = e["phases"], float(e["worst_cycles"])
    if e.get("overlapped"):
        stages = sorted(e.get("dataflow_stages") or [], key=lambda x: -x["cycles"])
        bars = "".join(f"<div class='stage'><span class='sname'>{html.escape(x['name'])}</span>"
                       f"<span class='sbar'><i style='width:{min(100, 100 * x['cycles'] / tot):.1f}%'></i></span>"
                       f"<span class='sval'>{fmt_cycles(x['cycles'])}</span></div>" for x in stages)
        viz = f"<div class='stages' title='DATAFLOW stages run at the same time; the longest sets the latency'>{bars}</div>"
    else:
        seg = [("load", ph["load_A"] + ph["load_B"] + ph["load_C"]), ("compute", ph["compute"]),
               ("store", ph["store_C"]), ("other", ph["other"])]
        viz = "<div class='pbar'>" + "".join(
            f"<i class='{k}' style='width:{100 * v / tot:.1f}%' title='{k}: {fmt_int(v)} cycles ({100 * v / tot:.0f}%)'></i>"
            for k, v in seg if v > 0) + "</div>"
        viz += ("<div class='plab'>" + " &middot; ".join(f"{k} {100 * v / tot:.0f}%" for k, v in seg[:3] if v > 0) + "</div>")
    return f"<td class='where'>{viz}<div class='bn'>{html.escape(e['bottleneck'])}</div></td>"


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def collect():
    jobs = dict(l.split()[:2] for l in (R / "jobs.txt").read_text().splitlines() if len(l.split()) >= 2)
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(jobs.values()), "-o", "JobID,State,Elapsed"],
                         capture_output=True, text=True).stdout
    st = {l.split("|")[0]: (l.split("|")[1].split()[0], l.split("|")[2]) for l in out.splitlines() if "|" in l}
    rows = {}
    for tag, j in jobs.items():
        m = re.match(r"ctl_(cc|ds)_(\w+?)_r(\d)$", tag)
        if not m:
            continue
        state, elapsed = st.get(j, ("?", ""))
        rec = {"tag": tag, "model": m[1], "variant": m[2], "rep": int(m[3]), "state": state, "elapsed": elapsed, "ok": False}
        cells = sorted(glob.glob(f"{R}/*_{tag}_*/autosa_mm"))
        if cells:
            c = cells[-1]
            ms = load(c + "/autosa_mm_multistep_results.json") or {}
            f = next((s for s in ms.get("steps", []) if s.get("step_name") == "flash"), {})
            rec["ok"] = bool(f.get("success"))
            if rec["ok"]:
                r = load(c + "/autosa_mm_selected_report.json") or {}
                r = r.get("report", r)
                rec.update({k: r.get(k) for k in CAP})
                rec["cycles"] = max(r.get("latency_cycles") or 0, r.get("latency_cycles_worst") or 0)
                rec["best"] = min(x for x in (r.get("latency_cycles"), r.get("latency_cycles_worst")) if x)
                rec["struct"] = structure(tag, c + "/autosa_mm_selected.cpp", r.get("work_dir"))
            cs = CC / "verify" / tag / "csim_summary.txt"
            if cs.exists():
                rec["csim"] = "Passed" if "correct=True" in cs.read_text() else "Failed"
        rows[tag] = rec
    return rows


def fmt_int(n):
    return f"{int(n):,}"


def fmt_cycles(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M" if n < 1e7 else f"{n / 1e6:.1f}M"
    return f"{n / 1e3:.0f}K"


def res(v, key):
    if v is None:
        return "<td class='num'></td>"
    pct = 100.0 * v / CAP[key]
    p = "&lt;0.1%" if 0 < pct < 0.1 else f"{pct:.1f}%"
    return f"<td class='num'>{fmt_int(v)} <span class='pct'>({p})</span></td>"


def outcome(r):
    if r["ok"]:
        return "<span class='chip ok'>Pass</span>"
    s = r["state"]
    if s in ("PENDING",):
        return "<span class='chip wait'>Queued</span>"
    if s in ("RUNNING",):
        return f"<span class='chip wait'>Running</span> <span class='why'>{html.escape(r['elapsed'])} so far</span>"
    note = NOTES.get(r["tag"])
    if not note and s == "TIMEOUT":
        h = int(r["elapsed"].split(":")[0]) if r["elapsed"] else 0
        note = f"Design never finished csynth; timed out at {h} h"
    return f"<span class='chip bad'>Fail</span> <span class='why'>{html.escape(note or s)}</span>"


def table(rows, variant):
    body = []
    for m in ("cc", "ds"):
        for rep in (1, 2, 3):
            r = rows.get(f"ctl_{m}_{variant}_r{rep}")
            if not r:
                body.append(f"<tr class='m-{m}'><th scope='row'><span class='dot {m}'></span>{MODELS[m]} r{rep}</th>"
                            f"<td><span class='chip wait'>Not run yet</span></td>" + "<td class='num'></td>" * 13 + "</tr>")
                continue
            if r["ok"]:
                cyc = r["cycles"]
                floor = MACS / (r["dsp"] / 5.0)
                cells = (f"<td class='num strong'>{fmt_int(cyc)}</td><td class='num'>{fmt_int(r['best'])}</td>"
                         f"<td class='num'>{cyc / floor:.2f}&times;</td>"
                         + res(r["dsp"], "dsp") + res(r["lut"], "lut") + res(r["bram"], "bram")
                         + res(r["uram"], "uram") + res(r["ff"], "ff")
                         + f"<td>{r.get('csim') or 'pending'}</td>"
                         + "".join(f"<td class='struct'>{html.escape(r['struct'][k])}</td>" for k in ("df", "pp", "st"))
                         + cycles_cell(r["tag"]))
            else:
                cells = "<td class='num'></td>" * 13
            body.append(f"<tr class='m-{m}'><th scope='row'><span class='dot {m}'></span>{MODELS[m]} r{rep}</th>"
                        f"<td>{outcome(r)}</td>{cells}</tr>")
    return ("<div class='tablewrap'><table><thead><tr><th scope='col'>Run</th><th scope='col'>Outcome</th>"
            "<th scope='col' class='num'>Worst-case cycles</th><th scope='col' class='num'>Best-case cycles</th>"
            "<th scope='col' class='num'>&times; floor</th><th scope='col' class='num'>DSP</th><th scope='col' class='num'>LUT</th><th scope='col' class='num'>BRAM_18K</th>"
            "<th scope='col' class='num'>URAM</th><th scope='col' class='num'>FF</th><th scope='col'>csim</th>"
            "<th scope='col'>DATAFLOW</th><th scope='col'>Ping-pong</th><th scope='col'>Streams</th>"
            "<th scope='col'>Where the cycles go</th></tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def tally(rows, variant):
    out = []
    for m in ("cc", "ds"):
        rs = [rows.get(f"ctl_{m}_{variant}_r{i}") for i in (1, 2, 3)]
        rs = [r for r in rs if r]
        done = [r for r in rs if r["ok"] or r["state"] not in ("RUNNING", "PENDING")]
        ok = [r for r in rs if r["ok"]]
        best = min((r["cycles"] for r in ok), default=None)
        txt = f"{len(ok)}/{len(done)} pass" if done else "pending"
        if len(done) < len(rs) and done:
            txt += f", {len(rs) - len(done)} running"
        if best:
            txt += f" &middot; best {fmt_cycles(best)}"
        out.append(f"<div class='tally {m}'><span class='dot {m}'></span><b>{MODELS[m]}</b> {txt}</div>")
    return "<div class='tallies'>" + "".join(out) + "</div>"


RUNGS = [
    ("zs", "Zero-shot",
     "One reply to a bare prompt. No Phase B, no skills, no DSP target.",
     "Every kernel is correct, so the baseline measures competence, not speed. Opus sizes its first guess at 80 to "
     "320 DSP. Two of DeepSeek's three designs use 3 DSP, a single multiply-add, and need 4.3 billion cycles."),
    ("noskills", "HLSmith Base <span class='alias'>(Flash, no skills)</span>",
     "The full flash prompt and its fix loop (up to 3 turns), starting from the shared baseline kernel. No skills.",
     "Every run passed on its first reply, so the fix loop never ran. The gap narrows: DeepSeek's best design (71M "
     "cycles) matches Opus's best (69M), but its other two use 6 and 18 DSP."),
    ("skills", "HLSmith-SingleStep-Expertise <span class='alias'>(Flash + skills)</span>",
     "Adds Expertise: the 90-skill pack with the gemm-flatten v1 entries. Still nothing asks for a DSP count.",
     "Skills help both models, and Opus reaches 19M cycles at 320 DSP. Every design runs within 1.1 to 2.2 times its "
     "own compute floor: the kernels are efficient, just small. One million cycles needs about 5,400 DSP, and "
     "nothing at this rung asks for it."),
    ("dspredo", "HLSmith-SingleStep-Expertise + Resource-Requirement <span class='alias'>(Skills + DSP floor)</span>",
     "Adds a Resource-Requirement, c2hls's DSP redo: fill at least 90% of the 9,024 DSP (unless another resource reaches 90%), with a floor "
     "of 300, a cap of 9,024, and prompt text telling the model to fill the chip. No refinement.",
     "This is where the models split. Opus fills the chip in two of three runs, at 0.90M and 2.85M cycles. The slower "
     "one is memory-starved at 4.4 times its floor, because nothing checks data movement yet. Opus's third design, "
     "a 1,664-MAC grid with no DATAFLOW, never finished csynth. DeepSeek's runs also stall in csynth for the full "
     "6 hours."),
    ("refneu", "HLSmith-SingleStep-Refined-Expertise <span class='alias'>(Refined skills)</span>",
     "Swaps Expertise for Refined-Expertise: the hold-back cues are removed from the prompts and skills, and the balance "
     "gate is added (see &ldquo;What refined means&rdquo; below). Nothing asks for DSP, in words or by a gate.",
     "Submitted on 13 October with a 1-hour limit; results pending."),
    ("refnf", "HLSmith-SingleStep-Refined-Expertise + Resource-Requirement in words <span class='alias'>(Refined skills, DSP asked for in the prompt, no gate)</span>",
     "The original refined wording, which also tells the model to size the datapath to the device and use most of the "
     "DSP, and two skills that say the same. The requirement is only in words: no floor or fill gate checks it.",
     "Asked in words but never checked, Opus lands on exactly 5,120 DSP in all three runs: 1,024 multiply-adds, the "
     "same as AutoSA's 128 &times; 8 array, at 1.25M to 1.73M cycles. DeepSeek reads the same words and stays at 768 to "
     "864 DSP, at 4.6M to 6.5M cycles. All three pairs got byte-identical first prompts."),
    ("refv2", "HLSmith-SingleStep-Refined-Expertise + Resource-Requirement <span class='alias'>(Refined skills + DSP floor)</span>",
     "Refined-Expertise plus the same Resource-Requirement as rung 4 (fill at least 90% of the DSP).",
     "Opus passes all three, at 0.69M to 0.85M cycles and 1.07 to 1.32 times its compute floor. The balance gate "
     "closes the bandwidth gap left at the DSP-floor rung. DeepSeek passes none: one run stayed at 13 to 15% DSP "
     "across three tries, and two built designs Vitis could not schedule in 6 hours."),
    ("win7080", "HLSmith-SingleStep-Refined-Expertise + Resource-Window <span class='alias'>(Refined skills + DSP floor and cap, 70&ndash;80%)</span>",
     "Resource-Window replaces the 90% fill: DSP must land between 6,317 and 7,219 (70 to 80%), a floor plus a cap "
     "that leaves headroom for place and route.",
     "Opus passes all three at 0.87M to 1.06M cycles, about 20% slower than the rung above with about 20% fewer "
     "DSP. DeepSeek reaches the window once (6,888 and 6,560 DSP), but the balance gate rejects both designs as "
     "memory-starved, at 3.0 and 3.7 times their floor."),
]


REFINED_BOX = """<div class='explain'>
  <h3>What &ldquo;refined&rdquo; means: every change, verbatim</h3>
  <p>Refined-Expertise changes the flash step in four places: the system prompt, the user prompt, the resource block the
  model sees, and the skill pack. It also adds one check after csynth, the balance gate. This step uses the neutral
  version, which never says how much of the chip to use.</p>
  __REFINED_CHANGES__
  <p class='meta'>The next step uses the original refined wording instead, which adds &ldquo;size the datapath to the
  device&rdquo;, &ldquo;use most of the device's DSP budget&rdquo; and &ldquo;a small datapath on a large device is a
  performance bug&rdquo;, plus two skills that say the same.</p>
</div>"""


def chart(rows, mode):
    key = 'cycles' if mode == 'worst' else 'best'
    W, H, L, Rm, T, B = 1260, 500, 96, 16, 80, 416
    cols = [v for v, *_ in RUNGS] + ["agent"]
    cw = (W - L - Rm) / len(cols)
    y = lambda c: B - (math.log10(c) - 5) * (B - T) / 5.0
    g = []
    for d, lab in zip(range(5, 11), ["100K", "1M", "10M", "100M", "1B", "10B"]):
        yy = y(10 ** d)
        g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{yy:.1f}' y2='{yy:.1f}' class='grid'/>"
                 f"<text x='{L - 8}' y='{yy + 4:.1f}' class='axis' text-anchor='end'>{lab}</text>")
    ya = y(1.054811e6 if mode == "worst" else 1.048845e6)
    g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{ya:.1f}' y2='{ya:.1f}' class='autosa'/>")
    g.append(f"<text x='{L + 6}' y='{ya - 8:.1f}' class='bandlab' text-anchor='start'>AutoSA st0_c6, {mode} case: "
             f"{'1,054,811' if mode == 'worst' else '1,048,845'}</text>")
    g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{y(1e6):.1f}' y2='{y(1e6):.1f}' class='target'/>")
    g.append(f"<text x='{L + 6}' y='{y(1e6) + 15:.1f}' class='targetlab'>1M cycles</text>")
    g.append(f"<text x='{L - 8}' y='{T - 38:.1f}' class='axis' text-anchor='end'>no design</text>")
    g.append(f"<text transform='translate(20,{(T + B) / 2:.1f}) rotate(-90)' class='ytitle' text-anchor='middle'>"
             f"csynth latency, {mode} case (cycles, log scale)</text>")
    g.append(f"<line x1='{L}' x2='{W - Rm}' y1='{T - 22:.1f}' y2='{T - 22:.1f}' class='grid'/>")
    names = ["Zero-shot", "Base", "Expertise", "Expertise +", "Refined", "Refined +", "Refined +", "Refined +", "Agentic"]
    names2 = ["", "no skills", "", "Res.-Req.", "Expertise", "Req. in words", "Res.-Req.", "Res.-Window", "ladder L2"]
    for i, v in enumerate(cols):
        cx = L + cw * (i + 0.5)
        if i:
            g.append(f"<line x1='{L + cw * i:.1f}' x2='{L + cw * i:.1f}' y1='{T - 56}' y2='{B}' class='{'sep' if v == 'agent' else 'colrule'}'/>")
        g.append(f"<text x='{cx:.1f}' y='{B + 26}' class='collab' text-anchor='middle'>{names[i]}</text>"
                 f"<text x='{cx:.1f}' y='{B + 46}' class='collab' text-anchor='middle'>{names2[i]}</text>")
        if v == "agent":
            yy = y(1066083)
            g.append(f"<path d='M{cx:.1f},{yy - 9:.1f} L{cx + 3:.1f},{yy - 3:.1f} L{cx + 9:.1f},{yy - 2:.1f} L{cx + 4:.1f},{yy + 2:.1f} "
                     f"L{cx + 6:.1f},{yy + 8:.1f} L{cx:.1f},{yy + 5:.1f} L{cx - 6:.1f},{yy + 8:.1f} L{cx - 4:.1f},{yy + 2:.1f} "
                     f"L{cx - 9:.1f},{yy - 2:.1f} L{cx - 3:.1f},{yy - 3:.1f} Z' class='opus-fill'/>")
            g.append(f"<text x='{cx:.1f}' y='{yy - 14:.1f}' class='ptlab' text-anchor='middle'>5,152 DSP</text>")
            continue
        pending = True
        for m, offs in (("cc", (-24, -16, -8)), ("ds", (8, 16, 24))):
            for rep, dx in zip((1, 2, 3), offs):
                r = rows.get(f"ctl_{m}_{v}_r{rep}")
                x = cx + dx
                if not r:
                    continue
                if r["ok"]:
                    pending = False
                    g.append(f"<circle cx='{x:.1f}' cy='{y(r[key]):.1f}' r='5' class='{m}-fill'>"
                             f"<title>{MODELS[m]} r{rep}: {fmt_int(r[key])} cycles ({mode} case), {fmt_int(r['dsp'])} DSP</title></circle>")
                elif r["state"] in ("RUNNING", "PENDING"):
                    g.append(f"<circle cx='{x:.1f}' cy='{T - 38:.1f}' r='4' class='{m}-open'><title>{MODELS[m]} r{rep}: running</title></circle>")
                else:
                    pending = False
                    yy = T - 38
                    g.append(f"<g class='{m}-stroke'><title>{MODELS[m]} r{rep}: no passing design</title>"
                             f"<line x1='{x - 4:.1f}' y1='{yy - 4:.1f}' x2='{x + 4:.1f}' y2='{yy + 4:.1f}'/>"
                             f"<line x1='{x - 4:.1f}' y1='{yy + 4:.1f}' x2='{x + 4:.1f}' y2='{yy - 4:.1f}'/></g>")
        if pending:
            g.append(f"<text x='{cx:.1f}' y='{(T + B) / 2:.1f}' class='pend' text-anchor='middle'>results</text>"
                     f"<text x='{cx:.1f}' y='{(T + B) / 2 + 15:.1f}' class='pend' text-anchor='middle'>pending</text>")
    return (f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='csynth cycles per rung, log scale, Opus and DeepSeek runs'>"
            + "".join(g) + "</svg>")


CSS = """
/* Layout: one vertical ladder. A thesis and the full-ladder chart up top, then one rung per section:
   what changed, what happened, the evidence table. */
:root{
  --bg:#f4f6f8; --surface:#ffffff; --ink:#15202b; --muted:#5a6573; --rule:#d9dfe6;
  --opus:#0b7a6e; --ds:#5a49c2; --target:#a8551b; --band:rgba(168,85,27,.10);
  --ok:#1d7a3a; --bad:#b42318; --wait:#7a6a1d; --old-bg:rgba(180,35,24,.07); --new-bg:rgba(29,122,58,.08); --c-load:#c47b2a; --c-compute:#2f6db3; --c-store:#8a5bb0;
  --display:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;
  --body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0e1319; --surface:#151c25; --ink:#e5eaf0; --muted:#97a2b1; --rule:#2a3441;
  --opus:#39c0ae; --ds:#9d8ff3; --target:#e3a15d; --band:rgba(227,161,93,.13);
  --ok:#5cc47e; --bad:#f2786f; --wait:#d8c46a; --old-bg:rgba(242,120,111,.10); --new-bg:rgba(92,196,126,.10); --c-load:#e0a25e; --c-compute:#6fa6e8; --c-store:#b892d8; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0e1319; --surface:#151c25; --ink:#e5eaf0; --muted:#97a2b1; --rule:#2a3441;
  --opus:#39c0ae; --ds:#9d8ff3; --target:#e3a15d; --band:rgba(227,161,93,.13);
  --ok:#5cc47e; --bad:#f2786f; --wait:#d8c46a; --old-bg:rgba(242,120,111,.10); --new-bg:rgba(92,196,126,.10); --c-load:#e0a25e; --c-compute:#6fa6e8; --c-store:#b892d8; color-scheme:dark}
html{font-size:20px}
body{background:var(--bg);color:var(--ink);font:19px/1.6 var(--body);margin:0}
.page{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:40px 64px;display:flex;flex-direction:column;gap:44px}
h1,h2,h3{font-family:var(--display);text-wrap:balance;margin:0;line-height:1.15}
h1{font-size:clamp(2rem,4.5vw,3rem);font-weight:600;letter-spacing:-.01em}
h2{font-size:1.55rem;font-weight:600;overflow-wrap:anywhere}
h3{font-size:1.05rem;font-weight:600}
p{margin:0;max-width:68ch}
.eyebrow{font:500 .75rem/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
header{display:flex;flex-direction:column;gap:14px}
.thesis{font-size:1.2rem;line-height:1.5;max-width:60ch}
.meta{color:var(--muted);font-size:.9rem}
.legend{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:.88rem;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:baseline}
.dot.cc{background:var(--opus)} .dot.ds{background:var(--ds)}
figure{margin:0;background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:18px 14px 8px;display:flex;flex-direction:column;gap:10px}
figure svg{width:100%;height:auto;display:block;min-width:760px}
.figscroll{overflow-x:auto}
figcaption{font-size:.85rem;color:var(--muted);padding-inline:6px}
svg text{font-family:var(--mono);font-size:15px}
.grid{stroke:var(--rule);stroke-width:1}
.colrule{stroke:var(--rule);stroke-width:1;stroke-dasharray:2 4}
.sep{stroke:var(--muted);stroke-width:1}
.axis,.collab{fill:var(--muted)}
.ytitle{fill:var(--muted);font-family:var(--body);font-size:15.5px}
.collab{font-family:var(--body);font-size:15.5px}
.band{fill:var(--band)}
.bandlab,.targetlab{fill:var(--target)}
.target{stroke:var(--target);stroke-width:1.5;stroke-dasharray:6 4}
.autosa{stroke:var(--target);stroke-width:1.5;opacity:.75}
.cc-fill,.opus-fill{fill:var(--opus)} .ds-fill{fill:var(--ds)}
.cc-open{fill:none;stroke:var(--opus);stroke-width:1.5;stroke-dasharray:2 2}
.ds-open{fill:none;stroke:var(--ds);stroke-width:1.5;stroke-dasharray:2 2}
.cc-stroke line{stroke:var(--opus);stroke-width:2} .ds-stroke line{stroke:var(--ds);stroke-width:2}
.ptlab{fill:var(--opus);font-size:14.5px}
.pend{fill:var(--muted);font-family:var(--body);font-size:16px;font-style:italic}
.primer{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}
.primer div{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:14px 16px;display:flex;flex-direction:column;gap:6px;min-width:0}
.primer code,.formula{font-family:var(--mono);font-size:.85rem}
.formula{background:var(--bg);border-radius:4px;padding:6px 8px;overflow-x:auto}
.rung{display:grid;grid-template-columns:56px minmax(0,1fr);gap:4px 18px;border-top:1px solid var(--rule);padding-top:26px}
.rung .num-badge{font:600 1.9rem/1 var(--display);color:var(--muted);grid-row:span 2}
.rung .body{display:flex;flex-direction:column;gap:12px;min-width:0}
.rung.pending .num-badge{opacity:.55}
.changed{font-size:.92rem;color:var(--muted)}
.alias{display:block;font:500 1rem/1.3 var(--body);color:var(--muted);margin-top:4px}
.explain{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:14px 18px;display:flex;flex-direction:column;gap:8px}
.explain ul{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:6px;max-width:74ch}
.explain code{font-family:var(--mono);font-size:.85em}
.explain details{border-top:1px solid var(--rule);padding-top:10px}
.explain summary{cursor:pointer;font-weight:600;font-family:var(--display);font-size:1.02rem;padding-block:4px}
.explain summary:focus-visible{outline:2px solid var(--opus);outline-offset:2px}
table.diff{table-layout:fixed;width:100%;min-width:720px}
table.diff th:first-child,table.diff td.kind{width:9.5em}
table.diff td{white-space:normal;vertical-align:top}
td.kind{font:500 .74rem/1.35 var(--mono);color:var(--muted);text-transform:none;word-break:break-word}
pre.verb{margin:0;white-space:pre-wrap;word-break:break-word;font:400 .78rem/1.5 var(--mono);padding:8px 10px;border-radius:4px;background:var(--bg)}
pre.verb.old{background:var(--old-bg)}
pre.verb.new{background:var(--new-bg)}
.none{color:var(--muted);font-style:italic}
td.where{white-space:normal;min-width:300px;max-width:420px}
.pbar{display:flex;height:12px;border-radius:3px;overflow:hidden;background:var(--rule)}
.pbar i{display:block;height:100%}
.pbar i.load,.key i.load{background:var(--c-load)} .pbar i.compute,.key i.compute{background:var(--c-compute)}
.pbar i.store,.key i.store{background:var(--c-store)} .pbar i.other,.key i.other{background:var(--muted);opacity:.5}
.plab{font:500 .72rem/1.4 var(--mono);color:var(--muted);margin-top:3px}
.stages{display:flex;flex-direction:column;gap:3px}
.stage{display:grid;grid-template-columns:7.5em 1fr 3.6em;gap:6px;align-items:center;font:500 .72rem/1.2 var(--mono);color:var(--muted)}
.sname{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sbar{height:8px;background:var(--rule);border-radius:2px;overflow:hidden}
.sbar i{display:block;height:100%;background:var(--c-compute)}
.sval{text-align:right}
.bn{font-size:.8rem;line-height:1.45;margin-top:6px}
.key{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.85rem;color:var(--muted)}
.key span{display:inline-flex;align-items:center;gap:6px}
.key i{display:inline-block;width:14px;height:10px;border-radius:2px}
.changed b{color:var(--ink);font-weight:600}
.tallies{display:flex;flex-wrap:wrap;gap:8px 22px;font-size:.92rem}
.tally{display:inline-flex;align-items:center}
.tally b{margin-right:6px;font-weight:600}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.84rem;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap;vertical-align:top}
thead th{font:600 .72rem/1.2 var(--mono);letter-spacing:.04em;text-transform:uppercase;color:var(--muted);background:var(--bg)}
tbody th{font-weight:500}
tr.m-ds:first-of-type th, tr.m-ds th{}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
tr.m-cc + tr.m-ds th, tr.m-cc + tr.m-ds td{border-top:2px solid var(--rule)}
td.num,th.num{text-align:right;font-family:var(--mono)}
td.strong{font-weight:600}
.pct{color:var(--muted);font-size:.92em}
td.struct{white-space:normal;min-width:120px}
.why{color:var(--muted);white-space:normal;display:inline-block;max-width:34ch;vertical-align:top;font-size:.92em}
td:nth-child(2){white-space:normal;min-width:150px}
.chip{display:inline-block;font:600 .7rem/1 var(--mono);letter-spacing:.04em;text-transform:uppercase;padding:4px 6px;border-radius:3px;border:1px solid currentColor;margin-right:4px}
.chip.ok{color:var(--ok)} .chip.bad{color:var(--bad)} .chip.wait{color:var(--wait)}
.aside{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:18px 20px;display:flex;flex-direction:column;gap:12px}
.aside ul,.aside ol{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:6px;max-width:72ch}
.steps{counter-reset:s}
.cols2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.cols2>div{min-width:0;display:flex;flex-direction:column;gap:8px}
footer{color:var(--muted);font-size:.82rem;border-top:1px solid var(--rule);padding-top:16px;display:flex;flex-direction:column;gap:4px}
footer code{font-family:var(--mono)}
@media (max-width:560px){.rung{grid-template-columns:1fr}.rung .num-badge{grid-row:auto}}
"""


def agent_struct(stage):
    c = glob.glob(str(CC / f"runs/ladder_agent_r3_*/skills_{stage}_*/autosa_mm"))[0]
    r = load(c + "/autosa_mm_selected_report.json") or {}
    r = r.get("report", r)
    st = structure(stage, c + "/autosa_mm_selected.cpp", r.get("work_dir"))
    return {k: st[k] for k in ("df", "pp", "st")}


def agent_table():
    stages = [("L1", "about 25% (1,805&ndash;2,707)", 2114147, 2592, 348220, 1175, 0, 420178),
              ("L2", "about 2&times; L1 (4,666&ndash;5,702)", 1066083, 5152, 642721, 1205, 0, 815483),
              ("L3", "70&ndash;80% (6,317&ndash;7,219)", 803619, 7073, 840932, 1205, 0, 1109875)]
    body = ""
    for s, win, cyc, dsp, lut, bram, uram, ff in stages:
        floor = MACS / (dsp / 5.0)
        cls = " class='hl'" if s == "L2" else ""
        body += (f"<tr{cls}><th scope='row'>{s}</th><td>{win}</td><td class='num strong'>{fmt_int(cyc)}</td><td class='num'>{fmt_int(cyc)}</td>"
                 f"<td class='num'>{cyc / floor:.2f}&times;</td>" + res(dsp, "dsp") + res(lut, "lut") + res(bram, "bram")
                 + res(uram, "uram") + res(ff, "ff") + "<td>Passed</td>"
                 + "".join(f"<td class='struct'>{html.escape(v)}</td>" for v in agent_struct(s).values())
                 + cycles_cell("agent_" + s) + "</tr>")
    return ("<div class='tablewrap'><table><thead><tr><th scope='col'>Stage</th><th scope='col'>DSP window</th>"
            "<th scope='col' class='num'>Worst-case cycles</th><th scope='col' class='num'>Best-case cycles</th><th scope='col' class='num'>&times; floor</th><th scope='col' class='num'>DSP</th>"
            "<th scope='col' class='num'>LUT</th><th scope='col' class='num'>BRAM_18K</th><th scope='col' class='num'>URAM</th>"
            "<th scope='col' class='num'>FF</th><th scope='col'>csim</th><th scope='col'>DATAFLOW</th><th scope='col'>Ping-pong</th>"
            "<th scope='col'>Streams</th><th scope='col'>Where the cycles go</th></tr></thead><tbody>" + body + "</tbody></table></div>")


def main():
    rows = collect()
    rungs = []
    for i, (v, name, changed, found) in enumerate(RUNGS, 1):
        has = any(rows.get(f"ctl_{m}_{v}_r{r}", {}).get("ok") or
                  rows.get(f"ctl_{m}_{v}_r{r}", {}).get("state") not in (None, "RUNNING", "PENDING")
                  for m in ("cc", "ds") for r in (1, 2, 3))
        rungs.append(f"""
<section class='rung{"" if has else " pending"}' id='rung-{v}'>
  <div class='num-badge'>{i}</div>
  <div class='body'>
    <h2>{name}</h2>
    <p class='changed'><b>What changes:</b> {changed}</p>
    <p>{found}</p>
    {REFINED_BOX.replace("__REFINED_CHANGES__", (Path(__file__).resolve().parent / "refined_changes.html").read_text()) if v == "refneu" else ""}
    {tally(rows, v)}
    {table(rows, v)}
  </div>
</section>""")
    stamp = time.strftime("%d %b %Y, %H:%M %Z")
    page = f"""<title>The 1M-Cycle Ladder</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
<style>{CSS}</style>
<main class='page'>
<header>
  <span class='eyebrow'>autosa_mm &middot; 1024&sup3; fp32 &middot; Alveo U280 (xcu280) &middot; 3.33 ns</span>
  <h1>The 1M-Cycle Ladder</h1>
  <p class='thesis'>Without a resource requirement, both models write correct but small matrix-multiply kernels. Once the flow asks
  them to fill the chip, Opus reaches about one million cycles, and DeepSeek has not yet produced a design that gets through.</p>
  <p class='meta'>Same c2hls flash flow, same prompts byte for byte, three runs per model per rung. Opus 5.5 is the model inside
  Claude Code, run here with no tools; DeepSeek v4.1-flash is called through its API.</p>
  <div class='legend'><span><span class='dot cc'></span>Opus 5.5</span><span><span class='dot ds'></span>DeepSeek v4.1-flash</span>
  <span>&times; in the top lane: run with no passing design</span><span>dashed circle: still running</span></div>
</header>

<figure>
  <h3>Worst-case latency</h3>
  <div class='figscroll'>{chart(rows, "worst")}</div>
  <figcaption>Worst-case csynth latency of the selected design for every run, log scale, against AutoSA st0_c6's worst case (1,054,811).</figcaption>
</figure>

<figure>
  <h3>Best-case latency</h3>
  <div class='figscroll'>{chart(rows, "best")}</div>
  <figcaption>Best-case csynth latency, log scale, against AutoSA st0_c6's best case (1,048,845). For most designs here, LLM and AutoSA alike, best and worst case are equal or close. Each rung adds one thing to the one before it.
  The last column is a separate flow, Claude Code with tools, explained at the end.</figcaption>
</figure>

<section class='primer' aria-label='How to read the numbers'>
  <div><h3>Compute floor</h3><p>The fewest cycles a design could take with its DSP count, if every multiply-add unit
  started a new operation each cycle.</p><div class='formula'>floor = 1024&sup3; / (DSP / 5)</div><p class='meta'>fp32
  multiply-add &asymp; 5 DSP. &ldquo;&times; floor&rdquo; = cycles / floor; 1.0&times; is perfect.</p></div>
  <div><h3>Reaching 1M cycles</h3><p>Needs about 1,070 multiply-adds per cycle, which is about 5,400 DSP. AutoSA's
  128 PE &times; SIMD 8 systolic array (config st0_c6, array partition 512 &times; 512 &times; 8) has 1,024: its csynth latency is 1,048,845 cycles best case and 1,054,811 worst case.</p></div>
  <div><h3>What the numbers are</h3><p>Cycles and resources are Vitis HLS csynth estimates. Percentages are of the whole
  U280: 9,024 DSP, 1,303,680 LUT, 4,032 BRAM_18K, 960 URAM, 2,607,360 FF. &ldquo;csim&rdquo; is a strict check of every
  output at 1024&sup3;.</p></div>
</section>

{''.join(rungs)}

<section class='aside' id='where-cycles'>
  <h2>Where the cycles go</h2>
  <p>Adding DSP only shortens the compute phase. Each design's csynth latency was split into loading A, B and C, computing,
  and storing C, using every loop's csynth latency times the number of times it runs; the parts add back up to the reported
  latency within 0.4% for all 35 designs. Three things keep latency from falling as DSP grows.</p>
  <div class='key'><span><i class='load'></i>load A, B, C</span><span><i class='compute'></i>compute</span>
  <span><i class='store'></i>store C</span><span><i class='other'></i>other</span>
  <span>DATAFLOW designs show each stage instead; stages run at the same time and the longest one sets the latency.</span></div>
  <ol>
    <li><b>Loads and stores that never overlap compute.</b> Opus's filled refined designs (8,192 to 8,326 DSP) spend 71 to
    77% of their time computing at about 2,000 multiply-adds per cycle, but the other 23 to 29% is B preload and per-tile
    load and store, done one after another. Doubling DSP would halve only the compute part. The refined designs with the
    requirement in words (5,120 DSP) compute at exactly 1,024 multiply-adds per cycle, their floor, for 80 to 84% of the
    time; serial loads are the rest.</li>
    <li><b>Memory loops that move one float per cycle.</b> A 512-bit port can carry 16 floats per cycle, but a loop that
    reads a plain <code>float</code> array runs at II=16. In Opus's DSP-floor r1 (8,320 DSP, 2.85M cycles), loading and
    storing C this way takes 75% of the latency; compute is 23%. In window r1, DATAFLOW overlaps everything, but the A
    loader runs at II=16 and sets the pace at 1.06M while compute needs only 0.79M.</li>
    <li><b>Compute loops that cannot start every cycle.</b> Adding into the same fp32 accumulator every cycle forces
    II=4 or more, so DSPs sit idle. DeepSeek's refined designs unroll about 1,024 multiply-adds, but the loop runs at II=4,
    so they finish only about 255 per cycle and three quarters of that hardware waits at any moment. Its zero-shot
    designs run one serial add chain at II=64: a quarter of a multiply-add per cycle.</li>
  </ol>
  <p>The agentic L2 design avoids all three: five DATAFLOW stages with a ping-pong B buffer, and its compute stage is
  98.4% of the latency at exactly 1,024 multiply-adds per cycle. That is why it lands next to AutoSA st0_c6.</p>
  <p class='meta'>Two worst-case numbers are pessimistic. Refined-in-words r3 charges its A load on all 64 tiles but runs it
  on 8 (true latency about 1.27M, not 1.73M); refined + floor r2 counts its one-time B load twice (about 820K, not 852K).
  Separately, the testbench checks C = A&middot;B<sup>T</sup> and never sets C beforehand, so a design may either read C and
  add to it (as <code>plain.cpp</code> is written) or overwrite it. Eight DeepSeek designs overwrite C and so skip the C
  load that Opus's designs pay for; both pass strict csim.</p>
</section>

<section class='aside' id='why-deepseek'>
  <h2>Why DeepSeek's large designs never finish</h2>
  <div class='cols2'>
    <div><h3>DeepSeek's stalled designs (9 of them)</h3><ul>
      <li>One flat function, no DATAFLOW.</li>
      <li>One pipelined loop fully unrolled into 5,600 to 9,000 fp32 multiply-adds, every accumulator its own register.</li>
      <li>Vitis reports 23K to 125K instructions before scheduling, then spends 5+ hours on that one loop until Slurm stops it.</li>
      <li>In 5 of 9, the accumulator is read and written back to back, so Vitis keeps retrying slower schedules.</li>
    </ul></div>
    <div><h3>Opus's high-DSP designs that pass</h3><ul>
      <li>Load, compute and store as separate DATAFLOW stages, or a moderate grid with explicit adder trees.</li>
      <li>Each accumulator reused only every 16 to 64 cycles.</li>
      <li>13K to 29K instructions; csynth takes 1.5 to 20 minutes.</li>
      <li>Opus is not immune: its one bare-floor failure was a 167K-instruction grid with no DATAFLOW.</li>
    </ul></div>
  </div>
</section>

<section class='aside' id='agentic'>
  <span class='eyebrow'>Outside the controlled comparison</span>
  <h2>HLSmith-Agentic-Refined-Expertise + Resource-Window <span class='alias'>(Agentic ladder L2: the 5,152-DSP design)</span></h2>
  <p>The controlled runs above give Opus no tools: it answers each prompt with code, exactly like DeepSeek. An earlier
  Claude-only campaign also ran the same c2hls flow in <b>agent mode</b>, and its stage-2 design is the one at 5,152 DSP.</p>
  <div class='cols2'>
    <div><h3>The DSP ladder</h3><p>Three flash runs in a row, each seeded with the previous stage's kernel, using the refined
    prompts, refined v2 skills, balance gate and a DSP window as floor plus cap. Stage 1 asks for about 25% of the DSP. Stage 2
    says: double the parallel MACs, keep the architecture, scale buffering and bandwidth with it, and be faster. Stage 3 grows
    toward the 80% roof.</p></div>
    <div><h3>Agent mode</h3><p>When c2hls sends a flash or repair prompt, Claude Code runs in a workspace holding
    <code>kernel.cpp</code>, <code>kernel.h</code> and the testbench. It can read and edit files and run one script,
    <code>./hls_check.sh compile|csim|csynth</code>, which runs real Vitis HLS on a compute node (12 checks per session).
    It can test a design before answering. Plain mode cannot.</p></div>
  </div>
  <h3>How the L2 design was made</h3>
  <ol>
    <li>Stage 1 produced a correct design at 2,592 DSP and 2.11M cycles.</li>
    <li>Stage 2's first two calls hit Claude's session limit. The shim then returned the unchanged stage-1 kernel, which the
    DSP window rejected as too small. (That shim bug is fixed; it returns an error now.)</li>
    <li>The rejection prompt started a real 71-minute agent session. Opus read the kernel, doubled the PE rows (16 to 32) and the
    row tile (32 to 64) with seven edits, ran csim (passed), then csynth (1,066,083 cycles, 5,152 DSP), and returned it. It used
    2 of its 12 checks.</li>
  </ol>
  <p>5,152 DSP is 1,024 fp32 multiply-adds (5,120 DSP) plus 32 DSP of index arithmetic: the same MAC count as AutoSA's
  128 &times; 8 array. At 1,066,083 cycles (best and worst case are equal) it runs within 1.02&times; of its compute floor. That is within 1.1% of
  AutoSA st0_c6's worst case (1,054,811) and 1.6% above its best case (1,048,845), with headroom left on every resource.</p>
  {agent_table()}
  <p class='meta'>DeepSeek has no tool mode in this flow, so these numbers are not part of the model comparison.</p>
</section>

<section class='aside' id='controls'>
  <h2>What was held equal</h2>
  <ul>
    <li>Identical prompts: every call flattened to one system and one user message; first prompts checked byte for byte.</li>
    <li>Same starting kernel (the bench baseline) for every rung after zero-shot.</li>
    <li>Answer taken from the reply content only, never from DeepSeek's reasoning text.</li>
    <li>Same output budget (262,144 tokens; Claude Code caps at 64,000, which no reply reached) and the same retries.</li>
    <li>Same harness fixes for both: continued pragma lines joined, Vitis headers available to the compile check.</li>
    <li>Same memory (128 GB), same wall-time rules; a design that never finishes csynth counts against the model that wrote it.</li>
    <li>Opus ran with no tools, no files, no memory and no subagents; its call logs show none were used.</li>
  </ul>
</section>

<footer>
  <span>Built {stamp} from <code>runs_controlled/</code> and <code>verify/</code> by <code>controlled/story/make_story.py</code>.</span>
  <span>Csynth estimates; Vitis cosim cycle checks for these designs are still to come. DeepSeek API model id: <code>deepseek-v4-flash</code>.</span>\n  <span>DATAFLOW: Vitis's own &ldquo;Applying dataflow&rdquo; report. Ping-pong: explicit double buffers in the source, or arrays passed between DATAFLOW stages (Vitis builds those as PIPO ping-pong). Streams: <code>hls::stream</code> channels declared.</span>
</footer>
</main>
"""
    OUT.write_text(page)
    print(OUT)


if __name__ == "__main__":
    main()

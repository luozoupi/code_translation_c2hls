#!/usr/bin/env python3
"""Build controlled/story/story_v11.html: the controlled ladder page (make_story.py, unchanged) plus the 13-14 Oct
study sections: fixed bench, plain 90-skill pack, DSP 50-65%, manual true dataflow, general agent, effort, dfdiag, LUD.

Every number is read from run outputs at build time; rerun after jobs finish. story.html and make_story.py are kept.
"""
import glob, html, json, re, subprocess, time
from pathlib import Path

import make_story as ms

CC = ms.CC
S = CC / "study"
OUT = Path(__file__).resolve().parent / "story_v11.html"
MM_BASE_WORST = 167503724616     # mm hls_baseline.cpp, the flow's fallback when no design is accepted
LUD_BASE_WORST = 1052705         # LUD hls_baseline.cpp worst case
LUD_N = 32
esc = html.escape


def j(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def states():
    led = [json.loads(l) for l in (S / "jobs/ledger.jsonl").read_text().splitlines() if l.strip()]
    last = {}
    for r in led:
        last[r["tag"]] = r
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(r["job"] for r in last.values()), "-o",
                          "JobID,State,Elapsed"], capture_output=True, text=True).stdout
    st = {l.split("|")[0]: (l.split("|")[1].split()[0], l.split("|")[2]) for l in out.splitlines() if "|" in l}
    return {t: dict(r, state=st.get(r["job"], ("?", ""))[0], elapsed=st.get(r["job"], ("?", ""))[1]) for t, r in last.items()}


LED = states()


LUT_CAP = 0.8 * ms.CAP["lut"]   # 2026-10-22: designs above 80% of the U280's LUTs are not counted anywhere on the page
UNCAPPED = {}                    # agentfw variant -> (worst, dsp, lut %) of its fastest design when that one is over the cap


def cell(root, tag, bench):
    """Result of one c2hls flash cell (latest non-archived dir for the tag)."""
    dirs = sorted(d for d in glob.glob(f"{root}/*_{tag}_2026*/{bench}") if "/_" not in d)
    rec = {"tag": tag, "state": LED.get(tag, {}).get("state", "?"), "elapsed": LED.get(tag, {}).get("elapsed", ""),
           "failed": LED.get(tag, {}).get("failed")}
    if not dirs:
        return rec
    d = j(f"{dirs[-1]}/{bench}_multistep_results.json")
    if not d:
        return rec
    fr = d.get("final_report") or {}
    rec.update(done=True, csim=(d.get("csim") or {}).get("passed"), best=fr.get("latency_cycles"),
               worst=fr.get("latency_cycles_worst") or fr.get("latency_cycles"),
               **{k: fr.get(k) for k in ("dsp", "lut", "ff", "bram", "uram")})
    base = MM_BASE_WORST if bench == "autosa_mm" else LUD_BASE_WORST
    rec["fallback"] = rec["worst"] == base
    rec["overcap"] = (rec.get("lut") or 0) > LUT_CAP
    return rec


def chip(kind, text):
    return f"<span class='chip {kind}'>{text}</span>"


def outcome(r, note=""):
    if r.get("done"):
        if r.get("overcap") and not r.get("fallback"):
            return chip("bad", "Over LUT cap") + f" <span class='why'>{r['lut'] / ms.CAP['lut']:.1%} LUT, above the 80% cap</span>"
        if r.get("fallback"):
            return chip("bad", "No design") + f" <span class='why'>{esc(note or 'No design was accepted; the flow returned the baseline kernel')}</span>"
        return chip("ok", "Pass") if r.get("csim") else chip("bad", "csim fail")
    if r["state"] in ("PENDING", "RUNNING"):
        return chip("wait", "Running" if r["state"] == "RUNNING" else "Queued") + (f" <span class='why'>{esc(r['elapsed'])} so far</span>" if r["state"] == "RUNNING" else "")
    if r.get("failed"):
        return chip("bad", "Fail") + f" <span class='why'>{esc(note or 'Never finished; timed out at 6 h (3rd try)')}</span>"
    return chip("wait", "Not run") if r["state"] == "?" else chip("bad", esc(r["state"]))


def res_cells(r, floor=True):
    if not r.get("done") or r.get("fallback"):
        return "<td class='num'></td>" * (8 if floor else 7)
    out = f"<td class='num strong'>{ms.fmt_int(r['worst'])}</td><td class='num'>{ms.fmt_int(r['best'])}</td>"
    if floor:
        out += f"<td class='num'>{r['worst'] / (ms.MACS / (r['dsp'] / 5.0)):.2f}&times;</td>" if r.get("dsp") else "<td class='num'></td>"
    return out + ms.res(r["dsp"], "dsp") + ms.res(r["lut"], "lut") + ms.res(r["bram"], "bram") + ms.res(r["uram"], "uram") + ms.res(r["ff"], "ff")


def run_table(rows, floor=True, first="Run"):
    head = (f"<th scope='col'>{first}</th><th scope='col'>Outcome</th><th scope='col' class='num'>Worst-case cycles</th>"
            "<th scope='col' class='num'>Best-case cycles</th>" + ("<th scope='col' class='num'>&times; floor</th>" if floor else "")
            + "".join(f"<th scope='col' class='num'>{h}</th>" for h in ("DSP", "LUT", "BRAM_18K", "URAM", "FF")))
    body = ""
    prev = None
    for label, m, r, note in rows:
        sep = " class='grp'" if prev is not None and m != prev else ""
        prev = m
        dot = f"<span class='dot {m}'></span>" if m in ("cc", "ds") else ""
        body += f"<tr{sep}><th scope='row'>{dot}{label}</th><td>{outcome(r, note)}</td>{res_cells(r, floor)}</tr>"
    return f"<div class='tablewrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def model_rows(root, fmt, bench, notes=None, models=("cc", "ds")):
    notes = notes or {}
    return [(f"{ms.MODELS[m]} r{r}", m, cell(root, fmt.format(m=m, r=r), bench), notes.get(fmt.format(m=m, r=r), ""))
            for m in models for r in (1, 2, 3)]


# ---------------------------------------------------------------- sections

def sec_bench():
    rows = []
    for d in sorted((S / "recsim").iterdir()):
        s = d / "csim_summary.txt"
        if d.is_dir() and s.exists():
            rows.append((d.name, "correct=True" in s.read_text()))
    def count(pref):
        rs = [ok for n, ok in rows if n.startswith(pref)]
        return sum(rs), len(rs)
    cc_ok, cc_n = count("ctl_cc_")
    ag_ok, ag_n = count("agent_ladder")
    ds_ok, ds_n = count("ctl_ds_")
    ds_pass = sorted({re.sub(r"ctl_ds_(\w+)_r\d", r"\1", n) for n, ok in rows if n.startswith("ctl_ds_") and ok})
    return f"""
<section class='aside alert' id='bench'>
  <span class='eyebrow'>Read this before the csim column</span>
  <h2>The testbench now checks C = A&middot;B<sup>T</sup> with C poisoned</h2>
  <p>AutoSA's kernel computes C = A&middot;B<sup>T</sup> and overwrites C. The original c2hls bench wrote
  <code>plain.cpp</code> as C = C + A&middot;B, but its testbench never set C before the call, so a kernel that read C and one
  that overwrote it both passed. The fixed bench (<code>study/bench_fixed/</code>) fills C with 12345.0 before the call and
  checks every one of the 1,048,576 outputs. Every result from 13 October on uses it.</p>
  <div class='cols3'>
    <div class='stat'><b>{cc_ok + ag_ok} / {cc_n + ag_n}</b><span>Opus kernels that pass the fixed bench ({cc_n} ladder + {ag_n} agentic). All
    {cc_n + ag_n - cc_ok - ag_ok} fail with 1,048,576 errors, because they read C as the old spec told them to.</span></div>
    <div class='stat'><b>{ds_ok} / {ds_n}</b><span>DeepSeek kernels that pass: every {', '.join(ds_pass)} run except one zero-shot
    run, and none of the skills runs. DeepSeek often ignored the C read.</span></div>
    <div class='stat'><b>Spec, not skill</b><span>The failures follow the old prompt, not the model's ability. The cycles in the
    ladder below still hold for the C&nbsp;+=&nbsp;A&middot;B kernels they measure; the new sections use the fixed spec.
    Controls: the fixed reference passes, the old C&nbsp;+= reference fails.</span></div>
  </div>
</section>"""


def sec_s90():
    t = run_table(model_rows(S / "s90/runs", "s90_{m}_r{r}", "autosa_mm"))
    t2 = run_table(model_rows(S / "s90/runs", "s90dsp_{m}_r{r}", "autosa_mm"))
    return f"""
<section class='rung' id='rung-s90'>
  <div class='num-badge'>3b</div>
  <div class='body'>
    <h2>HLSmith-SingleStep-Expertise, plain 90-skill pack <span class='alias'>(Flash + <code>skills_ii_target_miss_solutions_added(90skills).json</code>)</span></h2>
    <p class='changed'><b>What changes:</b> the pack. Rung 3 above used the 90-skill pack with the six gemm-flatten v1 entries
    (129 skills). Here the flow loads only the plain 90-skill file (123 skills, no gemm-flatten), confirmed by hashing each run's
    saved skill list. Fixed bench, so csim checks C = A&middot;B<sup>T</sup>.</p>
    <p>Without the gemm-flatten entries, Opus's designs stay small and slow: 23M to 86M cycles at 80 to 320 DSP, against
    19M at 320 DSP with gemm-flatten v1. DeepSeek lands at 158M to 305M cycles with 12 to 40 DSP. With the DSP floor added,
    Opus fills 92 to 96% of the DSP and reaches 1.05M to 2.85M cycles; its best (1,050,774) is 0.2% above AutoSA st0_c6's best
    case and below its worst case.</p>
    <h3>Plain 90-skill pack</h3>{t}
    <h3>Plain 90-skill pack + Resource-Requirement (DSP floor, 90% fill)</h3>{t2}
  </div>
</section>"""


def sec_dsp5065():
    notes = {"st_s5065_ds_r1": "DeepSeek's designs never reached the 4,512-DSP floor; rejected 3 times",
             "st_s5065_ds_r3b": "Stayed at 768 DSP; rejected 3 times by the 4,512-DSP floor",
             "st_r5065_ds_r1": "Never finished; timed out at 6 h (3rd try)", "st_r5065_ds_r3": "Never finished; timed out at 6 h (3rd try)"}
    sets = [("Skills + DSP 50&ndash;65% <span class='alias'>(gemm-flatten v1 pack, floor 4,512 and cap 5,865)</span>",
             [(f"{ms.MODELS[m]} r{r}", m, cell(S / 'dsp5065/runs', t, 'autosa_mm'), notes.get(t, ''))
              for m in ("cc", "ds") for r in (1, 2, 3)
              for t in [f"st_s5065_{m}_r{r}" + ("b" if (m, r) == ("ds", 3) else "")]]),
            ("Refined skills + DSP window 50&ndash;65%", model_rows(S / "dsp5065/runs", "st_r5065_{m}_r{r}", "autosa_mm", notes)),
            ("Agent mode, refined skills + DSP window 50&ndash;65% <span class='alias'>(Opus with tools and real csim/csynth checks)</span>",
             [(f"Opus 5.5 agent r{r}", "cc", cell(S / 'dsp5065/runs', t, 'autosa_mm'), "")
              for r, t in ((1, "st_a5065_ag_r1c"), (2, "st_a5065_ag_r2c"), (3, "st_a5065_ag_r3"))])]
    body = "".join(f"<h3>{n}</h3>{run_table(rows)}" for n, rows in sets)
    return f"""
<section class='aside' id='dsp5065'>
  <h2>A smaller DSP target: 50&ndash;65% of the chip</h2>
  <p>Every step whose designs used more than 60% of the DSP was rerun with a floor of 4,512 and a cap of 5,865 DSP (50 to
  65%), on the fixed bench. 5,120 DSP is 1,024 fp32 multiply-adds, the same as AutoSA's 128&nbsp;&times;&nbsp;8 array, so this
  asks whether the models can match AutoSA at AutoSA's own size.</p>
  <p>Opus's refined runs all land on exactly 5,120 DSP at 1.07M to 1.11M cycles, 1.02 to 1.06 times their floor and within
  1 to 6% of AutoSA st0_c6. With only the stock skills, Opus also reaches 5,120 DSP but at 1.27M to 7.4M cycles: the DSP is
  there, the data movement is not. DeepSeek reached the window in 2 of 6 runs.</p>
  {body}
  <p class='meta'>Three agent runs and one DeepSeek run from 13&ndash;14 October were invalid (Claude usage limit, then a full
  /scratch file quota), kept in <code>dsp5065/runs/_invalid_*</code>, and rerun under new tags.</p>
</section>"""


def sec_manual():
    rows = ""
    desc = {"v0": "refnf r2 rewritten to overwrite C (no C load); no DATAFLOW",
            "v1": "DATAFLOW over 16 row tiles: load A, compute, store C as stages; A and C tiles become PIPO ping-pong buffers; B streamed into URAM during tile 0",
            "v2": "v1 with the compute loops flattened into one pipeline"}
    diag = {"v0": "No DATAFLOW", "v1": "Ping-pong, full overlap", "v2": "Ping-pong, full overlap"}
    for v in ("v0", "v1", "v2"):
        s = j(S / f"manual_df/{v}/csynth_summary.json") or {}
        r = s.get("report") or {}
        c = (S / f"manual_df/{v}/csim_summary.txt")
        ok = c.exists() and "correct=True" in c.read_text()
        if not r:
            continue
        w = r.get("latency_cycles_worst") or r.get("latency_cycles")
        rows += (f"<tr><th scope='row'>{v}</th><td class='wrap'>{desc[v]}</td><td>{chip('ok', 'Pass') if ok else chip('bad', 'Fail')}</td>"
                 f"<td class='num strong'>{ms.fmt_int(w)}</td><td class='num'>{ms.fmt_int(r['latency_cycles'])}</td>"
                 f"<td class='num'>{w / (ms.MACS / (r['dsp'] / 5.0)):.2f}&times;</td>" + ms.res(r["dsp"], "dsp") + ms.res(r["lut"], "lut")
                 + ms.res(r["bram"], "bram") + ms.res(r["uram"], "uram") + ms.res(r["ff"], "ff") + f"<td>{diag[v]}</td></tr>")
    return f"""
<section class='aside' id='manual'>
  <h2>Making refnf r2 a true DATAFLOW ping-pong design by hand</h2>
  <p>Opus's refined-in-words r2 design uses exactly 5,120 DSP but runs its loads, compute and stores one after another.
  Rewriting only its structure, with the same 1,024 multiply-adds, takes it from 1.25M to 1.07M cycles, 1.02 times its
  compute floor and 1.6% above AutoSA st0_c6's worst case. Flattening the compute loops (v2) gains nothing more: the
  design is already compute-bound.</p>
  <div class='tablewrap'><table><thead><tr><th scope='col'>Version</th><th scope='col'>Change</th><th scope='col'>csim</th>
  <th scope='col' class='num'>Worst-case cycles</th><th scope='col' class='num'>Best-case cycles</th><th scope='col' class='num'>&times; floor</th>
  <th scope='col' class='num'>DSP</th><th scope='col' class='num'>LUT</th><th scope='col' class='num'>BRAM_18K</th><th scope='col' class='num'>URAM</th>
  <th scope='col' class='num'>FF</th><th scope='col'>dfdiag</th></tr></thead><tbody>{rows}</tbody></table></div>
</section>"""


POSTHOC = S / "posthoc_csim"     # 2026-10-22: post-hoc csim and csynth of unchanged session kernels, both sides


def hard_failed():
    """Kernel texts that fail the hardened bench (study/bench_hard/verify_hard_b.csv, study/posthoc_csim/hard.csv)."""
    import csv
    out = set()
    for f in (S / "bench_hard/verify_hard_b.csv", POSTHOC / "hard.csv"):
        if f.exists():
            for r in csv.DictReader(open(f)):
                k = S.parent / r["kernel"]
                if not r["verdict"].startswith("Passed") and k.exists():
                    out.add(k.read_text())
    return out


def _ended(w):
    import datetime as _dt
    try:
        return _dt.datetime.fromisoformat((w / "ended_at").read_text().strip().replace("Z", "+00:00")).timestamp()
    except Exception:
        return float("inf")


def session_best(w, window=None, in_session_only=False):
    """Best verified design of a Claude Code session: a csim pass (fixed bench) and a csynth of the byte-identical kernel,
    each from the session (exp/) or checked afterwards with the kernel unchanged (exp_post/, study/posthoc_csim/);
    fits the U280 and the 80% LUT cap (and the DSP window, if given), and does not fail the hardened bench. A design is post-session when either half was done
    after the session ended. in_session_only: matched-budget runs count only what the session itself finished.
    Returns best=(name, worst, report, post) or None, fastest=(worst, dsp, lut %) over the cap or None, nsyn."""
    end = _ended(w)
    bad = hard_failed()
    csim = {}
    for c in list(w.glob("exp/*/csim_summary.txt")) + list(w.glob("exp_post/*/csim_summary.txt")) \
            + list((POSTHOC / w.name).glob("*/csim_summary.txt")):
        k = c.parent / "kernel.cpp"
        if "correct=True" in c.read_text() and k.exists():
            post = "exp/" not in str(c.relative_to(S)).replace(w.name + "/", "", 1) or c.stat().st_mtime > end
            kt = k.read_text()
            csim[kt] = min(csim.get(kt, True), post)
    best, fastest, nsyn = None, None, 0
    for s in list(w.glob("exp/*/csynth_summary.json")) + list(w.glob("exp_post/*/csynth_summary.json")) \
            + list((POSTHOC / "csynth" / w.name).glob("*/csynth_summary.json")):
        d = j(s)
        k = s.parent / "kernel.cpp"
        if not d or not d.get("success") or not k.exists():
            continue
        spost = s.parent.parent.name != "exp" or s.stat().st_mtime > end
        nsyn += not spost
        kt = k.read_text()
        if kt not in csim or kt in bad:
            continue
        post = spost or csim[kt]
        if in_session_only and post:
            continue
        r = d["report"]
        if not all((r.get(x) or 0) <= ms.CAP[x] for x in ms.CAP):        # must fit the U280
            continue
        if window and not window[0] <= (r.get("dsp") or 0) <= window[1]:
            continue
        wc = r.get("latency_cycles_worst") or r.get("latency_cycles")
        if (r.get("lut") or 0) > LUT_CAP:
            if fastest is None or wc < fastest[0]:
                fastest = (wc, r.get("dsp"), 100 * (r.get("lut") or 0) / ms.CAP["lut"])
            continue
        if best is None or wc < best[1]:
            best = (s.parent.name, wc, r, post)
    return {"best": best, "fastest": fastest if fastest and (best is None or fastest[0] < best[1]) else None, "nsyn": nsyn}


def agentfw_runs():
    out = []
    for v in ("base", "know", "team"):
        w = S / f"agentfw/runs/mm1024_{v}_r1"
        cost = turns = 0.0
        tools = {}
        for t in sorted(w.glob("transcript*.jsonl")):
            for l in t.read_text().splitlines():
                try:
                    d = json.loads(l)
                except Exception:
                    continue
                if d.get("type") == "result":
                    cost = max(cost, d.get("total_cost_usd") or 0)   # cumulative across a resumed session
                    turns += d.get("num_turns") or 0
                if d.get("type") == "assistant":
                    for c in d["message"].get("content", []):
                        if c.get("type") == "tool_use":
                            tools[c["name"]] = tools.get(c["name"], 0) + 1
        sb = session_best(w)
        best, nsyn = sb["best"], sb["nsyn"]
        if sb["fastest"] and (best is None or sb["fastest"][0] < best[1]):
            UNCAPPED[v] = sb["fastest"]
        out.append((v, cost, int(turns), tools, nsyn, best))
    return out


def sec_agentfw():
    rows = ""
    q = {"base": "Read/Write/Edit/Glob/Grep, read-only shell, <code>bin/hls</code> (csim, csynth, report, dfdiag)",
         "know": "base + knowledge folder (skill packs, dataflow notes) + WebSearch/WebFetch",
         "team": "know + subagents"}
    for v, cost, turns, tools, nsyn, best in agentfw_runs():
        web = tools.get("WebSearch", 0) + tools.get("WebFetch", 0)
        if best:
            name, wc, r, post = best
            cells = (f"<td class='num strong'>{ms.fmt_int(wc)}</td><td class='num'>{wc / (ms.MACS / (r['dsp'] / 5.0)):.2f}&times;</td>"
                     + ms.res(r["dsp"], "dsp") + ms.res(r["lut"], "lut") + ms.res(r["bram"], "bram") + ms.res(r["ff"], "ff"))
        else:
            name, cells = "none", "<td class='num'></td>" * 6
        tagp = " <span class='pct'>post-session</span>" if best and best[3] else ""
        rows += (f"<tr><th scope='row'>{v}</th><td class='wrap'>{q[v]}</td><td><code>{esc(name)}</code>{tagp}</td>{cells}"
                 f"<td class='num'>{nsyn}</td><td class='num'>{tools.get('Read', 0) + tools.get('Grep', 0) + tools.get('Glob', 0)}</td>"
                 f"<td class='num'>{web}</td><td class='num'>{tools.get('Agent', 0)}</td><td class='num'>${cost:.2f}</td></tr>")
    return f"""
<section class='aside' id='agentfw'>
  <span class='eyebrow'>New framework, outside c2hls</span>
  <h2>A general Claude Code agent for HLS</h2>
  <p>No flash step, no ladder, no fixed prompts. Claude Code gets a workspace with the reference kernel, the fixed bench,
  a goal (fastest correct design that fits the U280 at 3.33&nbsp;ns) and one tool, <code>bin/hls</code>, that runs real Vitis
  csim and csynth through Slurm. It decides what to read, what to try and when to stop. Three variants differ only in what
  else it may use. One run each so far, on mm 1024&sup3;.</p>
  <div class='tablewrap'><table><thead><tr><th scope='col'>Variant</th><th scope='col'>Tools</th><th scope='col'>Best correct design</th>
  <th scope='col' class='num'>Worst-case cycles</th><th scope='col' class='num'>&times; floor</th><th scope='col' class='num'>DSP</th>
  <th scope='col' class='num'>LUT</th><th scope='col' class='num'>BRAM_18K</th><th scope='col' class='num'>FF</th>
  <th scope='col' class='num'>csynths</th><th scope='col' class='num'>File reads</th><th scope='col' class='num'>Web</th>
  <th scope='col' class='num'>Subagents</th><th scope='col' class='num'>Cost</th></tr></thead><tbody>{rows}</tbody></table></div>
  <ul>
    <li><b>base reaches 468,427 cycles</b> within the 80% LUT cap, 2.3 times faster than AutoSA st0_c6: 147 rows &times; 16
    lanes of multiply-add per cycle in one flat pipeline, B kept in URAM, A double-buffered, at 91% DSP and 70% LUT. It
    beats the 5-DSP compute floor because it puts 4 of each row's first-level tree adds in LUTs
    (<code>BIND_OP impl=fabric</code>). The kernel csim'd is byte-identical to the one synthesized. Its fastest design,
    404,451 cycles (171 rows, 7 fabric adds per row), uses 95% DSP and 87% LUT, so it is over the cap and not counted.
    Caveat: two <code>DEPENDENCE false</code> pragmas csim cannot check.</li>
    <li><b>know and team did not use their extra reach:</b> one web search and no subagent calls in total. know ended at
    658,279 cycles (79.6% LUT, just inside the cap). team has no design that is both csim-verified and synthesized inside its session: its verified kernels never
    finished csynth (time limit, then the file quota), and its fastest synthesized design (590,434 cycles) uses 10,240 DSP,
    113% of the chip, and 85% LUT. Checked after the session with its kernels unchanged, its csim-passing v3 synthesizes at
    527,571 cycles (8,192 DSP, 75.5% LUT), and its synthesized v3_ti4 (4,195,795 cycles) passes csim; both pass the
    hardened bench. The cost table counts 527,571, marked post-session.</li>
    <li>All three sessions were stopped once by Claude's usage limit and resumed; know and team were then cut short by the
    /scratch file quota. One run per variant, so these are pilots, not a comparison.</li>
  </ul>
  <p class='meta'>Cost is the session's reported total; &times; floor uses 5 DSP per multiply-add, which base does not need.</p>
</section>"""


def sec_effort():
    return """
<section class='aside' id='effort'>
  <h2>Effort and knowledge extraction</h2>
  <p>Measured from every call log and transcript. No flow before the general agent looked anything up: Opus answered from
  the prompt and its own reasoning, and read only its own kernel.</p>
  <div class='tablewrap'><table><thead><tr><th scope='col'>Flow</th><th scope='col' class='num'>Calls or cells</th>
  <th scope='col'>Knowledge acquisition</th><th scope='col'>Experiments</th><th scope='col' class='num'>Output tokens</th>
  <th scope='col' class='num'>Cost</th></tr></thead><tbody>
  <tr><th scope='row'>Controlled ladder, Opus, no tools</th><td class='num'>31 calls</td><td class='wrap'>None by construction</td><td>None</td><td class='num'>405,928 (294,420 thinking)</td><td class='num'>$25.63</td></tr>
  <tr><th scope='row'>Controlled ladder, DeepSeek</th><td class='num'>&mdash;</td><td class='wrap'>None</td><td>None</td><td class='num'>2.04M</td><td class='num'>&mdash;</td></tr>
  <tr><th scope='row'>c2hls agent mode (tools + <code>hls_check</code>)</th><td class='num'>25 calls</td><td class='wrap'>8 Reads (own kernel), 3 Greps, 0 web</td><td class='wrap'>37 checks: 18 csim, 17 csynth, 2 compile</td><td class='num'>283,292</td><td class='num'>$23.03</td></tr>
  <tr><th scope='row'>Clean U280 one-shots</th><td class='num'>6 runs</td><td class='wrap'>Glob 1 + Read 2 per run (seed files)</td><td>None</td><td class='num'>180,987</td><td class='num'>$5.13</td></tr>
  <tr><th scope='row'>DSE v4 st0_c1 design session</th><td class='num'>1 session</td><td class='wrap'>7 Reads + 23 read-only shell, 0 web</td><td class='wrap'>13 edits, 7 Slurm jobs</td><td class='num'>133,254</td><td class='num'>$14.13</td></tr>
  </tbody></table></div>
  <p class='meta'>Source: <code>study/effort/EFFORT.md</code>. The general agent's own counts are in its table above.</p>
</section>"""


def sec_dfdiag():
    rows = [("Refined + floor r3 (Opus)", "Ping-pong", "Full", "16 tiles overlap; compute-bound"),
            ("Refined + window r3 (Opus)", "Task concurrency", "Full", "Stages overlap; compute-bound"),
            ("Refined + window r1 (Opus)", "Task concurrency", "Partial", "A loader at II=16 sets the pace (1.06M vs 0.79M compute)"),
            ("Refined + floor r2 (Opus)", "Ineffective", "&mdash;", "DATAFLOW covers the loaders only; compute runs after them"),
            ("Refined in words r2 (Opus)", "No DATAFLOW", "&mdash;", "Load, compute and store in sequence"),
            ("Manual v1 / v2", "Ping-pong", "Full", "Tile loop interval 66,714 vs region latency 70,884"),
            ("General agent, base", "Task concurrency", "Full", "Compute stage is 100% of latency; manual A double buffer inside one loop is not recognized")]
    body = "".join(f"<tr><th scope='row'>{a}</th><td>{b}</td><td>{c}</td><td class='wrap'>{d}</td></tr>" for a, b, c, d in rows)
    return f"""
<section class='aside' id='dfdiag'>
  <h2>Is it really DATAFLOW? A diagnostic that reads Vitis, not the pragma</h2>
  <p>A <code>#pragma HLS DATAFLOW</code> line proves nothing. <code>study/dfdiag/dfdiag.py</code> reads what Vitis actually
  built: the &ldquo;Applying dataflow &hellip; extracted N process&rdquo; and &ldquo;Implementing PIPO&rdquo; log lines, each
  module's csynth report, and the loop tables. It then checks whether stages overlap: a dataflow loop of N iterations with
  latency close to N&middot;II + (L &minus; II) is overlapping; one close to N&middot;L is not. It also flags transfer loops
  at II&nbsp;&gt;&nbsp;1, compute loops at II&nbsp;&gt;&nbsp;1, serial work outside the region, and a non-compute bottleneck stage.</p>
  <div class='tablewrap'><table><thead><tr><th scope='col'>Design</th><th scope='col'>Class</th><th scope='col'>Overlap</th>
  <th scope='col'>Evidence</th></tr></thead><tbody>{body}</tbody></table></div>
</section>"""


def lud_dse():
    rows = j(S / "lud/check/dse_table.json") or []
    wrong = {0, 1, 3}
    by = {}
    for r in rows:
        n, st = map(int, re.match(r"n(\d+)_st(\d)", r["design"]).groups())
        tag = f"lud_{r['design']}_csynth"
        led = LED.get(tag, {})
        if st in wrong:
            status = chip("bad", "Wrong U")
        elif st == 2:
            status = chip("bad", "csynth error")
        elif r.get("lat_max"):
            status = chip("ok", "Pass")
        elif led.get("failed"):
            status = chip("bad", "Timed out at 6 h")
        elif led.get("state") in ("RUNNING", "PENDING"):
            status = chip("wait", "Running")
        else:
            status = chip("wait", led.get("state", "?"))
        by[(n, st)] = (r, status)
    ns = sorted({k[0] for k in by})
    body = ""
    for st in range(6):
        cells = ""
        for n in ns:
            r, status = by[(n, st)]
            val = (f"<div class='lat'>{ms.fmt_int(int(r['lat_max']))}</div><div class='sub'>{ms.fmt_int(r['DSP'])} DSP &middot; {r['PEs']} PE</div>"
                   if r.get("lat_max") else f"<div class='sub'>{r['PEs']} PE</div>")
            cells += f"<td class='dsecell'>{status}{val}</td>"
        body += f"<tr><th scope='row'>space_time {st}</th>{cells}</tr>"
    head = "".join(f"<th scope='col'>N = {n}</th>" for n in ns)
    return f"<div class='tablewrap'><table class='dse'><thead><tr><th scope='col'>AutoSA LU</th>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def sec_lud():
    variants = [("zs", "Zero-shot", "lud_zs_{m}_r{r}"), ("noskills", "HLSmith Base <span class='alias'>(Flash, no skills)</span>", "lud_noskills_{m}_r{r}"),
                ("skills", "HLSmith-SingleStep-Expertise <span class='alias'>(skills, gemm-flatten v1 pack)</span>", "lud_skills_{m}_r{r}"),
                ("s90", "HLSmith-SingleStep-Expertise <span class='alias'>(plain 90-skill pack)</span>", "lud_s90_{m}_r{r}", S / "s90/runs"),
                ("refneu", "HLSmith-SingleStep-Refined-Expertise <span class='alias'>(neutral refined skills)</span>", "lud_refneu_{m}_r{r}")]
    notes = {"lud_skills_ds_r3": "Rejected by c2hls's timing-regression gate (estimated slack &minus;0.64 ns); flow kept the baseline"}
    # summary strip: best worst-case per variant and model
    strip = ""
    variants = [x if len(x) == 4 else x + (S / "lud/runs",) for x in variants]
    for v, name, fmt, root in variants:
        cells = ""
        for m in ("cc", "ds"):
            rs = [cell(root, fmt.format(m=m, r=r), "lud") for r in (1, 2, 3)]
            ok = [r for r in rs if r.get("done") and not r.get("fallback") and r.get("csim")]
            best = min((r["worst"] for r in ok), default=None)
            med = sorted(r["worst"] for r in ok)[len(ok) // 2] if ok else None
            cells += (f"<td class='num'>{ms.fmt_int(best) if best else '&mdash;'}</td><td class='num'>{ms.fmt_int(med) if med else '&mdash;'}</td>"
                      f"<td class='num'>{len(ok)}/3</td>")
        strip += f"<tr><th scope='row'>{name}</th>{cells}</tr>"
    detail = "".join(f"<details><summary>{name}</summary>{run_table(model_rows(root, fmt, 'lud', notes), floor=False)}</details>"
                     for v, name, fmt, root in variants)
    return f"""
<section class='aside' id='lud'>
  <span class='eyebrow'>Second kernel &middot; LU decomposition &middot; fp32 &middot; U280 &middot; 3.33 ns</span>
  <h2>LUD: AutoSA's design space, then the same ablation</h2>
  <div class='cols2'>
    <div><h3>What AutoSA offers</h3><p>AutoSA ships one LU recipe (space_time 3, no tiling) and no U280 model. Its paper
    reports LU only for N = 12 to 24 on a U250, from RTL simulation at 250&nbsp;MHz, with no resources. Because LU cannot
    be tiled in AutoSA, the whole design space is N &times; space_time. All 30 points were generated at AutoSA rev
    <code>b61a1b4</code> and run through Vitis 2023.2 csim and csynth for xcu280 at 3.33&nbsp;ns.</p></div>
    <div><h3>What we found</h3><ul>
      <li><b>space_time 0, 1 and 3 compute U wrong</b> at every N: each row of U is shifted one column, while L is exact.
      Checked against a numpy LU. This includes the README recipe; regenerating without <code>--hls</code> gives the same kernel.</li>
      <li><b>space_time 2</b> passes csim but fails csynth: an internal stream is not produced and consumed exactly once.</li>
      <li><b>space_time 4 and 5</b> are correct; their csynth takes hours at large N.</li>
      <li>AutoSA's own LU testbench compares L only where the algorithm never writes it; our bench checks the real L and U
      with both poisoned.</li>
    </ul></div>
  </div>
  <h3>AutoSA design space: worst-case csynth latency</h3>
  {lud_dse()}
  <h3>The ablation at N = {LUD_N}, strict csim on L and U</h3>
  <p>Every completed run passes csim. Opus's median is 6 to 14 times lower than DeepSeek's at every step except neutral
  refined, where the medians tie (13.8K and 13.7K). Skills help Opus (median 10.5K to 5.3K cycles); the plain 90-skill
  pack helps less (8.3K). Neutral refined skills give the single best design, 2,264 cycles at 155 DSP,
  but also the widest spread.</p>
  <div class='tablewrap'><table><thead><tr><th scope='col' rowspan='2'>Step</th><th scope='colgroup' colspan='3'><span class='dot cc'></span>Opus 5.5</th>
  <th scope='colgroup' colspan='3'><span class='dot ds'></span>DeepSeek v4.1-flash</th></tr><tr>
  <th scope='col' class='num'>Best worst-case</th><th scope='col' class='num'>Median</th><th scope='col' class='num'>Pass</th>
  <th scope='col' class='num'>Best worst-case</th><th scope='col' class='num'>Median</th><th scope='col' class='num'>Pass</th></tr></thead>
  <tbody>{strip}</tbody></table></div>
  <div class='explain'><h3>Every run</h3>{detail}</div>
  <p class='meta'>AutoSA has no correct design at N = {LUD_N} that finishes csynth: space_time 4 and 5 both
  ran past 6 hours (the third and last time limit), as did space_time 5 at N = 20 and 24. Its largest correct result is
  N = 24, space_time 4: 3,141 cycles at 1,380 DSP. Every LLM design above is at N = {LUD_N} and finished csynth in minutes.
  The paper's N = 24 figure (10.44&nbsp;&micro;s at 250&nbsp;MHz, about 2,600 cycles) is a different device and method,
  so it is shown for scale only.</p>
</section>"""


EXTRA_CSS = """
.alert{border-color:var(--target)}
.cols3{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
.stat{display:flex;flex-direction:column;gap:6px;min-width:0;border-top:2px solid var(--rule);padding-top:10px}
.stat b{font:600 1.6rem/1.1 var(--display)}
.stat span{font-size:.9rem;color:var(--muted)}
td.wrap{white-space:normal;min-width:220px;max-width:420px}
tr.grp th,tr.grp td{border-top:2px solid var(--rule)}
table.dse td.dsecell{min-width:120px}
.dsecell .lat{font:600 .95rem/1.3 var(--mono);margin-top:6px}
.dsecell .sub{font:500 .72rem/1.3 var(--mono);color:var(--muted)}
.new{display:flex;flex-direction:column;gap:8px}
.new ul{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:4px;max-width:80ch}
.explain details .tablewrap{margin-top:8px}
"""


def main():
    rows = ms.collect()
    rungs = []
    for i, (v, name, changed, found) in enumerate(ms.RUNGS, 1):
        rungs.append(f"""
<section class='rung' id='rung-{v}'>
  <div class='num-badge'>{i}</div>
  <div class='body'>
    <h2>{name}</h2>
    <p class='changed'><b>What changes:</b> {changed}</p>
    <p>{found}</p>
    {ms.REFINED_BOX.replace("__REFINED_CHANGES__", (Path(ms.__file__).resolve().parent / "refined_changes.html").read_text()) if v == "refneu" else ""}
    {ms.tally(rows, v)}
    {ms.table(rows, v)}
  </div>
</section>""")
        if v == "skills":
            rungs.append(sec_s90())
    old = ms.OUT.read_text() if ms.OUT.exists() else ""
    def grab(sid):
        m = re.search(rf"<section class='aside' id='{sid}'>.*?</section>", old, re.S)
        return m.group(0) if m else ""
    stamp = time.strftime("%d %b %Y, %H:%M %Z")
    page = f"""<title>The 1M-Cycle Ladder</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap" rel="stylesheet">
<style>{ms.CSS}{EXTRA_CSS}
.page{{padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}</style>
<main class='page'>
<header>
  <span class='eyebrow'>autosa_mm &middot; 1024&sup3; fp32 &middot; LUD N = 32 &middot; Alveo U280 (xcu280) &middot; 3.33 ns</span>
  <h1>The 1M-Cycle Ladder</h1>
  <p class='thesis'>Without a resource requirement, both models write correct but small kernels. Asked to fill the chip,
  Opus reaches about one million cycles on matrix multiply, AutoSA's level, and DeepSeek rarely gets a large design through.
  A free-running Claude Code agent with real Vitis feedback went further, to 404K cycles.</p>
  <p class='meta'>Same c2hls flash flow, same prompts byte for byte, three runs per model per rung. Opus 5.5 is the model inside
  Claude Code, run with no tools in the controlled ladder; DeepSeek v4.1-flash is called through its API.</p>
  <div class='new explain'><h3>New in this version (13&ndash;14 October)</h3><ul>
    <li><a href='#bench'>Fixed testbench</a>: C = A&middot;B<sup>T</sup>, C poisoned; every older Opus kernel reads C and fails it.</li>
    <li><a href='#rung-s90'>Plain 90-skill pack</a> (no gemm-flatten), with and without the DSP floor.</li>
    <li><a href='#dsp5065'>DSP 50&ndash;65%</a> reruns of every step that used more than 60% of the DSP.</li>
    <li><a href='#manual'>Hand-made DATAFLOW ping-pong</a> version of refnf r2: 1.07M cycles at 5,120 DSP.</li>
    <li><a href='#agentfw'>General Claude Code agent</a>: 404K cycles on mm.</li>
    <li><a href='#effort'>Effort and knowledge extraction</a>, <a href='#dfdiag'>DATAFLOW diagnostic</a>, and <a href='#lud'>LUD</a> with AutoSA's design space.</li>
  </ul></div>
  <div class='legend'><span><span class='dot cc'></span>Opus 5.5</span><span><span class='dot ds'></span>DeepSeek v4.1-flash</span>
  <span>&times; in the top lane: run with no passing design</span><span>dashed circle: still running</span></div>
</header>

{sec_bench()}

<figure>
  <h3>Worst-case latency</h3>
  <div class='figscroll'>{ms.chart(rows, "worst")}</div>
  <figcaption>Worst-case csynth latency of the selected design for every run, log scale, against AutoSA st0_c6's worst case (1,054,811). Old bench (C&nbsp;+=&nbsp;A&middot;B).</figcaption>
</figure>

<figure>
  <h3>Best-case latency</h3>
  <div class='figscroll'>{ms.chart(rows, "best")}</div>
  <figcaption>Best-case csynth latency, log scale, against AutoSA st0_c6's best case (1,048,845). Each rung adds one thing to the one before it.
  The last column is a separate flow, Claude Code with tools, explained further down.</figcaption>
</figure>

<section class='primer' aria-label='How to read the numbers'>
  <div><h3>Compute floor</h3><p>The fewest cycles a design could take with its DSP count, if every multiply-add unit
  started a new operation each cycle.</p><div class='formula'>floor = 1024&sup3; / (DSP / 5)</div><p class='meta'>fp32
  multiply-add &asymp; 5 DSP. &ldquo;&times; floor&rdquo; = cycles / floor; 1.0&times; is perfect.</p></div>
  <div><h3>Reaching 1M cycles</h3><p>Needs about 1,070 multiply-adds per cycle, which is about 5,400 DSP. AutoSA's
  128 PE &times; SIMD 8 systolic array (config st0_c6) has 1,024: 1,048,845 cycles best case and 1,054,811 worst case.</p></div>
  <div><h3>What the numbers are</h3><p>Vitis HLS csynth estimates. Percentages are of the whole U280: 9,024 DSP,
  1,303,680 LUT, 4,032 BRAM_18K, 960 URAM, 2,607,360 FF. The ladder's csim column is the old bench; sections marked
  &ldquo;fixed bench&rdquo; check C = A&middot;B<sup>T</sup> with C poisoned.</p></div>
</section>

{''.join(rungs)}

{grab('where-cycles')}
{grab('why-deepseek')}
{grab('agentic')}
{sec_dsp5065()}
{sec_manual()}
{sec_agentfw()}
{sec_effort()}
{sec_dfdiag()}
{sec_lud()}
{grab('controls')}

<footer>
  <span>Built {stamp} by <code>controlled/story/make_story_v11.py</code> from <code>runs_controlled/</code>, <code>verify/</code> and <code>study/</code>.</span>
  <span>Csynth estimates; Vitis cosim cycle checks are still to come. DeepSeek API model id: <code>deepseek-v4-flash</code>.</span>
</footer>
</main>
"""
    OUT.write_text(page)
    print(OUT)


if __name__ == "__main__":
    main()

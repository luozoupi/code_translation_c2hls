#!/usr/bin/env python3
"""Summarise the controlled DeepSeek-vs-Claude campaign (runs_controlled/) into controlled/RESULTS.md.

Per run: flash pass, worst-case cycles, DSP/LUT/BRAM, est. clock, attempts and their reject stages, LLM-call
stats from ctrl_llm_calls.jsonl, pragma-join events, strict csim (verify/ctl_<model>_<variant>_r<n>/),
Slurm state. Per pair (same variant and replicate): is the first flash prompt identical between models after
normalising run-specific temp paths? Per variant and model: pass rate and median cycles.
"""
import glob, json, re, statistics, subprocess
from pathlib import Path

CC = Path("/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code")
R = CC / "runs_controlled"
MODEL = {"cc": "Claude Code (claude-opus-5-5)", "ds": "DeepSeek (deepseek-v4-flash)"}
VARIANT = {"zs": "zero-shot", "noskills": "flash, no skills", "skills": "flash + skills (no DSP gates)",
           "refneu": "neutral refined expertise (no resource requirement)", "refnf": "refined, DSP requirement in words only (no gate)", "dspredo": "flash + skills + DSP floor (DSP redo, fill >= 90%)", "refv2": "refined v2 (skills + floor + balance gate)",
           "win7080": "DSP window 70-80% (skills + floor + cap)"}


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def slurm_states():
    jobs = {}
    for line in (R / "jobs.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            jobs[parts[0]] = parts[1]
    ids = ",".join(jobs.values())
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ids, "-o", "JobID,State,Elapsed"],
                         capture_output=True, text=True).stdout
    st = {l.split("|")[0]: (l.split("|")[1].split()[0], l.split("|")[2]) for l in out.splitlines() if "|" in l}
    return {tag: st.get(j, ("?", "")) + (j,) for tag, j in jobs.items()}


def first_flash_prompt(cell):
    h = load(cell / "autosa_mm_history.json")
    msgs = h if isinstance(h, list) else (h or {}).get("history") or (h or {}).get("messages") or []
    for m in msgs:
        c = m.get("content") or ""
        if m.get("role") == "user" and ("FLASH" in c or "flash" in c[:400]):
            return c
    return None


def normalise(text):
    if text is None:
        return None
    text = re.sub(r"/scratch/\S+?/(hls_synth__|c2hls_compile__)\S*", "<TMP>", text)
    text = re.sub(r"(hls_synth__|c2hls_compile__)\w+", "<TMP>", text)
    text = re.sub(r"\d{8}T\d{6}Z", "<STAMP>", text)
    return text


def summarise(cell, tag, state):
    m = load(cell / "autosa_mm_multistep_results.json") or {}
    flash = next((s for s in m.get("steps", []) if s.get("step_name") == "flash"), {})
    sel = load(cell / "autosa_mm_selected_report.json") or {}
    rep = sel.get("report") or sel
    lat = max(rep.get("latency_cycles") or 0, rep.get("latency_cycles_worst") or 0) or None
    attempts = [f"{a.get('stage') or ('ok' if a.get('success') else '?')}" +
                (f"(DSP {a['report'].get('dsp')})" if isinstance(a.get("report"), dict) and a["report"].get("dsp") else "")
                for a in (flash.get("attempt_results") or [])]
    calls = [json.loads(l) for l in open(cell / "ctrl_llm_calls.jsonl")] if (cell / "ctrl_llm_calls.jsonl").exists() else []
    tries = [c for c in calls if "outcome" in c]
    k = {o: sum(1 for c in tries if c.get("outcome") == o) for o in ("ok", "truncated", "empty", "transport_error")}
    joins = sum(1 for c in calls if c.get("event") == "joined_pragma_continuations")
    rt = sum(c.get("reasoning_tokens") or 0 for c in tries)
    ct = sum(c.get("completion_tokens") or 0 for c in tries)
    csim = (CC / "verify" / tag / "csim_summary.txt")
    csim_s = csim.read_text() if csim.exists() else ""
    csim_v = "Passed!" if ("correct=True" in csim_s and "Passed!" in csim_s) else ("FAILED" if csim_s else "n/a")
    ok = bool(flash.get("success"))
    return {"tag": tag, "state": state[0], "elapsed": state[1], "ok": ok, "lat": lat if ok else None,
            "dsp": rep.get("dsp") if ok else None, "lut": rep.get("lut") if ok else None,
            "bram": rep.get("bram") if ok else None, "clk": rep.get("estimated_clock_period_ns") if ok else None,
            "attempts": attempts, "calls": k, "joins": joins, "ctok": ct, "rtok": rt, "csim": csim_v,
            "prompt": normalise(first_flash_prompt(cell))}


def main():
    states = slurm_states()
    rows = {}
    for tag, st in states.items():
        mm = re.match(r"ctl_(cc|ds)_(zs|noskills|skills|dspredo|refneu|refnf|refv2|win7080)_r(\d)$", tag)
        if not mm:
            continue
        cells = sorted(R.glob(f"*_{tag}_*/autosa_mm"))
        if cells:
            rows[tag] = summarise(cells[-1], tag, st)
        else:
            rows[tag] = {"tag": tag, "state": st[0], "elapsed": st[1], "ok": False, "lat": None, "attempts": [],
                         "calls": {}, "joins": 0, "ctok": 0, "rtok": 0, "csim": "n/a", "prompt": None,
                         "dsp": None, "lut": None, "bram": None, "clk": None}
    out = ["# Controlled DeepSeek vs Claude Code: results", "",
           "Generated by `controlled/compare.py` from `runs_controlled/`. Cycles are csynth worst-case latency. "
           "Calls = controlled-layer tries ok/truncated/empty/transport-error. Joins = pragma-continuation fixes applied.",
           ""]
    for v, vname in VARIANT.items():
        out += [f"## {vname}", "",
                "| Run | Slurm | Flash | Cycles | DSP | LUT | BRAM | Est. clk | csim | Attempts (stage) | Calls | Joins | Completion tok (reasoning) |",
                "|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|"]
        for m in ("cc", "ds"):
            for r in (1, 2, 3):
                x = rows.get(f"ctl_{m}_{v}_r{r}")
                if not x:
                    continue
                c = x["calls"]
                calls = "/".join(str(c.get(o, 0)) for o in ("ok", "truncated", "empty", "transport_error")) if c else "-"
                fmt = lambda n: f"{n:,}" if isinstance(n, (int, float)) and n else "-"
                out.append(f"| {m} r{r} | {x['state']} {x['elapsed']} | {'pass' if x['ok'] else ('running' if x['state'] in ('RUNNING', 'PENDING') else 'FAIL')} | {fmt(x['lat'])} | "
                           f"{fmt(x['dsp'])} | {fmt(x['lut'])} | {fmt(x['bram'])} | {x['clk'] or '-'} | {x['csim']} | "
                           f"{', '.join(x['attempts']) or '-'} | {calls} | {x['joins']} | {fmt(x['ctok'])} ({fmt(x['rtok'])}) |")
        out.append("")
        out += ["| Model | Flash pass | Median cycles (passing) | Best cycles |", "|---|---:|---:|---:|"]
        for m in ("cc", "ds"):
            xs = [rows[t] for t in rows if t.startswith(f"ctl_{m}_{v}_r")]
            lats = [x["lat"] for x in xs if x["ok"] and x["lat"]]
            out.append(f"| {MODEL[m]} | {sum(x['ok'] for x in xs)}/{len(xs)} | "
                       f"{f'{int(statistics.median(lats)):,}' if lats else '-'} | {f'{min(lats):,}' if lats else '-'} |")
        out.append("")
        idents = []
        for r in (1, 2, 3):
            a, b = rows.get(f"ctl_cc_{v}_r{r}"), rows.get(f"ctl_ds_{v}_r{r}")
            if a and b and a["prompt"] and b["prompt"]:
                idents.append(f"r{r}: {'identical' if a['prompt'] == b['prompt'] else 'DIFFERENT'}")
            else:
                idents.append(f"r{r}: n/a")
        out += [f"First flash prompt, Claude vs DeepSeek (temp paths normalised): {'; '.join(idents)}", ""]
    (CC / "controlled" / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()

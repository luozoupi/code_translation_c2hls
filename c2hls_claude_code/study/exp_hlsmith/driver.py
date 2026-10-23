#!/usr/bin/env python3
"""Two experiments on the contribution of HLSmith's agentic flow (supervisor's question, 2026-10-20). Serial Opus queue.

Reuses agentic_ablation/serial_driver.py (usage-limit probe and wait, invalid-run rerun, timeout escalation,
192G OOM rerun) with its own queue, state and log.

A. Plain HLSmith on the fixed bench (C = A*B^T, C poisoned), mm 1024^3, Opus 5.5 with no tools, every ladder rung
   that so far only had plain results on the old bench: pl_<rung>_cc_r<1..3>, rung in noskills, skills, dspredo,
   refneu, refnf, refv2, win7080 (same --export as agentic_ablation/submit.sh). Paired with the ag_<rung>_r<n> runs.
   (The DeepSeek plain runs pl_<rung>_ds_r<n> are submitted directly by submit_ds.sh; they use no Claude.)
B. Agentic HLSmith on the 58-bench kernels where the plain Opus Refined + Res.-Req run stayed at the baseline
   (bench_wide_runs/results_cc.csv, speedup 1.0): agb_<bench>_r1, same flow, floors and gates as bw_<bench>_cc
   (bench_wide_runs/params.py), with the agent shim. The shim is bench-specific (it copies the bench header and
   testbench into the agent workspace), so the driver starts one agent shim per bench on AGENT_PORT before
   submitting, and stops it after. machsuite_backprop is left out: its testbench needs support files the agent
   workspace does not copy.
C. (First; the supervisor's actual question.) Claude Code's own agent mode with HLSmith's knowledge and requirement but
   none of HLSmith's flow: agentfw variant `hlsm` (agentfw/new_task.py) = the free Claude Code agent with base tools,
   the same 138-skill refined v2 pack and the same DSP window 6,317-7,219 as ag_win7080, the same starting kernel and
   fixed bench, 10 h cap. Tags afw_hlsm_r<1..3>, workspaces agentfw/runs/mm1024_hlsm_r<n>. Compared with ag_win7080_r<n>.
   A session stopped by a usage limit is resumed (run_agent.sh RESUME=1) after the reset.
Order: C r1-r3, then A r1 (all rungs), A r2, A r3, then B.
"""
import importlib.util, json, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CC = Path("/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code")
spec = importlib.util.spec_from_file_location("serial_driver", CC / "study/agentic_ablation/serial_driver.py")
sd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd)
spec = importlib.util.spec_from_file_location("bw_params", CC / "study/bench_wide_runs/params.py")
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)

OUT = HERE / "runs"
FB = CC / "study/bench_fixed/n1024"
INC = "/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/vitis_hls_2023.2_include"
PY311 = "python3"
AGENT_PORT = 36016
HOST = "login5"
RUNGS_A = ["noskills", "skills", "dspredo", "refneu", "refnf", "refv2", "win7080"]
BENCH_B = ["fft", "hlsfactory_cholesky", "hlsfactory_fdtd-2d", "hlsfactory_gramschmidt", "hlsfactory_jacobi-2d",
           "hlsfactory_lu", "hlsfactory_ludcmp", "lud"]
AGENT_MATCH = (r"^(FLASH MODE:|HARD REJECT —|The current HLS code already synthesizes, but the controller flagged|"
               r"You are an expert HLS engineer\. Translate and optimize)")


def rung_kv(v):
    seed = f"SEED_CODE={FB}/autosa_mm/hls_baseline.cpp"
    return {"noskills": f"FLAVOR=noskills,{seed}", "skills": f"FLAVOR=skills,{seed}",
            "dspredo": f"FLAVOR=skills,DSP_REDO=1,{seed}", "refneu": f"FLAVOR=skills,REFINED=1,REFINED_NEUTRAL=1,{seed}",
            "refnf": f"FLAVOR=skills,REFINED=1,NO_DSP_FLOOR=1,{seed}", "refv2": f"FLAVOR=skills,REFINED=1,{seed}",
            "win7080": f"FLAVOR=skills,REFINED=1,DSP_WINDOW=6317:7219,{seed}"}[v]


def export_a(rung, model, tag):
    common = (f"ALL,CONTROLLED=1,C2HLS_THINKING=enabled,C2HLS_FLASH_MAX_TOKENS=262144,CTRL_MIN_MAX_TOKENS=262144,"
              f"OUT_ROOT={OUT},C2HLS_COMPILE_INCLUDE_PATHS={INC},BENCH_ROOT={FB}")
    url = (CC / ("shim_url.txt" if model == "cc" else "controlled/ds_proxy_url.txt")).read_text().strip()
    m = "claude-opus-5-5" if model == "cc" else "deepseek-v4-flash"
    return f"{common},{rung_kv(rung)},MODEL={m},CLAUDE_CODE_SHIM_URL={url},TAG={tag}"


def export_b(bench, tag):
    kv = bw.export(bench, "cc", tag).split(",")
    kv = [x for x in kv if not x.startswith(("CLAUDE_CODE_SHIM_URL=", "OUT_ROOT="))]
    kv += [f"OUT_ROOT={OUT}", f"CLAUDE_CODE_SHIM_URL=http://{HOST}:{AGENT_PORT}/v1", "CC_LLM_TIMEOUT=50400"]
    return ",".join(kv)


QUEUE = ([f"afw_hlsm_r{r}" for r in (1, 2, 3)] + [f"pl_{v}_cc_r{r}" for r in (1, 2, 3) for v in RUNGS_A]
         + [f"agb_{b}_r1" for b in BENCH_B])
# C r4 (added 2026-10-22): the third budgeted C run (r1 ran unbounded); workspace pre-created from tasks/mm1024_b12.
# It runs right after the A job the driver was waiting on when it was added.
QUEUE.insert(QUEUE.index("pl_refv2_cc_r2") + 1, "afw_hlsm_r4")
AFW = CC / "study/agentfw"
AFW_HOURS = 10


def run_afw(tag, st):
    """One free-agent session (run_agent.sh on this login node); resume it after usage-limit stops."""
    rec = st.setdefault(tag, {"resumes": 0})
    ws = AFW / f"runs/mm1024_hlsm_r{tag[-1]}"
    if not ws.exists():
        subprocess.run([sys.executable, str(AFW / "new_task.py"), str(AFW / "tasks/mm1024"), str(ws), "hlsm"], check=True)
    resume = (ws / "transcript.jsonl").exists()
    while True:
        sd.wait_until_usable()
        env = dict(__import__("os").environ, RESUME="1" if resume else "0")
        sd.log(f"{tag}: {'resuming' if resume else 'starting'} agent session in {ws} ({AFW_HOURS} h cap)")
        rec["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        sd.save_state(st)
        subprocess.run(["bash", str(AFW / "run_agent.sh"), str(ws), str(AFW_HOURS)], env=env)
        outs = sorted(ws.glob("transcript*.jsonl"), key=lambda p: p.stat().st_mtime)
        tail = outs[-1].read_text().splitlines()[-5:] if outs else []
        hit = [l for l in tail if '"type":"result"' in l.replace(" ", "") and sd.LIMIT_RE.search(l)]
        if hit and rec["resumes"] < 6:
            rec["resumes"] += 1
            sd.log(f"{tag}: session stopped by a usage limit; resume {rec['resumes']} after the reset")
            sd.wait_until_usable(hit[-1][:300])
            resume = True
            continue
        rec["final"] = f"exit {(ws / 'exit_code').read_text().strip() if (ws / 'exit_code').exists() else '?'}"
        sd.log(f"{tag}: agent session ended ({rec['final']}); workspace {ws}")
        return


def base_argv(tag):
    if tag.startswith("pl_"):
        _, v, m, _r = tag.split("_")
        return [f"--job-name={tag}", "--mem=64G", f"--export={export_a(v, m, tag)}"]
    bench = re.match(r"agb_(.+)_r\d$", tag).group(1)
    return [f"--job-name={tag}", "--mem=128G", f"--export={export_b(bench, tag)}"]


_shim = {"proc": None, "bench": None}


def agent_shim(bench):
    """Start the agent shim for this bench on AGENT_PORT (stopping the previous bench's shim)."""
    if _shim["bench"] == bench and _shim["proc"] and _shim["proc"].poll() is None:
        return
    stop_shim()
    meta = json.loads((CC / f"study/bench_wide/{bench}/metadata.json").read_text())
    logdir = CC / f"logs/shim_agent_exp_{bench}"
    if str(logdir) not in map(str, sd.SHIM_LOGS):
        sd.SHIM_LOGS.append(logdir)
    cmd = [PY311, str(CC / "claude_code_shim.py"), "--port", str(AGENT_PORT), "--workers", "2", "--log-dir", str(logdir),
           "--agent-match", AGENT_MATCH, "--agent-bench-dir", str(CC / f"study/bench_wide/{bench}"),
           "--agent-header", meta["header_file"], "--agent-top", meta["hls_top"], "--agent-max-checks", "12",
           "--agent-csim-time", "01:00:00", "--agent-csynth-time", "01:00:00",
           "--claude-bin", "/pc2/users/h/haqc2/.local/bin/claude"]
    with open(HERE / f"shim_{bench}.out", "w") as fp:
        _shim["proc"] = subprocess.Popen(cmd, cwd=CC, stdout=fp, stderr=subprocess.STDOUT, start_new_session=True)
    _shim["bench"] = bench
    time.sleep(10)
    sd.log(f"agent shim for {bench} on {HOST}:{AGENT_PORT} (pid {_shim['proc'].pid}, header {meta['header_file']}, "
           f"top {meta['hls_top']})")


def stop_shim():
    p = _shim["proc"]
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(60)
        except subprocess.TimeoutExpired:
            p.kill()
    _shim["proc"], _shim["bench"] = None, None


_run_tag = sd.run_tag


def adopt(tag, st):
    """After a driver restart: wait on the tag's last submitted job if it is still queued or running, instead of
    submitting a duplicate. Returns True when that job settled the tag."""
    rec = st.get(tag, {})
    job = rec.get("job")
    if not job or "final" in rec:
        return False
    state = sd.sacct_state(job)
    active = ("PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING")
    if state not in active:
        return False
    since = time.time()
    sd.log(f"{tag}: driver restarted; adopting job {job} ({state}) at {sd.TIERS[rec['tier']]}")
    while state in active + ("?",):
        time.sleep(120)
        state = sd.sacct_state(job)
    if sd.limit_hits(since):
        sd.log(f"{tag}: job {job} {state}, but a Claude call hit the usage limit; run invalid, will rerun")
        sd.move_runs(tag, "_invalid_usage_limit")
        return False
    if state.startswith("TIMEOUT"):
        sd.move_runs(tag, "_timeout")
        if rec["tier"] >= 2:
            rec["final"] = "failed: timeout at 6 h"
            sd.log(f"{tag}: TIMEOUT at 6 h; counted as failed")
            return True
        rec["tier"] += 1
        sd.log(f"{tag}: TIMEOUT; rerun at {sd.TIERS[rec['tier']]}")
        return False
    if state.startswith("OUT_OF_MEMORY") and not rec["mem"]:
        sd.move_runs(tag, "_oom")
        rec["mem"] = "192G"
        sd.log(f"{tag}: OUT_OF_MEMORY; rerun at 192G")
        return False
    rec["final"] = state
    sd.log(f"{tag}: job {job} ended {state}")
    return True


def run_tag(tag, st):
    if tag.startswith("afw_"):
        return run_afw(tag, st)
    if adopt(tag, st):
        return
    sd.save_state(st)
    if tag.startswith("agb_"):
        st.setdefault(tag, {"tier": 1, "mem": None, "tries": 0})   # agent runs start at the 3 h tier
        agent_shim(re.match(r"agb_(.+)_r\d$", tag).group(1))
    try:
        _run_tag(tag, st)
    finally:
        if tag.startswith("agb_"):
            stop_shim()


sd.QUEUE = QUEUE
sd.RUNS = OUT
sd.STATE = HERE / "driver_state.json"
sd.LOG = HERE / "driver.log"
sd.base_argv = base_argv
sd.run_tag = run_tag
sd.SHIM_LOGS = sd.SHIM_LOGS + [CC / "logs/shim_20261020_login2"]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sd.main()
    sd.log("driver done")

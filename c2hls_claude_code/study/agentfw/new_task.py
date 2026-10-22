#!/usr/bin/env python3
"""Create an agent workspace: new_task.py <task dir> <workspace> <variant: base|know|team|hlsm>

hlsm (2026-10-20, supervisor's question: HLSmith's agentic flow vs Claude Code's own agent mode): base tools only (no
web, no subagents), plus exactly what HLSmith's window rung (ag_win7080) gives the model and nothing of its flow:
the same 138-skill refined v2 pack (knowledge_hlsm/, copied from that run's skills_source.json), the same DSP window
6,317-7,219 (70-80%) as a hard requirement, and the same starting kernel (the bench's hls_baseline.cpp).
With task.json "budget" (task mm1024_b12): 12 Vitis submissions, one at a time, 1 h each, 3 h session: the allowance of
HLSmith's agent session (12 blocking hls_check.sh calls, 1 h csynth limit, inside a 3 h job)."""
import json, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
src, ws, variant = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
assert variant in ("base", "know", "team", "hlsm")
ws.mkdir(parents=True, exist_ok=False)
for f in src.iterdir():
    shutil.copy(f, ws / f.name)
t = json.loads((ws / "task.json").read_text())
shutil.copy(ws / (t.get("seed") if variant == "hlsm" and t.get("seed") else t["reference"]), ws / "kernel.cpp")
(ws / "bin").mkdir()
shutil.copy(HERE / "tools" / "hls", ws / "bin" / "hls")          # copies, not symlinks: Claude Code
shutil.copy(HERE / "tools" / "hls_eval_job.py", ws / "bin" / "hls_eval_job.py")   # blocks paths outside the workspace
if variant in ("know", "team"):
    shutil.copytree(HERE / "knowledge", ws / "knowledge")
if variant == "hlsm":
    shutil.copytree(HERE / "knowledge_hlsm", ws / "knowledge")
dev = t["device"]
know = """
## Knowledge you may use
- `knowledge/` holds reference material: HLS optimization skill packs (JSON), notes on true DATAFLOW / ping-pong and how
  to verify it from csynth module reports, and AutoSA systolic-array material. Read what helps; nothing is required.
- You may search the web (WebSearch, WebFetch) for Vitis HLS documentation and published FPGA designs.
""" if variant in ("know", "team") else ""
if variant == "hlsm":
    lo, hi = t["dsp_window"]
    bud = t.get("budget") or {}
    budget_txt = (f"\n- Budget: at most {bud['checks']} `bin/hls submit` calls (csim or csynth) in total, one job at a time "
                  f"(wait for a job before submitting the next), each limited to 1 h with no automatic retry, and "
                  f"{bud['hours']} h of session time. Plan your experiments within it." if bud else "")
    know = f"""
## Knowledge and resource requirement
- `knowledge/hls_skills_refined_v2.json` is a pack of HLS optimization skills (each: when it applies, what to do, how to
  check it). Use what helps.
- Resource requirement: the final design must use between {lo:,} and {hi:,} DSP (70-80% of the device's DSP), with every
  other resource strictly below 100%. A design outside this window does not count, however fast it is.{budget_txt}
"""
team = """
## Helpers
You may delegate self-contained subtasks to subagents (Agent tool): e.g. one explores an alternative architecture, one
reviews a report. You decide whether that helps. Subagents use the same tools and rules.
""" if variant == "team" else ""
(ws / "CLAUDE.md").write_text(f"""# Task: {t['name']}

{t['brief']}

Target: AMD Alveo U280, part `{t['part']}`, clock {t['clock_ns']} ns, Vitis HLS 2023.2. Device resources: DSP {dev['dsp']},
LUT {dev['lut']}, FF {dev['ff']}, BRAM_18K {dev['bram_18k']}, URAM {dev['uram']}.

## Goal
Write `kernel.cpp` (in this directory) with the lowest worst-case csynth latency (cycles) that
1. is correct: `hls submit csim` must end with `correct=True` (the testbench checks every output; C is pre-filled with
   garbage, so C must be overwritten, not accumulated into);
2. synthesizes, with every resource strictly below 100% of the device{" and DSP inside the window below" if variant == "hlsm" else ""};
3. keeps the top function and its argument types exactly as in `{t['reference']}` (the reference implementation).
You choose the architecture, the process, and when to stop. There is no fixed recipe or step order.

## Tools (run with Bash: `bin/hls ...`)
- `bin/hls submit csim|csynth --tag NAME` snapshots `kernel.cpp` to `exp/NAME/` and submits a Slurm job.
- `bin/hls wait JOB` waits up to 9 minutes; call again while it is still running. Csynth of a large design can take an
  hour or more; the wall time escalates 1 h -> 3 h -> 6 h automatically on timeout.
- `bin/hls result NAME`: csim verdict or csynth summary. `bin/hls diag NAME`: does the design really overlap load,
  compute and store (dataflow / ping-pong diagnosis from the module-level csynth reports, not the DATAFLOW label).
- `bin/hls floor DSP`: compute floor (total multiply-adds / (DSP/5)) to judge how close a design is to its DSP's limit.
- `bin/hls list`: experiments and their state.
You can run several csim/csynth jobs at once. Do not run Vitis or g++ yourself; use `bin/hls`.
Shell use: one command per Bash call (no `;`, `&&` or pipes into other programs); everything you need is inside this
directory. Use the Read/Grep/Glob tools to inspect files.
{know}{team}
## Finish
When you stop: leave the best correct design in `kernel.cpp`, and write `REPORT.md` with the final numbers (latency,
DSP/LUT/FF/BRAM/URAM, csim verdict, experiment name), the architecture, what you tried (each experiment: idea, result),
and what limits the design now.
""")
(ws / "variant.txt").write_text(variant + "\n")
print(ws)

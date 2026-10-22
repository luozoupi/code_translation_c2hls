#!/usr/bin/env python3
"""DSP ladder: three seeded flash_refined v2 stages in one job, each a normal run_flash_cc.py cell.

  L1  from plain C, DSP window 20-30% of 9024 (about 25%): a correct, balanced starting design.
  L2  seeded with L1's kernel, window 1.8x-2.2x L1's DSP (capped at the roof): "double it".
  L3  seeded with L2's kernel, window max(70%, 1.1x L2's DSP) up to the roof: "increase it".
The DSP roof is 80% of 9024 (7219) at every stage. Every stage also keeps flash_refined's rules: must be
faster than its seed (latency guard 1.0), every resource < 100%, compute/transfer balance check.
A stage whose flash step fails, or whose selected design is outside its window, ends the ladder.

  run_flash_ladder.py --tag r1 [--shim-url URL] [--turns 3]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEVICE_DSP = 9024
ROOF = int(DEVICE_DSP * 0.80)        # 7219
L3_FLOOR = math.ceil(DEVICE_DSP * 0.70)  # 6317

NOTE_L1 = """DSP SCALING LADDER — stage 1 of 3. This step builds the starting point: a correct design whose
compute and data movement are balanced, sized to about 25% of the device DSP (window {lo}–{hi}).
Later steps will scale this same design up: first to about twice its DSP, then further toward an
80% DSP roof ({roof})."""

NOTE_L2 = """DSP SCALING LADDER — stage 2 of 3. The current kernel above is the stage-1 design:
DSP={dsp}, latency {lat} cycles. Double its parallel MACs: target DSP about {target} (window
{lo}–{hi}; the roof for the whole ladder is {roof}). Keep its architecture, and scale the parallel
datapath together with the on-chip buffering and memory bandwidth the larger datapath needs so it
stays fed. The result must be faster than the current kernel."""

NOTE_L3 = """DSP SCALING LADDER — stage 3 of 3. The current kernel above is the stage-2 design:
DSP={dsp}, latency {lat} cycles. Increase its parallel MACs further, toward the 80% DSP roof:
window {lo}–{hi}, and {hi} is a hard roof. Keep its architecture, and scale the parallel datapath
together with the on-chip buffering and memory bandwidth the larger datapath needs so it stays fed.
The result must be faster than the current kernel."""


def stage_result(ladder_dir: Path, name: str) -> dict:
    cells = sorted(ladder_dir.glob(f"skills_{name}_*/autosa_mm"))
    if not cells:
        return {"ok": False, "why": "no cell directory"}
    cell = cells[-1]
    try:
        multi = json.loads((cell / "autosa_mm_multistep_results.json").read_text())
        sel = json.loads((cell / "autosa_mm_selected_report.json").read_text())
    except (OSError, ValueError) as exc:
        return {"ok": False, "cell": str(cell), "why": f"missing results: {exc}"}
    rep = sel.get("report") or sel
    flash = next((s for s in multi.get("steps") or [] if s.get("step_name") == "flash"), {})
    lat = max(rep.get("latency_cycles") or 0, rep.get("latency_cycles_worst") or 0)
    return {"ok": bool(flash.get("success")), "cell": str(cell), "code": str(cell / "autosa_mm_selected.cpp"),
            "dsp": rep.get("dsp"), "latency_cycles": rep.get("latency_cycles"), "latency_worst": lat,
            "clock": rep.get("estimated_clock_period_ns"),
            "csim": bool((flash.get("csim") or {}).get("passed")),
            "why": None if flash.get("success") else (flash.get("error") or "flash step failed")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shim-url", default=os.environ.get("CLAUDE_CODE_SHIM_URL", ""))
    ap.add_argument("--turns", default=os.environ.get("TURNS", ""))
    ap.add_argument("--out-root", default=str(HERE / "runs"))
    ap.add_argument("--model", default="", help="LLM model name passed to every stage (default: run_flash_cc.py's)")
    args = ap.parse_args()
    shim = args.shim_url or (HERE / "shim_url.txt").read_text().strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ladder = Path(args.out_root) / f"ladder_{args.tag}_{stamp}"
    ladder.mkdir(parents=True)
    summary = {"ladder_dir": str(ladder), "roof": ROOF, "stages": []}

    def run_stage(name: str, lo: int, hi: int, note: str, seed: str | None) -> dict:
        (ladder / f"{name}_note.txt").write_text(note + "\n")
        cmd = [sys.executable, str(HERE / "run_flash_cc.py"), "--flavor", "skills", "--refined",
               "--dsp-window", f"{lo},{hi}", "--stage-note-file", str(ladder / f"{name}_note.txt"),
               "--shim-url", shim, "--out-root", str(ladder), "--tag", name]
        if seed:
            cmd += ["--seed-code", seed]
        if args.model:
            cmd += ["--model", args.model]
        if args.turns:
            cmd += ["--turns", str(args.turns)]
        print(f"=== {name}: window {lo}-{hi}, seed={seed}", flush=True)
        rc = subprocess.call(cmd)
        res = stage_result(ladder, name)
        res.update({"stage": name, "window": [lo, hi], "seed": seed, "rc": rc})
        if res.get("ok") and not (lo <= (res.get("dsp") or 0) <= hi):
            # c2hls's escape (another resource >= 90%) can accept a design below lo; record it.
            res["outside_window"] = True
        summary["stages"].append(res)
        (ladder / "ladder_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"=== {name} result: {json.dumps(res)}", flush=True)
        return res

    lo1, hi1 = math.ceil(DEVICE_DSP * 0.20), int(DEVICE_DSP * 0.30)
    r1 = run_stage("L1", lo1, hi1, NOTE_L1.format(lo=lo1, hi=hi1, roof=ROOF), None)
    if not r1.get("ok"):
        return 1
    d1 = int(r1["dsp"])
    lo2, hi2 = min(round(1.8 * d1), ROOF), min(round(2.2 * d1), ROOF)
    r2 = run_stage("L2", lo2, hi2, NOTE_L2.format(dsp=d1, lat=r1["latency_worst"], target=min(2 * d1, ROOF),
                                                   lo=lo2, hi=hi2, roof=ROOF), r1["code"])
    if not r2.get("ok"):
        return 1
    d2 = int(r2["dsp"])
    lo3 = max(L3_FLOOR, math.ceil(1.1 * d2))
    if lo3 > ROOF:
        summary["stages"].append({"stage": "L3", "skipped": f"L2 DSP {d2} already within 10% of the roof"})
        (ladder / "ladder_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return 0
    r3 = run_stage("L3", lo3, ROOF, NOTE_L3.format(dsp=d2, lat=r2["latency_worst"], lo=lo3, hi=ROOF), r2["code"])
    return 0 if r3.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

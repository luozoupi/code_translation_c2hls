#!/usr/bin/env python3
"""Cosim speed test: run ../run_cosim_one.py unchanged, plus two optional knobs, and time the sim.sh step alone.

  run_speed.py NAME KERNEL.cpp --size N        env: XELAB_OPT (e.g. "--O3", added to the xelab line of run_xsim.sh)
cosim_design -O is passed through C2HLS_COSIM_EXTRA_ARGS by the sbatch script. Results land in ../NAME/ as usual;
the sim.sh wall time is written to ../NAME/sim_wall_s.txt.
"""
import os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, "/scratch/asa582/workspaces/code_translation_c2hls")
import hls_eval
import run_cosim_one

opt = os.environ.get("XELAB_OPT", "").strip()
orig_patch = hls_eval._patch_run_xsim_xelab_mt_off


def patch(work_dir):
    out = orig_patch(work_dir)
    if opt:
        for sh in Path(work_dir).rglob("run_xsim.sh"):
            t = sh.read_text()
            t2, n = re.subn(r"(\bxelab)(\s)", rf"\1 {opt}\2", t, count=1)
            if n:
                sh.write_text(t2)
                print(f"SPEEDTEST: xelab {opt} in {sh}", flush=True)
    return out


hls_eval._patch_run_xsim_xelab_mt_off = patch
orig_run = hls_eval._run_vitis_cmd
name = sys.argv[1]


def timed(cmd, timeout):
    t0 = time.time()
    r = orig_run(cmd, timeout)
    if "sim.sh" in cmd:
        (HERE.parent / name / "sim_wall_s.txt").write_text(f"{time.time() - t0:.0f}\n")
    return r


hls_eval._run_vitis_cmd = timed
sys.argv = [sys.argv[0]] + sys.argv[1:]
raise SystemExit(run_cosim_one.main())

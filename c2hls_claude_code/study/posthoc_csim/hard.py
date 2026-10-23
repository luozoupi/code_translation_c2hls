#!/usr/bin/env python3
"""Hardened-bench check (bench_hard/verify_hard.check, g++ -O2) of the post-hoc kernels and team v3. Writes hard.csv."""
import csv, glob, sys
from pathlib import Path
sys.path.insert(0, "/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/bench_hard")
import verify_hard as vh
H = Path(__file__).resolve().parent
vh.TMP.mkdir(parents=True, exist_ok=True)
ks = sorted(glob.glob(str(H / "*/*/kernel.cpp")) + glob.glob(str(H / "csynth/*/*/kernel.cpp"))) + [str(vh.CC / "study/agentfw/runs/mm1024_team_r1/exp_post/v3_csynth/kernel.cpp"),
                                                     str(vh.CC / "study/agentfw/runs/mm1024_team_r1/exp_post/v2_csynth/kernel.cpp")]
rows = [vh.check((k, None)) for k in ks if "/ws/" not in k]
with open(H / "hard.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["kernel", "verdict"]); w.writeheader(); [w.writerow({k: r[k] for k in ("kernel", "verdict")}) for r in rows]
for r in rows: print(r["verdict"], r["kernel"])

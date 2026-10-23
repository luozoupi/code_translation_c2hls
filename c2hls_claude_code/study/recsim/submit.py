#!/usr/bin/env python3
"""Strict csim of existing passing kernels on the fixed bench (C = A*B^T, C poisoned). One Slurm job per kernel.

Usage: submit.py [--dry]   -> writes study/recsim/<tag>/ and jobs.txt; never touches the original run dirs.
"""
import glob, json, re, subprocess, sys
from pathlib import Path
CC = Path("/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code")
BENCH = CC / "study/bench_fixed/n1024/autosa_mm"
TEMPLATE = CC / "verify/ctl_cc_zs_r1"          # run_eval.py + sbatch_csim.sh (strict n1024 csim harness)
HERE = Path(__file__).resolve().parent


def kernels():
    out = {}
    for c in sorted(glob.glob(str(CC / "runs_controlled/*_ctl_*_r[123]_*/autosa_mm"))):
        tag = re.search(r"(ctl_\w+?_r\d)_", c).group(1)
        out[tag] = Path(c)                       # latest stamp wins
    for s in ("L1", "L2", "L3"):
        out[f"agent_ladder_r3_{s}"] = Path(glob.glob(str(CC / f"runs/ladder_agent_r3_*/skills_{s}_*/autosa_mm"))[0])
    keep = {}
    for tag, c in out.items():
        try:
            ms = json.loads((c / "autosa_mm_multistep_results.json").read_text())
        except Exception:
            continue
        f = next((s for s in ms.get("steps", []) if s.get("step_name") == "flash"), {})
        if f.get("success") and (c / "autosa_mm_selected.cpp").exists():
            keep[tag] = c
    return keep


def main():
    dry = "--dry" in sys.argv
    jobs = []
    for tag, c in kernels().items():
        d = HERE / tag
        d.mkdir(exist_ok=True)
        (d / "kernel.cpp").write_text((c / "autosa_mm_selected.cpp").read_text())
        (d / "kernel.h").write_text((BENCH / "kernel.h").read_text())
        (d / "testbench.cpp").write_text((BENCH / "testbench.cpp").read_text())
        (d / "run_eval.py").write_text((TEMPLATE / "run_eval.py").read_text())
        (d / "SOURCE").write_text(str(c / "autosa_mm_selected.cpp") + "\n")
        sb = (TEMPLATE / "sbatch_csim.sh").read_text().replace(str(TEMPLATE), str(d))
        sb = (sb.replace("--cpus-per-task=8", "--cpus-per-task=2").replace("--mem=64G", "--mem=16G")
                .replace("--time=24:00:00", "--time=01:00:00").replace("C2HLS_CSIM_TIMEOUT=86400", "C2HLS_CSIM_TIMEOUT=3300")
                .replace("--job-name=verify_csim", "--job-name=recsim"))
        (d / "sbatch_csim.sh").write_text(sb)
        if not dry:
            j = subprocess.run(["sbatch", "--parsable", str(d / "sbatch_csim.sh")], capture_output=True, text=True).stdout.strip()
            jobs.append(f"{tag} {j}")
    if jobs:
        with open(HERE / "jobs.txt", "a") as fp:
            fp.write("\n".join(jobs) + "\n")
    print(len(kernels()), "kernels;", len(jobs), "submitted")


if __name__ == "__main__":
    main()

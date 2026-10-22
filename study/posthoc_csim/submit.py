#!/usr/bin/env python3
"""Post-hoc Vitis csim (fixed bench, ws/) of every candidate in candidates.json, both sides, same job as the agentfw tool.
One dir per kernel: <run>/<name>/{kernel.cpp, source.txt, job.sh}; results land there (csim_summary.txt). Writes submitted.txt."""
import json, subprocess
from pathlib import Path
H = Path(__file__).resolve().parent
EVAL = "/scratch/asa582/workspaces/c2hls_claude_code/study/agentfw/tools/hls_eval_job.py"
sub = []
for c in json.loads((H / "candidates.json").read_text()):
    d = H / c["run"] / c["name"].replace("_csynth.txt", "")
    d.mkdir(parents=True, exist_ok=True)
    (d / "kernel.cpp").write_bytes(Path(c["kernel"]).read_bytes())
    (d / "source.txt").write_text(json.dumps(c, indent=1) + "\n")
    (d / "job.sh").write_text(f"""#!/bin/bash
# Post-hoc csim, 2026-10-22, of {c['kernel']}: synthesized in its session, never csim'd there.
#SBATCH --account=def-zhenman
#SBATCH --job-name=posthoc_csim
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output={d}/csim-%j.out
#SBATCH --error={d}/csim-%j.err
set -eo pipefail
source /scratch/asa582/workspaces/dse_v3_claude/env.sh
dse_v3_source_vitis
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID C2HLS_SITE C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM DSE_V3_TRIAL_ROOT
export C2HLS_CSIM_TIMEOUT=10000 C2HLS_SYNTH_TIMEOUT=10000
"$PY" {EVAL} csim {d} {H / 'ws/task.json'}
""")
    j = subprocess.run(["sbatch", "--parsable", str(d / "job.sh")], capture_output=True, text=True).stdout.strip()
    sub.append(f"{c['run']} {c['name']} {j}")
(H / "submitted.txt").write_text("\n".join(sub) + "\n")
print("\n".join(sub))

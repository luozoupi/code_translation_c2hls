#!/usr/bin/env python3
"""Post-hoc Vitis csynth of candidates_csynth.json (csim'd in session, never synthesized), both sides; same job as the
agentfw tool (hls_eval_job.py csynth, allow_compile_jobs=False, node-local work dir). Dirs: csynth/<run>/<name>/."""
import json, subprocess
from pathlib import Path
H = Path(__file__).resolve().parent
EVAL = "/scratch/hpc-prf-llmfpga/asa582/projects/workspaces/c2hls_claude_code/study/agentfw/tools/hls_eval_job.py"
sub = []
for c in json.loads((H / "candidates_csynth.json").read_text()):
    if c["already_submitted"]:
        continue
    d = H / "csynth" / c["run"] / f"{c.get('call', 'exp')}_{c['name']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "kernel.cpp").write_bytes(Path(c["kernel"]).read_bytes())
    (d / "source.txt").write_text(json.dumps(c, indent=1) + "\n")
    (d / "job.sh").write_text(f"""#!/bin/bash
# Post-hoc csynth, 2026-10-22, of {c['kernel']}: csim passed in its session, never synthesized there.
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --job-name=posthoc_csynth
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output={d}/csynth-%j.out
#SBATCH --error={d}/csynth-%j.err
set -eo pipefail
source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/dse_v3_claude/env.sh
dse_v3_source_vitis
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID C2HLS_SITE C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM DSE_V3_TRIAL_ROOT
export C2HLS_CSIM_TIMEOUT=200000 C2HLS_SYNTH_TIMEOUT=200000
"$PY" {EVAL} csynth {d} {H / 'ws/task.json'}
""")
    j = subprocess.run(["sbatch", "--parsable", str(d / "job.sh")], capture_output=True, text=True)
    sub.append(f"{c['run']} {c.get('call', '')} {c['name']} {j.stdout.strip() or j.stderr.strip()}")
(H / "submitted_csynth.txt").write_text("\n".join(sub) + "\n")
print("\n".join(sub))

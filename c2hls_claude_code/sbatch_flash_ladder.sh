#!/bin/bash
# DSP ladder (flash_refined v2, seeded stages L1 25% -> L2 2x -> L3 toward the 80% roof) with Claude Code as the LLM.
#   sbatch --export=ALL,TAG=r1 sbatch_flash_ladder.sh
#SBATCH --job-name=flash_ladder
#SBATCH --account=def-zhenman
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --chdir=/scratch/asa582/workspaces/c2hls_claude_code
#SBATCH --output=/scratch/asa582/workspaces/c2hls_claude_code/logs/flash_ladder-%j.out
#SBATCH --error=/scratch/asa582/workspaces/c2hls_claude_code/logs/flash_ladder-%j.err
set -euo pipefail
D=/scratch/asa582/workspaces/c2hls_claude_code
source /scratch/asa582/workspaces/dse_v3_claude/env.sh
dse_v3_source_vitis
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID DSE_V3_TRIAL_ROOT C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM
export C2HLS_CSIM_TIMEOUT=86400 C2HLS_SYNTH_TIMEOUT=86400
export CC_LLM_TIMEOUT="${CC_LLM_TIMEOUT:-50400}"
"${PY}" "$D/run_flash_ladder.py" --tag "${TAG:?TAG}" --shim-url "${CLAUDE_CODE_SHIM_URL:-$(cat "$D/shim_url.txt")}" ${TURNS:+--turns "$TURNS"} ${MODEL:+--model "$MODEL"} ${OUT_ROOT:+--out-root "$OUT_ROOT"}

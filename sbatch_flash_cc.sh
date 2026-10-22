#!/bin/bash
# One c2hls flash cell with Claude Code as the LLM (via the login-node shim).
#   sbatch --export=ALL,FLAVOR=skills,BENCH=autosa_mm sbatch_flash_cc.sh
#SBATCH --job-name=flash_cc
#SBATCH --account=def-zhenman
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --chdir=/scratch/asa582/workspaces/c2hls_claude_code
#SBATCH --output=/scratch/asa582/workspaces/c2hls_claude_code/logs/flash_cc-%j.out
#SBATCH --error=/scratch/asa582/workspaces/c2hls_claude_code/logs/flash_cc-%j.err
set -euo pipefail
D=/scratch/asa582/workspaces/c2hls_claude_code
# PY, PYTHONPATH, temp dirs and the fir Vitis container; DSE-v3 workspace switches are cleared below.
source /scratch/asa582/workspaces/dse_v3_claude/env.sh
dse_v3_source_vitis
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID DSE_V3_TRIAL_ROOT C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM
export C2HLS_CSIM_TIMEOUT=86400 C2HLS_SYNTH_TIMEOUT=86400
# File-count quota on /scratch: Vitis work dirs go to node-local storage ($SLURM_TMPDIR, or $HLS_LOCAL_ROOT).
# On exit only reports, sources, json and logs are copied to /scratch/asa582/tmp/c2hls_kept/<TAG>/<bench>/<dir>
# (the run's recorded work_dir under $SLURM_TMPDIR/c2hls/ maps to that path).
LOCAL=${HLS_LOCAL_ROOT:-${SLURM_TMPDIR:-}}
if [ -n "$LOCAL" ]; then
  export C2HLS_TMP_ROOT=$LOCAL/c2hls C2HLS_TMP_RUN=${TAG:-job${SLURM_JOB_ID:-x}}
  mkdir -p "$C2HLS_TMP_ROOT"
  keep() { "$D/study/jobs/keep_copy.sh" "$C2HLS_TMP_ROOT" /scratch/asa582/tmp/c2hls_kept || true; }
  trap keep EXIT
  trap 'keep; exit 143' TERM
fi
# Agent sessions can run for hours (csynth checks); give c2hls's LLM client room to wait.
export CC_LLM_TIMEOUT="${CC_LLM_TIMEOUT:-50400}"
"${PY}" "$D/run_flash_cc.py" --flavor "${FLAVOR:-skills}" --bench "${BENCH:-autosa_mm}" \
  --shim-url "${CLAUDE_CODE_SHIM_URL:-$(cat "$D/shim_url.txt")}" ${TURNS:+--turns "$TURNS"} ${TAG:+--tag "$TAG"} \
  ${SKILLS_JSON:+--skills-json "$SKILLS_JSON"} ${POST_FLASH:+--post-flash} ${DSP_REDO:+--dsp-redo} ${REFINED:+--refined} ${NO_DSP_FLOOR:+--no-dsp-floor} ${REFINED_NEUTRAL:+--refined-neutral} ${REFINED_VERSION:+--refined-version "$REFINED_VERSION"} \
  ${DSP_WINDOW:+--dsp-window "$DSP_WINDOW"} ${STAGE_NOTE_FILE:+--stage-note-file "$STAGE_NOTE_FILE"} ${SEED_CODE:+--seed-code "$SEED_CODE"} ${OUT_ROOT:+--out-root "$OUT_ROOT"} ${MODEL:+--model "$MODEL"} ${BENCH_ROOT:+--bench-root "$BENCH_ROOT"} ${CONTROLLED:+--controlled} ${ZS_SKILLS_BLOCK:+--zs-skills-block "$ZS_SKILLS_BLOCK"}

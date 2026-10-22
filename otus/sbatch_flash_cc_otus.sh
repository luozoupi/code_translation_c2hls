#!/bin/bash
# Otus copy of ../sbatch_flash_cc.sh: one c2hls flash cell with Claude Code (or DeepSeek) as the LLM.
#   cd c2hls_claude_code && sbatch --export=ALL,FLAVOR=skills,BENCH=autosa_mm otus/sbatch_flash_cc_otus.sh
# Submit from c2hls_claude_code: the paths come from $SLURM_SUBMIT_DIR (logs/ is relative to it). Native Vitis 2023.2
# (C2HLS_VITIS_SETTINGS) instead of the Fir container; the dse_v3_claude env.sh settings are inlined; the node-local
# dir falls back to /tmp/$USER/$SLURM_JOB_ID when $SLURM_TMPDIR is unset. Can replace sbatch_flash_cc.sh on Otus.
#SBATCH --job-name=flash_cc
#SBATCH --account=hpc-prf-llmfpga
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/flash_cc-%j.out
#SBATCH --error=logs/flash_cc-%j.err
set -euo pipefail
D=${CC_DIR:-$SLURM_SUBMIT_DIR}
[ -f "$D/run_flash_cc.py" ] || { echo "submit from c2hls_claude_code (or set CC_DIR)"; exit 1; }
W=$(dirname "$D"); B=$(dirname "$W")              # workspaces, projects root
# what dse_v3_claude/env.sh set on Fir
export C2HLS_ROOT=$W/code_translation_c2hls
export PYTHONPATH="${C2HLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PY=${PY:-$B/packages/c2hls-venv/bin/python}
export C2HLS_TMP_ROOT=$B/tmp/c2hls C2HLS_VITIS_USER_HOME=$B/tmp/vitis_user_home
export PATH="${HOME}/.local/bin:${PATH}"
mkdir -p "${C2HLS_TMP_ROOT}" "${C2HLS_VITIS_USER_HOME}"
# Vitis, natively
export C2HLS_VITIS_SETTINGS=${OTUS_VITIS_SETTINGS:-/opt/software/FPGA/Xilinx/Vitis/2023.2/settings64.sh}
set +u; source "$C2HLS_VITIS_SETTINGS" >/dev/null; set -u
command -v vitis-run
export C2HLS_COMPILE_INCLUDE_PATHS=${C2HLS_COMPILE_INCLUDE_PATHS_OTUS:-$XILINX_HLS/include}
unset C2HLS_USE_CONTAINER XILINX_SIF C2HLS_SITE
unset C2HLS_DSE_V3 C2HLS_DSE_V3_HARNESS C2HLS_DSE_V4 C2HLS_DSE_V2_GRID DSE_V3_TRIAL_ROOT C2HLS_VITIS_JOBS C2HLS_VITIS_JOBS_FROM_SLURM
export C2HLS_CSIM_TIMEOUT=86400 C2HLS_SYNTH_TIMEOUT=86400
# File-count quota: Vitis work dirs go to node-local storage; on exit only reports, sources, json and logs are copied to
# <projects>/tmp/c2hls_kept/<TAG>/<bench>/<dir>.
LOCAL=${HLS_LOCAL_ROOT:-${SLURM_TMPDIR:-${OTUS_LOCAL_TMP:-/tmp}/$USER/$SLURM_JOB_ID}}
export C2HLS_TMP_ROOT=$LOCAL/c2hls C2HLS_TMP_RUN=${TAG:-job${SLURM_JOB_ID:-x}}
mkdir -p "$C2HLS_TMP_ROOT"
keep() { "$D/study/jobs/keep_copy.sh" "$C2HLS_TMP_ROOT" "$B/tmp/c2hls_kept" || true
         [ -n "${HLS_LOCAL_ROOT:-}${SLURM_TMPDIR:-}" ] || rm -rf "$LOCAL"; }
trap keep EXIT
trap 'keep; exit 143' TERM
# Agent sessions can run for hours (csynth checks); give c2hls's LLM client room to wait.
export CC_LLM_TIMEOUT="${CC_LLM_TIMEOUT:-50400}"
"${PY}" "$D/run_flash_cc.py" --flavor "${FLAVOR:-skills}" --bench "${BENCH:-autosa_mm}" \
  --shim-url "${CLAUDE_CODE_SHIM_URL:-$(cat "$D/shim_url.txt")}" ${TURNS:+--turns "$TURNS"} ${TAG:+--tag "$TAG"} \
  ${SKILLS_JSON:+--skills-json "$SKILLS_JSON"} ${POST_FLASH:+--post-flash} ${DSP_REDO:+--dsp-redo} ${REFINED:+--refined} ${NO_DSP_FLOOR:+--no-dsp-floor} ${REFINED_NEUTRAL:+--refined-neutral} ${REFINED_VERSION:+--refined-version "$REFINED_VERSION"} \
  ${DSP_WINDOW:+--dsp-window "$DSP_WINDOW"} ${STAGE_NOTE_FILE:+--stage-note-file "$STAGE_NOTE_FILE"} ${SEED_CODE:+--seed-code "$SEED_CODE"} ${OUT_ROOT:+--out-root "$OUT_ROOT"} ${MODEL:+--model "$MODEL"} ${BENCH_ROOT:+--bench-root "$BENCH_ROOT"} ${CONTROLLED:+--controlled} ${ZS_SKILLS_BLOCK:+--zs-skills-block "$ZS_SKILLS_BLOCK"}

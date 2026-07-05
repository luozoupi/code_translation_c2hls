#!/usr/bin/env bash
# Full 28-bench multistep pipelined: noskills, csynth+csim only, 1 repair round.
#
#   ./scripts/pc2/start_multistep_noskills_full_campaign.sh --dry-run
#   ./scripts/pc2/start_multistep_noskills_full_campaign.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
STAMP="${C2HLS_MULTISTEP_FIXED_COSIM_STAMP:-$(date +%Y%m%d)_fixed_cosim_multistep_noskills}"
VARIANT="noskills"
PY="${C2HLS_PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

export PC2_SLURM_ACCOUNT="hpc-prf-llmfpga"
export PC2_MULTISTEP_FULL_WALLTIME="${PC2_MULTISTEP_FULL_WALLTIME:-48:00:00}"
export PC2_MULTISTEP_VARIANT="${VARIANT}"
export C2HLS_MULTISTEP_FIXED_COSIM_STAMP="${STAMP}"
export C2HLS_MULTISTEP_VARIANT="${VARIANT}"
# 1 initial attempt + 1 repair on compile/synth/csim failure
export C2HLS_TURNS="${C2HLS_TURNS:-2}"
export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_PIPELINED_SYNTH_WORKERS="${C2HLS_PIPELINED_SYNTH_WORKERS:-4}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"

STAMP_SUFFIX="${STAMP}_pipelined"
DATE_PREFIX="$(printf '%s' "${STAMP}" | grep -oE '[0-9]{8}' | head -1)"
JSONL_OUT="${C2HLS_ROOT}/misc/hlsfactory_fixed_cosim_multistep_${VARIANT}_u280_${DATE_PREFIX}.jsonl"
POST_LOG="${C2HLS_ROOT}/artifacts/pc2/pipelines/wait_multistep_csynth_postprocess_${VARIANT}_${DATE_PREFIX}.log"

echo "=== multistep full campaign noskills (28 benches) ==="
echo "account=${PC2_SLURM_ACCOUNT} walltime=${PC2_MULTISTEP_FULL_WALLTIME} turns=${C2HLS_TURNS} (1 repair)"
echo "cosim=off (csynth+csim only)"
echo "stamp=${STAMP} -> artifacts/pc2/multistep_fixed_cosim_${VARIANT}_${STAMP_SUFFIX}"
echo "jsonl_out=${JSONL_OUT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  C2HLS_TURNS="${C2HLS_TURNS}" C2HLS_RUN_COSIM=0 C2HLS_COSIM_REQUIRED=0 \
    "${SCRIPT_DIR}/start_multistep_fixed_cosim_pipelined.sh" \
    --variant "${VARIANT}" \
    --stamp "${STAMP}" \
    --dry-run
  echo "dry-run ok (session not submitted)"
  exit 0
fi

export PC2_FORCE_WALLTIME="${PC2_MULTISTEP_FULL_WALLTIME}"
export PC2_COMPUTE_CPUS=64
export PC2_COMPUTE_MEM=256G
SESSION_ID="multistep_pipelined_cosim_${VARIANT}"

WORKER_CMD="C2HLS_TURNS=${C2HLS_TURNS} C2HLS_RUN_COSIM=0 C2HLS_COSIM_REQUIRED=0 C2HLS_MULTISTEP_FIXED_COSIM_STAMP=${STAMP} ${PY} scripts/pc2/run_multistep_fixed_cosim_pipelined.py --pc2 --variant ${VARIANT} --stamp ${STAMP}"

"${SCRIPT_DIR}/start_session.sh" \
  --session-id "${SESSION_ID}" \
  --worker-cmd "${WORKER_CMD}" \
  --auto-stop-on-complete

account_args=(--account="${PC2_SLURM_ACCOUNT}")
post_job="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --job-name="c2hls-multistep_post_${VARIANT}" \
    --output="${C2HLS_ROOT}/artifacts/pc2/pipelines/multistep_post_${VARIANT}_${DATE_PREFIX}-%j.out" \
    --error="${C2HLS_ROOT}/artifacts/pc2/pipelines/multistep_post_${VARIANT}_${DATE_PREFIX}-%j.err" \
    "${account_args[@]}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task=2 \
    --mem=8G \
    --time=72:00:00 \
    --wrap="bash ${SCRIPT_DIR}/wait_multistep_csynth_postprocess.sh --variant ${VARIANT} --stamp ${STAMP} --output ${JSONL_OUT} >> ${POST_LOG} 2>&1"
)"

echo "submitted postprocess watcher job ${post_job}"
echo "watch: tail -f artifacts/pc2/sessions/multistep_pipelined_cosim_${VARIANT}/watch.log"

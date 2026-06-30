#!/usr/bin/env bash
# Full 28-bench multistep pipelined campaign: nav_n (90 skills, no avoids).
# Submits GPU+compute session and a postprocess watcher (JSONL + MD summary).
#
#   ./scripts/pc2/start_multistep_nav_n_full_campaign.sh --dry-run
#   ./scripts/pc2/start_multistep_nav_n_full_campaign.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
STAMP="${C2HLS_MULTISTEP_FIXED_COSIM_STAMP:-$(date +%Y%m%d)_fixed_cosim_multistep_nav_n}"
VARIANT="nav_n"
PY="${C2HLS_PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

export PC2_SLURM_ACCOUNT="hpc-prf-haqc"
export PC2_MULTISTEP_FULL_WALLTIME="${PC2_MULTISTEP_FULL_WALLTIME:-48:00:00}"
export PC2_MULTISTEP_VARIANT="${VARIANT}"
export C2HLS_MULTISTEP_FIXED_COSIM_STAMP="${STAMP}"
export C2HLS_PIPELINED_SYNTH_WORKERS="${C2HLS_PIPELINED_SYNTH_WORKERS:-4}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"

STAMP_SUFFIX="${STAMP}_pipelined"
DATE_PREFIX="$(printf '%s' "${STAMP}" | grep -oE '[0-9]{8}' | head -1)"
JSONL_OUT="${C2HLS_ROOT}/misc/hlsfactory_fixed_cosim_multistep_${VARIANT}_u280_${DATE_PREFIX}.jsonl"
POST_LOG="${C2HLS_ROOT}/artifacts/pc2/pipelines/wait_multistep_csynth_postprocess_${VARIANT}_${DATE_PREFIX}.log"

echo "=== multistep full campaign nav_n (90 skills, no avoids) ==="
echo "account=${PC2_SLURM_ACCOUNT} walltime=${PC2_MULTISTEP_FULL_WALLTIME}"
echo "stamp=${STAMP} -> artifacts/pc2/multistep_fixed_cosim_${VARIANT}_${STAMP_SUFFIX}"
echo "jsonl_out=${JSONL_OUT}"
echo "summary_md=artifacts/pc2/analysis/${STAMP_SUFFIX}/summary.md"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  "${SCRIPT_DIR}/start_multistep_fixed_cosim_pipelined.sh" \
    --variant "${VARIANT}" \
    --stamp "${STAMP}" \
    --dry-run
  echo "dry-run ok (session not submitted)"
  exit 0
fi

"${SCRIPT_DIR}/start_multistep_fixed_cosim_pipelined.sh" \
  --variant "${VARIANT}" \
  --stamp "${STAMP}"

account_args=()
if [[ -n "${PC2_SLURM_ACCOUNT}" ]]; then
  account_args=(--account="${PC2_SLURM_ACCOUNT}")
fi

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

echo "submitted postprocess watcher job ${post_job} (JSONL + summary MD on completion)"
echo "watch session: tail -f artifacts/pc2/sessions/multistep_pipelined_cosim_${VARIANT}/watch.log"

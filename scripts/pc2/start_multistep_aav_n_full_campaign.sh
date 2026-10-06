#!/usr/bin/env bash
# Full 28-bench multistep pipelined: aav_n
# (gemm_flatten_v1 + flash_no_RMW overlay, all+avoids).
# Submits GPU+compute session and postprocess watcher (JSONL + MD summary).
#
#   ./scripts/pc2/start_multistep_aav_n_full_campaign.sh --dry-run
#   ./scripts/pc2/start_multistep_aav_n_full_campaign.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
STAMP="${C2HLS_MULTISTEP_FIXED_COSIM_STAMP:-$(date +%Y%m%d)_fixed_cosim_multistep_aav_n_gf_ovl}"
VARIANT="aav_n"
BENCHES=""
TURNS="${C2HLS_TURNS:-4}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --benches) shift; BENCHES="$1"; shift ;;
    --turns) shift; TURNS="$1"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# Scrub parent-shell skill/RAG/lat-opt so variant wiring wins.
unset C2HLS_PACKAGED_SKILLS_JSON C2HLS_FLASH_SKILL_ENTRIES_JSON || true
unset C2HLS_RAG_MODE C2HLS_RAG2_OPT_CORPUS C2HLS_RAG2_REPAIR_CORPUS C2HLS_RAG_SCRAPE_CORPUS || true
unset C2HLS_LATENCY_OPT_CHAIN_FLASH C2HLS_LATENCY_OPT_CHAIN_DATAFLOW || true
export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
export C2HLS_RAG2=0
export C2HLS_POST_FLASH_LATENCY_OPT=0

export PC2_SLURM_ACCOUNT="hpc-prf-llmfpga"
export PC2_MULTISTEP_FULL_WALLTIME="${PC2_MULTISTEP_FULL_WALLTIME:-48:00:00}"
export PC2_MULTISTEP_VARIANT="${VARIANT}"
export C2HLS_MULTISTEP_FIXED_COSIM_STAMP="${STAMP}"
export C2HLS_RUN_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_PIPELINED_SYNTH_WORKERS="${C2HLS_PIPELINED_SYNTH_WORKERS:-4}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_TURNS="${TURNS}"
export C2HLS_MAX_REPAIR_ATTEMPT="${C2HLS_MAX_REPAIR_ATTEMPT:-${TURNS}}"

STAMP_SUFFIX="${STAMP}_pipelined"
DATE_PREFIX="$(printf '%s' "${STAMP}" | grep -oE '[0-9]{8}' | head -1)"
JSONL_OUT="${C2HLS_ROOT}/misc/hlsfactory_fixed_cosim_multistep_${VARIANT}_u280_${DATE_PREFIX}.jsonl"
POST_LOG="${C2HLS_ROOT}/artifacts/pc2/pipelines/wait_multistep_csynth_postprocess_${VARIANT}_${DATE_PREFIX}.log"
EXPECTED_BENCHES=28
if [[ -n "${BENCHES}" ]]; then
  EXPECTED_BENCHES="$(awk -F',' '{print NF}' <<<"${BENCHES}")"
  JSONL_OUT="${C2HLS_ROOT}/misc/hlsfactory_fixed_cosim_multistep_${VARIANT}_u280_${STAMP}.jsonl"
  POST_LOG="${C2HLS_ROOT}/artifacts/pc2/pipelines/wait_multistep_csynth_postprocess_${VARIANT}_${STAMP}.log"
fi
START_EXTRA=()
if [[ -n "${BENCHES}" ]]; then
  START_EXTRA+=(--benches "${BENCHES}")
fi
START_EXTRA+=(--turns "${TURNS}")

echo "=== multistep full campaign aav_n (gemm_flatten_v1 + no_RMW overlay) ==="
echo "account=${PC2_SLURM_ACCOUNT} walltime=${PC2_MULTISTEP_FULL_WALLTIME}"
echo "cosim=off (csynth+csim only) rag=off rag2=off lat_opt=off turns=${TURNS}"
echo "stamp=${STAMP} -> artifacts/pc2/multistep_fixed_cosim_${VARIANT}_${STAMP_SUFFIX}"
echo "jsonl_out=${JSONL_OUT}"
echo "summary_md=artifacts/pc2/analysis/${STAMP_SUFFIX}/summary.md"
if [[ -n "${BENCHES}" ]]; then
  echo "benches=${BENCHES} expected=${EXPECTED_BENCHES}"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  C2HLS_RUN_COSIM=0 C2HLS_COSIM_REQUIRED=0 \
    "${SCRIPT_DIR}/start_multistep_fixed_cosim_pipelined.sh" \
    --variant "${VARIANT}" \
    --stamp "${STAMP}" \
    "${START_EXTRA[@]}" \
    --dry-run
  echo "dry-run ok (session not submitted)"
  exit 0
fi

C2HLS_RUN_COSIM=0 C2HLS_COSIM_REQUIRED=0 \
  "${SCRIPT_DIR}/start_multistep_fixed_cosim_pipelined.sh" \
  --variant "${VARIANT}" \
  --stamp "${STAMP}" \
  "${START_EXTRA[@]}"

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
    --wrap="bash ${SCRIPT_DIR}/wait_multistep_csynth_postprocess.sh --variant ${VARIANT} --stamp ${STAMP} --expected-benches ${EXPECTED_BENCHES} --output ${JSONL_OUT} >> ${POST_LOG} 2>&1"
)"

echo "submitted postprocess watcher job ${post_job}"
echo "watch: tail -f artifacts/pc2/sessions/multistep_pipelined_cosim_${VARIANT}_${STAMP}/watch.log"

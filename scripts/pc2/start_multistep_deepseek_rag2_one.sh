#!/usr/bin/env bash
# Start ONE multistep DeepSeek+RAG2+skills(+latency-opt) batch_parallel campaign.
#
# Usage:
#   ./scripts/pc2/start_multistep_deepseek_rag2_one.sh --corpus chathls|tier_a|tier_b \
#       --endpoint-url URL [--stamp STAMP] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CORPUS=""
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --corpus) shift; CORPUS="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "${CORPUS}" in
  chathls|tier_a|tier_b) ;;
  *)
    echo "ERROR: --corpus required (chathls|tier_a|tier_b)" >&2
    exit 2
    ;;
esac

SKILLS_JSON="${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"
if [[ ! -f "${SKILLS_JSON}" ]]; then
  echo "ERROR: missing skills file: ${SKILLS_JSON}" >&2
  exit 2
fi

KR="${C2HLS_ROOT}/artifacts/rag/knowledge_repo"
RAG2_OPT="${C2HLS_ROOT}/artifacts/rag/rag2_opt"
RAG2_REPAIR="${C2HLS_ROOT}/artifacts/rag/rag2_repair"
if [[ ! -f "${RAG2_OPT}/chunks.jsonl" ]] || [[ ! -f "${RAG2_REPAIR}/chunks.jsonl" ]]; then
  echo "building RAG2 indexes ..."
  "${C2HLS_PYTHON:-python3}" "${C2HLS_ROOT}/scripts/build_rag2_indexes.py" \
    --knowledge-repo "${KR}" \
    --opt-out "${RAG2_OPT}" \
    --repair-out "${RAG2_REPAIR}"
fi

case "${CORPUS}" in
  chathls)
    READY_ROOT="${C2HLS_ROOT}/related_work/benchmarks/HLSFactory_benchmarks/chathls_ready"
    if [[ ! -d "${READY_ROOT}/chathls_gemm" ]]; then
      echo "preparing chathls_ready corpus..."
      "${C2HLS_PYTHON:-python3}" "${C2HLS_ROOT}/scripts/prepare_chathls_ready.py" \
        --output-root "${READY_ROOT}"
    fi
    export BATCH_PARALLEL_CONFIG="${SCRIPT_DIR}/batch_parallel_chathls_multistep_deepseek_u280.json"
    export BATCH_PARALLEL_VARIANT="chathls_ms_aav_n"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_chathls_ms_ds_rag2_lat"
    export PC2_BATCH_JOB_PREFIX="bpchms"
    ;;
  tier_a)
    export BATCH_PARALLEL_CONFIG="${SCRIPT_DIR}/batch_parallel_tier_a_multistep_deepseek_u280.json"
    export BATCH_PARALLEL_VARIANT="tier_a_ms_aav_n"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_tier_a_ms_ds_rag2_lat"
    export PC2_BATCH_JOB_PREFIX="bptams"
    ;;
  tier_b)
    export BATCH_PARALLEL_CONFIG="${SCRIPT_DIR}/batch_parallel_tier_b_multistep_deepseek_u280.json"
    export BATCH_PARALLEL_VARIANT="tier_b_ms_aav_n"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_tier_b_ms_ds_rag2_lat"
    export PC2_BATCH_JOB_PREFIX="bptbms"
    ;;
esac

export C2HLS_MODEL=deepseek-chat
export BATCH_PARALLEL_EXTERNAL_MODEL=deepseek-chat
export C2HLS_COMBINED_HLS=1
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_DEEPSEEK_PEAK_PAUSE=0
export C2HLS_DEEPSEEK_SKIP_PEAK=1
export C2HLS_MULTISTEP_OPT_STEPS="${C2HLS_MULTISTEP_OPT_STEPS:-tiling,pipeline,unroll,coalescing,doublebuffer}"

# Intermediate synth is csim+csynth only; final selected cosim is enabled in-worker.
export C2HLS_RUN_COSIM=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_COSIM_REQUIRED="${C2HLS_COSIM_REQUIRED:-0}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-600}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-7200}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-604800}"
export C2HLS_LLM_TIMEOUT="${C2HLS_LLM_TIMEOUT:-172800}"

# RAG2 + gemm_flatten_v1 skills + latency-opt after every successful step.
export C2HLS_RAG2=1
export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
unset C2HLS_RAG_SCRAPE_CORPUS || true
export C2HLS_RAG_MODE="${C2HLS_RAG_MODE:-everywhere}"
export C2HLS_RAG2_OPT_CORPUS="${C2HLS_RAG2_OPT_CORPUS:-${RAG2_OPT}}"
export C2HLS_RAG2_REPAIR_CORPUS="${C2HLS_RAG2_REPAIR_CORPUS:-${RAG2_REPAIR}}"
export C2HLS_PACKAGED_SKILLS_JSON="${SKILLS_JSON}"
export C2HLS_PACKAGED_SKILLS_ONLY=1
export C2HLS_FORCE_SKILL_PROMPTS=1
export C2HLS_SKILL_PROMPT_MODE=all_skills_avoids_global
export C2HLS_POST_FLASH_LATENCY_OPT=1
export C2HLS_LATENCY_OPT_ROUNDS="${C2HLS_LATENCY_OPT_ROUNDS:-3}"
export C2HLS_LATENCY_OPT_REPAIR_ROUNDS="${C2HLS_LATENCY_OPT_REPAIR_ROUNDS:-3}"
# Multistep lat-opt uses multistep_* roles; flash/dataflow chain flags unused.
export C2HLS_LATENCY_OPT_CHAIN_FLASH=0
export C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=0

export PC2_BATCH_PARALLEL_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-72:00:00}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME}"
export C2HLS_TMP_RUN="${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:18094/v1"
  echo "WARNING: --endpoint-url not given; using placeholder ${ENDPOINT_URL} for --dry-run only" >&2
else
  echo "ERROR: --endpoint-url is required for a real (non-dry-run) start." >&2
  exit 2
fi

export BATCH_PARALLEL_EXTERNAL_LLM=1
export BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="${ENDPOINT_URL}"

echo "=== Multistep DeepSeek RAG2+skills+lat-opt U280: corpus=${CORPUS} ==="
echo "stamp=${STAMP}"
echo "config=${BATCH_PARALLEL_CONFIG}"
echo "variant=${BATCH_PARALLEL_VARIANT}"
echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
echo "part=${C2HLS_PART} clock_ns=${C2HLS_CLOCK_NS}"
echo "opt_steps=${C2HLS_MULTISTEP_OPT_STEPS}"
echo "skills=${C2HLS_PACKAGED_SKILLS_JSON}"
echo "latency_opt=1 rounds=${C2HLS_LATENCY_OPT_ROUNDS} repair=${C2HLS_LATENCY_OPT_REPAIR_ROUNDS}"
echo "skip_peak=${C2HLS_DEEPSEEK_SKIP_PEAK} wall=${PC2_FORCE_WALLTIME}"
echo "timeouts csim=${C2HLS_CSIM_TIMEOUT}s synth=${C2HLS_SYNTH_TIMEOUT}s llm=${C2HLS_LLM_TIMEOUT}s cosim=${C2HLS_COSIM_TIMEOUT}s"

EXTRA_ARGS=(--external-llm)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

env BATCH_PARALLEL_STAMP="${STAMP}" \
  "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
  --stamp "${STAMP}" \
  "${EXTRA_ARGS[@]}"

CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"
mkdir -p "${CAMPAIGN_ROOT}"
echo "${CORPUS}" > "${CAMPAIGN_ROOT}/corpus.txt"
echo "1" > "${CAMPAIGN_ROOT}/latency_opt.txt"
echo "${ENDPOINT_URL}" > "${CAMPAIGN_ROOT}/endpoint_url.txt"

echo "campaign=${CAMPAIGN_ROOT}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok"
fi

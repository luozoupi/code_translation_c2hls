#!/usr/bin/env bash
# Matched-pack 64³ Multistep tests (10 each). DeepSeek-v4-flash, no RAG/RAG2/lat-opt.
# 1) msss sliced from gemm_flatten_v1 + no-RMW overlay
# 2) aav_n dumping 90 ∪ gemm_flatten_v2 (no overlay)
# Compare later vs the existing 10v10: aav_n-v1 vs this msss-v1; msss-90v2 vs this aav_n-90v2.
#
# Usage:
#   ./scripts/pc2/start_autosa_mm_multistep_matched_packs.sh --dry-run
#   ./scripts/pc2/start_autosa_mm_multistep_matched_packs.sh --endpoint-url http://login5:18205/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""
PKG="${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package"
MSSS_V1_DIR="${PKG}/multistep_gf_v1_normw"
UNION_90_V2="${PKG}/skills_ii_target_miss_solutions_added(90skills)_plus_gemm_flatten_v2.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

unset C2HLS_RAG_MODE C2HLS_RAG2_OPT_CORPUS C2HLS_RAG2_REPAIR_CORPUS C2HLS_RAG_SCRAPE_CORPUS || true
unset C2HLS_LATENCY_OPT_CHAIN_FLASH C2HLS_LATENCY_OPT_CHAIN_DATAFLOW || true
unset C2HLS_LATENCY_OPT_ROUNDS C2HLS_LATENCY_OPT_REPAIR_ROUNDS || true
unset C2HLS_OPENAI_UPSTREAM C2HLS_XAI_HOSTED_URL C2HLS_REASONING_EFFORT || true
unset OPENAI_PROXY_MODEL ANTHROPIC_PROXY_MODEL || true
if [[ "${OPENAI_API_KEY:-}" == "EMPTY" || "${OPENAI_API_KEY:-}" == "empty" ]]; then
  unset OPENAI_API_KEY || true
fi
if [[ "${CHATHLS_API_KEY:-}" == "EMPTY" || "${CHATHLS_API_KEY:-}" == "empty" ]]; then
  unset CHATHLS_API_KEY || true
fi

export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
export C2HLS_RAG2=0
export C2HLS_POST_FLASH_LATENCY_OPT=0
export C2HLS_POST_FLASH_PRAGMA_OPT=0
export C2HLS_POST_FLASH_DATAFLOW=0
export C2HLS_POST_FLASH_DSE=0
export C2HLS_DSE_CHAIN_FLASH=0
export C2HLS_POST_FLASH_STREAM=0
export C2HLS_STREAM_CHAIN_FLASH=0
export C2HLS_ENFORCEMENT=0
export C2HLS_PHASEB_FROM_GOLD=0
export C2HLS_STRATEGY=static
export C2HLS_DYNAMIC_ROUTING=0
export C2HLS_MULTISTEP_SKIP_FINAL_COSIM=1
export C2HLS_MULTISTEP_OPT_STEPS="${C2HLS_MULTISTEP_OPT_STEPS:-tiling,pipeline,unroll,coalescing,doublebuffer}"
export C2HLS_RECORD_FLOW=1
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_RUN_COSIM=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_COSIM_TRACE_LEVEL=none
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-1800}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-1200}"
export C2HLS_LLM_TIMEOUT="${C2HLS_LLM_TIMEOUT:-1800}"
export C2HLS_MODEL=deepseek-v4-flash
export BATCH_PARALLEL_EXTERNAL_MODEL=deepseek-v4-flash
export DEEPSEEK_PROXY_MODEL=deepseek-v4-flash
export C2HLS_DEEPSEEK_PEAK_PAUSE=0
export C2HLS_DEEPSEEK_SKIP_PEAK=1
export CHATHLS_DEEPSEEK_MAX_QUEUE="${CHATHLS_DEEPSEEK_MAX_QUEUE:-32}"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

echo "=== building matched-pack skill files ==="
"${PY}" "${C2HLS_ROOT}/scripts/flash_shared/build_msss_step_skills.py" --preset all
if [[ ! -d "${MSSS_V1_DIR}" || ! -f "${UNION_90_V2}" ]]; then
  echo "ERROR: missing generated packs ${MSSS_V1_DIR} or ${UNION_90_V2}" >&2
  exit 2
fi

echo "=== preparing autosa_ready/autosa_mm (64 cubed) ==="
"${PY}" "${C2HLS_ROOT}/scripts/prepare_autosa_ready.py" --kernel mm

SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/autosa_mm_multistep_matched_packs_${STAMP}"
PROXY_ROOT="${SEQ_ROOT}/proxy"
mkdir -p "${PROXY_ROOT}"

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:18140/v1"
  echo "WARNING: --endpoint-url not given; using placeholder ${ENDPOINT_URL} for --dry-run only" >&2
else
  ENDPOINT_URL="$(
    DEEPSEEK_PROXY_MODEL=deepseek-v4-flash \
      "${SCRIPT_DIR}/start_dedicated_deepseek_proxy.sh" "${PROXY_ROOT}"
  )"
  if [[ -z "${ENDPOINT_URL}" || "${ENDPOINT_URL}" != http* ]]; then
    echo "ERROR: failed to start dedicated DeepSeek proxy in ${PROXY_ROOT}" >&2
    exit 2
  fi
fi

export BATCH_PARALLEL_EXTERNAL_LLM=1
export BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="${ENDPOINT_URL}"
export OPENAI_BASE_URL="${ENDPOINT_URL}"
export C2HLS_OPENAI_HOSTED_URL="${ENDPOINT_URL}"
export CHATHLS_API_BASE="${ENDPOINT_URL}"

if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" || "${OPENAI_API_KEY}" == "empty" ]]; then
  unset OPENAI_API_KEY || true
  unset CHATHLS_API_KEY || true
  CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
  # shellcheck disable=SC1091
  source "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh"
fi
if [[ "${DRY_RUN}" -eq 0 ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" ]]; then
    echo "ERROR: OPENAI_API_KEY missing after setup_deepseek_api.sh" >&2
    exit 2
  fi
fi
export CHATHLS_API_KEY="${CHATHLS_API_KEY:-${OPENAI_API_KEY:-}}"

EXTRA_ARGS=(--external-llm)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

export PC2_BATCH_PARALLEL_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME}"

submit_variant() {
  local variant="$1"
  local config="$2"
  local prefix="$3"
  local job_prefix="$4"
  local pack_note="$5"
  unset C2HLS_PACKAGED_SKILLS_JSON C2HLS_PACKAGED_SKILLS_ONLY C2HLS_FLASH_SKILL_ENTRIES_JSON C2HLS_SKILL_PROMPT_MODE || true
  export BATCH_PARALLEL_CONFIG="${config}"
  export BATCH_PARALLEL_VARIANT="${variant}"
  export BATCH_PARALLEL_ARTIFACT_PREFIX="${prefix}"
  export PC2_BATCH_JOB_PREFIX="${job_prefix}"
  export C2HLS_TMP_RUN="${prefix}_${STAMP}"
  echo "=== Multistep ${variant} autosa_mm x10 pack=${pack_note} ==="
  echo "stamp=${STAMP}"
  echo "config=${BATCH_PARALLEL_CONFIG}"
  echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
  echo "strategy=static rag=off rag2=off lat_opt=off dse=off stream=off cosim=off"
  env BATCH_PARALLEL_STAMP="${STAMP}" \
    "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
    --stamp "${STAMP}" \
    "${EXTRA_ARGS[@]}"
  local camp="${C2HLS_ROOT}/artifacts/pc2/${prefix}_${STAMP}"
  mkdir -p "${camp}"
  echo "${ENDPOINT_URL}" > "${camp}/endpoint_url.txt"
  echo "deepseek-v4-flash" > "${camp}/model.txt"
  echo "0" > "${camp}/latency_opt.txt"
  echo "0" > "${camp}/rag.txt"
  echo "0" > "${camp}/rag2.txt"
  echo "${variant}" > "${camp}/skill_prompt.txt"
  echo "${pack_note}" > "${camp}/skills_pack.txt"
  echo "${camp}" > "${SEQ_ROOT}/campaign_${variant}.txt"
  ln -sfn "${camp}" "${SEQ_ROOT}/campaign_${variant}"
  echo "campaign=${camp}"
}

# Test 1: msss on v1 + no-RMW (matches existing aav_n 10v10 pack).
unset C2HLS_AUTOSA_MS_AAV_PACK || true
export C2HLS_AUTOSA_MS_MSSS_PACK=v1_normw
export C2HLS_MSSS_SKILLS_DIR="${MSSS_V1_DIR}"
submit_variant \
  autosa_ms_msss \
  "${SCRIPT_DIR}/batch_parallel_autosa_mm_multistep_msss_v1n.json" \
  batch_parallel_autosa_mm_ms_msss_v1n \
  msv1ms \
  "gemm_flatten_v1+no_RMW"

# Test 2: aav_n on 90 ∪ v2 (matches existing msss 10v10 pack).
unset C2HLS_AUTOSA_MS_MSSS_PACK C2HLS_MSSS_SKILLS_DIR || true
export C2HLS_AUTOSA_MS_AAV_PACK=90_v2
submit_variant \
  autosa_ms_aav_n \
  "${SCRIPT_DIR}/batch_parallel_autosa_mm_multistep_aav_n_90v2.json" \
  batch_parallel_autosa_mm_ms_aav_n_90v2 \
  msv2aa \
  "90+gemm_flatten_v2"

{
  echo "stamp=${STAMP}"
  echo "seq_root=${SEQ_ROOT}"
  echo "endpoint=${ENDPOINT_URL}"
  echo "msss_v1n=$(cat "${SEQ_ROOT}/campaign_autosa_ms_msss.txt")"
  echo "aav_n_90v2=$(cat "${SEQ_ROOT}/campaign_autosa_ms_aav_n.txt")"
  echo "compare_pack_a=existing_aav_n_10v10 vs ${SEQ_ROOT}/campaign_autosa_ms_msss"
  echo "compare_pack_b=existing_msss_10v10 vs ${SEQ_ROOT}/campaign_autosa_ms_aav_n"
} | tee "${SEQ_ROOT}/launch_summary.txt"

echo "=== launched ==="
cat "${SEQ_ROOT}/launch_summary.txt"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok"
fi

#!/usr/bin/env bash
# Multistep aav_n vs msss on autosa_mm 1024³, 10-vs-10.
# DeepSeek-v4-flash, no RAG/RAG2/lat-opt. Does not edit the live 64³ bench.
# Every Slurm request is capped at 5 days.
#
# Usage:
#   ./scripts/pc2/start_autosa_mm_multistep_msss_1024.sh --dry-run
#   ./scripts/pc2/start_autosa_mm_multistep_msss_1024.sh --endpoint-url http://login5:18205/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""
# Exactly 5 days. Do not raise this.
WALLTIME_5D="5-00:00:00"

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
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-14400}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-28800}"
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

LIVE_H="${C2HLS_ROOT}/related_work/benchmarks/autosa_ready/autosa_mm/kernel.h"
if ! grep -qE '^#define I 64[[:space:]]*$' "${LIVE_H}"; then
  echo "ERROR: live 64³ header changed unexpectedly: ${LIVE_H}" >&2
  exit 2
fi

echo "=== staging autosa_mm 1024 cubed (live 64³ left alone) ==="
READY_ROOT="$("${PY}" - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "scripts/pc2")
from autosa_mm_ijk_sweep import BENCHES_ROOT, SOURCE_BENCH, stage_ijk_bench

dest = stage_ijk_bench(1024, BENCHES_ROOT, source=SOURCE_BENCH)
ready = dest.parent
print(ready)
PY
)"
export C2HLS_AUTOSA_READY_ROOT="${READY_ROOT}"
if [[ ! -f "${READY_ROOT}/autosa_mm/metadata.json" ]]; then
  echo "ERROR: staged 1024³ bench missing under ${READY_ROOT}" >&2
  exit 2
fi
if ! grep -qE '^#define I 1024[[:space:]]*$' "${READY_ROOT}/autosa_mm/kernel.h"; then
  echo "ERROR: staged kernel is not 1024³: ${READY_ROOT}/autosa_mm/kernel.h" >&2
  exit 2
fi
if ! grep -qE '^#define I 64[[:space:]]*$' "${LIVE_H}"; then
  echo "ERROR: staging mutated the live 64³ header: ${LIVE_H}" >&2
  exit 2
fi
echo "ready_root=${C2HLS_AUTOSA_READY_ROOT}"

SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/autosa_mm1024_multistep_msss_${STAMP}"
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

export PC2_BATCH_PARALLEL_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-${WALLTIME_5D}}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME}"
export PC2_HELPER_WALLTIME="${PC2_HELPER_WALLTIME:-${WALLTIME_5D}}"

submit_variant() {
  local variant="$1"
  local config="$2"
  local prefix="$3"
  local job_prefix="$4"
  unset C2HLS_PACKAGED_SKILLS_JSON C2HLS_PACKAGED_SKILLS_ONLY C2HLS_FLASH_SKILL_ENTRIES_JSON C2HLS_SKILL_PROMPT_MODE || true
  export BATCH_PARALLEL_CONFIG="${config}"
  export BATCH_PARALLEL_VARIANT="${variant}"
  export BATCH_PARALLEL_ARTIFACT_PREFIX="${prefix}"
  export PC2_BATCH_JOB_PREFIX="${job_prefix}"
  export C2HLS_TMP_RUN="${prefix}_${STAMP}"
  echo "=== Multistep ${variant} autosa_mm 1024³ x10 ==="
  echo "stamp=${STAMP}"
  echo "config=${BATCH_PARALLEL_CONFIG}"
  echo "ready_root=${C2HLS_AUTOSA_READY_ROOT}"
  echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
  echo "walltime=${PC2_FORCE_WALLTIME} helper=${PC2_HELPER_WALLTIME}"
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
  echo "1024" > "${camp}/ijk.txt"
  echo "${C2HLS_AUTOSA_READY_ROOT}" > "${camp}/autosa_ready_root.txt"
  echo "${camp}" > "${SEQ_ROOT}/campaign_${variant}.txt"
  ln -sfn "${camp}" "${SEQ_ROOT}/campaign_${variant}"
  echo "campaign=${camp}"
}

submit_variant \
  autosa_ms_aav_n \
  "${SCRIPT_DIR}/batch_parallel_autosa_mm1024_multistep_aav_n.json" \
  batch_parallel_autosa_mm1024_ms_aav_n \
  m1kaav

submit_variant \
  autosa_ms_msss \
  "${SCRIPT_DIR}/batch_parallel_autosa_mm1024_multistep_msss.json" \
  batch_parallel_autosa_mm1024_ms_msss \
  m1kmss

{
  echo "stamp=${STAMP}"
  echo "ijk=1024"
  echo "seq_root=${SEQ_ROOT}"
  echo "ready_root=${C2HLS_AUTOSA_READY_ROOT}"
  echo "endpoint=${ENDPOINT_URL}"
  echo "walltime=${PC2_FORCE_WALLTIME}"
  echo "helper_walltime=${PC2_HELPER_WALLTIME}"
  echo "aav_n=$(cat "${SEQ_ROOT}/campaign_autosa_ms_aav_n.txt")"
  echo "msss=$(cat "${SEQ_ROOT}/campaign_autosa_ms_msss.txt")"
} | tee "${SEQ_ROOT}/launch_summary.txt"

echo "=== launched ==="
cat "${SEQ_ROOT}/launch_summary.txt"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok"
fi

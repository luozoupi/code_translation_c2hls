#!/usr/bin/env bash
# Smoke: LLM select-then-code (aav_sel) on 2mm + heat-3d with deepseek-v4-flash.
# Flash + selected cosim only. No dataflow / lat_opt / RAG / RAG2.
#
# Usage:
#   ./scripts/pc2/start_hlsfactory_llm_sel_smoke.sh [--stamp STAMP] [--endpoint-url URL] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""
DEFAULT_CONFIG="${SCRIPT_DIR}/batch_parallel_hlsfactory_llm_sel_smoke.json"
SKILLS_GEMM="${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"
PROXY_PORT="${C2HLS_DEEPSEEK_PROXY_PORT_LLM_SEL_SMOKE:-18098}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Scrub parent-shell RAG / lat_opt poison.
unset C2HLS_RAG_MODE C2HLS_RAG2_OPT_CORPUS C2HLS_RAG2_REPAIR_CORPUS C2HLS_RAG_SCRAPE_CORPUS || true
unset C2HLS_LATENCY_OPT_CHAIN_FLASH C2HLS_LATENCY_OPT_CHAIN_DATAFLOW || true
unset C2HLS_LATENCY_OPT_ROUNDS C2HLS_LATENCY_OPT_REPAIR_ROUNDS || true
export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
export C2HLS_RAG2=0
export C2HLS_POST_FLASH_LATENCY_OPT=0

export BATCH_PARALLEL_CONFIG="${C2HLS_LLM_SEL_SMOKE_BATCH_CONFIG:-${DEFAULT_CONFIG}}"
export C2HLS_COMBINED_HLS=1
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33

export C2HLS_FLASH_DEFER_COSIM=1
export C2HLS_RUN_COSIM=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_COSIM_XELAB_MT_OFF=1
export C2HLS_COSIM_TRACE_LEVEL=none
export C2HLS_FLASH_COSIM_FULL_SIZE=1
export C2HLS_COSIM_BENCHMARKS_ROOT="${C2HLS_ROOT}/benchmarks_cosim"
export C2HLS_CSIM_USE_COSIM_TB="${C2HLS_CSIM_USE_COSIM_TB:-1}"

export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-600}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"
export C2HLS_MAX_REPAIR_ATTEMPT="${C2HLS_MAX_REPAIR_ATTEMPT:-7}"
export C2HLS_FLASH_MAX_TOKENS="${C2HLS_FLASH_MAX_TOKENS:-65536}"
export C2HLS_LLM_MAX_TOKENS="${C2HLS_LLM_MAX_TOKENS:-65536}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
POST_WALLTIME="${PC2_HLSFACTORY_POST_WALLTIME:-7-00:00:00}"

export C2HLS_PER_BENCH_PROXY=0
export C2HLS_RANKED_COSIM_AFTER_DATAFLOW=0
export C2HLS_DEEPSEEK_PEAK_PAUSE=0
export C2HLS_DEEPSEEK_SKIP_PEAK=1

unset C2HLS_BARE_OPT_PROMPTS C2HLS_CHATHLS_NOSKILLS C2HLS_DATAFLOW_NO_SKILLS C2HLS_FLASH_OPT_PROMPT_MODE || true

export C2HLS_MODEL=deepseek-v4-flash
export BATCH_PARALLEL_EXTERNAL_MODEL=deepseek-v4-flash
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bphfsel}"

export BATCH_PARALLEL_VARIANT="aav_sel"
export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_hlsfactory_ds_v4f_llm_sel_smoke"
export C2HLS_PACKAGED_SKILLS_JSON="${SKILLS_GEMM}"
export C2HLS_PACKAGED_SKILLS_ONLY=1
export C2HLS_FORCE_SKILL_PROMPTS=1
export C2HLS_SKILL_PROMPT_MODE=llm_select_then_code
export C2HLS_POST_FLASH_PROMPT_POLICY=system_skills
# aav_sel: gemm_flatten pack + flash_no_RMW overlay (same as aav_n skill-on).
export C2HLS_TMP_RUN="${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"

PROXY_ROOT="${C2HLS_ROOT}/artifacts/pc2/hlsfactory_ds_v4f_llm_sel_smoke_${STAMP}/proxy_aav_sel"
mkdir -p "${PROXY_ROOT}"

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:${PROXY_PORT}/v1"
  echo "WARNING: dry-run placeholder endpoint ${ENDPOINT_URL}" >&2
else
  export CHATHLS_DEEPSEEK_PROXY_PORT="${PROXY_PORT}"
  export DEEPSEEK_PROXY_MODEL=deepseek-v4-flash
  ENDPOINT_URL="$(bash "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${PROXY_ROOT}")"
fi

export BATCH_PARALLEL_EXTERNAL_LLM=1
export BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="${ENDPOINT_URL}"
export OPENAI_BASE_URL="${ENDPOINT_URL}"
export C2HLS_OPENAI_HOSTED_URL="${ENDPOINT_URL}"
export CHATHLS_API_BASE="${ENDPOINT_URL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export CHATHLS_API_KEY="${CHATHLS_API_KEY:-${OPENAI_API_KEY}}"

echo "=== HLSFactory deepseek-v4-flash LLM-SEL SMOKE (aav_sel) ==="
echo "stamp=${STAMP} variant=aav_sel mode=llm_select_then_code"
echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
echo "benches=2mm,heat-3d defer_cosim=1 no_dataflow=1 lat_opt=0 rag=0 rag2=0"

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
if [[ -f "${PROXY_ROOT}/llm_endpoint.json" ]]; then
  cp -f "${PROXY_ROOT}/llm_endpoint.json" "${CAMPAIGN_ROOT}/llm_endpoint.json"
fi
echo "aav_sel" > "${CAMPAIGN_ROOT}/flavor.txt"
echo "llm_sel_smoke_flash_only" > "${CAMPAIGN_ROOT}/test_mode.txt"
echo "0" > "${CAMPAIGN_ROOT}/latency_opt.txt"
echo "0" > "${CAMPAIGN_ROOT}/rag2.txt"
echo "0" > "${CAMPAIGN_ROOT}/dataflow.txt"

if [[ -f "${CAMPAIGN_ROOT}/campaign.json" ]]; then
  "${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["skip_peak_pause"] = True
doc["flavor"] = "aav_sel"
doc["test_mode"] = "llm_sel_smoke_flash_only"
doc["flash_defer_cosim"] = True
doc["model"] = "deepseek-v4-flash"
doc["latency_opt"] = False
doc["dataflow"] = False
doc["rag"] = False
doc["rag2"] = False
doc["skill_prompt_mode"] = "llm_select_then_code"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok campaign=${CAMPAIGN_ROOT}"
  exit 0
fi

WATCH_LOG="${CAMPAIGN_ROOT}/flow/post_flash_cosim_only_watcher.log"
mkdir -p "${CAMPAIGN_ROOT}/flow"

POST_EXPORT="ALL,C2HLS_RAG=0,C2HLS_RAG_ENABLE=0,C2HLS_RAG_SCRAPE=0,C2HLS_RAG2=0,C2HLS_RAG_MODE=,C2HLS_MODEL=${C2HLS_MODEL},C2HLS_PART=${C2HLS_PART},C2HLS_CLOCK_NS=${C2HLS_CLOCK_NS},C2HLS_TMP_RUN=${C2HLS_TMP_RUN},C2HLS_RUN_COSIM=0,C2HLS_REFERENCE_COSIM=0,C2HLS_COSIM_XELAB_MT_OFF=1,C2HLS_FLASH_COSIM_FULL_SIZE=1,C2HLS_FLASH_MAX_TOKENS=${C2HLS_FLASH_MAX_TOKENS},C2HLS_LLM_MAX_TOKENS=${C2HLS_LLM_MAX_TOKENS},C2HLS_COSIM_BENCHMARKS_ROOT=${C2HLS_COSIM_BENCHMARKS_ROOT},C2HLS_MAX_REPAIR_ATTEMPT=${C2HLS_MAX_REPAIR_ATTEMPT},C2HLS_COSIM_TIMEOUT=${C2HLS_COSIM_TIMEOUT},C2HLS_POST_FLASH_LATENCY_OPT=0,C2HLS_LATENCY_OPT_CHAIN_FLASH=0,C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=0,C2HLS_RANKED_COSIM_AFTER_DATAFLOW=0,C2HLS_PER_BENCH_PROXY=0,C2HLS_POST_FLASH_RESULTS_SUFFIX=${C2HLS_POST_FLASH_RESULTS_SUFFIX:-hlsfactory_cosim_repairs},BATCH_PARALLEL_EXTERNAL_LLM=1,BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL=${ENDPOINT_URL},BATCH_PARALLEL_EXTERNAL_MODEL=${BATCH_PARALLEL_EXTERNAL_MODEL},C2HLS_PACKAGED_SKILLS_JSON=${C2HLS_PACKAGED_SKILLS_JSON},C2HLS_PACKAGED_SKILLS_ONLY=1,C2HLS_FORCE_SKILL_PROMPTS=1,C2HLS_SKILL_PROMPT_MODE=llm_select_then_code,C2HLS_POST_FLASH_PROMPT_POLICY=${C2HLS_POST_FLASH_PROMPT_POLICY},PC2_BATCH_JOB_PREFIX=${PC2_BATCH_JOB_PREFIX},OPENAI_API_KEY=${OPENAI_API_KEY},CHATHLS_API_KEY=${CHATHLS_API_KEY},OPENAI_BASE_URL=${ENDPOINT_URL},C2HLS_OPENAI_HOSTED_URL=${ENDPOINT_URL}"

POST_JOB="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --job-name="${PC2_BATCH_JOB_PREFIX}-post" \
    --output="${CAMPAIGN_ROOT}/flow/post_watcher-%j.out" \
    --error="${CAMPAIGN_ROOT}/flow/post_watcher-%j.err" \
    --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task=2 \
    --mem=8G \
    --time="${POST_WALLTIME}" \
    --export="${POST_EXPORT}" \
    --wrap="bash ${SCRIPT_DIR}/wait_hlsfactory_flash_cosim_only.sh --campaign-root ${CAMPAIGN_ROOT} >> ${WATCH_LOG} 2>&1"
)"

"${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["post_watcher_job_id"] = "${POST_JOB}"
doc["waiter_script"] = "wait_hlsfactory_flash_cosim_only.sh"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "submitted post watcher job ${POST_JOB} (flash cosim only; no dataflow)"
echo "campaign=${CAMPAIGN_ROOT}"
echo "watch: tail -f ${CAMPAIGN_ROOT}/flow/watch.log"
echo "post:  tail -f ${WATCH_LOG}"

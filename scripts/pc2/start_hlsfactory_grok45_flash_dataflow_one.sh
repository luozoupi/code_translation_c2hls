#!/usr/bin/env bash
# HLSFactory flash→dataflow with grok-4.5 (high/max reasoning), top-5 benches.
# Flavors: skills | noskills
#
# Usage:
#   ./scripts/pc2/start_hlsfactory_grok45_flash_dataflow_one.sh --flavor skills \
#       [--stamp STAMP] [--dry-run] [--bench-set top5|rest23]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

FLAVOR=""
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""
BENCH_SET="top5"
MODEL_ID="grok-4.5"
REASONING_EFFORT="${C2HLS_REASONING_EFFORT:-high}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flavor) shift; FLAVOR="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    --bench-set) shift; BENCH_SET="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "${FLAVOR}" in
  skills|noskills) ;;
  "")
    echo "ERROR: --flavor required (skills|noskills)" >&2
    exit 2
    ;;
  *)
    echo "ERROR: unknown --flavor '${FLAVOR}'" >&2
    exit 2
    ;;
esac

case "${BENCH_SET}" in
  top5)
    DEFAULT_CONFIG="${SCRIPT_DIR}/batch_parallel_hlsfactory_grok45_top5.json"
    SET_TAG="top5"
    ;;
  rest23)
    DEFAULT_CONFIG="${SCRIPT_DIR}/batch_parallel_hlsfactory_grok45_rest23.json"
    SET_TAG="rest23"
    ;;
  *)
    echo "ERROR: --bench-set must be top5|rest23" >&2
    exit 2
    ;;
esac

# Scrub parent-shell RAG poison before any sbatch --export=ALL.
unset C2HLS_RAG_MODE C2HLS_RAG2_OPT_CORPUS C2HLS_RAG2_REPAIR_CORPUS C2HLS_RAG_SCRAPE_CORPUS || true
export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
export C2HLS_RAG2=0

# Match Sonnet top5: no flash/dataflow lat_opt chain. RAG/RAG2 already scrubbed above.
export C2HLS_POST_FLASH_LATENCY_OPT=0
unset C2HLS_LATENCY_OPT_CHAIN_FLASH C2HLS_LATENCY_OPT_CHAIN_DATAFLOW || true
unset C2HLS_LATENCY_OPT_ROUNDS C2HLS_LATENCY_OPT_REPAIR_ROUNDS || true
export C2HLS_REASONING_EFFORT="${REASONING_EFFORT}"

SKILLS_GEMM="${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"

export BATCH_PARALLEL_CONFIG="${C2HLS_GROK45_BATCH_CONFIG:-${DEFAULT_CONFIG}}"
export C2HLS_MODEL="${MODEL_ID}"
export BATCH_PARALLEL_EXTERNAL_MODEL="${MODEL_ID}"
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
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
POST_WALLTIME="${PC2_HLSFACTORY_POST_WALLTIME:-7-00:00:00}"

export C2HLS_DATAFLOW_EXCLUSIVE=0
export C2HLS_DATAFLOW_WORKER_CPUS="${C2HLS_DATAFLOW_WORKER_CPUS:-32}"
export C2HLS_DATAFLOW_WORKER_MEM_GB="${C2HLS_DATAFLOW_WORKER_MEM_GB:-128}"
export C2HLS_PER_BENCH_PROXY=0
export C2HLS_PROXY_BACKEND=openai
export C2HLS_OPENAI_QUEUE_WORKERS="${C2HLS_OPENAI_QUEUE_WORKERS:-16}"
# Headroom for xhigh reasoning tokens.
export C2HLS_FLASH_MAX_TOKENS="${C2HLS_FLASH_MAX_TOKENS:-65536}"
export C2HLS_LLM_MAX_TOKENS="${C2HLS_LLM_MAX_TOKENS:-65536}"

unset C2HLS_BARE_OPT_PROMPTS C2HLS_CHATHLS_NOSKILLS C2HLS_DATAFLOW_NO_SKILLS C2HLS_FLASH_OPT_PROMPT_MODE || true

case "${FLAVOR}" in
  skills)
    export BATCH_PARALLEL_VARIANT="aav_n"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_hlsfactory_grok45_${SET_TAG}_skills"
    export PC2_BATCH_JOB_PREFIX="bphfgs"
    export C2HLS_PACKAGED_SKILLS_JSON="${SKILLS_GEMM}"
    export C2HLS_PACKAGED_SKILLS_ONLY=1
    export C2HLS_FORCE_SKILL_PROMPTS=1
    export C2HLS_SKILL_PROMPT_MODE=all_skills_avoids_global
    export C2HLS_POST_FLASH_PROMPT_POLICY=system_skills
    PROXY_PORT="${C2HLS_OPENAI_PROXY_PORT_GROK_SKILLS:-18210}"
    FLAVOR_DESC="skills(gemm_flatten)+full HLS-opt prompts"
    ;;
  noskills)
    export BATCH_PARALLEL_VARIANT="noskills"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_hlsfactory_grok45_${SET_TAG}_noskills"
    export PC2_BATCH_JOB_PREFIX="bphfgn"
    export C2HLS_DATAFLOW_NO_SKILLS=1
    unset C2HLS_PACKAGED_SKILLS_JSON C2HLS_FORCE_SKILL_PROMPTS C2HLS_SKILL_PROMPT_MODE || true
    export C2HLS_POST_FLASH_PROMPT_POLICY=system_skills
    PROXY_PORT="${C2HLS_OPENAI_PROXY_PORT_GROK_NOSKILLS:-18211}"
    FLAVOR_DESC="noskills + keep flash/dataflow HLS-opt wording"
    ;;
esac

export C2HLS_TMP_RUN="${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"

_load_grok_key_from_bashrc() {
  local line val name
  for name in Grok_API XAI_API_KEY GROK_API; do
    line="$(grep -E "^[[:space:]]*${name}=" "${HOME}/.bashrc" 2>/dev/null | tail -1 || true)"
    [[ -n "${line}" ]] || continue
    val="${line#*=}"
    val="${val%\"}"; val="${val#\"}"
    val="${val%\'}"; val="${val#\'}"
    if [[ -n "${val}" ]]; then
      export OPENAI_API_KEY="${val}"
      export Grok_API="${val}"
      export XAI_API_KEY="${val}"
      return 0
    fi
  done
  return 1
}

# Always prefer Grok_API / XAI over any inherited OPENAI_API_KEY (e.g. Luna).
if [[ -n "${Grok_API:-}" ]]; then
  export OPENAI_API_KEY="${Grok_API}"
  export XAI_API_KEY="${Grok_API}"
elif [[ -n "${XAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${XAI_API_KEY}"
else
  _load_grok_key_from_bashrc || true
fi
unset C2HLS_RAG_MODE C2HLS_RAG2_OPT_CORPUS C2HLS_RAG2_REPAIR_CORPUS || true
export C2HLS_RAG=0 C2HLS_RAG_ENABLE=0 C2HLS_RAG_SCRAPE=0 C2HLS_RAG2=0
if [[ -z "${OPENAI_API_KEY:-}" || ${#OPENAI_API_KEY} -lt 20 ]]; then
  echo "ERROR: Grok_API / XAI_API_KEY missing" >&2
  exit 2
fi
if [[ "${OPENAI_API_KEY}" != xai-* ]]; then
  echo "WARNING: Grok key does not look like xAI (prefix=${OPENAI_API_KEY:0:4}...)" >&2
fi
export CHATHLS_API_KEY="${OPENAI_API_KEY}"
export Grok_API="${Grok_API:-${OPENAI_API_KEY}}"
export XAI_API_KEY="${XAI_API_KEY:-${OPENAI_API_KEY}}"
# xAI upstream for login-node proxy
export C2HLS_OPENAI_UPSTREAM="${C2HLS_OPENAI_UPSTREAM:-https://api.x.ai/v1}"
export C2HLS_XAI_HOSTED_URL="${C2HLS_XAI_HOSTED_URL:-https://api.x.ai/v1}"

TRIPLE_OR_PROXY_ROOT="${C2HLS_ROOT}/artifacts/pc2/hlsfactory_grok45_${SET_TAG}_${STAMP}/proxy_${FLAVOR}"
mkdir -p "${TRIPLE_OR_PROXY_ROOT}"

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:${PROXY_PORT}/v1"
  echo "WARNING: dry-run placeholder endpoint ${ENDPOINT_URL}" >&2
else
  export C2HLS_OPENAI_PROXY_PORT="${PROXY_PORT}"
  export C2HLS_OPENAI_QUEUE_WORKERS="${C2HLS_OPENAI_QUEUE_WORKERS:-16}"
  export OPENAI_PROXY_MODEL="${MODEL_ID}"
  export C2HLS_REASONING_EFFORT="${REASONING_EFFORT}"
  ENDPOINT_URL="$(bash "${SCRIPT_DIR}/c2hls_openai_proxy.sh" "${TRIPLE_OR_PROXY_ROOT}")"
fi

export BATCH_PARALLEL_EXTERNAL_LLM=1
export BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="${ENDPOINT_URL}"
# Hosted openai client prefers OPENAI_BASE_URL when set (login proxy).
export OPENAI_BASE_URL="${ENDPOINT_URL}"
export C2HLS_OPENAI_HOSTED_URL="${ENDPOINT_URL}"
export CHATHLS_API_BASE="${ENDPOINT_URL}"

echo "=== HLSFactory grok-4.5 flash+dataflow (${FLAVOR}/${SET_TAG}) ==="
echo "stamp=${STAMP} flavor=${FLAVOR} desc=${FLAVOR_DESC}"
echo "model=${C2HLS_MODEL} effort=${REASONING_EFFORT} endpoint=${ENDPOINT_URL}"
echo "defer_cosim=1 df_exclusive=0 shared_proxy=1 lat_opt=0 rag=0 rag2=0"

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
if [[ -f "${TRIPLE_OR_PROXY_ROOT}/llm_endpoint.json" ]]; then
  cp -f "${TRIPLE_OR_PROXY_ROOT}/llm_endpoint.json" "${CAMPAIGN_ROOT}/llm_endpoint.json"
fi
echo "${FLAVOR}" > "${CAMPAIGN_ROOT}/flavor.txt"
echo "grok45_${SET_TAG}_stream_df" > "${CAMPAIGN_ROOT}/test_mode.txt"
echo "0" > "${CAMPAIGN_ROOT}/latency_opt.txt"
echo "0" > "${CAMPAIGN_ROOT}/rag2.txt"
echo "${REASONING_EFFORT}" > "${CAMPAIGN_ROOT}/reasoning_effort.txt"

if [[ -f "${CAMPAIGN_ROOT}/campaign.json" ]]; then
  "${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["skip_peak_pause"] = True
doc["flavor"] = "${FLAVOR}"
doc["test_mode"] = "grok45_${SET_TAG}_stream_df"
doc["flash_defer_cosim"] = True
doc["model"] = "${MODEL_ID}"
doc["reasoning_effort"] = "${REASONING_EFFORT}"
doc["proxy_backend"] = "xai"
doc["dataflow_exclusive"] = False
doc["latency_opt"] = False
doc["no_cosim_holdout_before_dataflow"] = True
doc["bench_set"] = "${SET_TAG}"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok campaign=${CAMPAIGN_ROOT}"
  exit 0
fi

WATCH_LOG="${CAMPAIGN_ROOT}/flow/post_flash_dataflow_watcher.log"
mkdir -p "${CAMPAIGN_ROOT}/flow"

POST_EXPORT="ALL,C2HLS_RAG=0,C2HLS_RAG_ENABLE=0,C2HLS_RAG_SCRAPE=0,C2HLS_RAG2=0,C2HLS_RAG_MODE=,C2HLS_MODEL=${C2HLS_MODEL},C2HLS_REASONING_EFFORT=${REASONING_EFFORT},C2HLS_OPENAI_UPSTREAM=${C2HLS_OPENAI_UPSTREAM},C2HLS_XAI_HOSTED_URL=${C2HLS_XAI_HOSTED_URL:-https://api.x.ai/v1},Grok_API=${OPENAI_API_KEY},XAI_API_KEY=${OPENAI_API_KEY},C2HLS_PART=${C2HLS_PART},C2HLS_CLOCK_NS=${C2HLS_CLOCK_NS},C2HLS_TMP_RUN=${C2HLS_TMP_RUN},C2HLS_RUN_COSIM=0,C2HLS_REFERENCE_COSIM=0,C2HLS_COSIM_XELAB_MT_OFF=1,C2HLS_FLASH_COSIM_FULL_SIZE=1,C2HLS_FLASH_MAX_TOKENS=${C2HLS_FLASH_MAX_TOKENS},C2HLS_LLM_MAX_TOKENS=${C2HLS_LLM_MAX_TOKENS},C2HLS_COSIM_BENCHMARKS_ROOT=${C2HLS_COSIM_BENCHMARKS_ROOT},C2HLS_DATAFLOW_REPAIR_ROUNDS=${C2HLS_DATAFLOW_REPAIR_ROUNDS},C2HLS_DATAFLOW_CONTRACT_ROUNDS=${C2HLS_DATAFLOW_CONTRACT_ROUNDS},C2HLS_MAX_REPAIR_ATTEMPT=${C2HLS_MAX_REPAIR_ATTEMPT},C2HLS_COSIM_TIMEOUT=${C2HLS_COSIM_TIMEOUT},C2HLS_POST_FLASH_LATENCY_OPT=0,C2HLS_LATENCY_OPT_CHAIN_FLASH=0,C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=0,C2HLS_RANKED_COSIM_AFTER_DATAFLOW=1,C2HLS_DATAFLOW_EXCLUSIVE=0,C2HLS_PER_BENCH_PROXY=0,C2HLS_PROXY_BACKEND=openai,C2HLS_OPENAI_QUEUE_WORKERS=${C2HLS_OPENAI_QUEUE_WORKERS},C2HLS_POST_FLASH_RESULTS_SUFFIX=${C2HLS_POST_FLASH_RESULTS_SUFFIX:-hlsfactory_cosim_repairs},BATCH_PARALLEL_EXTERNAL_LLM=1,BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL=${ENDPOINT_URL},BATCH_PARALLEL_EXTERNAL_MODEL=${BATCH_PARALLEL_EXTERNAL_MODEL},C2HLS_DATAFLOW_NO_SKILLS=${C2HLS_DATAFLOW_NO_SKILLS:-0},C2HLS_PACKAGED_SKILLS_JSON=${C2HLS_PACKAGED_SKILLS_JSON:-},C2HLS_PACKAGED_SKILLS_ONLY=${C2HLS_PACKAGED_SKILLS_ONLY:-0},C2HLS_FORCE_SKILL_PROMPTS=${C2HLS_FORCE_SKILL_PROMPTS:-0},C2HLS_SKILL_PROMPT_MODE=${C2HLS_SKILL_PROMPT_MODE:-},C2HLS_POST_FLASH_PROMPT_POLICY=${C2HLS_POST_FLASH_PROMPT_POLICY},PC2_BATCH_JOB_PREFIX=${PC2_BATCH_JOB_PREFIX},OPENAI_API_KEY=${OPENAI_API_KEY},CHATHLS_API_KEY=${CHATHLS_API_KEY},OPENAI_BASE_URL=${ENDPOINT_URL},C2HLS_OPENAI_HOSTED_URL=${ENDPOINT_URL}"

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
    --wrap="bash ${SCRIPT_DIR}/wait_hlsfactory_flash_lat_dataflow.sh --campaign-root ${CAMPAIGN_ROOT} >> ${WATCH_LOG} 2>&1"
)"

"${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["post_watcher_job_id"] = "${POST_JOB}"
doc["waiter_script"] = "wait_hlsfactory_flash_lat_dataflow.sh"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "submitted post watcher job ${POST_JOB} (stream DF; lat_opt off; grok45 high)"
echo "campaign=${CAMPAIGN_ROOT}"
echo "watch: tail -f ${CAMPAIGN_ROOT}/flow/watch.log"
echo "post:  tail -f ${WATCH_LOG}"

#!/usr/bin/env bash
# Start tier_A DeepSeek RAG2+skills flash→dataflow A/B arm for one suite:
#   forgebench | hp_fft | spector
#
# Usage:
#   ./scripts/pc2/start_tier_a_deepseek_rag2_skills_u280.sh --suite forgebench \
#       --endpoint-url URL [--stamp STAMP] [--latency-opt] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

SUITE=""
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""
LATENCY_OPT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) shift; SUITE="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    --latency-opt) LATENCY_OPT=1; shift ;;
    --no-latency-opt) LATENCY_OPT=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "${SUITE}" in
  forgebench|hp_fft|spector) ;;
  *)
    echo "ERROR: --suite required (forgebench|hp_fft|spector)" >&2
    exit 2
    ;;
esac

TIER_A_READY="${C2HLS_ROOT}/related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready"
if [[ ! -d "${TIER_A_READY}" ]]; then
  echo "ERROR: missing tier_A_ready at ${TIER_A_READY}" >&2
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

export BATCH_PARALLEL_CONFIG="${SCRIPT_DIR}/batch_parallel_${SUITE}_deepseek_u280.json"
export BATCH_PARALLEL_VARIANT="tier_a_90"
export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_${SUITE}_ds_rag2"
export PC2_BATCH_JOB_PREFIX="bpta${SUITE:0:4}"
export C2HLS_MODEL=deepseek-chat
export BATCH_PARALLEL_EXTERNAL_MODEL=deepseek-chat
export C2HLS_COMBINED_HLS=1
export C2HLS_RAG2=1
export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
unset C2HLS_RAG_SCRAPE_CORPUS || true
export C2HLS_RAG_MODE="${C2HLS_RAG_MODE:-everywhere}"
export C2HLS_RAG2_OPT_CORPUS="${C2HLS_RAG2_OPT_CORPUS:-${RAG2_OPT}}"
export C2HLS_RAG2_REPAIR_CORPUS="${C2HLS_RAG2_REPAIR_CORPUS:-${RAG2_REPAIR}}"
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_DEEPSEEK_PEAK_PAUSE=1
export C2HLS_DEEPSEEK_SKIP_PEAK="1"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
export C2HLS_RUN_COSIM=1
export C2HLS_REFERENCE_COSIM=1
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-600}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"
export C2HLS_MAX_REPAIR_ATTEMPT="${C2HLS_MAX_REPAIR_ATTEMPT:-7}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_POST_FLASH_RESULTS_SUFFIX="${SUITE}_cosim_repairs"

if [[ "${LATENCY_OPT}" -eq 1 ]]; then
  export C2HLS_POST_FLASH_LATENCY_OPT=1
  export C2HLS_LATENCY_OPT_CHAIN_FLASH=1
  export C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=1
  export C2HLS_LATENCY_OPT_ROUNDS="${C2HLS_LATENCY_OPT_ROUNDS:-3}"
  export C2HLS_LATENCY_OPT_REPAIR_ROUNDS="${C2HLS_LATENCY_OPT_REPAIR_ROUNDS:-3}"
  export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX}_lat"
else
  unset C2HLS_POST_FLASH_LATENCY_OPT || true
  unset C2HLS_LATENCY_OPT_CHAIN_FLASH || true
  unset C2HLS_LATENCY_OPT_CHAIN_DATAFLOW || true
  export C2HLS_POST_FLASH_LATENCY_OPT=0
fi
export C2HLS_TMP_RUN="${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"
POST_WALLTIME="${PC2_TIER_A_POST_WALLTIME:-7-00:00:00}"

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:18092/v1"
else
  echo "ERROR: --endpoint-url required" >&2
  exit 2
fi
export BATCH_PARALLEL_EXTERNAL_LLM=1
export BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="${ENDPOINT_URL}"

echo "=== tier_A DeepSeek RAG2+skills suite=${SUITE} latency_opt=${LATENCY_OPT} ==="
echo "stamp=${STAMP} config=${BATCH_PARALLEL_CONFIG} endpoint=${ENDPOINT_URL}"

EXTRA_ARGS=(--external-llm)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

env BATCH_PARALLEL_STAMP="${STAMP}" \
  "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
  --stamp "${STAMP}" \
  "${EXTRA_ARGS[@]}"

CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"
mkdir -p "${CAMPAIGN_ROOT}/flow"
echo "rag2_skills" > "${CAMPAIGN_ROOT}/flavor.txt"
echo "${LATENCY_OPT}" > "${CAMPAIGN_ROOT}/latency_opt.txt"
echo "${SUITE}" > "${CAMPAIGN_ROOT}/suite.txt"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok campaign=${CAMPAIGN_ROOT}"
  exit 0
fi

WATCH_LOG="${CAMPAIGN_ROOT}/flow/post_flash_dataflow_watcher.log"
POST_EXPORT="ALL,C2HLS_RAG=0,C2HLS_RAG_ENABLE=0,C2HLS_RAG_SCRAPE=0,C2HLS_RAG2=1,C2HLS_RAG_MODE=${C2HLS_RAG_MODE},C2HLS_RAG2_OPT_CORPUS=${C2HLS_RAG2_OPT_CORPUS},C2HLS_RAG2_REPAIR_CORPUS=${C2HLS_RAG2_REPAIR_CORPUS},C2HLS_MODEL=${C2HLS_MODEL},C2HLS_PART=${C2HLS_PART},C2HLS_CLOCK_NS=${C2HLS_CLOCK_NS},C2HLS_DEEPSEEK_PEAK_PAUSE=${C2HLS_DEEPSEEK_PEAK_PAUSE},C2HLS_TMP_RUN=${C2HLS_TMP_RUN},C2HLS_RUN_COSIM=1,C2HLS_REFERENCE_COSIM=1,C2HLS_DATAFLOW_REPAIR_ROUNDS=${C2HLS_DATAFLOW_REPAIR_ROUNDS},C2HLS_DATAFLOW_CONTRACT_ROUNDS=${C2HLS_DATAFLOW_CONTRACT_ROUNDS},C2HLS_MAX_REPAIR_ATTEMPT=${C2HLS_MAX_REPAIR_ATTEMPT},C2HLS_COSIM_TIMEOUT=${C2HLS_COSIM_TIMEOUT},C2HLS_POST_FLASH_LATENCY_OPT=${C2HLS_POST_FLASH_LATENCY_OPT:-0},C2HLS_LATENCY_OPT_CHAIN_FLASH=${C2HLS_LATENCY_OPT_CHAIN_FLASH:-0},C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=${C2HLS_LATENCY_OPT_CHAIN_DATAFLOW:-0},C2HLS_LATENCY_OPT_ROUNDS=${C2HLS_LATENCY_OPT_ROUNDS:-3},C2HLS_LATENCY_OPT_REPAIR_ROUNDS=${C2HLS_LATENCY_OPT_REPAIR_ROUNDS:-3},C2HLS_POST_FLASH_RESULTS_SUFFIX=${C2HLS_POST_FLASH_RESULTS_SUFFIX},BATCH_PARALLEL_EXTERNAL_LLM=1,BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL=${ENDPOINT_URL},BATCH_PARALLEL_EXTERNAL_MODEL=${BATCH_PARALLEL_EXTERNAL_MODEL},C2HLS_PACKAGED_SKILLS_JSON=${C2HLS_PACKAGED_SKILLS_JSON:-},C2HLS_PACKAGED_SKILLS_ONLY=${C2HLS_PACKAGED_SKILLS_ONLY:-1},C2HLS_FORCE_SKILL_PROMPTS=${C2HLS_FORCE_SKILL_PROMPTS:-1},C2HLS_SKILL_PROMPT_MODE=${C2HLS_SKILL_PROMPT_MODE:-all_skills_avoids_global}"

POST_JOB="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --job-name="bpta-${SUITE}-post" \
    --output="${CAMPAIGN_ROOT}/flow/post_watcher-%j.out" \
    --error="${CAMPAIGN_ROOT}/flow/post_watcher-%j.err" \
    --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task=2 \
    --mem=8G \
    --time="${POST_WALLTIME}" \
    --export="${POST_EXPORT}" \
    --wrap="bash ${SCRIPT_DIR}/wait_hlsfactory_flash_then_dataflow.sh --campaign-root ${CAMPAIGN_ROOT} >> ${WATCH_LOG} 2>&1"
)"

"${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["post_watcher_job_id"] = "${POST_JOB}"
doc["flavor"] = "rag2_skills"
doc["suite"] = "${SUITE}"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "submitted post watcher ${POST_JOB}"
echo "campaign=${CAMPAIGN_ROOT}"

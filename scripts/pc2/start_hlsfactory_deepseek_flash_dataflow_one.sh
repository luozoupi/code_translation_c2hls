#!/usr/bin/env bash
# HLSFactory flash→dataflow with DeepSeek-v4-flash.
# Flavors: skills | noskills | bare
#
# Optional --test modes (lat_opt / rag2_lat / rag2):
#   flash (+lat_opt) → rank → async ranked cosim + immediate dataflow (no cosim)
#   dataflow (+lat_opt) → rank → async ranked cosim
# Without --test: legacy waiter (selected cosim then dataflow+cosim).
#
# Usage:
#   ./scripts/pc2/start_hlsfactory_deepseek_flash_dataflow_one.sh --flavor skills \
#       --endpoint-url http://login:18092/v1 [--stamp STAMP] [--test lat_opt|rag2_lat|rag2] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

FLAVOR=""
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""
TEST_MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flavor) shift; FLAVOR="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    --test) shift; TEST_MODE="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "${FLAVOR}" in
  skills|noskills|bare) ;;
  "")
    echo "ERROR: --flavor required (skills|noskills|bare)" >&2
    exit 2
    ;;
  *)
    echo "ERROR: unknown --flavor '${FLAVOR}'" >&2
    exit 2
    ;;
esac

case "${TEST_MODE}" in
  ""|lat_opt|rag2_lat|rag2) ;;
  *)
    echo "ERROR: unknown --test '${TEST_MODE}' (lat_opt|rag2_lat|rag2)" >&2
    exit 2
    ;;
esac

SKILLS_GEMM="${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"
KR="${C2HLS_ROOT}/artifacts/rag/knowledge_repo"
RAG2_OPT="${C2HLS_ROOT}/artifacts/rag/rag2_opt"
RAG2_REPAIR="${C2HLS_ROOT}/artifacts/rag/rag2_repair"

_rag2_ensure_indexes() {
  if [[ ! -f "${RAG2_OPT}/chunks.jsonl" ]] || [[ ! -f "${RAG2_REPAIR}/chunks.jsonl" ]]; then
    echo "building RAG2 indexes under artifacts/rag/rag2_{opt,repair} ..."
    "${C2HLS_PYTHON:-python3}" "${C2HLS_ROOT}/scripts/build_rag2_indexes.py" \
      --knowledge-repo "${KR}" \
      --opt-out "${RAG2_OPT}" \
      --repair-out "${RAG2_REPAIR}"
  fi
}

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_hlsfactory_deepseek_u280.json}"
export C2HLS_MODEL=deepseek-v4-flash
export BATCH_PARALLEL_EXTERNAL_MODEL=deepseek-v4-flash
export C2HLS_COMBINED_HLS=1
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33

# Defaults: no RAG / RAG2 / latency-opt (overridden by --test)
export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
export C2HLS_RAG2=0
unset C2HLS_RAG_SCRAPE_CORPUS || true
unset C2HLS_RAG2_OPT_CORPUS || true
unset C2HLS_RAG2_REPAIR_CORPUS || true
export C2HLS_POST_FLASH_LATENCY_OPT=0
unset C2HLS_LATENCY_OPT_CHAIN_FLASH || true
unset C2HLS_LATENCY_OPT_CHAIN_DATAFLOW || true
# Only set RAG_MODE when RAG/RAG2 is on — otherwise codegen raises:
# "C2HLS_RAG_MODE set but C2HLS_RAG is not enabled"
unset C2HLS_RAG_MODE || true

# Skip Beijing peak
export C2HLS_DEEPSEEK_PEAK_PAUSE=0
export C2HLS_DEEPSEEK_SKIP_PEAK=1

# Flash: defer cosim; csim+csynth in synth path
export C2HLS_FLASH_DEFER_COSIM=1
export C2HLS_RUN_COSIM=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_COSIM_REQUIRED=0

# Cosim xelab -mt off (selected + dataflow final)
export C2HLS_COSIM_XELAB_MT_OFF=1
export C2HLS_COSIM_TRACE_LEVEL=none

export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-600}"
# Gold-check TB for functional csim (dump PolyBench TB always returns 0).
export C2HLS_CSIM_USE_COSIM_TB="${C2HLS_CSIM_USE_COSIM_TB:-1}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"
export C2HLS_MAX_REPAIR_ATTEMPT="${C2HLS_MAX_REPAIR_ATTEMPT:-7}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export C2HLS_LATENCY_OPT_ROUNDS="${C2HLS_LATENCY_OPT_ROUNDS:-3}"
export C2HLS_LATENCY_OPT_REPAIR_ROUNDS="${C2HLS_LATENCY_OPT_REPAIR_ROUNDS:-3}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
POST_WALLTIME="${PC2_HLSFACTORY_POST_WALLTIME:-7-00:00:00}"

unset C2HLS_BARE_OPT_PROMPTS || true
unset C2HLS_CHATHLS_NOSKILLS || true
unset C2HLS_DATAFLOW_NO_SKILLS || true
unset C2HLS_FLASH_OPT_PROMPT_MODE || true

TEST_TAG=""
case "${TEST_MODE}" in
  lat_opt)
    TEST_TAG="lat_opt"
    export C2HLS_POST_FLASH_LATENCY_OPT=1
    export C2HLS_LATENCY_OPT_CHAIN_FLASH=1
    export C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=1
    export C2HLS_RAG2=0
    unset C2HLS_RAG_MODE || true
    ;;
  rag2_lat)
    TEST_TAG="rag2_lat"
    _rag2_ensure_indexes
    export C2HLS_RAG2=1
    export C2HLS_RAG_MODE="${C2HLS_RAG_MODE:-everywhere}"
    export C2HLS_RAG2_OPT_CORPUS="${C2HLS_RAG2_OPT_CORPUS:-${RAG2_OPT}}"
    export C2HLS_RAG2_REPAIR_CORPUS="${C2HLS_RAG2_REPAIR_CORPUS:-${RAG2_REPAIR}}"
    export C2HLS_POST_FLASH_LATENCY_OPT=1
    export C2HLS_LATENCY_OPT_CHAIN_FLASH=1
    export C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=1
    ;;
  rag2)
    TEST_TAG="rag2"
    _rag2_ensure_indexes
    export C2HLS_RAG2=1
    export C2HLS_RAG_MODE="${C2HLS_RAG_MODE:-everywhere}"
    export C2HLS_RAG2_OPT_CORPUS="${C2HLS_RAG2_OPT_CORPUS:-${RAG2_OPT}}"
    export C2HLS_RAG2_REPAIR_CORPUS="${C2HLS_RAG2_REPAIR_CORPUS:-${RAG2_REPAIR}}"
    export C2HLS_POST_FLASH_LATENCY_OPT=0
    unset C2HLS_LATENCY_OPT_CHAIN_FLASH || true
    unset C2HLS_LATENCY_OPT_CHAIN_DATAFLOW || true
    ;;
esac

case "${FLAVOR}" in
  skills)
    export BATCH_PARALLEL_VARIANT="aav_n"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_hlsfactory_ds_v4f_skills"
    export PC2_BATCH_JOB_PREFIX="bphfs"
    export C2HLS_PACKAGED_SKILLS_JSON="${SKILLS_GEMM}"
    export C2HLS_PACKAGED_SKILLS_ONLY=1
    export C2HLS_FORCE_SKILL_PROMPTS=1
    export C2HLS_SKILL_PROMPT_MODE=all_skills_avoids_global
    FLAVOR_DESC="skills(gemm_flatten)+full HLS-opt prompts"
    ;;
  noskills)
    export BATCH_PARALLEL_VARIANT="noskills"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_hlsfactory_ds_v4f_noskills"
    export PC2_BATCH_JOB_PREFIX="bphfn"
    export C2HLS_DATAFLOW_NO_SKILLS=1
    unset C2HLS_PACKAGED_SKILLS_JSON || true
    unset C2HLS_FORCE_SKILL_PROMPTS || true
    unset C2HLS_SKILL_PROMPT_MODE || true
    FLAVOR_DESC="noskills + keep flash/dataflow HLS-opt wording"
    ;;
  bare)
    export BATCH_PARALLEL_VARIANT="noskills"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_hlsfactory_ds_v4f_bare"
    export PC2_BATCH_JOB_PREFIX="bphfb"
    export C2HLS_DATAFLOW_NO_SKILLS=1
    export C2HLS_BARE_OPT_PROMPTS=1
    unset C2HLS_PACKAGED_SKILLS_JSON || true
    unset C2HLS_FORCE_SKILL_PROMPTS || true
    unset C2HLS_SKILL_PROMPT_MODE || true
    FLAVOR_DESC="bare noskills (no HLS technique prompts)"
    ;;
esac

if [[ -n "${TEST_TAG}" ]]; then
  export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX}_${TEST_TAG}"
  case "${TEST_TAG}" in
    lat_opt) export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX}lo" ;;
    rag2_lat) export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX}rl" ;;
    rag2) export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX}r2" ;;
  esac
  FLAVOR_DESC="${FLAVOR_DESC} +test=${TEST_TAG}"
fi

export C2HLS_TMP_RUN="${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:18092/v1"
  echo "WARNING: --endpoint-url not given; using placeholder ${ENDPOINT_URL} for --dry-run only" >&2
else
  echo "ERROR: --endpoint-url is required for a real start." >&2
  exit 2
fi

export BATCH_PARALLEL_EXTERNAL_LLM=1
export BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="${ENDPOINT_URL}"

# External-llm campaigns require a real API key in the submit shell (workers inherit ALL).
if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" || "${OPENAI_API_KEY}" == "empty" ]]; then
  unset OPENAI_API_KEY || true
  unset CHATHLS_API_KEY || true
  CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
  # shellcheck disable=SC1091
  source "${CHATHLS_ROOT}/scripts/pc2/setup_deepseek_api.sh"
fi
if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" ]]; then
  echo "ERROR: OPENAI_API_KEY missing after setup_deepseek_api.sh" >&2
  exit 2
fi
export CHATHLS_API_KEY="${CHATHLS_API_KEY:-${OPENAI_API_KEY}}"

echo "=== HLSFactory DeepSeek-v4-flash flash+dataflow (${FLAVOR}) ==="
echo "stamp=${STAMP} flavor=${FLAVOR} desc=${FLAVOR_DESC}"
echo "variant=${BATCH_PARALLEL_VARIANT} prefix=${BATCH_PARALLEL_ARTIFACT_PREFIX}"
echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
echo "defer_cosim=1 run_cosim_flash=0 mt_off=${C2HLS_COSIM_XELAB_MT_OFF} skip_peak=1"
echo "test=${TEST_MODE:-none} rag2=${C2HLS_RAG2} lat_opt=${C2HLS_POST_FLASH_LATENCY_OPT} bare=${C2HLS_BARE_OPT_PROMPTS:-0}"

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
echo "${FLAVOR}" > "${CAMPAIGN_ROOT}/flavor.txt"
echo "${TEST_MODE:-legacy}" > "${CAMPAIGN_ROOT}/test_mode.txt"
echo "${C2HLS_POST_FLASH_LATENCY_OPT}" > "${CAMPAIGN_ROOT}/latency_opt.txt"
echo "${C2HLS_RAG2}" > "${CAMPAIGN_ROOT}/rag2.txt"

# Plant skip_peak_pause on campaign.json when present
if [[ -f "${CAMPAIGN_ROOT}/campaign.json" ]]; then
  "${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["skip_peak_pause"] = True
doc["flavor"] = "${FLAVOR}"
doc["test_mode"] = "${TEST_MODE}"
doc["flash_defer_cosim"] = True
doc["c2hls_cosim_xelab_mt_off"] = "1"
doc["rag2"] = "${C2HLS_RAG2}"
doc["latency_opt"] = "${C2HLS_POST_FLASH_LATENCY_OPT}"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok campaign=${CAMPAIGN_ROOT}"
  exit 0
fi

WATCH_LOG="${CAMPAIGN_ROOT}/flow/post_flash_dataflow_watcher.log"
mkdir -p "${CAMPAIGN_ROOT}/flow"

if [[ -n "${TEST_MODE}" ]]; then
  WAITER_SCRIPT="wait_hlsfactory_flash_lat_dataflow.sh"
  # Dataflow RUN_COSIM deferred; ranked cosim async after flash rank + after each df bench.
  POST_EXPORT="ALL,C2HLS_RAG=0,C2HLS_RAG_ENABLE=0,C2HLS_RAG_SCRAPE=0,C2HLS_RAG2=${C2HLS_RAG2},C2HLS_RAG_MODE=${C2HLS_RAG_MODE:-},C2HLS_RAG2_OPT_CORPUS=${C2HLS_RAG2_OPT_CORPUS:-},C2HLS_RAG2_REPAIR_CORPUS=${C2HLS_RAG2_REPAIR_CORPUS:-},C2HLS_MODEL=${C2HLS_MODEL},C2HLS_PART=${C2HLS_PART},C2HLS_CLOCK_NS=${C2HLS_CLOCK_NS},C2HLS_DEEPSEEK_PEAK_PAUSE=0,C2HLS_DEEPSEEK_SKIP_PEAK=1,C2HLS_TMP_RUN=${C2HLS_TMP_RUN},C2HLS_RUN_COSIM=0,C2HLS_REFERENCE_COSIM=0,C2HLS_COSIM_XELAB_MT_OFF=1,C2HLS_FLASH_COSIM_FULL_SIZE=1,C2HLS_COSIM_BENCHMARKS_ROOT=${C2HLS_ROOT}/benchmarks_cosim,C2HLS_DATAFLOW_REPAIR_ROUNDS=${C2HLS_DATAFLOW_REPAIR_ROUNDS},C2HLS_DATAFLOW_CONTRACT_ROUNDS=${C2HLS_DATAFLOW_CONTRACT_ROUNDS},C2HLS_MAX_REPAIR_ATTEMPT=${C2HLS_MAX_REPAIR_ATTEMPT},C2HLS_COSIM_TIMEOUT=${C2HLS_COSIM_TIMEOUT},C2HLS_POST_FLASH_LATENCY_OPT=${C2HLS_POST_FLASH_LATENCY_OPT},C2HLS_LATENCY_OPT_CHAIN_FLASH=${C2HLS_LATENCY_OPT_CHAIN_FLASH:-0},C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=${C2HLS_LATENCY_OPT_CHAIN_DATAFLOW:-0},C2HLS_LATENCY_OPT_ROUNDS=${C2HLS_LATENCY_OPT_ROUNDS},C2HLS_LATENCY_OPT_REPAIR_ROUNDS=${C2HLS_LATENCY_OPT_REPAIR_ROUNDS},C2HLS_RANKED_COSIM_AFTER_DATAFLOW=1,C2HLS_DATAFLOW_EXCLUSIVE=0,C2HLS_POST_FLASH_RESULTS_SUFFIX=${C2HLS_POST_FLASH_RESULTS_SUFFIX:-hlsfactory_cosim_repairs},BATCH_PARALLEL_EXTERNAL_LLM=1,BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL=${ENDPOINT_URL},BATCH_PARALLEL_EXTERNAL_MODEL=${BATCH_PARALLEL_EXTERNAL_MODEL},C2HLS_DATAFLOW_NO_SKILLS=${C2HLS_DATAFLOW_NO_SKILLS:-0},C2HLS_BARE_OPT_PROMPTS=${C2HLS_BARE_OPT_PROMPTS:-0},C2HLS_PACKAGED_SKILLS_JSON=${C2HLS_PACKAGED_SKILLS_JSON:-},C2HLS_PACKAGED_SKILLS_ONLY=${C2HLS_PACKAGED_SKILLS_ONLY:-0},C2HLS_FORCE_SKILL_PROMPTS=${C2HLS_FORCE_SKILL_PROMPTS:-0},C2HLS_SKILL_PROMPT_MODE=${C2HLS_SKILL_PROMPT_MODE:-},PC2_BATCH_JOB_PREFIX=${PC2_BATCH_JOB_PREFIX}"
else
  WAITER_SCRIPT="wait_hlsfactory_flash_then_dataflow.sh"
  POST_EXPORT="ALL,C2HLS_RAG=0,C2HLS_RAG_ENABLE=0,C2HLS_RAG_SCRAPE=0,C2HLS_RAG2=0,C2HLS_MODEL=${C2HLS_MODEL},C2HLS_PART=${C2HLS_PART},C2HLS_CLOCK_NS=${C2HLS_CLOCK_NS},C2HLS_DEEPSEEK_PEAK_PAUSE=0,C2HLS_DEEPSEEK_SKIP_PEAK=1,C2HLS_TMP_RUN=${C2HLS_TMP_RUN},C2HLS_RUN_COSIM=1,C2HLS_REFERENCE_COSIM=1,C2HLS_COSIM_XELAB_MT_OFF=1,C2HLS_FLASH_COSIM_KERNEL=selected,C2HLS_FLASH_COSIM_FULL_SIZE=1,C2HLS_COSIM_BENCHMARKS_ROOT=${C2HLS_ROOT}/benchmarks_cosim,C2HLS_DATAFLOW_REPAIR_ROUNDS=${C2HLS_DATAFLOW_REPAIR_ROUNDS},C2HLS_DATAFLOW_CONTRACT_ROUNDS=${C2HLS_DATAFLOW_CONTRACT_ROUNDS},C2HLS_MAX_REPAIR_ATTEMPT=${C2HLS_MAX_REPAIR_ATTEMPT},C2HLS_COSIM_TIMEOUT=${C2HLS_COSIM_TIMEOUT},C2HLS_POST_FLASH_LATENCY_OPT=0,C2HLS_POST_FLASH_RESULTS_SUFFIX=${C2HLS_POST_FLASH_RESULTS_SUFFIX:-hlsfactory_cosim_repairs},BATCH_PARALLEL_EXTERNAL_LLM=1,BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL=${ENDPOINT_URL},BATCH_PARALLEL_EXTERNAL_MODEL=${BATCH_PARALLEL_EXTERNAL_MODEL},C2HLS_DATAFLOW_NO_SKILLS=${C2HLS_DATAFLOW_NO_SKILLS:-0},C2HLS_BARE_OPT_PROMPTS=${C2HLS_BARE_OPT_PROMPTS:-0},C2HLS_PACKAGED_SKILLS_JSON=${C2HLS_PACKAGED_SKILLS_JSON:-},C2HLS_PACKAGED_SKILLS_ONLY=${C2HLS_PACKAGED_SKILLS_ONLY:-0},C2HLS_FORCE_SKILL_PROMPTS=${C2HLS_FORCE_SKILL_PROMPTS:-0},C2HLS_SKILL_PROMPT_MODE=${C2HLS_SKILL_PROMPT_MODE:-}"
fi

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
    --wrap="bash ${SCRIPT_DIR}/${WAITER_SCRIPT} --campaign-root ${CAMPAIGN_ROOT} >> ${WATCH_LOG} 2>&1"
)"

"${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["post_watcher_job_id"] = "${POST_JOB}"
doc["flavor"] = "${FLAVOR}"
doc["test_mode"] = "${TEST_MODE}"
doc["skip_peak_pause"] = True
doc["waiter_script"] = "${WAITER_SCRIPT}"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "submitted post watcher job ${POST_JOB} (${WAITER_SCRIPT})"
echo "campaign=${CAMPAIGN_ROOT}"
echo "watch: tail -f ${CAMPAIGN_ROOT}/flow/watch.log"
echo "post:  tail -f ${WATCH_LOG}"

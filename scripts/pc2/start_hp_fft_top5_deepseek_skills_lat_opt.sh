#!/usr/bin/env bash
# hp_fft top-5 skills+lat_opt flash→ranked-cosim→dataflow (same knobs as
# start_hlsfactory_deepseek_flash_dataflow_one.sh --flavor skills --test lat_opt),
# but variant/workflow/corpus are tier_A so hp_fft benches resolve.
#
# Usage:
#   BATCH_PARALLEL_CONFIG=scripts/pc2/batch_parallel_hp_fft_top5_lat_opt.json \
#   ./scripts/pc2/start_hp_fft_top5_deepseek_skills_lat_opt.sh \
#     --endpoint-url http://login5:18097/v1 \
#     --stamp YYYYMMDD_HHMMSS_hp_fft_top5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)_hp_fft_top5}"
DRY_RUN=0
ENDPOINT_URL_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

SKILLS_GEMM="${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"
TIER_A_READY="${C2HLS_ROOT}/related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready"
if [[ ! -d "${TIER_A_READY}" ]]; then
  echo "ERROR: missing tier_A_ready at ${TIER_A_READY}" >&2
  exit 2
fi
if [[ ! -f "${SKILLS_GEMM}" ]]; then
  echo "ERROR: missing skills pack ${SKILLS_GEMM}" >&2
  exit 2
fi

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_hp_fft_top5_lat_opt.json}"
export BATCH_PARALLEL_VARIANT="tier_a_90"
export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_hp_fft_ds_v4f_skills_lat_opt"
export PC2_BATCH_JOB_PREFIX="bptahp5lo"
export C2HLS_MODEL=deepseek-v4-flash
export BATCH_PARALLEL_EXTERNAL_MODEL=deepseek-v4-flash
export C2HLS_COMBINED_HLS=1
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33

# skills + lat_opt, no RAG/RAG2 (matches hlsfactory --flavor skills --test lat_opt)
export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
export C2HLS_RAG2=0
unset C2HLS_RAG_SCRAPE_CORPUS || true
unset C2HLS_RAG2_OPT_CORPUS || true
unset C2HLS_RAG2_REPAIR_CORPUS || true
unset C2HLS_RAG_MODE || true
export C2HLS_POST_FLASH_LATENCY_OPT=1
export C2HLS_LATENCY_OPT_CHAIN_FLASH=1
export C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=1
export C2HLS_LATENCY_OPT_ROUNDS="${C2HLS_LATENCY_OPT_ROUNDS:-3}"
export C2HLS_LATENCY_OPT_REPAIR_ROUNDS="${C2HLS_LATENCY_OPT_REPAIR_ROUNDS:-3}"

export C2HLS_DEEPSEEK_PEAK_PAUSE=0
export C2HLS_DEEPSEEK_SKIP_PEAK=1

export C2HLS_FLASH_DEFER_COSIM=1
export C2HLS_RUN_COSIM=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_COSIM_XELAB_MT_OFF=1
export C2HLS_COSIM_TRACE_LEVEL=none

export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-600}"
export C2HLS_CSIM_USE_COSIM_TB="${C2HLS_CSIM_USE_COSIM_TB:-1}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-43200}"
export C2HLS_MAX_REPAIR_ATTEMPT="${C2HLS_MAX_REPAIR_ATTEMPT:-7}"
export C2HLS_DATAFLOW_REPAIR_ROUNDS="${C2HLS_DATAFLOW_REPAIR_ROUNDS:-4}"
export C2HLS_DATAFLOW_CONTRACT_ROUNDS="${C2HLS_DATAFLOW_CONTRACT_ROUNDS:-4}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
POST_WALLTIME="${PC2_HP_FFT_POST_WALLTIME:-7-00:00:00}"

export C2HLS_PACKAGED_SKILLS_JSON="${SKILLS_GEMM}"
export C2HLS_PACKAGED_SKILLS_ONLY=1
export C2HLS_FORCE_SKILL_PROMPTS=1
export C2HLS_SKILL_PROMPT_MODE=all_skills_avoids_global
# Ranked cosim TB lookup: prefer tier_A_ready for hp_fft (not hlsfactory-only benchmarks_cosim).
export C2HLS_COSIM_BENCHMARKS_ROOT="${C2HLS_COSIM_BENCHMARKS_ROOT:-${TIER_A_READY}}"
export C2HLS_POST_FLASH_RESULTS_SUFFIX="${C2HLS_POST_FLASH_RESULTS_SUFFIX:-hp_fft_top5_cosim_repairs}"
export C2HLS_TMP_RUN="${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:18097/v1"
  echo "WARNING: --endpoint-url not given; placeholder for --dry-run" >&2
else
  echo "ERROR: --endpoint-url is required for a real start." >&2
  exit 2
fi
export BATCH_PARALLEL_EXTERNAL_LLM=1
export BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="${ENDPOINT_URL}"

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

echo "=== hp_fft top-5 DeepSeek-v4-flash skills+lat_opt ==="
echo "stamp=${STAMP} config=${BATCH_PARALLEL_CONFIG}"
echo "variant=${BATCH_PARALLEL_VARIANT} prefix=${BATCH_PARALLEL_ARTIFACT_PREFIX}"
echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
echo "skills=${C2HLS_PACKAGED_SKILLS_JSON}"
echo "lat_opt=1 rounds=${C2HLS_LATENCY_OPT_ROUNDS} repair=${C2HLS_LATENCY_OPT_REPAIR_ROUNDS} rag2=0"

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
echo "skills" > "${CAMPAIGN_ROOT}/flavor.txt"
echo "lat_opt" > "${CAMPAIGN_ROOT}/test_mode.txt"
echo "1" > "${CAMPAIGN_ROOT}/latency_opt.txt"
echo "0" > "${CAMPAIGN_ROOT}/rag2.txt"
echo "hp_fft_top5" > "${CAMPAIGN_ROOT}/suite.txt"

if [[ -f "${CAMPAIGN_ROOT}/campaign.json" ]]; then
  "${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["skip_peak_pause"] = True
doc["flavor"] = "skills"
doc["test_mode"] = "lat_opt"
doc["flash_defer_cosim"] = True
doc["c2hls_cosim_xelab_mt_off"] = "1"
doc["rag2"] = "0"
doc["latency_opt"] = "1"
doc["suite"] = "hp_fft_top5"
doc["benches"] = [
    "hp_fft_n1024__UF1",
    "hp_fft_n256__UF1",
    "hp_fft_n256__UF2",
    "hp_fft_n1024__UF2",
    "hp_fft_n256__UF4",
]
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok campaign=${CAMPAIGN_ROOT}"
  exit 0
fi

WATCH_LOG="${CAMPAIGN_ROOT}/flow/post_flash_dataflow_watcher.log"
WAITER_SCRIPT="wait_hlsfactory_flash_lat_dataflow.sh"
POST_EXPORT="ALL,C2HLS_RAG=0,C2HLS_RAG_ENABLE=0,C2HLS_RAG_SCRAPE=0,C2HLS_RAG2=0,C2HLS_MODEL=${C2HLS_MODEL},C2HLS_PART=${C2HLS_PART},C2HLS_CLOCK_NS=${C2HLS_CLOCK_NS},C2HLS_DEEPSEEK_PEAK_PAUSE=0,C2HLS_DEEPSEEK_SKIP_PEAK=1,C2HLS_TMP_RUN=${C2HLS_TMP_RUN},C2HLS_RUN_COSIM=0,C2HLS_REFERENCE_COSIM=0,C2HLS_COSIM_XELAB_MT_OFF=1,C2HLS_FLASH_COSIM_FULL_SIZE=1,C2HLS_COSIM_BENCHMARKS_ROOT=${C2HLS_COSIM_BENCHMARKS_ROOT},C2HLS_DATAFLOW_REPAIR_ROUNDS=${C2HLS_DATAFLOW_REPAIR_ROUNDS},C2HLS_DATAFLOW_CONTRACT_ROUNDS=${C2HLS_DATAFLOW_CONTRACT_ROUNDS},C2HLS_MAX_REPAIR_ATTEMPT=${C2HLS_MAX_REPAIR_ATTEMPT},C2HLS_COSIM_TIMEOUT=${C2HLS_COSIM_TIMEOUT},C2HLS_POST_FLASH_LATENCY_OPT=1,C2HLS_LATENCY_OPT_CHAIN_FLASH=1,C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=1,C2HLS_LATENCY_OPT_ROUNDS=${C2HLS_LATENCY_OPT_ROUNDS},C2HLS_LATENCY_OPT_REPAIR_ROUNDS=${C2HLS_LATENCY_OPT_REPAIR_ROUNDS},C2HLS_RANKED_COSIM_AFTER_DATAFLOW=1,C2HLS_DATAFLOW_EXCLUSIVE=0,C2HLS_POST_FLASH_RESULTS_SUFFIX=${C2HLS_POST_FLASH_RESULTS_SUFFIX},BATCH_PARALLEL_EXTERNAL_LLM=1,BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL=${ENDPOINT_URL},BATCH_PARALLEL_EXTERNAL_MODEL=${BATCH_PARALLEL_EXTERNAL_MODEL},C2HLS_PACKAGED_SKILLS_JSON=${C2HLS_PACKAGED_SKILLS_JSON},C2HLS_PACKAGED_SKILLS_ONLY=1,C2HLS_FORCE_SKILL_PROMPTS=1,C2HLS_SKILL_PROMPT_MODE=${C2HLS_SKILL_PROMPT_MODE},PC2_BATCH_JOB_PREFIX=${PC2_BATCH_JOB_PREFIX}"

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
doc["waiter_script"] = "${WAITER_SCRIPT}"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "submitted post watcher job ${POST_JOB} (${WAITER_SCRIPT})"
echo "campaign=${CAMPAIGN_ROOT}"
echo "watch: tail -f ${CAMPAIGN_ROOT}/flow/watch.log"
echo "post:  tail -f ${WATCH_LOG}"

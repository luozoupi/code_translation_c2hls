#!/usr/bin/env bash
# Start MachSuite tier_B flash+dataflow batch_parallel with DeepSeek external_llm
# and RAG2+90-skills (U280 3.33 ns, combined-HLS nodes).
#
# Model: deepseek-chat via login-node OpenAI-compatible proxy (--endpoint-url).
# After flash completes, post watcher runs wait_machsuite_flash_then_dataflow.sh
# (dataflow + cosim) with RAG2 env propagated via sbatch --export.
#
# Usage:
#   ./scripts/pc2/start_machsuite_deepseek_rag2_skills_u280.sh --dry-run [--endpoint-url URL]
#   ./scripts/pc2/start_machsuite_deepseek_rag2_skills_u280.sh --endpoint-url http://127.0.0.1:PORT/v1
#
# Artifacts: artifacts/pc2/batch_parallel_machsuite_ds_rag2_<stamp>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""
LATENCY_OPT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    --latency-opt) LATENCY_OPT=1; shift ;;
    --no-latency-opt) LATENCY_OPT=0; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

TIER_B_READY="${C2HLS_ROOT}/related_work/benchmarks/HLSFactory_benchmarks/tier_B_ready"
if [[ ! -d "${TIER_B_READY}/machsuite_aes_table" ]]; then
  echo "preparing tier_B_ready MachSuite corpus..."
  "${C2HLS_PYTHON:-python3}" "${C2HLS_ROOT}/scripts/prepare_tier_b_machsuite_ready.py" \
    --output-root "${TIER_B_READY}"
fi

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

_rag2_ensure_indexes

export BATCH_PARALLEL_CONFIG="${SCRIPT_DIR}/batch_parallel_machsuite_deepseek_u280.json"
export BATCH_PARALLEL_VARIANT="tier_b_aav_n"
export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_machsuite_ds_rag2"
export PC2_BATCH_JOB_PREFIX="bpmds"
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
unset C2HLS_CHATHLS_NOSKILLS || true
unset C2HLS_DATAFLOW_NO_SKILLS || true
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
POST_WALLTIME="${PC2_MACHSUITE_POST_WALLTIME:-7-00:00:00}"

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:18092/v1"
  echo "WARNING: --endpoint-url not given; using placeholder ${ENDPOINT_URL} for --dry-run only" >&2
else
  echo "ERROR: --endpoint-url is required for a real (non-dry-run) start." >&2
  echo "       Start the shared DeepSeek proxy first, e.g.:" >&2
  echo "         ./scripts/pc2/c2hls_deepseek_proxy.sh <some_dir>" >&2
  exit 2
fi

export BATCH_PARALLEL_EXTERNAL_LLM=1
export BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="${ENDPOINT_URL}"

echo "=== MachSuite DeepSeek RAG2+skills U280 batch_parallel ==="
echo "stamp=${STAMP}"
echo "config=${BATCH_PARALLEL_CONFIG}"
echo "variant=${BATCH_PARALLEL_VARIANT}"
echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
echo "part=${C2HLS_PART} clock_ns=${C2HLS_CLOCK_NS} combined_hls=1 gpu_policy=always_on"
echo "rag2=1 rag2_opt=${C2HLS_RAG2_OPT_CORPUS} rag2_repair=${C2HLS_RAG2_REPAIR_CORPUS}"
echo "post_watcher_walltime=${POST_WALLTIME}"
echo "tmp_run=${C2HLS_TMP_RUN}"

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
echo "rag2_skills" > "${CAMPAIGN_ROOT}/flavor.txt"
echo "${LATENCY_OPT}" > "${CAMPAIGN_ROOT}/latency_opt.txt"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok"
  echo "campaign=${CAMPAIGN_ROOT}"
  exit 0
fi

WATCH_LOG="${CAMPAIGN_ROOT}/flow/post_flash_dataflow_watcher.log"
mkdir -p "${CAMPAIGN_ROOT}/flow"

POST_EXPORT="ALL,C2HLS_RAG=${C2HLS_RAG},C2HLS_RAG_ENABLE=${C2HLS_RAG_ENABLE},C2HLS_RAG_MODE=${C2HLS_RAG_MODE},C2HLS_RAG_SCRAPE=${C2HLS_RAG_SCRAPE},C2HLS_RAG2=${C2HLS_RAG2},C2HLS_RAG2_OPT_CORPUS=${C2HLS_RAG2_OPT_CORPUS},C2HLS_RAG2_REPAIR_CORPUS=${C2HLS_RAG2_REPAIR_CORPUS},C2HLS_MODEL=${C2HLS_MODEL},C2HLS_PART=${C2HLS_PART},C2HLS_CLOCK_NS=${C2HLS_CLOCK_NS},C2HLS_DEEPSEEK_PEAK_PAUSE=${C2HLS_DEEPSEEK_PEAK_PAUSE},C2HLS_TMP_RUN=${C2HLS_TMP_RUN},C2HLS_RUN_COSIM=${C2HLS_RUN_COSIM},C2HLS_REFERENCE_COSIM=${C2HLS_REFERENCE_COSIM},C2HLS_DATAFLOW_REPAIR_ROUNDS=${C2HLS_DATAFLOW_REPAIR_ROUNDS},C2HLS_DATAFLOW_CONTRACT_ROUNDS=${C2HLS_DATAFLOW_CONTRACT_ROUNDS},C2HLS_MAX_REPAIR_ATTEMPT=${C2HLS_MAX_REPAIR_ATTEMPT},C2HLS_COSIM_TIMEOUT=${C2HLS_COSIM_TIMEOUT},C2HLS_POST_FLASH_LATENCY_OPT=${C2HLS_POST_FLASH_LATENCY_OPT:-0},C2HLS_LATENCY_OPT_CHAIN_FLASH=${C2HLS_LATENCY_OPT_CHAIN_FLASH:-0},C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=${C2HLS_LATENCY_OPT_CHAIN_DATAFLOW:-0},C2HLS_LATENCY_OPT_ROUNDS=${C2HLS_LATENCY_OPT_ROUNDS:-3},C2HLS_LATENCY_OPT_REPAIR_ROUNDS=${C2HLS_LATENCY_OPT_REPAIR_ROUNDS:-3},C2HLS_POST_FLASH_RESULTS_SUFFIX=${C2HLS_POST_FLASH_RESULTS_SUFFIX:-machsuite_cosim_repairs},BATCH_PARALLEL_EXTERNAL_LLM=1,BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL=${ENDPOINT_URL},BATCH_PARALLEL_EXTERNAL_MODEL=${BATCH_PARALLEL_EXTERNAL_MODEL},C2HLS_PACKAGED_SKILLS_JSON=${C2HLS_PACKAGED_SKILLS_JSON:-},C2HLS_PACKAGED_SKILLS_ONLY=${C2HLS_PACKAGED_SKILLS_ONLY:-1},C2HLS_FORCE_SKILL_PROMPTS=${C2HLS_FORCE_SKILL_PROMPTS:-1},C2HLS_SKILL_PROMPT_MODE=${C2HLS_SKILL_PROMPT_MODE:-all_skills_avoids_global}"

POST_JOB="$(
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --job-name="bpmds-post" \
    --output="${CAMPAIGN_ROOT}/flow/post_watcher-%j.out" \
    --error="${CAMPAIGN_ROOT}/flow/post_watcher-%j.err" \
    --account="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task=2 \
    --mem=8G \
    --time="${POST_WALLTIME}" \
    --export="${POST_EXPORT}" \
    --wrap="bash ${SCRIPT_DIR}/wait_machsuite_flash_then_dataflow.sh --campaign-root ${CAMPAIGN_ROOT} >> ${WATCH_LOG} 2>&1"
)"

"${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["post_watcher_job_id"] = "${POST_JOB}"
doc["flavor"] = "rag2_skills"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY

echo "submitted post flash→dataflow watcher job ${POST_JOB}"
echo "campaign=${CAMPAIGN_ROOT}"
echo "watch: tail -f ${CAMPAIGN_ROOT}/flow/watch.log"
echo "post:  tail -f ${WATCH_LOG}"

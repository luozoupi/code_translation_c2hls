#!/usr/bin/env bash
# HLSFactory FLASH-ONLY next5 (gemm, floyd-warshall, bicg, doitgen, fdtd-2d).
# Models: sonnet5 | luna | grok45 | haiku45
# Flavors: skills | noskills
#
# Policy: no dataflow, no lat_opt, no RAG, no RAG2.
# Does NOT modify existing flash→dataflow launchers.
#
# Usage:
#   ./scripts/pc2/start_hlsfactory_flash_only_one.sh \
#       --model sonnet5 --flavor skills [--stamp STAMP] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

MODEL=""
FLAVOR=""
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""
SET_TAG="next5_flash"
DEFAULT_CONFIG="${SCRIPT_DIR}/batch_parallel_hlsfactory_next5_flash.json"
SKILLS_GEMM="${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) shift; MODEL="$1"; shift ;;
    --flavor) shift; FLAVOR="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "${MODEL}" in
  sonnet5|luna|grok45|haiku45) ;;
  "")
    echo "ERROR: --model required (sonnet5|luna|grok45|haiku45)" >&2
    exit 2
    ;;
  *)
    echo "ERROR: unknown --model '${MODEL}'" >&2
    exit 2
    ;;
esac

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

# Scrub parent-shell poison before ANY model/proxy setup.
# Cursor/agent shells retain exports across launches (Grok upstream/key → Luna, etc.).
unset C2HLS_RAG_MODE C2HLS_RAG2_OPT_CORPUS C2HLS_RAG2_REPAIR_CORPUS C2HLS_RAG_SCRAPE_CORPUS || true
unset C2HLS_LATENCY_OPT_CHAIN_FLASH C2HLS_LATENCY_OPT_CHAIN_DATAFLOW || true
unset C2HLS_LATENCY_OPT_ROUNDS C2HLS_LATENCY_OPT_REPAIR_ROUNDS || true
unset C2HLS_OPENAI_UPSTREAM C2HLS_XAI_HOSTED_URL C2HLS_REASONING_EFFORT || true
unset OPENAI_PROXY_MODEL ANTHROPIC_PROXY_MODEL || true
unset OPENAI_API_KEY ANTHROPIC_API_KEY CHATHLS_API_KEY OPENAI_BASE_URL || true
unset Claude_API OPEN_AI_API Grok_API XAI_API_KEY || true
export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
export C2HLS_RAG2=0
export C2HLS_POST_FLASH_LATENCY_OPT=0

export BATCH_PARALLEL_CONFIG="${C2HLS_NEXT5_FLASH_BATCH_CONFIG:-${DEFAULT_CONFIG}}"
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
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
POST_WALLTIME="${PC2_HLSFACTORY_POST_WALLTIME:-7-00:00:00}"

export C2HLS_PER_BENCH_PROXY=0
export C2HLS_RANKED_COSIM_AFTER_DATAFLOW=0

unset C2HLS_BARE_OPT_PROMPTS C2HLS_CHATHLS_NOSKILLS C2HLS_DATAFLOW_NO_SKILLS C2HLS_FLASH_OPT_PROMPT_MODE || true

# Model + proxy + keys
PROXY_BACKEND=""
REASONING_EFFORT=""
case "${MODEL}" in
  sonnet5)
    MODEL_ID="claude-sonnet-5"
    PROXY_BACKEND="anthropic"
    export C2HLS_ANTHROPIC_QUEUE_WORKERS="${C2HLS_ANTHROPIC_QUEUE_WORKERS:-32}"
    export C2HLS_FLASH_MAX_TOKENS="${C2HLS_FLASH_MAX_TOKENS:-32768}"
    export C2HLS_LLM_MAX_TOKENS="${C2HLS_LLM_MAX_TOKENS:-${C2HLS_FLASH_MAX_TOKENS}}"
    ARTIFACT_MODEL="sonnet5"
    case "${FLAVOR}" in
      skills) PROXY_PORT="${C2HLS_ANTHROPIC_PROXY_PORT_NEXT5_SONNET_SKILLS:-18230}"; JOB_PREFIX="bpn5s" ;;
      noskills) PROXY_PORT="${C2HLS_ANTHROPIC_PROXY_PORT_NEXT5_SONNET_NOSKILLS:-18231}"; JOB_PREFIX="bpn5n" ;;
    esac
    ;;
  haiku45)
    MODEL_ID="claude-haiku-4-5-20251001"
    PROXY_BACKEND="anthropic"
    export C2HLS_ANTHROPIC_QUEUE_WORKERS="${C2HLS_ANTHROPIC_QUEUE_WORKERS:-32}"
    export C2HLS_FLASH_MAX_TOKENS="${C2HLS_FLASH_MAX_TOKENS:-32768}"
    if [[ "${C2HLS_FLASH_MAX_TOKENS}" =~ ^[0-9]+$ ]] && (( C2HLS_FLASH_MAX_TOKENS > 64000 )); then
      export C2HLS_FLASH_MAX_TOKENS=64000
    fi
    export C2HLS_LLM_MAX_TOKENS="${C2HLS_LLM_MAX_TOKENS:-${C2HLS_FLASH_MAX_TOKENS}}"
    if [[ "${C2HLS_LLM_MAX_TOKENS}" =~ ^[0-9]+$ ]] && (( C2HLS_LLM_MAX_TOKENS > 64000 )); then
      export C2HLS_LLM_MAX_TOKENS=64000
    fi
    ARTIFACT_MODEL="haiku45"
    case "${FLAVOR}" in
      skills) PROXY_PORT="${C2HLS_ANTHROPIC_PROXY_PORT_NEXT5_HAIKU_SKILLS:-18236}"; JOB_PREFIX="bpnhs" ;;
      noskills) PROXY_PORT="${C2HLS_ANTHROPIC_PROXY_PORT_NEXT5_HAIKU_NOSKILLS:-18237}"; JOB_PREFIX="bpnhn" ;;
    esac
    ;;
  luna)
    MODEL_ID="gpt-5.6-luna"
    PROXY_BACKEND="openai"
    # Force xhigh for Luna (ignore poisoned parent C2HLS_REASONING_EFFORT=high).
    REASONING_EFFORT="${C2HLS_LUNA_REASONING_EFFORT:-xhigh}"
    export C2HLS_REASONING_EFFORT="${REASONING_EFFORT}"
    # Critical: never inherit Grok's C2HLS_OPENAI_UPSTREAM=xai.
    export C2HLS_OPENAI_UPSTREAM="https://api.openai.com/v1"
    unset C2HLS_XAI_HOSTED_URL || true
    export C2HLS_OPENAI_QUEUE_WORKERS="${C2HLS_OPENAI_QUEUE_WORKERS:-16}"
    export C2HLS_FLASH_MAX_TOKENS="${C2HLS_FLASH_MAX_TOKENS:-65536}"
    export C2HLS_LLM_MAX_TOKENS="${C2HLS_LLM_MAX_TOKENS:-65536}"
    ARTIFACT_MODEL="luna"
    case "${FLAVOR}" in
      skills) PROXY_PORT="${C2HLS_OPENAI_PROXY_PORT_NEXT5_LUNA_SKILLS:-18232}"; JOB_PREFIX="bpnls" ;;
      noskills) PROXY_PORT="${C2HLS_OPENAI_PROXY_PORT_NEXT5_LUNA_NOSKILLS:-18233}"; JOB_PREFIX="bpnln" ;;
    esac
    ;;
  grok45)
    MODEL_ID="grok-4.5"
    PROXY_BACKEND="openai"
    REASONING_EFFORT="${C2HLS_GROK_REASONING_EFFORT:-high}"
    export C2HLS_REASONING_EFFORT="${REASONING_EFFORT}"
    export C2HLS_OPENAI_UPSTREAM="${C2HLS_OPENAI_UPSTREAM:-https://api.x.ai/v1}"
    # Normalize short alias.
    if [[ "${C2HLS_OPENAI_UPSTREAM}" == "xai" ]]; then
      export C2HLS_OPENAI_UPSTREAM="https://api.x.ai/v1"
    fi
    export C2HLS_XAI_HOSTED_URL="${C2HLS_XAI_HOSTED_URL:-https://api.x.ai/v1}"
    export C2HLS_OPENAI_QUEUE_WORKERS="${C2HLS_OPENAI_QUEUE_WORKERS:-16}"
    export C2HLS_FLASH_MAX_TOKENS="${C2HLS_FLASH_MAX_TOKENS:-65536}"
    export C2HLS_LLM_MAX_TOKENS="${C2HLS_LLM_MAX_TOKENS:-65536}"
    ARTIFACT_MODEL="grok45"
    case "${FLAVOR}" in
      skills) PROXY_PORT="${C2HLS_OPENAI_PROXY_PORT_NEXT5_GROK_SKILLS:-18234}"; JOB_PREFIX="bpngs" ;;
      noskills) PROXY_PORT="${C2HLS_OPENAI_PROXY_PORT_NEXT5_GROK_NOSKILLS:-18235}"; JOB_PREFIX="bpngn" ;;
    esac
    ;;
esac

export C2HLS_MODEL="${MODEL_ID}"
export BATCH_PARALLEL_EXTERNAL_MODEL="${MODEL_ID}"
export C2HLS_PROXY_BACKEND="${PROXY_BACKEND}"
export PC2_BATCH_JOB_PREFIX="${JOB_PREFIX}"

case "${FLAVOR}" in
  skills)
    export BATCH_PARALLEL_VARIANT="aav_n"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_hlsfactory_${ARTIFACT_MODEL}_${SET_TAG}_skills"
    export C2HLS_PACKAGED_SKILLS_JSON="${SKILLS_GEMM}"
    export C2HLS_PACKAGED_SKILLS_ONLY=1
    export C2HLS_FORCE_SKILL_PROMPTS=1
    export C2HLS_SKILL_PROMPT_MODE=all_skills_avoids_global
    export C2HLS_POST_FLASH_PROMPT_POLICY=system_skills
    FLAVOR_DESC="skills(gemm_flatten)+full HLS-opt prompts"
    ;;
  noskills)
    export BATCH_PARALLEL_VARIANT="noskills"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_hlsfactory_${ARTIFACT_MODEL}_${SET_TAG}_noskills"
    export C2HLS_DATAFLOW_NO_SKILLS=1
    unset C2HLS_PACKAGED_SKILLS_JSON C2HLS_FORCE_SKILL_PROMPTS C2HLS_SKILL_PROMPT_MODE || true
    export C2HLS_POST_FLASH_PROMPT_POLICY=system_skills
    FLAVOR_DESC="noskills + keep flash HLS-opt wording"
    ;;
esac

export C2HLS_TMP_RUN="${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"

_load_key_line() {
  local name="$1"
  local line val
  line="$(grep -E "^[[:space:]]*${name}=" "${HOME}/.bashrc" 2>/dev/null | tail -1 || true)"
  [[ -n "${line}" ]] || return 0
  val="${line#*${name}=}"
  val="${val%\"}"; val="${val#\"}"
  val="${val%\'}"; val="${val#\'}"
  printf '%s' "${val}"
}

if [[ "${PROXY_BACKEND}" == "anthropic" ]]; then
  # Prefer Claude_API / sk-ant-* over poisoned ANTHROPIC_API_KEY (e.g. xai-*).
  if [[ -z "${Claude_API:-}" ]]; then
    Claude_API="$(_load_key_line Claude_API)"
    export Claude_API
  fi
  if [[ -n "${Claude_API:-}" && "${Claude_API}" == sk-ant-* ]]; then
    export ANTHROPIC_API_KEY="${Claude_API}"
  elif [[ -z "${ANTHROPIC_API_KEY:-}" || "${ANTHROPIC_API_KEY}" == "EMPTY" || "${ANTHROPIC_API_KEY}" != sk-ant-* ]]; then
    if [[ -n "${Claude_API:-}" ]]; then
      export ANTHROPIC_API_KEY="${Claude_API}"
    fi
  fi
  if [[ -z "${ANTHROPIC_API_KEY:-}" || ${#ANTHROPIC_API_KEY} -lt 20 || "${ANTHROPIC_API_KEY}" != sk-ant-* ]]; then
    echo "ERROR: ANTHROPIC_API_KEY / Claude_API missing or not sk-ant-*" >&2
    exit 2
  fi
  export OPENAI_API_KEY="${ANTHROPIC_API_KEY}"
  export CHATHLS_API_KEY="${ANTHROPIC_API_KEY}"
else
  # OpenAI / xAI
  if [[ "${MODEL}" == "grok45" ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" ]]; then
      for name in Grok_API XAI_API_KEY GROK_API; do
        val="$(_load_key_line "${name}")"
        if [[ -n "${val}" ]]; then
          export OPENAI_API_KEY="${val}"
          break
        fi
      done
    fi
    export Grok_API="${OPENAI_API_KEY}"
    export XAI_API_KEY="${OPENAI_API_KEY}"
  else
    if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" ]]; then
      for name in OPENAI_API_KEY OPEN_AI_API; do
        val="$(_load_key_line "${name}")"
        if [[ -n "${val}" ]]; then
          export OPENAI_API_KEY="${val}"
          break
        fi
      done
    fi
  fi
  if [[ -z "${OPENAI_API_KEY:-}" || ${#OPENAI_API_KEY} -lt 20 ]]; then
    echo "ERROR: API key missing for ${MODEL}" >&2
    exit 2
  fi
  export CHATHLS_API_KEY="${OPENAI_API_KEY}"
fi

TRIPLE_OR_PROXY_ROOT="${C2HLS_ROOT}/artifacts/pc2/hlsfactory_${ARTIFACT_MODEL}_${SET_TAG}_${STAMP}/proxy_${FLAVOR}"
mkdir -p "${TRIPLE_OR_PROXY_ROOT}"

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:${PROXY_PORT}/v1"
  echo "WARNING: dry-run placeholder endpoint ${ENDPOINT_URL}" >&2
else
  if [[ "${PROXY_BACKEND}" == "anthropic" ]]; then
    export C2HLS_ANTHROPIC_PROXY_PORT="${PROXY_PORT}"
    export ANTHROPIC_PROXY_MODEL="${MODEL_ID}"
    ENDPOINT_URL="$(bash "${SCRIPT_DIR}/c2hls_anthropic_proxy.sh" "${TRIPLE_OR_PROXY_ROOT}")"
  else
    export C2HLS_OPENAI_PROXY_PORT="${PROXY_PORT}"
    export OPENAI_PROXY_MODEL="${MODEL_ID}"
    ENDPOINT_URL="$(bash "${SCRIPT_DIR}/c2hls_openai_proxy.sh" "${TRIPLE_OR_PROXY_ROOT}")"
  fi
fi

export BATCH_PARALLEL_EXTERNAL_LLM=1
export BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="${ENDPOINT_URL}"
export OPENAI_BASE_URL="${ENDPOINT_URL}"
export C2HLS_OPENAI_HOSTED_URL="${ENDPOINT_URL}"
export CHATHLS_API_BASE="${ENDPOINT_URL}"

echo "=== HLSFactory ${MODEL_ID} FLASH-ONLY (${FLAVOR}/${SET_TAG}) ==="
echo "stamp=${STAMP} flavor=${FLAVOR} desc=${FLAVOR_DESC}"
echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
echo "benches=gemm,floyd-warshall,bicg,doitgen,fdtd-2d"
echo "defer_cosim=1 no_dataflow=1 lat_opt=0 rag=0 rag2=0"

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
echo "${ARTIFACT_MODEL}_${SET_TAG}_flash_only" > "${CAMPAIGN_ROOT}/test_mode.txt"
echo "0" > "${CAMPAIGN_ROOT}/latency_opt.txt"
echo "0" > "${CAMPAIGN_ROOT}/rag2.txt"
echo "0" > "${CAMPAIGN_ROOT}/dataflow.txt"
[[ -n "${REASONING_EFFORT}" ]] && echo "${REASONING_EFFORT}" > "${CAMPAIGN_ROOT}/reasoning_effort.txt"

if [[ -f "${CAMPAIGN_ROOT}/campaign.json" ]]; then
  "${C2HLS_PYTHON:-python3}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["skip_peak_pause"] = True
doc["flavor"] = "${FLAVOR}"
doc["test_mode"] = "${ARTIFACT_MODEL}_${SET_TAG}_flash_only"
doc["flash_defer_cosim"] = True
doc["model"] = "${MODEL_ID}"
doc["proxy_backend"] = "${PROXY_BACKEND}"
doc["latency_opt"] = False
doc["dataflow"] = False
doc["bench_set"] = "${SET_TAG}"
if "${REASONING_EFFORT}":
    doc["reasoning_effort"] = "${REASONING_EFFORT}"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run ok campaign=${CAMPAIGN_ROOT}"
  exit 0
fi

WATCH_LOG="${CAMPAIGN_ROOT}/flow/post_flash_cosim_only_watcher.log"
mkdir -p "${CAMPAIGN_ROOT}/flow"

POST_EXPORT="ALL,C2HLS_RAG=0,C2HLS_RAG_ENABLE=0,C2HLS_RAG_SCRAPE=0,C2HLS_RAG2=0,C2HLS_RAG_MODE=,C2HLS_MODEL=${C2HLS_MODEL},C2HLS_PART=${C2HLS_PART},C2HLS_CLOCK_NS=${C2HLS_CLOCK_NS},C2HLS_TMP_RUN=${C2HLS_TMP_RUN},C2HLS_RUN_COSIM=0,C2HLS_REFERENCE_COSIM=0,C2HLS_COSIM_XELAB_MT_OFF=1,C2HLS_FLASH_COSIM_FULL_SIZE=1,C2HLS_FLASH_MAX_TOKENS=${C2HLS_FLASH_MAX_TOKENS},C2HLS_LLM_MAX_TOKENS=${C2HLS_LLM_MAX_TOKENS},C2HLS_COSIM_BENCHMARKS_ROOT=${C2HLS_COSIM_BENCHMARKS_ROOT},C2HLS_MAX_REPAIR_ATTEMPT=${C2HLS_MAX_REPAIR_ATTEMPT},C2HLS_COSIM_TIMEOUT=${C2HLS_COSIM_TIMEOUT},C2HLS_POST_FLASH_LATENCY_OPT=0,C2HLS_LATENCY_OPT_CHAIN_FLASH=0,C2HLS_LATENCY_OPT_CHAIN_DATAFLOW=0,C2HLS_RANKED_COSIM_AFTER_DATAFLOW=0,C2HLS_PER_BENCH_PROXY=0,C2HLS_PROXY_BACKEND=${PROXY_BACKEND},C2HLS_POST_FLASH_RESULTS_SUFFIX=${C2HLS_POST_FLASH_RESULTS_SUFFIX:-hlsfactory_cosim_repairs},BATCH_PARALLEL_EXTERNAL_LLM=1,BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL=${ENDPOINT_URL},BATCH_PARALLEL_EXTERNAL_MODEL=${BATCH_PARALLEL_EXTERNAL_MODEL},C2HLS_DATAFLOW_NO_SKILLS=${C2HLS_DATAFLOW_NO_SKILLS:-0},C2HLS_PACKAGED_SKILLS_JSON=${C2HLS_PACKAGED_SKILLS_JSON:-},C2HLS_PACKAGED_SKILLS_ONLY=${C2HLS_PACKAGED_SKILLS_ONLY:-0},C2HLS_FORCE_SKILL_PROMPTS=${C2HLS_FORCE_SKILL_PROMPTS:-0},C2HLS_SKILL_PROMPT_MODE=${C2HLS_SKILL_PROMPT_MODE:-},C2HLS_POST_FLASH_PROMPT_POLICY=${C2HLS_POST_FLASH_PROMPT_POLICY},PC2_BATCH_JOB_PREFIX=${PC2_BATCH_JOB_PREFIX},OPENAI_API_KEY=${OPENAI_API_KEY},CHATHLS_API_KEY=${CHATHLS_API_KEY},OPENAI_BASE_URL=${ENDPOINT_URL},C2HLS_OPENAI_HOSTED_URL=${ENDPOINT_URL}"

if [[ "${PROXY_BACKEND}" == "anthropic" ]]; then
  POST_EXPORT="${POST_EXPORT},ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY},C2HLS_ANTHROPIC_QUEUE_WORKERS=${C2HLS_ANTHROPIC_QUEUE_WORKERS}"
else
  POST_EXPORT="${POST_EXPORT},C2HLS_OPENAI_QUEUE_WORKERS=${C2HLS_OPENAI_QUEUE_WORKERS},C2HLS_REASONING_EFFORT=${REASONING_EFFORT}"
  if [[ "${MODEL}" == "grok45" ]]; then
    POST_EXPORT="${POST_EXPORT},C2HLS_OPENAI_UPSTREAM=${C2HLS_OPENAI_UPSTREAM},C2HLS_XAI_HOSTED_URL=${C2HLS_XAI_HOSTED_URL},Grok_API=${OPENAI_API_KEY},XAI_API_KEY=${OPENAI_API_KEY}"
  fi
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

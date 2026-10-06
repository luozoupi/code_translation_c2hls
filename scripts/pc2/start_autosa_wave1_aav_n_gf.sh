#!/usr/bin/env bash
# AutoSA wave-1: 64^3 dense GEMM benches except autosa_mm.
# flash (gemm_flatten_v1 + overlay + avoids) -> multi-PE -> stream.
# DeepSeek-v4-flash, U280 3.33 ns, csim+csynth, no cosim/lat-opt/dataflow.
#
# Usage:
#   ./scripts/pc2/start_autosa_wave1_aav_n_gf.sh --dry-run
#   ./scripts/pc2/start_autosa_wave1_aav_n_gf.sh --endpoint-url http://login5:18092/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""
PROXY_PORT="${CHATHLS_DEEPSEEK_PROXY_PORT_AUTOSA_WAVE1:-18124}"

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
export C2HLS_POST_FLASH_DSE=1
export C2HLS_DSE_CHAIN_FLASH=1
export C2HLS_POST_FLASH_STREAM=1
export C2HLS_STREAM_CHAIN_FLASH=1
export C2HLS_STREAM_MAX_TOKENS="${C2HLS_STREAM_MAX_TOKENS:-65536}"
export C2HLS_DSE_MAX_TOKENS="${C2HLS_DSE_MAX_TOKENS:-65536}"
export C2HLS_PHASEB_FROM_GOLD=0
export C2HLS_SKILL_PROMPT_MODE=all_skills_avoids_global

export C2HLS_STRATEGY=flash
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_RUN_COSIM=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_COSIM_TRACE_LEVEL=none
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-1800}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-14400}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-1200}"
export C2HLS_MODEL=deepseek-v4-flash
export BATCH_PARALLEL_EXTERNAL_MODEL=deepseek-v4-flash
export C2HLS_DEEPSEEK_PEAK_PAUSE=0
export C2HLS_DEEPSEEK_SKIP_PEAK=1

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

echo "=== preparing autosa_ready wave-1 kernels ==="
for kid in mm_hcl mm_hcl_intel mm_intel mm_int16 mm_catapult mm_getting_started; do
  "${PY}" "${C2HLS_ROOT}/scripts/prepare_autosa_ready.py" --kernel "${kid}"
done

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_autosa_wave1_aav_n_gf.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-autosa_aav_n_gf}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_autosa_wave1_aav_n_gf}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-bpautw1}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-24:00:00}"
export C2HLS_PACKAGED_SKILLS_JSON="${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"

PROXY_ROOT="${C2HLS_ROOT}/artifacts/pc2/autosa_wave1_aav_n_gf_${STAMP}/proxy"
mkdir -p "${PROXY_ROOT}"

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:${PROXY_PORT}/v1"
  echo "WARNING: --endpoint-url not given; using placeholder ${ENDPOINT_URL} for --dry-run only" >&2
else
  export CHATHLS_DEEPSEEK_PROXY_PORT="${PROXY_PORT}"
  export CHATHLS_DEEPSEEK_QUEUE_WORKERS="${CHATHLS_DEEPSEEK_QUEUE_WORKERS:-1}"
  export DEEPSEEK_PROXY_MODEL=deepseek-v4-flash
  export C2HLS_MODEL=deepseek-v4-flash
  bash "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${PROXY_ROOT}" >/tmp/c2hls_deepseek_proxy_autosa_w1_$$.log 2>&1 || {
    cat /tmp/c2hls_deepseek_proxy_autosa_w1_$$.log >&2
    exit 1
  }
  cat /tmp/c2hls_deepseek_proxy_autosa_w1_$$.log >&2
  ENDPOINT_URL="$(
    "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${PROXY_ROOT}") / "llm_endpoint.json"
print(json.loads(p.read_text())["url"])
PY
  )"
  if [[ -z "${ENDPOINT_URL}" || "${ENDPOINT_URL}" != http* ]]; then
    echo "ERROR: failed to resolve DeepSeek proxy URL from ${PROXY_ROOT}/llm_endpoint.json" >&2
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

echo "=== autosa wave-1 aav_n_gf flash+multi-PE+stream (DeepSeek-v4-flash) ==="
echo "stamp=${STAMP}"
echo "benches=mm_hcl mm_hcl_intel mm_intel mm_int16 mm_catapult mm_getting_started"
echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
echo "clock=${C2HLS_CLOCK_NS} ns part=${C2HLS_PART}"
echo "skills=gemm_flatten_v1 + overlay, prompt=all_skills_avoids_global"
echo "cosim=off lat-opt=off rag=off dataflow=off dse=on stream=on stream_tokens=${C2HLS_STREAM_MAX_TOKENS}"
echo "gate=csynth <= rank1 * 1.02 (U280 3.33ns)"
echo "external_llm=1 (no gpu_h100)"

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
cp -f "${SCRIPT_DIR}/autosa_rank1_u280_targets.json" "${CAMPAIGN_ROOT}/autosa_rank1_u280_targets.json"
echo "deepseek-v4-flash" > "${CAMPAIGN_ROOT}/model.txt"
echo "0" > "${CAMPAIGN_ROOT}/latency_opt.txt"
echo "0" > "${CAMPAIGN_ROOT}/rag2.txt"
echo "0" > "${CAMPAIGN_ROOT}/dataflow.txt"
echo "1" > "${CAMPAIGN_ROOT}/dse.txt"
echo "1" > "${CAMPAIGN_ROOT}/stream.txt"
echo "aav_n_gf" > "${CAMPAIGN_ROOT}/skill_prompt.txt"
echo "gemm_flatten_v1" > "${CAMPAIGN_ROOT}/skills_pack.txt"
echo "wave1" > "${CAMPAIGN_ROOT}/wave.txt"

if [[ -f "${CAMPAIGN_ROOT}/campaign.json" ]]; then
  "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["skip_peak_pause"] = True
doc["model"] = "deepseek-v4-flash"
doc["latency_opt"] = False
doc["rag"] = False
doc["rag2"] = False
doc["dataflow"] = False
doc["dse"] = True
doc["stream"] = True
doc["stream_max_tokens"] = int("${C2HLS_STREAM_MAX_TOKENS}")
doc["skill_prompt_mode"] = "all_skills_avoids_global"
doc["skills_pack"] = "gemm_flatten_v1"
doc["external_llm"] = True
doc["wave"] = 1
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
fi

echo "campaign=${CAMPAIGN_ROOT}"

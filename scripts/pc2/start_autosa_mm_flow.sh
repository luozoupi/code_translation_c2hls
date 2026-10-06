#!/usr/bin/env bash
# autosa_mm only: flash then compute (DSE) then stream I/O. No ping-pong enforcement.
# Same lock as Aug 23/25: DeepSeek-v4-flash, U280 3.33 ns, aav_n_gf.
#
# Usage:
#   ./scripts/pc2/start_autosa_mm_flow.sh --dry-run
#   ./scripts/pc2/start_autosa_mm_flow.sh --endpoint-url http://login5:18092/v1
#   ./scripts/pc2/start_autosa_mm_flow.sh --pe-recipe autosa_mm_32x8 --endpoint-url http://login5:18092/v1
#   ./scripts/pc2/start_autosa_mm_flow.sh --flavor noskills --endpoint-url http://login5:18092/v1
#     job prefix mmns, artifacts batch_parallel_autosa_mm_flow_noskills
#   ./scripts/pc2/start_autosa_mm_flow.sh --flavor zero_shot --endpoint-url http://login5:18092/v1
#     job prefix mmzs, artifacts batch_parallel_autosa_mm_flow_zero_shot
#   ./scripts/pc2/start_autosa_mm_flow.sh --flavor one_shot --endpoint-url http://login5:18092/v1
#     job prefix mm1s, artifacts batch_parallel_autosa_mm_flow_one_shot
#     skip phase B, zero-shot prompt, no FLASH MODE extras, no DSE/stream/floor
#   ./scripts/pc2/start_autosa_mm_flow.sh --dse-v2 --endpoint-url http://login5:18092/v1
#     flash → DSE 2.0 PE×SIMD sweep → stop (no stream; no locked 16×4)
#   ./scripts/pc2/start_autosa_mm_flow.sh --flavor aav_n_90 --dse-v2 --endpoint-url http://login5:18092/v1
#     same DSE 2.0, but flash uses the plain 90-skill pack only (no gemm_flatten, no no-RMW overlay)
#   ./scripts/pc2/start_autosa_mm_flow.sh --dse-v2 --no-thinking --endpoint-url http://login5:18092/v1
#     hosted DeepSeek-v4-flash with thinking.type=disabled (default is thinking on)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""
PROXY_PORT="${CHATHLS_DEEPSEEK_PROXY_PORT_AUTOSA_MM_FLOW:-18126}"
PE_RECIPE=""
FLAVOR="skills"
DSE_V2=0
NO_THINKING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    --pe-recipe) shift; PE_RECIPE="$1"; shift ;;
    --flavor) shift; FLAVOR="$1"; shift ;;
    --dse-v2) DSE_V2=1; shift ;;
    --no-thinking) NO_THINKING=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ "${NO_THINKING}" -eq 1 ]]; then
  export C2HLS_THINKING=disabled
fi

case "${PE_RECIPE}" in
  "")
    ;;
  32x8|autosa_mm_32x8)
    export C2HLS_PE_RECIPE=autosa_mm_32x8
    export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-mm32x8}"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_autosa_mm_32x8_flow_aav_n_gf}"
    ;;
  *)
    echo "ERROR: unknown --pe-recipe '${PE_RECIPE}' (use autosa_mm_32x8 or omit for locked 16x4)" >&2
    exit 2
    ;;
esac

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
export C2HLS_PHASEB_FROM_GOLD=0
export C2HLS_SKILL_PROMPT_MODE=all_skills_avoids_global

export C2HLS_STRATEGY=flash
export C2HLS_AUTOSA_FLOW=1
export C2HLS_ENFORCEMENT=0
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_RUN_COSIM=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_COSIM_TRACE_LEVEL=none
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-1800}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-1200}"
export C2HLS_MODEL=deepseek-v4-flash
export BATCH_PARALLEL_EXTERNAL_MODEL=deepseek-v4-flash
export C2HLS_DEEPSEEK_PEAK_PAUSE=0
export C2HLS_DEEPSEEK_SKIP_PEAK=1
# DeepSeek-v4-flash spent the old 16384-token cap on CoT and never closed a kernel.
export C2HLS_FLASH_MAX_TOKENS="${C2HLS_FLASH_MAX_TOKENS:-65536}"
export C2HLS_LLM_MAX_TOKENS="${C2HLS_LLM_MAX_TOKENS:-${C2HLS_FLASH_MAX_TOKENS}}"
export C2HLS_CPP_CONTINUATIONS="${C2HLS_CPP_CONTINUATIONS:-8}"

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

if [[ -n "${C2HLS_AUTOSA_READY_ROOT:-}" ]]; then
  echo "=== skipping prepare_autosa_ready (C2HLS_AUTOSA_READY_ROOT=${C2HLS_AUTOSA_READY_ROOT}) ==="
else
  echo "=== preparing autosa_ready GEMM kernels ==="
  # shellcheck disable=SC2206
  KERNELS=(${AUTOSA_ENFORCEMENT_KERNELS:-mm})
  for kid in "${KERNELS[@]}"; do
    "${PY}" "${C2HLS_ROOT}/scripts/prepare_autosa_ready.py" --kernel "${kid}"
  done
fi

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_autosa_mm_flow.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-autosa_aav_n_gf}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_autosa_mm_flow_aav_n_gf}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-mmflow}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
export C2HLS_PACKAGED_SKILLS_JSON="${C2HLS_PACKAGED_SKILLS_JSON:-${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json}"

eval "$("${PY}" - "${FLAVOR}" "${SCRIPT_DIR}" <<'PY'
import shlex
import sys

sys.path.insert(0, sys.argv[2])
from autosa_flash_lib import apply_mm_flow_flavor

try:
    apply_mm_flow_flavor(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"ERROR: {exc}") from exc

_KEYS = (
    "C2HLS_MM_FLOW_FLAVOR",
    "BATCH_PARALLEL_VARIANT",
    "PC2_BATCH_JOB_PREFIX",
    "BATCH_PARALLEL_ARTIFACT_PREFIX",
    "C2HLS_POST_FLASH_DSE",
    "C2HLS_DSE_CHAIN_FLASH",
    "C2HLS_DSE_V2",
    "C2HLS_DSE_V2_CHAIN_FLASH",
    "C2HLS_DSE_V2_GRID",
    "C2HLS_POST_FLASH_STREAM",
    "C2HLS_STREAM_CHAIN_FLASH",
    "C2HLS_POST_FLASH_NO_SKILLS",
    "C2HLS_SKIP_PHASE_B",
    "C2HLS_ONE_SHOT",
    "C2HLS_FLASH_OPT_PROMPT_MODE",
    "C2HLS_TURNS",
    "C2HLS_SKILL_MODE",
    "C2HLS_FORCE_SKILL_PROMPTS",
    "C2HLS_SKILL_PROMPT_MODE",
    "C2HLS_PACKAGED_SKILLS_JSON",
    "C2HLS_PACKAGED_SKILLS_ONLY",
    "C2HLS_SKILL_PROMPT_ORDER_JSON",
    "C2HLS_FLASH_SKILL_ENTRIES_JSON",
    "C2HLS_DSE_SKILL_ENTRIES_JSON",
    "C2HLS_STREAM_SKILL_ENTRIES_JSON",
    "C2HLS_FLASH_MIN_DSP",
    "C2HLS_FLASH_MAX_DSP",
    "C2HLS_FLASH_DSP_REDO",
    "C2HLS_FLASH_DSP_FILL_PCT",
    "C2HLS_CANDIDATES_PER_STEP",
    "C2HLS_FLASH_ROW_UF",
    "C2HLS_FLASH_PE_BLK",
    "C2HLS_FLASH_TILE_PP",
    "C2HLS_FLASH_ONCHIP",
)
for key in _KEYS:
    val = __import__("os").environ.get(key)
    if val is None:
        print(f"unset {key} || true")
    else:
        print(f"export {key}={shlex.quote(val)}")
PY
)"

if [[ "${C2HLS_FLASH_ONLY:-0}" == "1" ]]; then
  export C2HLS_POST_FLASH_DSE=0
  export C2HLS_DSE_CHAIN_FLASH=0
  export C2HLS_POST_FLASH_STREAM=0
  export C2HLS_STREAM_CHAIN_FLASH=0
fi

if [[ "${DSE_V2}" == "1" ]]; then
  # After flavor apply: PE×SIMD sweep instead of locked recipe; no stream.
  export C2HLS_DSE_V2=1
  export C2HLS_DSE_V2_CHAIN_FLASH=1
  export C2HLS_POST_FLASH_DSE=0
  export C2HLS_DSE_CHAIN_FLASH=0
  export C2HLS_POST_FLASH_STREAM=0
  export C2HLS_STREAM_CHAIN_FLASH=0
  if [[ "${C2HLS_MM_FLOW_FLAVOR:-}" == "aav_n_90" ]]; then
    export PC2_BATCH_JOB_PREFIX="mmdsev290"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="batch_parallel_autosa_mm_flow_dse_v2_90only"
  else
    export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-mmdsev2}"
    export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_autosa_mm_flow_dse_v2_aav_n_gf}"
  fi
fi

if [[ -n "${C2HLS_SWEEP_JOB_PREFIX:-}" ]]; then
  export PC2_BATCH_JOB_PREFIX="${C2HLS_SWEEP_JOB_PREFIX}"
fi
if [[ -n "${C2HLS_SWEEP_ARTIFACT_PREFIX:-}" ]]; then
  export BATCH_PARALLEL_ARTIFACT_PREFIX="${C2HLS_SWEEP_ARTIFACT_PREFIX}"
fi

FLAVOR_TAG="${C2HLS_MM_FLOW_FLAVOR:-skills}"
if [[ "${FLAVOR_TAG}" == "skills" ]]; then
  FLAVOR_TAG="aav_n_gf"
fi
PROXY_ROOT="${C2HLS_ROOT}/artifacts/pc2/autosa_mm_flow_${FLAVOR_TAG}_${STAMP}/proxy"
mkdir -p "${PROXY_ROOT}"

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:${PROXY_PORT}/v1"
  echo "WARNING: --endpoint-url not given; using placeholder ${ENDPOINT_URL} for --dry-run only" >&2
else
  echo "ERROR: --endpoint-url is required for a real start (e.g. http://login5:18092/v1)." >&2
  exit 2
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

echo "=== autosa_mm flash then compute then I/O (no ping-pong enforcement) ==="
echo "stamp=${STAMP}"
echo "flavor=${C2HLS_MM_FLOW_FLAVOR:-skills} variant=${BATCH_PARALLEL_VARIANT}"
echo "benches=${AUTOSA_ENFORCEMENT_KERNELS:-mm}"
echo "pe_recipe=${C2HLS_PE_RECIPE:-autosa_mm (16x4 locked default)}"
echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
echo "clock=${C2HLS_CLOCK_NS} ns part=${C2HLS_PART}"
echo "autosa_flow=1 enforcement=0 dse=${C2HLS_POST_FLASH_DSE:-0} dse_v2=${C2HLS_DSE_V2:-0} stream=${C2HLS_POST_FLASH_STREAM:-0}"
echo "synth_timeout=${C2HLS_SYNTH_TIMEOUT}s turns=${C2HLS_TURNS:-4} flash_min_dsp=${C2HLS_FLASH_MIN_DSP:-off} flash_max_dsp=${C2HLS_FLASH_MAX_DSP:-off} flash_dsp_redo=${C2HLS_FLASH_DSP_REDO:-off} flash_row_uf=${C2HLS_FLASH_ROW_UF:-off} flash_pe_blk=${C2HLS_FLASH_PE_BLK:-off} flash_tile_pp=${C2HLS_FLASH_TILE_PP:-off} flash_onchip=${C2HLS_FLASH_ONCHIP:-off}"
if [[ -n "${C2HLS_PACKAGED_SKILLS_JSON:-}" ]]; then
  echo "skills=${C2HLS_PACKAGED_SKILLS_JSON} only=${C2HLS_PACKAGED_SKILLS_ONLY:-} prompt=${C2HLS_SKILL_PROMPT_MODE:-all_skills_avoids_global} overlay=${C2HLS_FLASH_SKILL_ENTRIES_JSON:-off}"
else
  echo "skills=none (no packaged JSON, no DSE/stream skill files)"
fi
if [[ -n "${C2HLS_SKILL_PROMPT_ORDER_JSON:-}" ]]; then
  echo "skill_order=${C2HLS_SKILL_PROMPT_ORDER_JSON}"
fi
echo "cosim=off lat-opt=off rag=off skip_phase_b=${C2HLS_SKIP_PHASE_B:-0}"
echo "flash_max_tokens=${C2HLS_FLASH_MAX_TOKENS} continuations=${C2HLS_CPP_CONTINUATIONS}"
echo "thinking=${C2HLS_THINKING:-default-on} flash_only=${C2HLS_FLASH_ONLY:-0}"

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
echo "deepseek-v4-flash" > "${CAMPAIGN_ROOT}/model.txt"
echo "1" > "${CAMPAIGN_ROOT}/autosa_flow.txt"
echo "0" > "${CAMPAIGN_ROOT}/enforcement.txt"
echo "${C2HLS_POST_FLASH_DSE:-0}" > "${CAMPAIGN_ROOT}/dse.txt"
echo "${C2HLS_DSE_V2:-0}" > "${CAMPAIGN_ROOT}/dse_v2.txt"
echo "${C2HLS_POST_FLASH_STREAM:-0}" > "${CAMPAIGN_ROOT}/stream.txt"
echo "${C2HLS_MM_FLOW_FLAVOR:-skills}" > "${CAMPAIGN_ROOT}/flavor.txt"
echo "${C2HLS_SKILL_PROMPT_MODE:-none}" > "${CAMPAIGN_ROOT}/skill_prompt.txt"
echo "${C2HLS_PE_RECIPE:-autosa_mm}" > "${CAMPAIGN_ROOT}/pe_recipe.txt"
echo "${C2HLS_THINKING:-}" > "${CAMPAIGN_ROOT}/thinking.txt"

if [[ -f "${CAMPAIGN_ROOT}/campaign.json" ]]; then
  "${PY}" - <<PY
import json
import os
from pathlib import Path

def _on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["skip_peak_pause"] = True
doc["model"] = "deepseek-v4-flash"
doc["autosa_flow"] = True
doc["enforcement"] = False
doc["post_flash_dse"] = _on("C2HLS_POST_FLASH_DSE")
doc["dse_chain_flash"] = _on("C2HLS_DSE_CHAIN_FLASH")
doc["dse_v2"] = _on("C2HLS_DSE_V2")
doc["dse_v2_chain_flash"] = _on("C2HLS_DSE_V2_CHAIN_FLASH")
doc["post_flash_stream"] = _on("C2HLS_POST_FLASH_STREAM")
doc["stream_chain_flash"] = _on("C2HLS_STREAM_CHAIN_FLASH")
doc["post_flash_no_skills"] = _on("C2HLS_POST_FLASH_NO_SKILLS")
doc["skip_phase_b"] = _on("C2HLS_SKIP_PHASE_B")
doc["one_shot"] = _on("C2HLS_ONE_SHOT")
doc["synth_timeout"] = int("${C2HLS_SYNTH_TIMEOUT}")
doc["latency_opt"] = False
doc["dse"] = _on("C2HLS_POST_FLASH_DSE")
doc["dse_v2_enabled"] = _on("C2HLS_DSE_V2")
doc["stream"] = _on("C2HLS_POST_FLASH_STREAM")
doc["mm_flow_flavor"] = os.environ.get("C2HLS_MM_FLOW_FLAVOR", "skills")
doc["skill_prompt_mode"] = os.environ.get("C2HLS_SKILL_PROMPT_MODE", "")
pack_name = os.environ.get("C2HLS_PACKAGED_SKILLS_JSON", "").strip()
if "gemm_flatten" in pack_name:
    doc["skills_pack"] = "gemm_flatten_v1"
elif "(90skills).json" in pack_name:
    doc["skills_pack"] = "90skills"
elif pack_name:
    doc["skills_pack"] = Path(pack_name).name
else:
    doc["skills_pack"] = "none"
doc["flash_skill_overlay"] = bool(os.environ.get("C2HLS_FLASH_SKILL_ENTRIES_JSON", "").strip())
doc["flash_opt_prompt_mode"] = os.environ.get("C2HLS_FLASH_OPT_PROMPT_MODE", "")
if os.environ.get("C2HLS_TURNS", "").strip():
    doc["turns"] = int(os.environ["C2HLS_TURNS"])
thinking = os.environ.get("C2HLS_THINKING", "").strip()
if thinking:
    doc["thinking"] = thinking
if os.environ.get("C2HLS_FLASH_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
    doc["flash_only"] = True
if os.environ.get("C2HLS_FLASH_MIN_DSP", "").strip().isdigit():
    doc["flash_min_dsp"] = int(os.environ["C2HLS_FLASH_MIN_DSP"])
if os.environ.get("C2HLS_FLASH_MAX_DSP", "").strip().isdigit():
    doc["flash_max_dsp"] = int(os.environ["C2HLS_FLASH_MAX_DSP"])
if os.environ.get("C2HLS_FLASH_DSP_REDO", "").strip().lower() in {"1", "true", "yes", "on"}:
    doc["flash_dsp_redo"] = 1
if os.environ.get("C2HLS_FLASH_DSP_FILL_PCT", "").strip().isdigit():
    doc["flash_dsp_fill_pct"] = int(os.environ["C2HLS_FLASH_DSP_FILL_PCT"])
cands = os.environ.get("C2HLS_CANDIDATES_PER_STEP", "").strip()
if cands:
    doc["candidates_per_step"] = cands
if os.environ.get("C2HLS_FLASH_ROW_UF", "").strip().isdigit():
    doc["flash_row_uf"] = int(os.environ["C2HLS_FLASH_ROW_UF"])
if os.environ.get("C2HLS_FLASH_PE_BLK", "").strip().isdigit():
    doc["flash_pe_blk"] = int(os.environ["C2HLS_FLASH_PE_BLK"])
if os.environ.get("C2HLS_FLASH_TILE_PP", "").strip().lower() in {"1", "true", "yes", "on"}:
    doc["flash_tile_pp"] = 1
if os.environ.get("C2HLS_FLASH_ONCHIP", "").strip().lower() in {"1", "true", "yes", "on"}:
    doc["flash_onchip"] = 1
pack = os.environ.get("C2HLS_PACKAGED_SKILLS_JSON", "").strip()
if pack:
    doc["packaged_skills_json"] = pack
if os.environ.get("C2HLS_PACKAGED_SKILLS_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
    doc["packaged_skills_only"] = 1
order = os.environ.get("C2HLS_SKILL_PROMPT_ORDER_JSON", "").strip()
if order:
    doc["skill_prompt_order_json"] = order
for env, key in (
    ("C2HLS_FLASH_MAX_TOKENS", "flash_max_tokens"),
    ("C2HLS_LLM_MAX_TOKENS", "llm_max_tokens"),
    ("C2HLS_CPP_CONTINUATIONS", "cpp_continuations"),
):
    raw = os.environ.get(env, "").strip()
    if raw.isdigit():
        doc[key] = int(raw)
doc["external_llm"] = True
variant = os.environ.get("BATCH_PARALLEL_VARIANT", "").strip()
if variant:
    doc["active_variants"] = [variant]
    cfg_doc = doc.setdefault("config", {})
    pilot = cfg_doc.setdefault("pilot", {})
    if isinstance(pilot, dict):
        pilot["variant"] = variant
if "${C2HLS_PE_RECIPE:-}":
    doc["pe_recipe"] = "${C2HLS_PE_RECIPE:-}"
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
fi

echo "campaign=${CAMPAIGN_ROOT}"

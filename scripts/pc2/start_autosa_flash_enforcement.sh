#!/usr/bin/env bash
# Flash + ping-pong/DATAFLOW enforcement vs previous sequential flash.
# Same lock as Aug 23/25: DeepSeek-v4-flash, U280 3.33 ns, aav_n_gf, no DSE/stream.
#
# Usage:
#   ./scripts/pc2/start_autosa_flash_enforcement.sh --dry-run
#   ./scripts/pc2/start_autosa_flash_enforcement.sh --endpoint-url http://login5:18092/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""
PROXY_PORT="${CHATHLS_DEEPSEEK_PROXY_PORT_AUTOSA_FLASH_ENF:-18125}"

SEED_FLASH_DIR="${C2HLS_FLASH_SEED_DIR:-}"
LOAD_B_IN_DF=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    --seed-flash) shift; SEED_FLASH_DIR="$1"; shift ;;
    --load-b-in-df) LOAD_B_IN_DF=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "${SEED_FLASH_DIR}" ]]; then
  if [[ ! -f "${SEED_FLASH_DIR}/autosa_mm_flash_opt.cpp" || ! -f "${SEED_FLASH_DIR}/autosa_mm_flash_opt_report.json" ]]; then
    echo "ERROR: --seed-flash needs autosa_mm_flash_opt.cpp and autosa_mm_flash_opt_report.json in ${SEED_FLASH_DIR}" >&2
    exit 2
  fi
  export C2HLS_SKIP_FLASH=1
  export C2HLS_FLASH_SEED_DIR="${SEED_FLASH_DIR}"
  export C2HLS_SKIP_PHASE_B=1
fi

_loadb_raw="$(echo "${C2HLS_PP_LOAD_B_IN_DF:-}" | tr '[:upper:]' '[:lower:]')"
if [[ "${LOAD_B_IN_DF}" -eq 1 || "${_loadb_raw}" == "1" || "${_loadb_raw}" == "true" || "${_loadb_raw}" == "yes" || "${_loadb_raw}" == "on" ]]; then
  export C2HLS_PP_LOAD_B_IN_DF=1
else
  unset C2HLS_PP_LOAD_B_IN_DF || true
fi

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
export C2HLS_PHASEB_FROM_GOLD=0
export C2HLS_SKILL_PROMPT_MODE=all_skills_avoids_global

export C2HLS_STRATEGY=flash
export C2HLS_ENFORCEMENT=1
export C2HLS_ENFORCEMENT_ROUNDS="${C2HLS_ENFORCEMENT_ROUNDS:-20}"
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

echo "=== preparing autosa_ready GEMM kernels ==="
# shellcheck disable=SC2206
KERNELS=(${AUTOSA_ENFORCEMENT_KERNELS:-mm mm_hcl mm_hcl_intel mm_intel mm_int16 mm_catapult mm_getting_started})
for kid in "${KERNELS[@]}"; do
  "${PY}" "${C2HLS_ROOT}/scripts/prepare_autosa_ready.py" --kernel "${kid}"
done

export BATCH_PARALLEL_CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_autosa_flash_enforcement.json}"
export BATCH_PARALLEL_VARIANT="${BATCH_PARALLEL_VARIANT:-autosa_aav_n_gf}"
export BATCH_PARALLEL_ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel_autosa_flash_enf_aav_n_gf}"
export PC2_BATCH_JOB_PREFIX="${PC2_BATCH_JOB_PREFIX:-flshenf}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
export C2HLS_PACKAGED_SKILLS_JSON="${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/skills_ii_target_miss_solutions_added(90skills)_gemm_flatten_v1.json"

PROXY_ROOT="${C2HLS_ROOT}/artifacts/pc2/autosa_flash_enf_aav_n_gf_${STAMP}/proxy"
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

echo "=== autosa flash + ping-pong/DATAFLOW enforcement ==="
echo "stamp=${STAMP}"
echo "benches=${AUTOSA_ENFORCEMENT_KERNELS:-autosa_mm + 6 wave1 GEMMs (serial, mm first)}"
echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
echo "clock=${C2HLS_CLOCK_NS} ns part=${C2HLS_PART}"
echo "enforcement=${C2HLS_ENFORCEMENT} rounds=${C2HLS_ENFORCEMENT_ROUNDS}"
echo "skip_flash=${C2HLS_SKIP_FLASH:-0} seed_dir=${C2HLS_FLASH_SEED_DIR:-}"
echo "pp_load_b_in_dataflow=${C2HLS_PP_LOAD_B_IN_DF:-0}"
echo "synth_timeout=${C2HLS_SYNTH_TIMEOUT}s"
echo "skills=gemm_flatten_v1 + flash_no_RMW overlay, prompt=all_skills_avoids_global"
echo "cosim=off lat-opt=off rag=off dse=off stream=off"
echo "baseline mm flash=24745 cycles interval=24746 dsp=80"

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
echo "1" > "${CAMPAIGN_ROOT}/enforcement.txt"
echo "${C2HLS_ENFORCEMENT_ROUNDS}" > "${CAMPAIGN_ROOT}/enforcement_rounds.txt"
echo "0" > "${CAMPAIGN_ROOT}/dse.txt"
echo "0" > "${CAMPAIGN_ROOT}/stream.txt"
echo "aav_n_gf" > "${CAMPAIGN_ROOT}/skill_prompt.txt"

if [[ -f "${CAMPAIGN_ROOT}/campaign.json" ]]; then
  "${PY}" - <<PY
import json
import os
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["skip_peak_pause"] = True
doc["model"] = "deepseek-v4-flash"
doc["enforcement"] = True
doc["enforcement_rounds"] = int("${C2HLS_ENFORCEMENT_ROUNDS}")
doc["synth_timeout"] = int("${C2HLS_SYNTH_TIMEOUT}")
doc["latency_opt"] = False
doc["dse"] = False
doc["stream"] = False
doc["skill_prompt_mode"] = "all_skills_avoids_global"
doc["skills_pack"] = "gemm_flatten_v1"
doc["keep_flash_skills"] = (
    "hls_full_optimization_skills_schema_1_1_package/"
    "flash_enforcement_keep_flash_skill_entries.json"
)
doc["overlap_judge"] = True
doc["overlap_judge_spec"] = "docs/pc2/2026-09-09-dataflow-pingpong-judge.md"
doc["keep_flash"] = True
doc["pp_load_b_in_dataflow"] = os.environ.get("C2HLS_PP_LOAD_B_IN_DF", "").strip().lower() in {
    "1", "true", "yes", "on",
}
if doc["pp_load_b_in_dataflow"]:
    doc["keep_flash_skills_overlay"] = (
        "hls_full_optimization_skills_schema_1_1_package/"
        "flash_enforcement_loadb_in_dataflow_skill_entries.json"
    )
doc["frozen_old_enforcement"] = "batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043"
doc["frozen_scalar_rewrite"] = "batch_parallel_autosa_mm_enf_aav_n_gf_20260909_085740"
if os.environ.get("C2HLS_SKIP_FLASH", "").strip() in {"1", "true", "yes", "on"}:
    doc["skip_flash"] = True
    doc["skip_phase_b"] = True
    seed_dir = os.environ.get("C2HLS_FLASH_SEED_DIR", "")
    doc["flash_seed_dir"] = seed_dir
    doc["frozen_mmflow_flash"] = "batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow"
    seed_lat = seed_dsp = None
    seed_cpp = ""
    rpt_path = Path(seed_dir) / "autosa_mm_flash_opt_report.json" if seed_dir else None
    cpp_path = Path(seed_dir) / "autosa_mm_flash_opt.cpp" if seed_dir else None
    if rpt_path is not None and rpt_path.is_file():
        try:
            seed_rpt = json.loads(rpt_path.read_text())
            seed_lat = seed_rpt.get("latency_cycles")
            seed_dsp = seed_rpt.get("dsp")
        except (OSError, json.JSONDecodeError):
            seed_rpt = {}
    if cpp_path is not None and cpp_path.is_file():
        try:
            seed_cpp = cpp_path.read_text(encoding="utf-8")
        except OSError:
            seed_cpp = ""
    doc["seed_latency_cycles"] = seed_lat
    doc["seed_dsp"] = seed_dsp
    if "#define PE" in seed_cpp and "#define SIMD" in seed_cpp:
        doc["seed_kind"] = "compute_rewrite"
    else:
        doc["seed_kind"] = "flash"
    wrap_note = (
        "load_B is an INLINE-off tile DATAFLOW task (C2HLS_PP_LOAD_B_IN_DF). "
        if doc.get("pp_load_b_in_dataflow")
        else "B loaded once outside the tile loop. "
    )
    doc["note"] = (
        f"Enforcement-only on seeded {doc['seed_kind']} {seed_lat}/{seed_dsp}. "
        + wrap_note
        + "No flash LLM. Does not overwrite 123043, 085740, 101836, 114519, "
        "121934, 175131, 234214, mmflow, repro2, or repro2_enf."
    )
else:
    wrap_note = (
        "load_B inside the tile DATAFLOW (C2HLS_PP_LOAD_B_IN_DF). "
        if doc.get("pp_load_b_in_dataflow")
        else "B loaded once outside the tile loop. "
    )
    doc["note"] = (
        "Keep-flash enforcement: wrap LANES=16 flash load/compute with tile "
        "DATAFLOW. " + wrap_note +
        "Reject scalar AXI walks and kernel latency > flash x 1.10. "
        "Does not overwrite 12893 or 20260909_085740."
    )
doc["external_llm"] = True
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
fi

echo "campaign=${CAMPAIGN_ROOT}"

#!/usr/bin/env bash
# HLSFactory one-bench flash with the AutoSA onchip pack, then compute rewrite
# / hide load-store (POST_FLASH_DSE + POST_FLASH_STREAM) with the same JSON.
# Not the 90-skill dump. Cosim off. Transfer test: PE_BLK / 64^3 assumptions
# may be wrong for PolyBench; still run.
#
# DSP: no FLASH_MIN_DSP=5000 (would reject non-GEMM). Ceiling 9024.
# Fused A+B reject on via C2HLS_FLASH_ONCHIP=1.
#
# Usage:
#   ./scripts/pc2/start_hlsfactory_kernel_onchip.sh --bench hlsfactory_gemm \
#     --endpoint-url http://login5:18400/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

BENCH=""
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench) shift; BENCH="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${BENCH}" ]]; then
  echo "ERROR: --bench is required (e.g. hlsfactory_gemm)" >&2
  exit 2
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

ONCHIP_JSON="${C2HLS_ROOT}/hls_full_optimization_skills_schema_1_1_package/flash_onchip_wide_gemm_skill_entries.json"

unset BATCH_PARALLEL_ARTIFACT_PREFIX C2HLS_FLASH_SKILL_ENTRIES_JSON C2HLS_PE_RECIPE || true
unset C2HLS_FLASH_MIN_DSP C2HLS_FLASH_PE_BLK C2HLS_FLASH_ROW_UF C2HLS_FLASH_K_TILE || true
unset C2HLS_RAG_MODE C2HLS_RAG2_OPT_CORPUS C2HLS_RAG2_REPAIR_CORPUS || true

eval "$("${PY}" - "${BENCH}" "${SCRIPT_DIR}" <<'PY'
import shlex
import sys

sys.path.insert(0, sys.argv[2])
from hlsfactory_onchip_lib import dsp_policy, hlsfactory_job_prefix

bench = sys.argv[1].strip()
if not bench.startswith("hlsfactory_"):
    raise SystemExit(f"ERROR: expected hlsfactory_* bench, got {bench}")
pol = dsp_policy(bench=bench)
job = hlsfactory_job_prefix(bench)
os_env = __import__("os").environ
os_env["C2HLS_HLSFACTORY_BENCH"] = bench
os_env["BATCH_PARALLEL_VARIANT"] = "onchip"
os_env["C2HLS_FLASH_SKILL_BIN"] = "onchip"
os_env["C2HLS_FLASH_ONCHIP"] = "1"
os_env["C2HLS_PACKAGED_SKILLS_ONLY"] = "1"
os_env["C2HLS_FORCE_SKILL_PROMPTS"] = "1"
os_env["C2HLS_SKILL_PROMPT_MODE"] = "all_skills_avoids_global"
os_env["C2HLS_FLASH_ONLY"] = "0"
os_env["C2HLS_POST_FLASH_DSE"] = "1"
os_env["C2HLS_DSE_CHAIN_FLASH"] = "1"
os_env["C2HLS_POST_FLASH_STREAM"] = "1"
os_env["C2HLS_STREAM_CHAIN_FLASH"] = "1"
os_env.setdefault("C2HLS_TURNS", "7")
os_env["BATCH_PARALLEL_ARTIFACT_PREFIX"] = f"batch_parallel_{bench}_onchip_compute"
os_env["PC2_BATCH_JOB_PREFIX"] = job
if pol["flash_max_dsp"] is not None:
    os_env["C2HLS_FLASH_MAX_DSP"] = str(pol["flash_max_dsp"])
os_env["C2HLS_DSE_MIN_DSP"] = str(pol["dse_min_dsp"])
keys = (
    "C2HLS_HLSFACTORY_BENCH",
    "BATCH_PARALLEL_VARIANT",
    "C2HLS_FLASH_SKILL_BIN",
    "C2HLS_FLASH_ONCHIP",
    "C2HLS_PACKAGED_SKILLS_ONLY",
    "C2HLS_FORCE_SKILL_PROMPTS",
    "C2HLS_SKILL_PROMPT_MODE",
    "C2HLS_FLASH_ONLY",
    "C2HLS_POST_FLASH_DSE",
    "C2HLS_DSE_CHAIN_FLASH",
    "C2HLS_POST_FLASH_STREAM",
    "C2HLS_STREAM_CHAIN_FLASH",
    "C2HLS_TURNS",
    "BATCH_PARALLEL_ARTIFACT_PREFIX",
    "PC2_BATCH_JOB_PREFIX",
    "C2HLS_FLASH_MAX_DSP",
    "C2HLS_DSE_MIN_DSP",
)
import os
for key in keys:
    val = os.environ.get(key)
    if val is None:
        print(f"unset {key} || true")
    else:
        print(f"export {key}={shlex.quote(val)}")
PY
)"

export C2HLS_PACKAGED_SKILLS_JSON="${ONCHIP_JSON}"
export C2HLS_DSE_SKILL_ENTRIES_JSON="${ONCHIP_JSON}"
export C2HLS_STREAM_SKILL_ENTRIES_JSON="${ONCHIP_JSON}"

if [[ -z "${BATCH_PARALLEL_ARTIFACT_PREFIX}" ]]; then
  echo "ERROR: artifact prefix missing" >&2
  exit 2
fi
case "${BATCH_PARALLEL_ARTIFACT_PREFIX}" in
  *20260830_mmflow*|*20260830_mm32x8*|*pe16_20260904_131622*)
    echo "ERROR: refusing frozen mm artifact prefix ${BATCH_PARALLEL_ARTIFACT_PREFIX}" >&2
    exit 2
    ;;
esac

export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
export C2HLS_RAG2=0
export C2HLS_POST_FLASH_LATENCY_OPT=0
export C2HLS_POST_FLASH_PRAGMA_OPT=0
export C2HLS_POST_FLASH_DATAFLOW=0
export C2HLS_COMBINED_HLS=1
export C2HLS_PART=xcu280-fsvh2892-2L-e
export C2HLS_CLOCK_NS=3.33
export C2HLS_FLASH_DEFER_COSIM=1
export C2HLS_RUN_COSIM=0
export C2HLS_REFERENCE_COSIM=0
export C2HLS_COSIM_REQUIRED=0
export C2HLS_COSIM_TRACE_LEVEL=none
export C2HLS_COSIM_XELAB_MT_OFF=1
export C2HLS_CSIM_USE_COSIM_TB="${C2HLS_CSIM_USE_COSIM_TB:-1}"
export C2HLS_CSIM_TIMEOUT="${C2HLS_CSIM_TIMEOUT:-600}"
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-3600}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-1200}"
export C2HLS_MAX_REPAIR_ATTEMPT="${C2HLS_MAX_REPAIR_ATTEMPT:-7}"
export C2HLS_MODEL=deepseek-v4-flash
export BATCH_PARALLEL_EXTERNAL_MODEL=deepseek-v4-flash
export C2HLS_DEEPSEEK_PEAK_PAUSE=0
export C2HLS_DEEPSEEK_SKIP_PEAK=1
export C2HLS_FLASH_MAX_TOKENS="${C2HLS_FLASH_MAX_TOKENS:-65536}"
export C2HLS_LLM_MAX_TOKENS="${C2HLS_LLM_MAX_TOKENS:-${C2HLS_FLASH_MAX_TOKENS}}"
export C2HLS_CPP_CONTINUATIONS="${C2HLS_CPP_CONTINUATIONS:-8}"
export C2HLS_AUTOSA_FLOW=1
export C2HLS_ENFORCEMENT=0
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"
# Do not inherit a 90-skill overlay onto the onchip pack.
unset C2HLS_FLASH_SKILL_ENTRIES_JSON || true

CFG_DIR="${C2HLS_ROOT}/artifacts/pc2/hlsfactory_onchip_configs"
mkdir -p "${CFG_DIR}"
export BATCH_PARALLEL_CONFIG="${CFG_DIR}/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}.json"
"${PY}" - <<PY
import os
import sys
sys.path.insert(0, "${SCRIPT_DIR}")
from hlsfactory_onchip_lib import write_hlsfactory_onchip_config
from pathlib import Path
write_hlsfactory_onchip_config(
    bench=os.environ["C2HLS_HLSFACTORY_BENCH"],
    dest=Path(os.environ["BATCH_PARALLEL_CONFIG"]),
    job_prefix=os.environ["PC2_BATCH_JOB_PREFIX"],
)
print("config=" + os.environ["BATCH_PARALLEL_CONFIG"])
PY

if [[ -n "${ENDPOINT_URL_ARG}" ]]; then
  ENDPOINT_URL="${ENDPOINT_URL_ARG}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  ENDPOINT_URL="http://127.0.0.1:18127/v1"
  echo "WARNING: --endpoint-url not given; using placeholder ${ENDPOINT_URL} for --dry-run only" >&2
else
  echo "ERROR: --endpoint-url is required for a real start (e.g. http://login5:18400/v1)." >&2
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

echo "=== HLSFactory onchip flash + compute rewrite / hide load-store ==="
echo "stamp=${STAMP} bench=${C2HLS_HLSFACTORY_BENCH} variant=${BATCH_PARALLEL_VARIANT}"
echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
echo "skills=${C2HLS_PACKAGED_SKILLS_JSON} only=${C2HLS_PACKAGED_SKILLS_ONLY}"
echo "stages=flash then compute-rewrite then hide-load-store (not dataflow, not 90-skill dump)"
echo "flash_min_dsp=off flash_max_dsp=${C2HLS_FLASH_MAX_DSP:-off} dse_min_dsp=${C2HLS_DSE_MIN_DSP}"
echo "fused_ab_reject=onchip (C2HLS_FLASH_ONCHIP=1)"
echo "prefix=${BATCH_PARALLEL_ARTIFACT_PREFIX} job=${PC2_BATCH_JOB_PREFIX}"
echo "cosim=off"

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
echo "${C2HLS_HLSFACTORY_BENCH}" > "${CAMPAIGN_ROOT}/kernel.txt"
echo "onchip" > "${CAMPAIGN_ROOT}/skill_bin.txt"
echo "1" > "${CAMPAIGN_ROOT}/dse.txt"
echo "1" > "${CAMPAIGN_ROOT}/stream.txt"
echo "${ONCHIP_JSON}" > "${CAMPAIGN_ROOT}/skills_json.txt"

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
doc["kernel"] = os.environ.get("C2HLS_HLSFACTORY_BENCH", "")
doc["flash_skill_bin"] = "onchip"
doc["flash_onchip"] = 1
doc["packaged_skills_json"] = os.environ.get("C2HLS_PACKAGED_SKILLS_JSON", "")
doc["packaged_skills_only"] = 1
doc["dse_skill_entries_json"] = os.environ.get("C2HLS_DSE_SKILL_ENTRIES_JSON", "")
doc["stream_skill_entries_json"] = os.environ.get("C2HLS_STREAM_SKILL_ENTRIES_JSON", "")
doc["post_flash_dse"] = True
doc["dse_chain_flash"] = True
doc["post_flash_stream"] = True
doc["stream_chain_flash"] = True
doc["external_llm"] = True
doc["synth_timeout"] = int("${C2HLS_SYNTH_TIMEOUT}")
if os.environ.get("C2HLS_FLASH_MAX_DSP", "").strip().isdigit():
    doc["flash_max_dsp"] = int(os.environ["C2HLS_FLASH_MAX_DSP"])
if os.environ.get("C2HLS_DSE_MIN_DSP", "").strip().lstrip("-").isdigit():
    doc["dse_min_dsp"] = int(os.environ["C2HLS_DSE_MIN_DSP"])
doc["dsp_floor_note"] = (
    "No FLASH_MIN_DSP=5000 on HLSFactory; ceiling 9024; compute-rewrite min DSP 1."
)
doc["active_variants"] = ["onchip"]
cfg_doc = doc.setdefault("config", {})
pilot = cfg_doc.setdefault("pilot", {})
if isinstance(pilot, dict):
    pilot["variant"] = "onchip"
    pilot["benches"] = [os.environ.get("C2HLS_HLSFACTORY_BENCH", "")]
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
fi

echo "campaign=${CAMPAIGN_ROOT}"

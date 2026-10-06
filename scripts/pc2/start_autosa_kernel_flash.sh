#!/usr/bin/env bash
# Flash-only AutoSA-ready kernel with a curated skill bin.
# Primary pack is spend-DSP on-chip (940-class). Does not load the 90-skill dump.
# Does not chain compute/stream. Does not touch 20260830_mmflow or mm champion stamps.
#
# Usage:
#   ./scripts/pc2/start_autosa_kernel_flash.sh --kernel autosa_mm_hcl --pack onchip \
#     --endpoint-url http://login5:18092/v1
#   ./scripts/pc2/start_autosa_kernel_flash.sh --kernel autosa_mm_hcl --pack generic --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

KERNEL=""
PACK="${C2HLS_FLASH_SKILL_BIN:-onchip}"
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
ENDPOINT_URL_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel) shift; KERNEL="$1"; shift ;;
    --pack) shift; PACK="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --endpoint-url) shift; ENDPOINT_URL_ARG="$1"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

KERNEL="${KERNEL:-${C2HLS_AUTOSA_KERNEL:-}}"
if [[ -z "${KERNEL}" ]]; then
  echo "ERROR: --kernel is required (e.g. autosa_mm_hcl)" >&2
  exit 2
fi

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

unset BATCH_PARALLEL_ARTIFACT_PREFIX C2HLS_FLASH_SKILL_ENTRIES_JSON C2HLS_PE_RECIPE || true

eval "$("${PY}" - "${KERNEL}" "${PACK}" "${SCRIPT_DIR}" <<'PY'
import shlex
import sys

sys.path.insert(0, sys.argv[3])
from autosa_skill_bins import (
    apply_skill_bin,
    default_min_dsp,
    default_pe_blk,
    job_short,
    kernel_record,
    normalize_bench,
    normalize_pack,
    prepare_kernel_id,
    variant_for_pack,
    default_max_dsp,
    default_row_uf,
    default_k_tile,
    needs_onchip_tile,
)

bench = normalize_bench(sys.argv[1])
pack = normalize_pack(sys.argv[2])
rec = kernel_record(bench)
if rec.get("skip"):
    raise SystemExit(f"ERROR: {bench} is skipped (frozen mm study)")
apply_skill_bin(pack)
short = job_short(bench)
os_env = __import__("os").environ
os_env["C2HLS_AUTOSA_KERNEL"] = bench
os_env["C2HLS_FLASH_SKILL_BIN"] = pack
os_env["BATCH_PARALLEL_VARIANT"] = variant_for_pack(pack)
os_env.setdefault("C2HLS_TURNS", "7")
os_env["C2HLS_FLASH_ONLY"] = "1"
os_env["C2HLS_POST_FLASH_DSE"] = "0"
os_env["C2HLS_DSE_CHAIN_FLASH"] = "0"
os_env["C2HLS_POST_FLASH_STREAM"] = "0"
os_env["C2HLS_STREAM_CHAIN_FLASH"] = "0"
if pack == "onchip":
    min_dsp = default_min_dsp(bench, pack)
    pe = default_pe_blk(bench, pack)
    if min_dsp is not None:
        os_env.setdefault("C2HLS_FLASH_MIN_DSP", str(min_dsp))
    if pe is not None:
        os_env.setdefault("C2HLS_FLASH_PE_BLK", str(pe))
    max_dsp = default_max_dsp(bench, pack)
    if max_dsp is not None:
        os_env.setdefault("C2HLS_FLASH_MAX_DSP", str(max_dsp))
    row_uf = default_row_uf(bench, pack)
    if row_uf is not None:
        os_env.setdefault("C2HLS_FLASH_ROW_UF", str(row_uf))
    if needs_onchip_tile(bench):
        os_env.setdefault("C2HLS_FLASH_ONCHIP_TILE", "1")
        k_tile = default_k_tile(bench, pack)
        if k_tile is not None:
            os_env.setdefault("C2HLS_FLASH_K_TILE", str(k_tile))
turns = os_env.get("C2HLS_TURNS", "7")
retry = os_env.get("C2HLS_FLASH_RETRY_TAG", "").strip()
prefix = f"batch_parallel_{bench}_flash_{pack}_t{turns}"
job = f"{short}{retry}{pack[:3]}{turns}"[:10]
os_env["BATCH_PARALLEL_ARTIFACT_PREFIX"] = prefix
os_env["PC2_BATCH_JOB_PREFIX"] = job
os_env["AUTOSA_PREPARE_KERNEL"] = prepare_kernel_id(bench)
keys = (
    "C2HLS_AUTOSA_KERNEL",
    "C2HLS_FLASH_SKILL_BIN",
    "BATCH_PARALLEL_VARIANT",
    "C2HLS_FLASH_ONLY",
    "C2HLS_FLASH_ONCHIP",
    "C2HLS_FLASH_MIN_DSP",
    "C2HLS_FLASH_MAX_DSP",
    "C2HLS_FLASH_PE_BLK",
    "C2HLS_FLASH_ROW_UF",
    "C2HLS_FLASH_ONCHIP_TILE",
    "C2HLS_FLASH_K_TILE",
    "C2HLS_FLASH_RETRY_TAG",
    "C2HLS_TURNS",
    "C2HLS_PACKAGED_SKILLS_JSON",
    "C2HLS_PACKAGED_SKILLS_ONLY",
    "C2HLS_POST_FLASH_DSE",
    "C2HLS_DSE_CHAIN_FLASH",
    "C2HLS_POST_FLASH_STREAM",
    "C2HLS_STREAM_CHAIN_FLASH",
    "C2HLS_POST_FLASH_NO_SKILLS",
    "C2HLS_SKIP_PHASE_B",
    "C2HLS_FLASH_OPT_PROMPT_MODE",
    "C2HLS_SKILL_MODE",
    "C2HLS_FORCE_SKILL_PROMPTS",
    "C2HLS_SKILL_PROMPT_MODE",
    "BATCH_PARALLEL_ARTIFACT_PREFIX",
    "PC2_BATCH_JOB_PREFIX",
    "AUTOSA_PREPARE_KERNEL",
    "C2HLS_FLASH_SKILL_ENTRIES_JSON",
)
import os
for key in keys:
    val = os.environ.get(key)
    if val is None:
        print(f"unset {key} || true")
    else:
        print(f"export {key}={shlex.quote(val)}")
print(f"export C2HLS_KERNEL_FLASH_BENCH={shlex.quote(bench)}")
print(f"export C2HLS_KERNEL_FLASH_PACK={shlex.quote(pack)}")
PY
)"

if [[ -z "${BATCH_PARALLEL_ARTIFACT_PREFIX}" ]]; then
  echo "ERROR: artifact prefix missing after pack apply" >&2
  exit 2
fi
case "${BATCH_PARALLEL_ARTIFACT_PREFIX}" in
  *20260830_mmflow*|*20260830_mm32x8*|*pe16_20260904_131622*)
    echo "ERROR: refusing frozen mm artifact prefix ${BATCH_PARALLEL_ARTIFACT_PREFIX}" >&2
    exit 2
    ;;
esac

"${PY}" "${C2HLS_ROOT}/scripts/prepare_autosa_ready.py" --kernel "${AUTOSA_PREPARE_KERNEL}"

export C2HLS_RAG=0
export C2HLS_RAG_ENABLE=0
export C2HLS_RAG_SCRAPE=0
export C2HLS_RAG2=0
export C2HLS_POST_FLASH_LATENCY_OPT=0
export C2HLS_POST_FLASH_PRAGMA_OPT=0
export C2HLS_POST_FLASH_DATAFLOW=0
export C2HLS_PHASEB_FROM_GOLD=0
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
export C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-14400}"
export C2HLS_COSIM_TIMEOUT="${C2HLS_COSIM_TIMEOUT:-1200}"
export C2HLS_MODEL=deepseek-v4-flash
export BATCH_PARALLEL_EXTERNAL_MODEL=deepseek-v4-flash
export C2HLS_DEEPSEEK_PEAK_PAUSE=0
export C2HLS_DEEPSEEK_SKIP_PEAK=1
export C2HLS_FLASH_MAX_TOKENS="${C2HLS_FLASH_MAX_TOKENS:-65536}"
export C2HLS_LLM_MAX_TOKENS="${C2HLS_LLM_MAX_TOKENS:-${C2HLS_FLASH_MAX_TOKENS}}"
export C2HLS_CPP_CONTINUATIONS="${C2HLS_CPP_CONTINUATIONS:-8}"
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-48:00:00}"

# Config must live outside the campaign root: start_batch_parallel_campaign.sh
# does rm -rf on artifacts/pc2/${PREFIX}_${STAMP}.
CFG_DIR="${C2HLS_ROOT}/artifacts/pc2/kernel_flash_configs"
mkdir -p "${CFG_DIR}"
export BATCH_PARALLEL_CONFIG="${CFG_DIR}/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}.json"
"${PY}" - <<PY
import os
import sys
sys.path.insert(0, "${SCRIPT_DIR}")
from autosa_skill_bins import write_kernel_flash_config
from pathlib import Path
write_kernel_flash_config(
    bench=os.environ["C2HLS_KERNEL_FLASH_BENCH"],
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

echo "=== autosa kernel flash (curated bin, flash-only) ==="
echo "stamp=${STAMP}"
echo "kernel=${C2HLS_KERNEL_FLASH_BENCH} pack=${C2HLS_KERNEL_FLASH_PACK} variant=${BATCH_PARALLEL_VARIANT}"
echo "model=${C2HLS_MODEL} endpoint=${ENDPOINT_URL}"
echo "clock=${C2HLS_CLOCK_NS} ns part=${C2HLS_PART}"
echo "flash_min_dsp=${C2HLS_FLASH_MIN_DSP:-off} flash_max_dsp=${C2HLS_FLASH_MAX_DSP:-off} flash_pe_blk=${C2HLS_FLASH_PE_BLK:-off} flash_onchip=${C2HLS_FLASH_ONCHIP:-off} row_uf=${C2HLS_FLASH_ROW_UF:-off} k_tile=${C2HLS_FLASH_K_TILE:-off} onchip_tile=${C2HLS_FLASH_ONCHIP_TILE:-off}"
echo "fused_ab_reject=onchip-source-gate (two loops load_A then load_B; never load_A_B)"
echo "skills=${C2HLS_PACKAGED_SKILLS_JSON:-none} only=${C2HLS_PACKAGED_SKILLS_ONLY:-}"
echo "prefix=${BATCH_PARALLEL_ARTIFACT_PREFIX} job=${PC2_BATCH_JOB_PREFIX}"
echo "cosim=off lat-opt=off rag=off"

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
echo "${C2HLS_KERNEL_FLASH_BENCH}" > "${CAMPAIGN_ROOT}/kernel.txt"
echo "${C2HLS_KERNEL_FLASH_PACK}" > "${CAMPAIGN_ROOT}/skill_bin.txt"
echo "1" > "${CAMPAIGN_ROOT}/autosa_flow.txt"
echo "0" > "${CAMPAIGN_ROOT}/dse.txt"
echo "0" > "${CAMPAIGN_ROOT}/stream.txt"

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
doc["post_flash_dse"] = False
doc["dse_chain_flash"] = False
doc["post_flash_stream"] = False
doc["stream_chain_flash"] = False
doc["kernel"] = os.environ.get("C2HLS_KERNEL_FLASH_BENCH", "")
doc["flash_skill_bin"] = os.environ.get("C2HLS_KERNEL_FLASH_PACK", "")
doc["synth_timeout"] = int("${C2HLS_SYNTH_TIMEOUT}")
if os.environ.get("C2HLS_TURNS", "").strip():
    doc["turns"] = int(os.environ["C2HLS_TURNS"])
if os.environ.get("C2HLS_FLASH_MIN_DSP", "").strip().isdigit():
    doc["flash_min_dsp"] = int(os.environ["C2HLS_FLASH_MIN_DSP"])
if os.environ.get("C2HLS_FLASH_MAX_DSP", "").strip().isdigit():
    doc["flash_max_dsp"] = int(os.environ["C2HLS_FLASH_MAX_DSP"])
if os.environ.get("C2HLS_FLASH_PE_BLK", "").strip().isdigit():
    doc["flash_pe_blk"] = int(os.environ["C2HLS_FLASH_PE_BLK"])
if os.environ.get("C2HLS_FLASH_ROW_UF", "").strip().isdigit():
    doc["flash_row_uf"] = int(os.environ["C2HLS_FLASH_ROW_UF"])
if os.environ.get("C2HLS_FLASH_K_TILE", "").strip().isdigit():
    doc["flash_k_tile"] = int(os.environ["C2HLS_FLASH_K_TILE"])
if _on("C2HLS_FLASH_ONCHIP"):
    doc["flash_onchip"] = 1
if _on("C2HLS_FLASH_ONCHIP_TILE"):
    doc["flash_onchip_tile"] = 1
pack = os.environ.get("C2HLS_PACKAGED_SKILLS_JSON", "").strip()
if pack:
    doc["packaged_skills_json"] = pack
if _on("C2HLS_PACKAGED_SKILLS_ONLY"):
    doc["packaged_skills_only"] = 1
doc["external_llm"] = True
variant = os.environ.get("BATCH_PARALLEL_VARIANT", "").strip()
if variant:
    doc["active_variants"] = [variant]
    cfg_doc = doc.setdefault("config", {})
    pilot = cfg_doc.setdefault("pilot", {})
    if isinstance(pilot, dict):
        pilot["variant"] = variant
        pilot["benches"] = [os.environ.get("C2HLS_KERNEL_FLASH_BENCH", "")]
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
fi

echo "campaign=${CAMPAIGN_ROOT}"

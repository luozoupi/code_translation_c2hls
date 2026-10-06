#!/usr/bin/env bash
# Gold-gate csynth of one autosa_ready seed (plain.cpp / hls_baseline.cpp /
# kernel.h ABI). No LLM, no flash rewrite, no onchip 940 pack.
#
# Usage:
#   ./scripts/pc2/start_autosa_kernel_seed_synth.sh --kernel autosa_mm_hcl
#   ./scripts/pc2/start_autosa_kernel_seed_synth.sh --kernel autosa_lu --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

KERNEL=""
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel) shift; KERNEL="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
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
unset C2HLS_FLASH_MIN_DSP C2HLS_FLASH_MAX_DSP C2HLS_FLASH_PE_BLK C2HLS_FLASH_ONCHIP || true
unset C2HLS_FLASH_ROW_UF C2HLS_FLASH_K_TILE C2HLS_FLASH_ONCHIP_TILE || true
unset C2HLS_PACKAGED_SKILLS_JSON C2HLS_PACKAGED_SKILLS_ONLY C2HLS_FLASH_SKILL_BIN || true

eval "$("${PY}" - "${KERNEL}" "${SCRIPT_DIR}" <<'PY'
import shlex
import sys

sys.path.insert(0, sys.argv[2])
from autosa_skill_bins import job_short, kernel_record, normalize_bench, prepare_kernel_id

bench = normalize_bench(sys.argv[1])
rec = kernel_record(bench)
if rec.get("skip"):
    raise SystemExit(f"ERROR: {bench} is skipped (frozen mm study)")
short = job_short(bench)
os_env = __import__("os").environ
os_env["C2HLS_AUTOSA_KERNEL"] = bench
os_env["BATCH_PARALLEL_VARIANT"] = "autosa_gold"
os_env["C2HLS_FLASH_ONLY"] = "0"
os_env["C2HLS_REFERENCE_ONLY"] = "1"
os_env["C2HLS_SKIP_PHASE_B"] = "1"
os_env["C2HLS_POST_FLASH_DSE"] = "0"
os_env["C2HLS_DSE_CHAIN_FLASH"] = "0"
os_env["C2HLS_POST_FLASH_STREAM"] = "0"
os_env["C2HLS_STREAM_CHAIN_FLASH"] = "0"
os_env["C2HLS_TURNS"] = "0"
prefix = f"batch_parallel_{bench}_seed_synth"
job = f"{short}sd"[:10]
os_env["BATCH_PARALLEL_ARTIFACT_PREFIX"] = prefix
os_env["PC2_BATCH_JOB_PREFIX"] = job
os_env["AUTOSA_PREPARE_KERNEL"] = prepare_kernel_id(bench)
keys = (
    "C2HLS_AUTOSA_KERNEL",
    "BATCH_PARALLEL_VARIANT",
    "C2HLS_FLASH_ONLY",
    "C2HLS_REFERENCE_ONLY",
    "C2HLS_SKIP_PHASE_B",
    "C2HLS_POST_FLASH_DSE",
    "C2HLS_DSE_CHAIN_FLASH",
    "C2HLS_POST_FLASH_STREAM",
    "C2HLS_STREAM_CHAIN_FLASH",
    "C2HLS_TURNS",
    "BATCH_PARALLEL_ARTIFACT_PREFIX",
    "PC2_BATCH_JOB_PREFIX",
    "AUTOSA_PREPARE_KERNEL",
)
import os
for key in keys:
    val = os.environ.get(key)
    if val is None:
        print(f"unset {key} || true")
    else:
        print(f"export {key}={shlex.quote(val)}")
print(f"export C2HLS_KERNEL_SEED_BENCH={shlex.quote(bench)}")
PY
)"

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
export C2HLS_MODEL=none
export BATCH_PARALLEL_EXTERNAL_MODEL=none
export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME:-24:00:00}"

CFG_DIR="${C2HLS_ROOT}/artifacts/pc2/kernel_seed_synth_configs"
mkdir -p "${CFG_DIR}"
export BATCH_PARALLEL_CONFIG="${CFG_DIR}/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}.json"
"${PY}" - <<PY
import os
import sys
sys.path.insert(0, "${SCRIPT_DIR}")
from autosa_skill_bins import write_kernel_seed_synth_config
from pathlib import Path
write_kernel_seed_synth_config(
    bench=os.environ["C2HLS_KERNEL_SEED_BENCH"],
    dest=Path(os.environ["BATCH_PARALLEL_CONFIG"]),
    job_prefix=os.environ["PC2_BATCH_JOB_PREFIX"],
)
print("config=" + os.environ["BATCH_PARALLEL_CONFIG"])
PY

echo "=== autosa kernel seed synth (no LLM, no flash) ==="
echo "stamp=${STAMP}"
echo "kernel=${C2HLS_KERNEL_SEED_BENCH} variant=${BATCH_PARALLEL_VARIANT}"
echo "clock=${C2HLS_CLOCK_NS} ns part=${C2HLS_PART}"
echo "prefix=${BATCH_PARALLEL_ARTIFACT_PREFIX} job=${PC2_BATCH_JOB_PREFIX}"
echo "cosim=off llm=off flash=off gold=hls_baseline.cpp"

EXTRA_ARGS=(--no-gpu)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

env BATCH_PARALLEL_STAMP="${STAMP}" \
  "${SCRIPT_DIR}/start_batch_parallel_campaign.sh" \
  --stamp "${STAMP}" \
  "${EXTRA_ARGS[@]}"

CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${BATCH_PARALLEL_ARTIFACT_PREFIX}_${STAMP}"
mkdir -p "${CAMPAIGN_ROOT}"
echo "none" > "${CAMPAIGN_ROOT}/model.txt"
echo "${C2HLS_KERNEL_SEED_BENCH}" > "${CAMPAIGN_ROOT}/kernel.txt"
echo "seed_synth" > "${CAMPAIGN_ROOT}/skill_bin.txt"
echo "0" > "${CAMPAIGN_ROOT}/dse.txt"
echo "0" > "${CAMPAIGN_ROOT}/stream.txt"
echo "1" > "${CAMPAIGN_ROOT}/reference_only.txt"

if [[ -f "${CAMPAIGN_ROOT}/campaign.json" ]]; then
  "${PY}" - <<PY
import json
import os
from pathlib import Path

p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text()) if p.is_file() else {}
doc["no_gpu"] = True
doc["model"] = "none"
doc["autosa_flow"] = True
doc["reference_only"] = True
doc["kernel"] = os.environ.get("C2HLS_KERNEL_SEED_BENCH", "")
doc["synth_timeout"] = int("${C2HLS_SYNTH_TIMEOUT}")
doc["post_flash_dse"] = False
doc["post_flash_stream"] = False
doc["external_llm"] = False
doc["active_variants"] = ["autosa_gold"]
cfg_doc = doc.setdefault("config", {})
pilot = cfg_doc.setdefault("pilot", {})
if isinstance(pilot, dict):
    pilot["variant"] = "autosa_gold"
    pilot["workflow"] = "autosa_gold"
    pilot["benches"] = [os.environ.get("C2HLS_KERNEL_SEED_BENCH", "")]
p.write_text(json.dumps(doc, indent=2) + "\n")
PY
fi

echo "campaign=${CAMPAIGN_ROOT}"

#!/usr/bin/env bash
# Start a batch_parallel campaign (pilot or full) from a JSON config.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

CONFIG="${BATCH_PARALLEL_CONFIG:-${SCRIPT_DIR}/batch_parallel_pilot.json}"
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
VARIANT="${BATCH_PARALLEL_VARIANT:-}"
DRY_RUN=0
FOREGROUND_COORD=0
BORROW_GPU=0
EXTERNAL_LLM=0
NO_GPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) shift; CONFIG="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --variant) shift; VARIANT="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --foreground-coordinator) FOREGROUND_COORD=1; shift ;;
    --borrow-gpu) BORROW_GPU=1; shift ;;
    --no-borrow-gpu) BORROW_GPU=0; shift ;;
    --external-llm) EXTERNAL_LLM=1; shift ;;
    --no-gpu) NO_GPU=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: BATCH_PARALLEL_CONFIG is not a file: ${CONFIG}" >&2
  exit 2
fi

if [[ "${BATCH_PARALLEL_EXTERNAL_LLM:-0}" == "1" ]]; then
  EXTERNAL_LLM=1
fi
if [[ "${BATCH_PARALLEL_NO_GPU:-0}" == "1" ]]; then
  NO_GPU=1
fi
if [[ "${EXTERNAL_LLM}" -eq 1 && "${BORROW_GPU}" -eq 1 ]]; then
  echo "ERROR: --external-llm and --borrow-gpu are mutually exclusive" >&2
  exit 2
fi
if [[ "${NO_GPU}" -eq 1 && "${EXTERNAL_LLM}" -eq 1 ]]; then
  echo "ERROR: --no-gpu and --external-llm are mutually exclusive" >&2
  exit 2
fi
if [[ "${NO_GPU}" -eq 1 && "${BORROW_GPU}" -eq 1 ]]; then
  echo "ERROR: --no-gpu and --borrow-gpu are mutually exclusive" >&2
  exit 2
fi

EXTERNAL_ENDPOINT_URL="${BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL:-}"
EXTERNAL_MODEL="${BATCH_PARALLEL_EXTERNAL_MODEL:-deepseek-chat}"
if [[ "${EXTERNAL_LLM}" -eq 1 ]]; then
  if [[ -z "${EXTERNAL_ENDPOINT_URL}" ]]; then
    echo "ERROR: --external-llm requires a non-empty BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL" >&2
    exit 1
  fi
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      echo "WARNING: OPENAI_API_KEY is not set (dry-run; required for real external-llm runs)" >&2
    else
      echo "ERROR: OPENAI_API_KEY must be set for real external-llm runs" >&2
      exit 1
    fi
  fi
fi

# PY must be resolved before use below (PC2_BATCH_JOB_PREFIX default lookup).
PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

export BATCH_PARALLEL_CONFIG="${CONFIG}"
export PC2_JOB_TAG="${PC2_JOB_TAG:-${STAMP}}"
if [[ -z "${PC2_BATCH_JOB_PREFIX:-}" ]]; then
  PC2_BATCH_JOB_PREFIX="$("${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["BATCH_PARALLEL_CONFIG"])
d = json.loads(p.read_text())
print(d.get("job_prefix", "bpcplx"))
PY
)"
fi
export PC2_BATCH_JOB_PREFIX
# Slurm walltime must exceed cosim_timeout_s (default 12h); common.sh defaults to 3h.
# Prefer PC2_BATCH_PARALLEL_WALLTIME; else keep a caller-set PC2_FORCE_WALLTIME
# (e.g. Devstral one-shot exports 48h); else default 13h.
# Do NOT clobber an already-set PC2_FORCE_WALLTIME when BATCH_PARALLEL is unset —
# that previously forced Devstral GPUs down to 13h despite the one-shot's 48h.
if [[ -n "${PC2_BATCH_PARALLEL_WALLTIME:-}" ]]; then
  export PC2_FORCE_WALLTIME="${PC2_BATCH_PARALLEL_WALLTIME}"
elif [[ -z "${PC2_FORCE_WALLTIME:-}" ]]; then
  export PC2_FORCE_WALLTIME="13:00:00"
fi
# common.sh was sourced above; refresh WALLTIME to match the resolved FORCE.
export PC2_WALLTIME="${PC2_FORCE_WALLTIME}"

if [[ -z "${VARIANT}" ]]; then
  VARIANT="$("${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["BATCH_PARALLEL_CONFIG"])
print(json.loads(p.read_text())["pilot"]["variant"])
PY
)"
fi

ARTIFACT_PREFIX="${BATCH_PARALLEL_ARTIFACT_PREFIX:-batch_parallel}"
CAMPAIGN_ROOT="${C2HLS_ROOT}/artifacts/pc2/${ARTIFACT_PREFIX}_${STAMP}"
export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
rm -rf "${CAMPAIGN_ROOT}"
mkdir -p "${CAMPAIGN_ROOT}/flow/snapshots" "${CAMPAIGN_ROOT}/variants"

read -r BENCH_COUNT SYNTH_NODES SYNTH_WPN COSIM_NODES COSIM_WPN <<<"$(
  "${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["BATCH_PARALLEL_CONFIG"])
d = json.loads(p.read_text())
print(len(d["pilot"]["benches"]), d["synth_nodes_per_variant"], d["synth_workers_per_node"], d["cosim_nodes_per_variant"], d["cosim_workers_per_node"])
PY
)"

"${PY}" - <<PY
import sys
sys.path.insert(0, "${C2HLS_ROOT}/scripts/pc2")
from batch_parallel_config import init_campaign_json, load_config, campaign_paths, benches_for_config, seed_kwargs_for_workflow
from batch_parallel_queue import BatchParallelQueue
cfg = load_config()
paths = campaign_paths(__import__("pathlib").Path("${CAMPAIGN_ROOT}"))
init_campaign_json(paths["root"], cfg, stamp="${STAMP}", active_variants=["${VARIANT}"])
queue = BatchParallelQueue(paths["queue_db"])
benches = cfg.sort_benches(benches_for_config(cfg))
queue.register_benches("${VARIANT}", benches)
seed_kw = seed_kwargs_for_workflow(cfg.pilot_workflow)
seeded = queue.seed_initial_wave("${VARIANT}", benches, max_inflight=cfg.max_inflight_benches, seed_kwargs=seed_kw)
print("campaign_root=${CAMPAIGN_ROOT}")
print("config=${CONFIG}")
print("variant=${VARIANT}")
print("benches:", len(benches))
print("seeded:", seeded)
print("deferred:", [b for b in benches if b not in seeded])
PY

echo "synth: ${SYNTH_NODES} nodes x ${SYNTH_WPN} workers"
echo "cosim: ${COSIM_NODES} nodes x ${COSIM_WPN} workers"

if [[ "${EXTERNAL_LLM}" -eq 1 && "${DRY_RUN}" -eq 0 ]]; then
  # Reuse a caller-provided endpoint (shared aav_n/msss proxy). Otherwise
  # start one dedicated proxy per campaign. Do not fall back to :18092.
  if [[ -n "${BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL:-}" ]]; then
    EXTERNAL_ENDPOINT_URL="${BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL}"
    echo "reusing external llm endpoint: ${EXTERNAL_ENDPOINT_URL}"
  else
    EXTERNAL_ENDPOINT_URL="$("${SCRIPT_DIR}/start_dedicated_deepseek_proxy.sh" "${CAMPAIGN_ROOT}/node_llm_proxy")"
    export BATCH_PARALLEL_EXTERNAL_ENDPOINT_URL="${EXTERNAL_ENDPOINT_URL}"
    echo "dedicated llm endpoint: ${EXTERNAL_ENDPOINT_URL}"
  fi
fi

if [[ "${EXTERNAL_LLM}" -eq 1 ]]; then
  "${PY}" - <<PY
import json
from pathlib import Path
root = Path("${CAMPAIGN_ROOT}")
endpoint = {
    "url": "${EXTERNAL_ENDPOINT_URL}",
    "model": "${EXTERNAL_MODEL}",
    "job_id": None,
    "borrowed": True,
    "external_llm": True,
    "queued": True,
}
(root / "llm_endpoint.json").write_text(json.dumps(endpoint, indent=2) + "\\n")
p = root / "campaign.json"
doc = json.loads(p.read_text())
doc["gpu_job_id"] = None
doc["gpu_borrowed"] = True
doc["gpu_mode"] = "up"
doc["external_llm"] = True
import os
if os.environ.get("C2HLS_DEEPSEEK_SKIP_PEAK", "0") == "1":
    doc["skip_peak_pause"] = True
cfg_doc = doc.setdefault("config", {})
cfg_doc["gpu_policy"] = "always_on"
doc["gpu_policy"] = "always_on"
p.write_text(json.dumps(doc, indent=2) + "\\n")
PY
  echo "external_llm endpoint: ${EXTERNAL_ENDPOINT_URL} (model=${EXTERNAL_MODEL})"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" BATCH_PARALLEL_VARIANT="${VARIANT}" \
    "${SCRIPT_DIR}/start_batch_parallel_variant.sh" --dry-run
  echo "dry-run ok"
  exit 0
fi

GPU_JOB=""
GPU_BORROWED=0
if [[ "${NO_GPU}" -eq 1 ]]; then
  echo "no_gpu mode: skipping GPU / external LLM (gold-gate / seed csynth)"
  "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text())
doc["no_gpu"] = True
doc["gpu_job_id"] = None
doc["gpu_borrowed"] = False
doc["gpu_mode"] = "parked"
doc["external_llm"] = False
p.write_text(json.dumps(doc, indent=2) + "\\n")
PY
elif [[ "${EXTERNAL_LLM}" -eq 1 ]]; then
  GPU_BORROWED=1
  echo "external_llm mode: skipping GPU submit and --borrow-gpu discovery (endpoint=${EXTERNAL_ENDPOINT_URL})"
elif [[ "${BORROW_GPU}" -eq 1 ]]; then
  export BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}"
  export PC2_SESSION_DIR="${CAMPAIGN_ROOT}"
  export PC2_ENDPOINT_FILE="${CAMPAIGN_ROOT}/llm_endpoint.json"
  export PC2_WATCH_LOG="${CAMPAIGN_ROOT}/flow/watch.log"
  mkdir -p "${CAMPAIGN_ROOT}/flow"
  if ! "${PY}" "${SCRIPT_DIR}/pc2_llm_discovery.py" adopt "${PC2_ENDPOINT_FILE}" --require-job-running; then
    echo "ERROR: --borrow-gpu set but no healthy borrowable endpoint found" >&2
    exit 1
  fi
  GPU_JOB="$("${PY}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["PC2_ENDPOINT_FILE"])
doc = json.loads(p.read_text())
print(doc.get("job_id") or "")
PY
)"
  GPU_BORROWED=1
  echo "borrowed GPU endpoint (job=${GPU_JOB:-unknown}); no new gpu_h100 job submitted"
else
  GPU_JOB="$(
    BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
      "${SCRIPT_DIR}/batch_parallel_submit_gpu.sh"
  )"
fi
if [[ "${NO_GPU}" -ne 1 ]]; then
  "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text())
doc["gpu_job_id"] = "${GPU_JOB}" or None
doc["gpu_mode"] = "up"
doc["gpu_borrowed"] = bool(int("${GPU_BORROWED}"))
p.write_text(json.dumps(doc, indent=2) + "\\n")
PY
fi

# Login-node nohup dies unpredictably on long campaigns; run helpers on Slurm.
_submit_bp_helper() {
  local role="$1"
  local wrap_cmd="$2"
  local out_log="$3"
  local err_log="$4"
  local mem="${5:-8G}"
  local cpus="${6:-2}"
  sbatch --parsable \
    --chdir="${C2HLS_ROOT}" \
    --job-name="${PC2_BATCH_JOB_PREFIX}-${role}" \
    --output="${out_log}" \
    --error="${err_log}" \
    --account="${PC2_SLURM_ACCOUNT}" \
    --partition="${PC2_COMPUTE_PARTITION}" \
    --cpus-per-task="${cpus}" \
    --mem="${mem}" \
    --time="${PC2_HELPER_WALLTIME:-72:00:00}" \
    --export=ALL,BATCH_PARALLEL_CAMPAIGN_ROOT="${CAMPAIGN_ROOT}",BATCH_PARALLEL_CONFIG="${CONFIG}",C2HLS_PYTHON="${PY}",OPENAI_BASE_URL="",C2HLS_MODEL="${C2HLS_MODEL}",BATCH_PARALLEL_EXTERNAL_MODEL="${BATCH_PARALLEL_EXTERNAL_MODEL:-${EXTERNAL_MODEL}}",C2HLS_ENFORCEMENT="${C2HLS_ENFORCEMENT:-0}",C2HLS_ENFORCEMENT_ROUNDS="${C2HLS_ENFORCEMENT_ROUNDS:-20}",C2HLS_AUTOSA_FLOW="${C2HLS_AUTOSA_FLOW:-0}",C2HLS_POST_FLASH_DSE="${C2HLS_POST_FLASH_DSE:-0}",C2HLS_DSE_CHAIN_FLASH="${C2HLS_DSE_CHAIN_FLASH:-0}",C2HLS_DSE_V2="${C2HLS_DSE_V2:-0}",C2HLS_DSE_V2_CHAIN_FLASH="${C2HLS_DSE_V2_CHAIN_FLASH:-0}",C2HLS_DSE_V2_GRID="${C2HLS_DSE_V2_GRID:-}",C2HLS_POST_FLASH_STREAM="${C2HLS_POST_FLASH_STREAM:-0}",C2HLS_STREAM_CHAIN_FLASH="${C2HLS_STREAM_CHAIN_FLASH:-0}",C2HLS_POST_FLASH_NO_SKILLS="${C2HLS_POST_FLASH_NO_SKILLS:-0}",C2HLS_SKIP_PHASE_B="${C2HLS_SKIP_PHASE_B:-0}",C2HLS_ONE_SHOT="${C2HLS_ONE_SHOT:-0}",C2HLS_SKIP_FLASH="${C2HLS_SKIP_FLASH:-0}",C2HLS_PP_LOAD_B_IN_DF="${C2HLS_PP_LOAD_B_IN_DF:-}",C2HLS_FLASH_SEED_DIR="${C2HLS_FLASH_SEED_DIR:-}",C2HLS_REFERENCE_ONLY="${C2HLS_REFERENCE_ONLY:-0}",C2HLS_FLASH_OPT_PROMPT_MODE="${C2HLS_FLASH_OPT_PROMPT_MODE:-}",C2HLS_TURNS="${C2HLS_TURNS:-}",C2HLS_SYNTH_TIMEOUT="${C2HLS_SYNTH_TIMEOUT:-14400}",C2HLS_PE_RECIPE="${C2HLS_PE_RECIPE:-}",C2HLS_FLASH_MAX_TOKENS="${C2HLS_FLASH_MAX_TOKENS:-}",C2HLS_LLM_MAX_TOKENS="${C2HLS_LLM_MAX_TOKENS:-}",C2HLS_CPP_CONTINUATIONS="${C2HLS_CPP_CONTINUATIONS:-}",C2HLS_FLASH_MIN_DSP="${C2HLS_FLASH_MIN_DSP:-}",C2HLS_FLASH_MAX_DSP="${C2HLS_FLASH_MAX_DSP:-}",C2HLS_FLASH_ROW_UF="${C2HLS_FLASH_ROW_UF:-}",C2HLS_FLASH_PE_BLK="${C2HLS_FLASH_PE_BLK:-}",C2HLS_FLASH_K_TILE="${C2HLS_FLASH_K_TILE:-}",C2HLS_FLASH_TILE_PP="${C2HLS_FLASH_TILE_PP:-}",C2HLS_FLASH_ONCHIP="${C2HLS_FLASH_ONCHIP:-}",C2HLS_FLASH_ONCHIP_TILE="${C2HLS_FLASH_ONCHIP_TILE:-}",C2HLS_FLASH_SKILL_BIN="${C2HLS_FLASH_SKILL_BIN:-}",C2HLS_PACKAGED_SKILLS_JSON="${C2HLS_PACKAGED_SKILLS_JSON:-}",C2HLS_PACKAGED_SKILLS_ONLY="${C2HLS_PACKAGED_SKILLS_ONLY:-}",C2HLS_SKILL_PROMPT_ORDER_JSON="${C2HLS_SKILL_PROMPT_ORDER_JSON:-}",C2HLS_DSE_SKILL_ENTRIES_JSON="${C2HLS_DSE_SKILL_ENTRIES_JSON:-}",C2HLS_STREAM_SKILL_ENTRIES_JSON="${C2HLS_STREAM_SKILL_ENTRIES_JSON:-}",C2HLS_DSE_MIN_DSP="${C2HLS_DSE_MIN_DSP:-}",C2HLS_THINKING="${C2HLS_THINKING:-}",C2HLS_FLASH_ONLY="${C2HLS_FLASH_ONLY:-0}",C2HLS_LLM_TIMEOUT="${C2HLS_LLM_TIMEOUT:-3600}",C2HLS_LLM_EMPTY_RETRIES="${C2HLS_LLM_EMPTY_RETRIES:-1}" \
    --wrap="${wrap_cmd}"
}

if [[ "${EXTERNAL_LLM}" -eq 1 ]]; then
  export C2HLS_MODEL="${C2HLS_MODEL:-${EXTERNAL_MODEL}}"
  export BATCH_PARALLEL_EXTERNAL_MODEL="${BATCH_PARALLEL_EXTERNAL_MODEL:-${EXTERNAL_MODEL}}"
fi

WATCH_JOB="$(_submit_bp_helper watch \
  "bash ${SCRIPT_DIR}/batch_parallel_watch_session.sh >> ${CAMPAIGN_ROOT}/flow/watch.log 2>&1" \
  "${CAMPAIGN_ROOT}/flow/helper_watch-%j.out" \
  "${CAMPAIGN_ROOT}/flow/helper_watch-%j.err" \
  4G 1)"

DRAIN_JOB=""
if [[ "${NO_GPU}" -ne 1 ]]; then
  DRAIN_JOB="$(_submit_bp_helper drain \
    "source ${SCRIPT_DIR}/common.sh && source ${SCRIPT_DIR}/setup_vitis_env.sh && pc2_setup_vitis_env && ${PY} ${SCRIPT_DIR}/batch_parallel_gpu_drain.py --campaign-root ${CAMPAIGN_ROOT} >> ${CAMPAIGN_ROOT}/flow/gpu_drain.log 2>&1" \
    "${CAMPAIGN_ROOT}/flow/helper_drain-%j.out" \
    "${CAMPAIGN_ROOT}/flow/helper_drain-%j.err" \
    16G 4)"
fi

COORD_JOB=""
if [[ "${FOREGROUND_COORD}" -ne 1 ]]; then
  COORD_JOB="$(_submit_bp_helper coord \
    "${PY} ${SCRIPT_DIR}/batch_parallel_coordinator.py --campaign-root ${CAMPAIGN_ROOT} >> ${CAMPAIGN_ROOT}/flow/coordinator.log 2>&1" \
    "${CAMPAIGN_ROOT}/flow/helper_coord-%j.out" \
    "${CAMPAIGN_ROOT}/flow/helper_coord-%j.err" \
    4G 1)"
fi

"${PY}" - <<PY
import json
from pathlib import Path
p = Path("${CAMPAIGN_ROOT}") / "campaign.json"
doc = json.loads(p.read_text())
doc["helper_jobs"] = {
    "watch": "${WATCH_JOB}" or None,
    "drain": "${DRAIN_JOB}" or None,
    "coord": "${COORD_JOB}" or None,
}
p.write_text(json.dumps(doc, indent=2) + "\\n")
PY

echo "helpers: watch=${WATCH_JOB} drain=${DRAIN_JOB} coord=${COORD_JOB:-foreground}"
echo "compute: deferred until GPU RUNNING (batch_parallel_watch_session.sh)"
echo "campaign=${CAMPAIGN_ROOT}"
echo "tail -f ${CAMPAIGN_ROOT}/flow/events.jsonl"
echo "stop: BATCH_PARALLEL_CAMPAIGN_ROOT=${CAMPAIGN_ROOT} ${SCRIPT_DIR}/stop_batch_parallel_campaign.sh"

if [[ "${FOREGROUND_COORD}" -eq 1 ]]; then
  exec env BATCH_PARALLEL_CONFIG="${CONFIG}" \
    "${PY}" "${SCRIPT_DIR}/batch_parallel_coordinator.py" --campaign-root "${CAMPAIGN_ROOT}"
fi

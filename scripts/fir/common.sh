#!/usr/bin/env bash
# Shared setup for Fir open-weight runs (source, do not execute).
set -euo pipefail

_FIR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export C2HLS_ROOT="${C2HLS_ROOT:-$(cd "${_FIR_DIR}/../.." && pwd)}"
export C2HLS_SITE=fir

# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/source_local_env.sh"

_fir_configure_session_paths() {
  local campaign_root="${BATCH_PARALLEL_CAMPAIGN_ROOT:-${FIR_BATCH_CAMPAIGN_ROOT:-}}"
  if [[ -n "${campaign_root}" ]]; then
    FIR_SESSION_DIR="${campaign_root}"
    FIR_SESSION_FILE="${campaign_root}/campaign.json"
    FIR_ENDPOINT_FILE="${campaign_root}/llm_endpoint.json"
    FIR_WATCH_LOG="${campaign_root}/flow/watch.log"
  elif [[ -n "${FIR_SESSION_ID:-}" ]]; then
    FIR_SESSION_DIR="${C2HLS_ROOT}/artifacts/fir/sessions/${FIR_SESSION_ID}"
    FIR_SESSION_FILE="${FIR_SESSION_DIR}/session.json"
    FIR_ENDPOINT_FILE="${FIR_SESSION_DIR}/llm_endpoint.json"
    FIR_WATCH_LOG="${FIR_SESSION_DIR}/watch.log"
  else
    FIR_SESSION_DIR="${C2HLS_ROOT}/artifacts/fir"
    FIR_SESSION_FILE="${FIR_SESSION_DIR}/session.json"
    FIR_ENDPOINT_FILE="${FIR_SESSION_DIR}/llm_endpoint.json"
    FIR_WATCH_LOG="${FIR_SESSION_DIR}/watch.log"
  fi
}
_fir_configure_session_paths

FIR_GPU_PARTITION="${FIR_GPU_PARTITION:-gpubase_bynode_b1}"
FIR_COMPUTE_PARTITION="${FIR_COMPUTE_PARTITION:-}"
FIR_SLURM_ACCOUNT="${FIR_SLURM_ACCOUNT:-def-zhenman_gpu}"
FIR_COMPUTE_SLURM_ACCOUNT="${FIR_COMPUTE_SLURM_ACCOUNT:-def-zhenman}"
if [[ -n "${FIR_FORCE_WALLTIME:-}" ]]; then
  FIR_WALLTIME="${FIR_FORCE_WALLTIME}"
else
  FIR_WALLTIME="${FIR_WALLTIME:-3:00:00}"
fi
FIR_LLM_PORT="${FIR_LLM_PORT:-8000}"
FIR_LLM_MODEL="${FIR_LLM_MODEL:-${C2HLS_MODEL:-mistralai/Devstral-2-123B-Instruct-2512}}"
FIR_GPU_GPUS="${FIR_GPU_GPUS:-4}"
FIR_GPU_NODES="${FIR_GPU_NODES:-1}"
FIR_GPU_CPUS_PER_TASK="${FIR_GPU_CPUS_PER_TASK:-16}"
FIR_GPU_MEM="${FIR_GPU_MEM:-128G}"
FIR_COMPUTE_CPUS="${FIR_COMPUTE_CPUS:-8}"
FIR_COMPUTE_MEM="${FIR_COMPUTE_MEM:-32G}"
FIR_MAX_RESTARTS="${FIR_MAX_RESTARTS:-10}"
FIR_WATCH_INTERVAL_SEC="${FIR_WATCH_INTERVAL_SEC:-60}"
# Pre-submit next GPU job when the serving job has this many seconds left (b1 handoff).
FIR_GPU_PRESUBMIT_SEC="${FIR_GPU_PRESUBMIT_SEC:-600}"
FIR_COMPUTE_WALLTIME="${FIR_COMPUTE_WALLTIME:-}"
FIR_COMPUTE_LLM_WAIT_SEC="${FIR_COMPUTE_LLM_WAIT_SEC:-1800}"
FIR_AUTO_STOP_ON_COMPLETE="${FIR_AUTO_STOP_ON_COMPLETE:-1}"
FIR_AUTO_STOP_DELAY_SEC="${FIR_AUTO_STOP_DELAY_SEC:-120}"
FIR_VLLM_TENSOR_PARALLEL_SIZE="${FIR_VLLM_TENSOR_PARALLEL_SIZE:-4}"
FIR_VLLM_EXTRA_ARGS="${FIR_VLLM_EXTRA_ARGS:-}"
FIR_GPU_MODULES="${FIR_GPU_MODULES:-python/3.11.5 cuda/12.6}"
FIR_COMPUTE_MODULES="${FIR_COMPUTE_MODULES:-apptainer/1.3.5}"
FIR_INFERENCE_ROOT="${FIR_INFERENCE_ROOT:-/scratch/asa582/workspaces/inference}"
FIR_VLLM_VENV="${FIR_VLLM_VENV:-}"
FIR_COMPUTE_VENV="${FIR_COMPUTE_VENV:-}"
FIR_LLM_SERVE_CMD="${FIR_LLM_SERVE_CMD:-}"

FIR_WORKER_CMD="${FIR_WORKER_CMD:-${C2HLS_PYTHON:-python3} scripts/fir/run_flash_smoke_batch.py --fir --benches hlsfactory_2mm,hlsfactory_lu,hlsfactory_3mm}"

fir_apply_path_defaults() {
  [[ "${C2HLS_SITE}" == "fir" ]] || return 0
  # shellcheck disable=SC1091
  source "${_FIR_DIR}/vitis_paths.env"
}
fir_apply_path_defaults

fir_load_inference_env() {
  local loader="${FIR_INFERENCE_ROOT}/scripts/load_inference_env.sh"
  if [[ -f "${loader}" ]]; then
    # shellcheck disable=SC1090
    source "${loader}"
  fi
}

fir_log() {
  mkdir -p "${FIR_SESSION_DIR}"
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${FIR_WATCH_LOG}"
}

fir_session_py() {
  "${C2HLS_PYTHON:-python3}" "${_FIR_DIR}/session_ctl.py" "$@"
}

fir_job_active() {
  local job_id="$1"
  [[ -n "${job_id}" && "${job_id}" != "null" && "${job_id}" != "None" ]] || return 1
  squeue -h -j "${job_id}" 2>/dev/null | grep -q .
}

fir_job_is_running() {
  local job_id="$1"
  [[ -n "${job_id}" && "${job_id}" != "null" && "${job_id}" != "None" ]] || return 1
  squeue -h -j "${job_id}" -t RUNNING,COMPLETING 2>/dev/null | grep -q .
}

fir_job_is_pending() {
  local job_id="$1"
  [[ -n "${job_id}" && "${job_id}" != "null" && "${job_id}" != "None" ]] || return 1
  squeue -h -j "${job_id}" -t PENDING,CONFIGURING 2>/dev/null | grep -q .
}

# Slurm TimeLeft for a running/pending job, in seconds (empty if unknown).
fir_job_time_left_sec() {
  local job_id="$1"
  [[ -n "${job_id}" && "${job_id}" != "null" && "${job_id}" != "None" ]] || return 1
  local tl
  tl="$(squeue -h -j "${job_id}" -o "%L" 2>/dev/null | head -1 | tr -d ' ')"
  [[ -n "${tl}" && "${tl}" != "NOT_SET" && "${tl}" != "N/A" && "${tl}" != "UNLIMITED" ]] || return 1
  "${C2HLS_PYTHON:-python3}" - "${tl}" <<'PY'
import sys

def parse_timeleft(raw: str) -> int | None:
    t = raw.strip()
    if not t or t in ("NOT_SET", "UNLIMITED", "N/A"):
        return None
    days = 0
    if "-" in t:
        day_part, t = t.split("-", 1)
        days = int(day_part)
    parts = t.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = (int(parts[0]), int(parts[1]), int(parts[2]))
    elif len(parts) == 2:
        hours, minutes, seconds = (0, int(parts[0]), int(parts[1]))
    else:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds

sec = parse_timeleft(sys.argv[1])
if sec is None:
    raise SystemExit(1)
print(sec)
PY
}

fir_job_state() {
  local job_id="$1"
  [[ -n "${job_id}" && "${job_id}" != "null" && "${job_id}" != "None" ]] || { echo "none"; return; }
  if fir_job_is_running "${job_id}"; then
    echo "running"
    return
  fi
  if fir_job_is_pending "${job_id}"; then
    echo "pending"
    return
  fi
  local state
  state="$(sacct -n -X -j "${job_id}" -o State 2>/dev/null | head -1 | tr -d ' ')"
  [[ -n "${state}" ]] || { echo "unknown"; return; }
  echo "${state}"
}

fir_cancel_job() {
  local job_id="$1"
  if fir_job_active "${job_id}"; then
    scancel "${job_id}" 2>/dev/null || true
  fi
}

fir_endpoint_healthy() {
  [[ -f "${FIR_ENDPOINT_FILE}" ]] || return 1
  local url
  url="$("${C2HLS_PYTHON:-python3}" -c "
import json
from pathlib import Path
p = Path('${FIR_ENDPOINT_FILE}')
print(json.loads(p.read_text()).get('url', '').rstrip('/'))
" 2>/dev/null || echo "")"
  [[ -n "${url}" ]] || return 1
  curl -sf --max-time 10 "${url}/models" >/dev/null 2>&1 \
    || curl -sf --max-time 10 "${url}/health" >/dev/null 2>&1
}

fir_llm_ready() {
  fir_endpoint_healthy
}

fir_gpu_serving() {
  local gpu_job_id="$1"
  fir_endpoint_healthy || return 1
  if fir_session_is_borrowed_gpu; then
    return 0
  fi
  fir_job_is_running "${gpu_job_id}"
}

fir_session_is_borrowed_gpu() {
  if [[ -f "${FIR_ENDPOINT_FILE}" ]]; then
    local borrowed
    borrowed="$("${C2HLS_PYTHON:-python3}" -c "
import json
from pathlib import Path
p = Path('${FIR_ENDPOINT_FILE}')
if not p.is_file():
    print('false')
else:
    print('true' if json.loads(p.read_text()).get('borrowed') else 'false')
" 2>/dev/null || echo false)"
    [[ "${borrowed}" == "true" ]] && return 0
  fi
  if [[ -f "${FIR_SESSION_FILE}" ]]; then
    local borrowed
    borrowed="$(fir_session_py get gpu_borrowed 2>/dev/null || echo false)"
    [[ "${borrowed}" == "True" || "${borrowed}" == "true" || "${borrowed}" == "1" ]] && return 0
  fi
  return 1
}

fir_llm_ready() {
  fir_endpoint_healthy
}

# Slurm job-name prefix for batch_parallel campaigns.
fir_batch_job_prefix() {
  if [[ -n "${FIR_BATCH_JOB_PREFIX:-}" ]]; then
    echo "${FIR_BATCH_JOB_PREFIX}"
    return 0
  fi
  local root="${1:-${BATCH_PARALLEL_CAMPAIGN_ROOT:-${FIR_BATCH_CAMPAIGN_ROOT:-}}}"
  if [[ -n "${root}" && -f "${root}/campaign.json" ]]; then
    "${C2HLS_PYTHON:-python3}" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "${C2HLS_ROOT}/scripts/fir")
from batch_parallel.config import campaign_job_prefix
doc = json.loads(Path("${root}/campaign.json").read_text())
print(campaign_job_prefix(doc))
PY
    return 0
  fi
  echo "firbp"
}

fir_cancel_batch_parallel_named_jobs() {
  local prefix="${1:?job prefix required}"
  local job_id
  while IFS= read -r job_id; do
    [[ -n "${job_id}" ]] || continue
    fir_cancel_job "${job_id}"
  done < <(squeue -u "$(whoami)" -h -o "%i %j" 2>/dev/null | awk -v p="${prefix}" '$2 ~ "^"p"-" {print $1}' || true)
}

#!/usr/bin/env bash
# Shared setup for PC2 supervised batch session (source, do not execute).
set -euo pipefail

_PC2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export C2HLS_ROOT="${C2HLS_ROOT:-$(cd "${_PC2_DIR}/../.." && pwd)}"
export C2HLS_SITE=pc2

# shellcheck disable=SC1091
source "${C2HLS_ROOT}/scripts/source_local_env.sh"

_pc2_configure_session_paths() {
  if [[ -n "${PC2_SESSION_ID:-}" ]]; then
    PC2_SESSION_DIR="${C2HLS_ROOT}/artifacts/pc2/sessions/${PC2_SESSION_ID}"
    PC2_SESSION_FILE="${PC2_SESSION_DIR}/session.json"
    PC2_ENDPOINT_FILE="${PC2_SESSION_DIR}/llm_endpoint.json"
    PC2_WATCH_LOG="${PC2_SESSION_DIR}/watch.log"
  else
    PC2_SESSION_DIR="${C2HLS_ROOT}/artifacts/pc2"
    PC2_SESSION_FILE="${PC2_SESSION_DIR}/session.json"
    PC2_ENDPOINT_FILE="${PC2_SESSION_DIR}/llm_endpoint.json"
    PC2_WATCH_LOG="${PC2_SESSION_DIR}/watch.log"
  fi
}
_pc2_configure_session_paths

PC2_GPU_PARTITION="${PC2_GPU_PARTITION:-gpu_h100}"
PC2_COMPUTE_PARTITION="${PC2_COMPUTE_PARTITION:-normal}"
PC2_SLURM_ACCOUNT="${PC2_SLURM_ACCOUNT:-hpc-prf-llmfpga}"
if [[ -n "${PC2_FORCE_WALLTIME:-}" ]]; then
  PC2_WALLTIME="${PC2_FORCE_WALLTIME}"
else
  PC2_WALLTIME="${PC2_WALLTIME:-3:00:00}"
fi
PC2_MAX_RESTARTS="${PC2_MAX_RESTARTS:-10}"
PC2_WATCH_INTERVAL_SEC="${PC2_WATCH_INTERVAL_SEC:-60}"
PC2_LLM_PORT="${PC2_LLM_PORT:-8000}"
PC2_LLM_MODEL="${PC2_LLM_MODEL:-${C2HLS_MODEL:-}}"
PC2_GPU_GPUS="${PC2_GPU_GPUS:-4}"
PC2_GPU_NODES="${PC2_GPU_NODES:-1}"
PC2_GPU_CPUS_PER_TASK="${PC2_GPU_CPUS_PER_TASK:-48}"
PC2_GPU_MEM="${PC2_GPU_MEM:-0}"
PC2_COMPUTE_CPUS="${PC2_COMPUTE_CPUS:-16}"
PC2_COMPUTE_MEM="${PC2_COMPUTE_MEM:-64G}"
# Max seconds compute polls for a healthy LLM after Slurm allocation (not a fixed delay).
PC2_COMPUTE_LLM_WAIT_SEC="${PC2_COMPUTE_LLM_WAIT_SEC:-1800}"
# When set to 1, watch_session stops GPU/compute after worker success (see delay).
PC2_AUTO_STOP_ON_COMPLETE="${PC2_AUTO_STOP_ON_COMPLETE:-0}"
PC2_AUTO_STOP_DELAY_SEC="${PC2_AUTO_STOP_DELAY_SEC:-120}"

# Command run on the compute node after the LLM endpoint is ready.
PC2_WORKER_CMD="${PC2_WORKER_CMD:-${C2HLS_PYTHON:-python3} run_agentic_sweep.py --pc2}"

# Optional: space-separated module names loaded by setup_vitis_env / setup_vllm_env.
PC2_GPU_MODULES="${PC2_GPU_MODULES:-lang system CUDA/12.6.0 Python/3.11.5-GCCcore-13.2.0}"
PC2_COMPUTE_MODULES="${PC2_COMPUTE_MODULES:-fpga xilinx/xrt/2.16}"
PC2_COMPUTE_U280_SWAP_TO="${PC2_COMPUTE_U280_SWAP_TO:-xilinx/u280/xdma_202211_1}"
PC2_VLLM_VENV="${PC2_VLLM_VENV:-}"
PC2_COMPUTE_VENV="${PC2_COMPUTE_VENV:-}"
PC2_VLLM_TENSOR_PARALLEL_SIZE="${PC2_VLLM_TENSOR_PARALLEL_SIZE:-4}"
PC2_VLLM_EXTRA_ARGS="${PC2_VLLM_EXTRA_ARGS:-}"

# PC2 Xilinx paths (bash jobs do not run c2hls_paths.configure_site).
pc2_apply_path_defaults() {
  [[ "${C2HLS_SITE}" == "pc2" ]] || return 0
  export C2HLS_VITIS_SETTINGS="${C2HLS_VITIS_SETTINGS:-/opt/software/FPGA/Xilinx/Vitis/2023.2/settings64.sh}"
  export C2HLS_XRT_SETUP="${C2HLS_XRT_SETUP:-/opt/software/FPGA/Xilinx/XRT/xrt_2.16/setup.sh}"
  export C2HLS_PLATFORM_REPO_PATHS="${C2HLS_PLATFORM_REPO_PATHS:-/opt/software/FPGA/Xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1}"
  export C2HLS_TMP_ROOT="${C2HLS_TMP_ROOT:-${C2HLS_ROOT}/c2hls_tmp}"
  export C2HLS_VITIS_VERSION="${C2HLS_VITIS_VERSION:-2023.2}"
}
pc2_apply_path_defaults

# Override the default vLLM launch if your stack differs (e.g. multi-node Devstral).
PC2_LLM_SERVE_CMD="${PC2_LLM_SERVE_CMD:-}"

pc2_log() {
  mkdir -p "${PC2_SESSION_DIR}"
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${PC2_WATCH_LOG}"
}

pc2_session_py() {
  "${C2HLS_PYTHON:-python3}" "${_PC2_DIR}/session_ctl.py" "$@"
}

pc2_job_active() {
  local job_id="$1"
  [[ -n "${job_id}" && "${job_id}" != "null" && "${job_id}" != "None" ]] || return 1
  squeue -h -j "${job_id}" 2>/dev/null | grep -q .
}

pc2_job_is_running() {
  local job_id="$1"
  [[ -n "${job_id}" && "${job_id}" != "null" && "${job_id}" != "None" ]] || return 1
  squeue -h -j "${job_id}" -t RUNNING,COMPLETING 2>/dev/null | grep -q .
}

pc2_job_is_pending() {
  local job_id="$1"
  [[ -n "${job_id}" && "${job_id}" != "null" && "${job_id}" != "None" ]] || return 1
  squeue -h -j "${job_id}" -t PENDING,CONFIGURING 2>/dev/null | grep -q .
}

pc2_job_state() {
  local job_id="$1"
  [[ -n "${job_id}" && "${job_id}" != "null" && "${job_id}" != "None" ]] || { echo "none"; return; }
  if pc2_job_is_running "${job_id}"; then
    echo "running"
    return
  fi
  if pc2_job_is_pending "${job_id}"; then
    echo "pending"
    return
  fi
  local state
  state="$(sacct -n -X -j "${job_id}" -o State 2>/dev/null | head -1 | tr -d ' ')"
  [[ -n "${state}" ]] || { echo "unknown"; return; }
  echo "${state}"
}

pc2_cancel_job() {
  local job_id="$1"
  if pc2_job_active "${job_id}"; then
    scancel "${job_id}" 2>/dev/null || true
  fi
}

# Slurm TIME_LEFT for a running job, in seconds (empty if unknown/unlimited).
pc2_job_time_left_sec() {
  local job_id="$1"
  [[ -n "${job_id}" && "${job_id}" != "null" && "${job_id}" != "None" ]] || return 1
  "${C2HLS_PYTHON:-python3}" - "${job_id}" <<'PY'
import subprocess, sys

def parse_time_left(text: str):
    text = (text or "").strip()
    if not text or text in {"NOT_SET", "UNLIMITED", "N/A"}:
        return None
    if "-" in text:
        days, rest = text.split("-", 1)
        h, m, s = (rest + ":00").split(":")[:3]
        return int(days) * 86400 + int(h) * 3600 + int(m) * 60 + int(s)
    parts = (text + ":00").split(":")[:3]
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    return None

job_id = sys.argv[1]
try:
    out = subprocess.check_output(
        ["squeue", "-h", "-j", job_id, "-o", "%L"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except subprocess.CalledProcessError:
    raise SystemExit(1)
left = parse_time_left(out.splitlines()[0] if out else "")
if left is None:
    raise SystemExit(1)
print(left)
PY
}

# Resolve Slurm job-name prefix for batch_parallel (env > campaign.json > default).
pc2_batch_job_prefix() {
  if [[ -n "${PC2_BATCH_JOB_PREFIX:-}" ]]; then
    echo "${PC2_BATCH_JOB_PREFIX}"
    return 0
  fi
  local root="${1:-${BATCH_PARALLEL_CAMPAIGN_ROOT:-}}"
  if [[ -n "${root}" && -f "${root}/campaign.json" ]]; then
    "${C2HLS_PYTHON:-python3}" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "${C2HLS_ROOT}/scripts/pc2")
from batch_parallel_config import campaign_job_prefix
doc = json.loads(Path("${root}/campaign.json").read_text())
print(campaign_job_prefix(doc))
PY
    return 0
  fi
  echo "bpcplx"
}

pc2_cancel_batch_parallel_named_jobs() {
  local prefix="${1:?job prefix required}"
  local name
  for name in \
    "${prefix}-synth" "${prefix}-cosim" "${prefix}-gpu" \
    "${prefix}-watch" "${prefix}-drain" "${prefix}-coord" "${prefix}-post"
  do
    while IFS= read -r job_id; do
      [[ -n "${job_id}" ]] || continue
      pc2_cancel_job "${job_id}"
    done < <(squeue -u "$(whoami)" -h -n "${name}" -o "%i" 2>/dev/null || true)
  done
  # Legacy prefixes from older runs.
  for name in bp-synth bp-cosim bpcplx-synth bpcplx-cosim bpcplx-gpu; do
    while IFS= read -r job_id; do
      [[ -n "${job_id}" ]] || continue
      pc2_cancel_job "${job_id}"
    done < <(squeue -u "$(whoami)" -h -n "${name}" -o "%i" 2>/dev/null || true)
  done
}

pc2_gpu_serving() {
  local gpu_job_id="$1"
  pc2_endpoint_healthy || return 1
  if pc2_session_is_borrowed_gpu; then
    return 0
  fi
  pc2_job_is_running "${gpu_job_id}"
}

pc2_session_is_borrowed_gpu() {
  local borrowed
  borrowed="$(pc2_session_py get gpu_borrowed 2>/dev/null || echo false)"
  [[ "${borrowed}" == "True" || "${borrowed}" == "true" || "${borrowed}" == "1" ]]
}

pc2_llm_ready() {
  pc2_endpoint_healthy
}

pc2_endpoint_healthy() {
  [[ -f "${PC2_ENDPOINT_FILE}" ]] || return 1
  local url
  url="$("${C2HLS_PYTHON:-python3}" -c "
import json, sys
from pathlib import Path
p = Path('${PC2_ENDPOINT_FILE}')
print(json.loads(p.read_text()).get('url', '').rstrip('/'))
" 2>/dev/null || echo "")"
  [[ -n "${url}" ]] || return 1
  curl -sf --max-time 10 "${url}/models" >/dev/null 2>&1 \
    || curl -sf --max-time 10 "${url}/health" >/dev/null 2>&1
}

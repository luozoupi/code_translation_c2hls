#!/usr/bin/env bash
# Port 46 benches, start shared DeepSeek proxy, smoke, then parallel submit:
#   - c2hls machsuite RAG2+skills DeepSeek U280
#   - c2hls hlsfactory RAG2+skills DeepSeek U280
#   - ChatHLS hybrid c2hls-port 46
#
# Usage:
#   ./scripts/pc2/start_c2hls_chathls_dual_track_u280.sh [--dry-run] [--skip-export] [--skip-chathls] [--skip-c2hls]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

DRY_RUN=0
SKIP_EXPORT=0
SKIP_CHATHLS=0
SKIP_C2HLS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    --skip-chathls) SKIP_CHATHLS=1; shift ;;
    --skip-c2hls) SKIP_C2HLS=1; shift ;;
    *) echo "unknown: $1" >&2; exit 2 ;;
  esac
done

SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/dual_track_u280_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${SEQ_ROOT}"

CHATHLS_ROOT="${CHATHLS_ROOT:-/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26}"
export CHATHLS_ROOT
SMOKE_FILE="${CHATHLS_ROOT}/benchmark/benchmark_optimization/hlsfactory_atax/kernel_info.txt"
CHATHLS_SESSION_DIR="${CHATHLS_ROOT}/artifacts/pc2/sessions/hybrid-u280-c2hlsport-$(date +%Y%m%d-%H%M%S)"
PLACEHOLDER_URL="http://127.0.0.1:18092/v1"

_write_dual_track_state() {
  local url="$1"
  local dry_run_flag="$2"
  "${C2HLS_PYTHON:-python3}" - <<PY
import json
import time
from pathlib import Path

state = {
    "seq_root": "${SEQ_ROOT}",
    "endpoint_url": "${url}",
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "dry_run": bool(${dry_run_flag}),
    "skip_export": bool(${SKIP_EXPORT}),
    "skip_chathls": bool(${SKIP_CHATHLS}),
    "skip_c2hls": bool(${SKIP_C2HLS}),
    "chathls_root": "${CHATHLS_ROOT}",
    "chathls_session_dir": "${CHATHLS_SESSION_DIR}",
}
p = Path("${SEQ_ROOT}") / "dual_track_state.json"
p.write_text(json.dumps(state, indent=2) + "\n")
PY
}

if [[ "${SKIP_EXPORT}" -eq 0 ]]; then
  echo "exporting 46 prefixed benches to ${CHATHLS_ROOT}/benchmark/benchmark_optimization ..."
  "${C2HLS_PYTHON:-python3}" "${SCRIPT_DIR}/export_c2hls_bench_to_chathls.py" \
    --all-prefixed \
    --benchmarks-root "${C2HLS_ROOT}/benchmarks" \
    --out-root "${CHATHLS_ROOT}/benchmark/benchmark_optimization"
else
  echo "skipping export (--skip-export)"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  URL="${PLACEHOLDER_URL}"
  echo "[dry-run] using placeholder endpoint ${URL} (no DeepSeek proxy started)"
else
  echo "starting shared DeepSeek proxy under ${SEQ_ROOT} ..."
  "${SCRIPT_DIR}/c2hls_deepseek_proxy.sh" "${SEQ_ROOT}"
  URL="$("${C2HLS_PYTHON:-python3}" -c "import json; print(json.load(open('${SEQ_ROOT}/llm_endpoint.json'))['url'])")"
fi

echo "smoke-check: ${SMOKE_FILE}"
test -f "${SMOKE_FILE}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] would launch parallel tracks with endpoint ${URL}"
  if [[ "${SKIP_C2HLS}" -eq 0 ]]; then
    echo "[dry-run]   nohup ${SCRIPT_DIR}/start_machsuite_deepseek_rag2_skills_u280.sh --endpoint-url ${URL}"
    echo "[dry-run]   nohup ${SCRIPT_DIR}/start_hlsfactory_deepseek_rag2_skills_u280.sh --endpoint-url ${URL}"
  else
    echo "[dry-run]   skip c2hls tracks (--skip-c2hls)"
  fi
  if [[ "${SKIP_CHATHLS}" -eq 0 ]]; then
    echo "[dry-run]   CHATHLS_SKIP_DEEPSEEK_PROXY=1 OPENAI_BASE_URL=${URL} CHATHLS_SESSION_DIR=${CHATHLS_SESSION_DIR}"
    echo "[dry-run]   nohup bash ${CHATHLS_ROOT}/scripts/pc2/submit_chathls_hybrid_c2hls_port_u280.sh"
  else
    echo "[dry-run]   skip ChatHLS track (--skip-chathls)"
  fi
  echo "${URL}" > "${SEQ_ROOT}/endpoint.url"
  _write_dual_track_state "${URL}" 1
  echo "dual_track seq_root=${SEQ_ROOT} (dry-run)"
  exit 0
fi

if [[ "${SKIP_C2HLS}" -eq 0 ]]; then
  nohup "${SCRIPT_DIR}/start_machsuite_deepseek_rag2_skills_u280.sh" --endpoint-url "${URL}" \
    > "${SEQ_ROOT}/machsuite_launch.log" 2>&1 &
  echo $! > "${SEQ_ROOT}/machsuite_launcher.pid"
  nohup "${SCRIPT_DIR}/start_hlsfactory_deepseek_rag2_skills_u280.sh" --endpoint-url "${URL}" \
    > "${SEQ_ROOT}/hlsfactory_launch.log" 2>&1 &
  echo $! > "${SEQ_ROOT}/hlsfactory_launcher.pid"
fi

if [[ "${SKIP_CHATHLS}" -eq 0 ]]; then
  export CHATHLS_SKIP_DEEPSEEK_PROXY=1
  export OPENAI_BASE_URL="${URL}"
  export CHATHLS_SESSION_DIR
  nohup bash "${CHATHLS_ROOT}/scripts/pc2/submit_chathls_hybrid_c2hls_port_u280.sh" \
    > "${SEQ_ROOT}/chathls_launch.log" 2>&1 &
  echo $! > "${SEQ_ROOT}/chathls_launcher.pid"
  echo "${CHATHLS_SESSION_DIR}" > "${SEQ_ROOT}/chathls_session_dir.txt"
fi

_write_dual_track_state "${URL}" 0
echo "dual_track seq_root=${SEQ_ROOT}"

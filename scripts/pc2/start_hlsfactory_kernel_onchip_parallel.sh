#!/usr/bin/env bash
# Parallel HLSFactory benches with AutoSA onchip pack + compute rewrite /
# hide load-store. One DeepSeek-v4-flash queue proxy per bench (workers=1).
# Does not reuse 18082 / 18092. Default base port 18400 (leave 18340-18381 retry
# wave alone if still up).
#
# Usage:
#   ./scripts/pc2/start_hlsfactory_kernel_onchip_parallel.sh
#   ./scripts/pc2/start_hlsfactory_kernel_onchip_parallel.sh --benches hlsfactory_gemm,hlsfactory_atax
#   ./scripts/pc2/start_hlsfactory_kernel_onchip_parallel.sh --dry-run --base-port 18400
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

BENCHES_CSV=""
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
BASE_PORT="${C2HLS_PER_KERNEL_PROXY_BASE_PORT:-18400}"
LOGIN_HOST="${CHATHLS_LOGIN_HOST:-$(hostname -s)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benches) shift; BENCHES_CSV="$1"; shift ;;
    --stamp) shift; STAMP="$1"; shift ;;
    --base-port) shift; BASE_PORT="$1"; shift ;;
    --login-host) shift; LOGIN_HOST="$1"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PY="${C2HLS_PYTHON:-${C2HLS_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="${C2HLS_PYTHON:-python3}"
fi

if [[ -n "${BENCHES_CSV}" ]]; then
  IFS=',' read -r -a BENCHES <<< "${BENCHES_CSV}"
else
  mapfile -t BENCHES < <("${PY}" -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from hlsfactory_onchip_lib import hlsfactory_benches
for b in hlsfactory_benches():
    print(b)
")
fi

_trimmed=()
for b in "${BENCHES[@]}"; do
  b="${b#"${b%%[![:space:]]*}"}"
  b="${b%"${b##*[![:space:]]}"}"
  [[ -n "${b}" ]] && _trimmed+=("${b}")
done
BENCHES=("${_trimmed[@]:-}")

if [[ ${#BENCHES[@]} -eq 0 ]]; then
  echo "ERROR: no HLSFactory benches" >&2
  exit 2
fi

SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/hlsfactory_onchip_parallel_${STAMP}"
PROXY_ROOT="${SEQ_ROOT}/proxies"
mkdir -p "${SEQ_ROOT}"

echo "=== parallel HLSFactory onchip + compute rewrite ==="
echo "stamp=${STAMP} n=${#BENCHES[@]} login=${LOGIN_HOST}"
echo "benches=${BENCHES[*]}"
echo "base_port=${BASE_PORT} (skip busy; leave 18082/18092 alone)"
echo "seq_root=${SEQ_ROOT}"
echo "skills=flash_onchip_wide_gemm_skill_entries.json stages=flash,compute-rewrite,hide-load-store"

BENCHES_JOINED=$(IFS=','; echo "${BENCHES[*]}")

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] would start per-bench proxies then submit:"
  port="${BASE_PORT}"
  for bench in "${BENCHES[@]}"; do
    url="http://${LOGIN_HOST}:${port}/v1"
    echo "[dry-run] ${bench} endpoint=${url}"
    (
      unset BATCH_PARALLEL_ARTIFACT_PREFIX PC2_BATCH_JOB_PREFIX BATCH_PARALLEL_VARIANT || true
      unset C2HLS_FLASH_MIN_DSP C2HLS_FLASH_PE_BLK C2HLS_FLASH_ONCHIP_TILE || true
      "${SCRIPT_DIR}/start_hlsfactory_kernel_onchip.sh" \
        --bench "${bench}" --stamp "${STAMP}_${bench}" \
        --endpoint-url "${url}" --dry-run
    )
    port=$((port + 1))
  done
  echo "dry-run ok"
  echo "seq_root=${SEQ_ROOT}"
  exit 0
fi

"${SCRIPT_DIR}/start_hlsfactory_per_bench_proxies.sh" \
  --proxy-root "${PROXY_ROOT}" \
  --benches "${BENCHES_JOINED}" \
  --base-port "${BASE_PORT}" \
  --login-host "${LOGIN_HOST}" \
  --workers 1

MAP_JSON="${PROXY_ROOT}/port_map.json"
if [[ ! -f "${MAP_JSON}" ]]; then
  echo "ERROR: ${MAP_JSON} missing after proxy start" >&2
  exit 2
fi

{
  echo "stamp=${STAMP}"
  echo "login=${LOGIN_HOST}"
  echo "skills=flash_onchip_wide_gemm_skill_entries.json"
  echo "stages=flash,compute-rewrite,hide-load-store"
} > "${SEQ_ROOT}/launch_summary.txt"

for bench in "${BENCHES[@]}"; do
  url="$("${PY}" -c "import json; print(json.load(open('${MAP_JSON}'))['${bench}']['url'])")"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submit ${bench} endpoint=${url}"
  (
    unset BATCH_PARALLEL_ARTIFACT_PREFIX PC2_BATCH_JOB_PREFIX BATCH_PARALLEL_VARIANT || true
    unset C2HLS_FLASH_MIN_DSP C2HLS_FLASH_PE_BLK C2HLS_FLASH_ONCHIP_TILE || true
    "${SCRIPT_DIR}/start_hlsfactory_kernel_onchip.sh" \
      --bench "${bench}" --stamp "${STAMP}_${bench}" \
      --endpoint-url "${url}"
  )
  echo "${bench} ${url}" >> "${SEQ_ROOT}/launch_summary.txt"
done

echo "=== launched ${#BENCHES[@]} HLSFactory onchip campaigns ==="
cat "${SEQ_ROOT}/launch_summary.txt"
echo "seq_root=${SEQ_ROOT}"
echo "port_map=${MAP_JSON}"

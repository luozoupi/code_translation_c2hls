#!/usr/bin/env bash
# Parallel flash-only AutoSA kernels: one DeepSeek-v4-flash queue proxy per
# kernel (unique login-node port, workers=1) + one Slurm synth campaign each.
#
# HLS already uses cluster nodes. The old "one flash job" rule was only the
# single login5:18092 proxy. This script starts new ports from 18340 and
# does not reuse 18092.
#
# Usage:
#   ./scripts/pc2/start_autosa_kernel_flash_parallel.sh --wave gemm64
#   ./scripts/pc2/start_autosa_kernel_flash_parallel.sh --wave all
#   ./scripts/pc2/start_autosa_kernel_flash_parallel.sh --kernels autosa_mm_hcl,autosa_mm_intel
#   ./scripts/pc2/start_autosa_kernel_flash_parallel.sh --wave gemm64 --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
cd "${C2HLS_ROOT}"

WAVE="gemm64"
KERNELS_CSV=""
PACK="${C2HLS_FLASH_SKILL_BIN:-onchip}"
STAMP="${BATCH_PARALLEL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN=0
BASE_PORT="${C2HLS_PER_KERNEL_PROXY_BASE_PORT:-18340}"
LOGIN_HOST="${CHATHLS_LOGIN_HOST:-$(hostname -s)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wave) shift; WAVE="$1"; shift ;;
    --kernels) shift; KERNELS_CSV="$1"; shift ;;
    --pack) shift; PACK="$1"; shift ;;
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

if [[ -n "${KERNELS_CSV}" ]]; then
  IFS=',' read -r -a KERNELS <<< "${KERNELS_CSV}"
else
  mapfile -t KERNELS < <("${PY}" -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from autosa_skill_bins import kernels_for_wave
for k in kernels_for_wave('${WAVE}'):
    print(k)
")
fi

# trim whitespace
_trimmed=()
for k in "${KERNELS[@]}"; do
  k="${k#"${k%%[![:space:]]*}"}"
  k="${k%"${k##*[![:space:]]}"}"
  [[ -n "${k}" ]] && _trimmed+=("${k}")
done
KERNELS=("${_trimmed[@]:-}")

if [[ ${#KERNELS[@]} -eq 0 ]]; then
  echo "ERROR: no kernels (wave=${WAVE})" >&2
  exit 2
fi

SEQ_ROOT="${C2HLS_ROOT}/artifacts/pc2/kernel_flash_parallel_${PACK}_${STAMP}"
PROXY_ROOT="${SEQ_ROOT}/proxies"
mkdir -p "${SEQ_ROOT}"

echo "=== parallel autosa kernel flash ==="
echo "stamp=${STAMP} pack=${PACK} wave=${WAVE} n=${#KERNELS[@]} login=${LOGIN_HOST}"
echo "kernels=${KERNELS[*]}"
echo "base_port=${BASE_PORT} (skip busy ports; leave 18092 alone unless it is the chosen free port)"
echo "seq_root=${SEQ_ROOT}"

BENCHES_JOINED=$(IFS=','; echo "${KERNELS[*]}")

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] would start per-kernel proxies then submit:"
  port="${BASE_PORT}"
  for bench in "${KERNELS[@]}"; do
    url="http://${LOGIN_HOST}:${port}/v1"
    echo "[dry-run] ${bench} endpoint=${url}"
    (
      unset BATCH_PARALLEL_ARTIFACT_PREFIX PC2_BATCH_JOB_PREFIX BATCH_PARALLEL_VARIANT || true
      unset C2HLS_FLASH_MIN_DSP C2HLS_FLASH_MAX_DSP C2HLS_FLASH_PE_BLK C2HLS_FLASH_ONCHIP || true
      unset C2HLS_FLASH_ROW_UF C2HLS_FLASH_K_TILE C2HLS_FLASH_ONCHIP_TILE || true
      unset C2HLS_FLASH_SKILL_ENTRIES_JSON C2HLS_PACKAGED_SKILLS_JSON || true
      "${SCRIPT_DIR}/start_autosa_kernel_flash.sh" \
        --kernel "${bench}" --pack "${PACK}" --stamp "${STAMP}_${bench}" \
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
  echo "pack=${PACK}"
  echo "login=${LOGIN_HOST}"
} > "${SEQ_ROOT}/launch_summary.txt"

for bench in "${KERNELS[@]}"; do
  url="$("${PY}" -c "import json; print(json.load(open('${MAP_JSON}'))['${bench}']['url'])")"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submit ${bench} endpoint=${url}"
  (
    unset BATCH_PARALLEL_ARTIFACT_PREFIX PC2_BATCH_JOB_PREFIX BATCH_PARALLEL_VARIANT || true
    unset C2HLS_FLASH_MIN_DSP C2HLS_FLASH_MAX_DSP C2HLS_FLASH_PE_BLK C2HLS_FLASH_ROW_UF || true
    unset C2HLS_FLASH_K_TILE C2HLS_FLASH_ONCHIP_TILE || true
    unset C2HLS_FLASH_SKILL_ENTRIES_JSON C2HLS_PACKAGED_SKILLS_JSON || true
    "${SCRIPT_DIR}/start_autosa_kernel_flash.sh" \
      --kernel "${bench}" --pack "${PACK}" --stamp "${STAMP}_${bench}" \
      --endpoint-url "${url}"
  )
  echo "${bench} ${url}" >> "${SEQ_ROOT}/launch_summary.txt"
done

echo "=== launched ${#KERNELS[@]} campaigns ==="
cat "${SEQ_ROOT}/launch_summary.txt"
echo "seq_root=${SEQ_ROOT}"
echo "port_map=${MAP_JSON}"

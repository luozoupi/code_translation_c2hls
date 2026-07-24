#!/usr/bin/env bash
# Dry-run Fir Vitis/U280 path checks (no synthesis).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export C2HLS_ROOT="${C2HLS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export C2HLS_SITE=fir

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup_vitis_env.sh"
fir_setup_vitis_env

ok=0
fail=0

check_file() {
  local label="$1" path="$2"
  if [[ -f "${path}" ]]; then
    echo "OK  ${label}: ${path}"
    ok=$((ok + 1))
  else
    echo "MISS ${label}: ${path}"
    fail=$((fail + 1))
  fi
}

check_dir() {
  local label="$1" path="$2"
  if [[ -d "${path}" ]]; then
    echo "OK  ${label}: ${path}"
    ok=$((ok + 1))
  else
    echo "MISS ${label}: ${path}"
    fail=$((fail + 1))
  fi
}

echo "=== Fir Vitis preflight (site=${C2HLS_SITE}) ==="

if [[ "${C2HLS_USE_CONTAINER:-}" == "1" && -n "${XILINX_SIF:-}" && -f "${XILINX_SIF}" ]]; then
  echo "OK  container SIF: ${XILINX_SIF}"
  ok=$((ok + 1))
  if command -v vitis-run >/dev/null 2>&1; then
    echo "OK  vitis-run wrapper: $(command -v vitis-run)"
    ok=$((ok + 1))
  else
    echo "MISS vitis-run not on PATH (source fir_container_env.sh)"
    fail=$((fail + 1))
  fi
else
  check_file "Vitis settings" "${C2HLS_VITIS_SETTINGS:-}"
  check_file "XRT setup" "${C2HLS_XRT_SETUP:-}"
  check_dir  "U280 platforms" "${C2HLS_PLATFORM_REPO_PATHS:-}"
  check_dir  "OpenCL headers" "${C2HLS_OPENCL_HEADERS:-}"

  if command -v vitis_hls >/dev/null 2>&1; then
    echo "OK  vitis_hls: $(command -v vitis_hls)"
    ok=$((ok + 1))
  else
    echo "MISS vitis_hls not on PATH (source setup_vitis_env.sh after install)"
    fail=$((fail + 1))
  fi

  platform="${C2HLS_DEVICE_PLATFORM:-xilinx_u280_gen3x16_xdma_1_202211_1}"
  xpfm="${C2HLS_PLATFORM_REPO_PATHS:-}/${platform}/${platform}.xpfm"
  if [[ -f "${xpfm}" ]]; then
    echo "OK  platform xpfm: ${xpfm}"
    ok=$((ok + 1))
  else
    echo "MISS platform xpfm: ${xpfm}"
    fail=$((fail + 1))
  fi
fi

check_dir  "TMP root" "${C2HLS_TMP_ROOT:-}"

echo ""
echo "checks passed=${ok} failed=${fail}"
[[ "${fail}" -eq 0 ]]

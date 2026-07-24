#!/usr/bin/env bash
# Fir-only: user-installed Vitis 2023.2 + XRT under /scratch (no Lmod FPGA modules).
# Usage: source scripts/fir/setup_vitis_env.sh

fir_setup_vitis_env() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local root="${C2HLS_ROOT:-$(cd "${script_dir}/../.." && pwd)}"

  # shellcheck disable=SC1091
  source "${root}/scripts/source_local_env.sh"
  # shellcheck disable=SC1091
  source "${script_dir}/vitis_paths.env"

  local sif="${C2HLS_XILINX_SIF:-${FIR_SCRATCH_ROOT:-/scratch/${USER}}/containers/xilinx_vitis_2023.2.standalone.sif}"
  if [[ "${C2HLS_USE_CONTAINER:-1}" != "0" && -f "${sif}" ]]; then
    export XILINX_SIF="${sif}"
    # shellcheck disable=SC1091
    source "${script_dir}/fir_container_env.sh"
    mkdir -p "${C2HLS_TMP_ROOT:-${root}/c2hls_tmp}"
    return 0
  fi

  if [[ -n "${C2HLS_VITIS_SETTINGS:-}" && -f "${C2HLS_VITIS_SETTINGS}" ]]; then
    # shellcheck disable=SC1090
    source "${C2HLS_VITIS_SETTINGS}" >/dev/null 2>&1 || true
  fi

  if [[ -n "${C2HLS_XRT_SETUP:-}" && -f "${C2HLS_XRT_SETUP}" ]]; then
    # shellcheck disable=SC1090
    source "${C2HLS_XRT_SETUP}" >/dev/null 2>&1 || true
  fi

  if [[ -n "${C2HLS_PLATFORM_REPO_PATHS:-}" ]]; then
    export PLATFORM_REPO_PATHS="${C2HLS_PLATFORM_REPO_PATHS}"
  fi

  if [[ -n "${C2HLS_OPENCL_HEADERS:-}" && -d "${C2HLS_OPENCL_HEADERS}" ]]; then
    export CPLUS_INCLUDE_PATH="${C2HLS_OPENCL_HEADERS}:/usr/include/x86_64-linux-gnu${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export C_INCLUDE_PATH="${C2HLS_OPENCL_HEADERS}:/usr/include/x86_64-linux-gnu${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  fi

  export C2HLS_DEVICE_PLATFORM="${C2HLS_DEVICE_PLATFORM:-xilinx_u280_gen3x16_xdma_1_202211_1}"
  export PLATFORM="${PLATFORM:-${C2HLS_DEVICE_PLATFORM}}"

  if [[ -n "${C2HLS_VITIS_USER_HOME:-}" ]]; then
    mkdir -p "${C2HLS_VITIS_USER_HOME}"
    export HOME="${C2HLS_VITIS_USER_HOME}"
    export XILINX_VITIS="${XILINX_VITIS:-$(dirname "$(dirname "${C2HLS_VITIS_SETTINGS}")")}"
  fi

  mkdir -p "${C2HLS_TMP_ROOT:-${root}/c2hls_tmp}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  export C2HLS_SITE=fir
  fir_setup_vitis_env
  command -v vitis_hls >/dev/null 2>&1 && echo "vitis_hls=$(command -v vitis_hls)" || echo "WARN: vitis_hls not on PATH"
fi

#!/usr/bin/env bash
# PC2-only: Vitis 2023.2 + XRT 2.16 for Alveo U280 (Otus/Noctua 2 modules).
# Source from setup_emu_env.sh or compute_worker.sbatch.sh — do not execute directly.

pc2_setup_vitis_env() {
  local modules="${PC2_COMPUTE_MODULES:-fpga xilinx/xrt/2.16}"
  local swap_to="${PC2_COMPUTE_U280_SWAP_TO:-xilinx/u280/xdma_202211_1}"

  if command -v module >/dev/null 2>&1 && [[ -n "${modules}" ]]; then
    module reset 2>/dev/null || module purge 2>/dev/null || true
    # shellcheck disable=SC2086
    module load ${modules}
    # Vitis 2023.2 Tcl stack expects libtinfo.so.5; PC2 ncurses ships libtinfo.so.6.
    if ! module is-loaded ncurses 2>/dev/null; then
      module load ncurses/6.4-GCCcore-13.2.0 2>/dev/null || true
    fi
    if [[ -n "${swap_to}" ]]; then
      if module is-loaded xilinx/u55c 2>/dev/null; then
        module swap xilinx/u55c "${swap_to}" 2>/dev/null || true
      elif module is-loaded xilinx/u280 2>/dev/null; then
        module swap xilinx/u280 "${swap_to}" 2>/dev/null || true
      fi
    fi
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

  if [[ -n "${C2HLS_OPENCL_HEADERS:-}" ]]; then
    export CPLUS_INCLUDE_PATH="${C2HLS_OPENCL_HEADERS}:/usr/include/x86_64-linux-gnu${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export C_INCLUDE_PATH="${C2HLS_OPENCL_HEADERS}:/usr/include/x86_64-linux-gnu${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  fi

  export C2HLS_DEVICE_PLATFORM="${C2HLS_DEVICE_PLATFORM:-xilinx_u280_gen3x16_xdma_1_202211_1}"
  export PLATFORM="${PLATFORM:-${C2HLS_DEVICE_PLATFORM}}"

  # libtinfo.so.5 compat for vitis_hls / libxv_commontasks.so on RHEL9-based nodes.
  local ncurses_lib="${EBROOTNCURSES:-/opt/software/pc2/EB-SW/software/ncurses/6.4-GCCcore-13.2.0}"
  local compat_dir="${C2HLS_TMP_ROOT:-${C2HLS_ROOT}/c2hls_tmp}/libtinfo_compat"
  if [[ -f "${ncurses_lib}/lib/libtinfo.so.6" ]]; then
    mkdir -p "${compat_dir}"
    ln -sfn "${ncurses_lib}/lib/libtinfo.so.6" "${compat_dir}/libtinfo.so.5"
    export LD_LIBRARY_PATH="${compat_dir}:${ncurses_lib}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi

  # shellcheck disable=SC1091
  source "$(dirname "${BASH_SOURCE[0]}")/setup_compute_env.sh"
  pc2_setup_compute_python_env
}

#!/usr/bin/env bash
# Fir: Apptainer wrapper for standalone Vitis 2023.2 SIF (protobuf + Khronos inside image).
# Usage:
#   source scripts/fir/fir_container_env.sh
#   run_vitis 'vitis-run --version'
#
# After sourcing, scripts/fir/bin/vitis-run is prepended to PATH so hls_eval.py
# invokes Vitis inside the container without a host settings64.sh install.
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source this file: source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

_FIR_CONTAINER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_FIR_SCRATCH="${FIR_SCRATCH_ROOT:-/scratch/${USER}}"

export C2HLS_USE_CONTAINER=1
export XILINX_SIF="${XILINX_SIF:-${_FIR_SCRATCH}/containers/xilinx_vitis_2023.2.standalone.sif}"

if ! command -v apptainer >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load apptainer/1.3.5 2>/dev/null || module load apptainer 2>/dev/null || true
  fi
fi
command -v apptainer >/dev/null 2>&1 || {
  echo "fir_container_env: apptainer not found (module load apptainer?)" >&2
  return 1
}

mkdir -p "${HOME}/.Xilinx" "${_FIR_SCRATCH}/.Xilinx" "${_FIR_SCRATCH}/.xilinx-local/Vivado/2023.2/commands" 2>/dev/null || true

export APPTAINER_NO_MOUNT="${APPTAINER_NO_MOUNT:-hostfs}"
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:-${_FIR_SCRATCH},${HOME}/.Xilinx}"

if [[ -n "${XILINXD_LICENSE_FILE:-}" ]]; then
  export APPTAINERENV_XILINXD_LICENSE_FILE="${XILINXD_LICENSE_FILE}"
fi

export APPTAINERENV_PLATFORM_REPO_PATHS="/opt/xilinx/platforms"
export APPTAINERENV_XILINX_VITIS="/opt/Xilinx/Vitis/2023.2"
export APPTAINERENV_XILINX_VIVADO="/opt/Xilinx/Vivado/2023.2"
export APPTAINERENV_XILINX_XRT="/opt/xilinx/xrt"
export APPTAINERENV_XILINX_VITIS_HLS="/opt/Xilinx/Vitis_HLS/2023.2"
export APPTAINERENV_PATH="/opt/Xilinx/Vitis/2023.2/bin:/opt/Xilinx/Vivado/2023.2/bin:/opt/Xilinx/Vitis_HLS/2023.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export APPTAINERENV_LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:/opt/Xilinx/Vitis/2023.2/lib/lnx64.o:/opt/Xilinx/Vivado/2023.2/lib/lnx64.o"
export APPTAINERENV_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"

export XILINX_LOCAL_USER_DATA="${XILINX_LOCAL_USER_DATA:-${_FIR_SCRATCH}/.xilinx-local}"
export APPTAINERENV_XILINX_LOCAL_USER_DATA="${XILINX_LOCAL_USER_DATA}"

export APPTAINERENV_CL_TARGET_OPENCL_VERSION="120"
export APPTAINERENV_C_INCLUDE_PATH="/usr/include/x86_64-linux-gnu"
export APPTAINERENV_CPLUS_INCLUDE_PATH="/usr/include/x86_64-linux-gnu"
export APPTAINERENV_TRILLIUM_OPENCL_CFLAGS="-I/opt/xilinx/khronos-opencl -I/opt/xilinx/khronos-clhpp/include"

# Host Lmod paths break in-container csim link (crt1.o); keep linker paths inside SIF.
unset LIBRARY_PATH LD_LIBRARY_PATH 2>/dev/null || true
unset C2HLS_VITIS_SETTINGS VITIS_SETTINGS 2>/dev/null || true

export PATH="${_FIR_CONTAINER_DIR}/bin:${PATH}"

run_vitis() {
  apptainer exec "${XILINX_SIF}" bash -lc "$*"
}

fir_container_env() {
  :
}

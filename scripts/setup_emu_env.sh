#!/bin/bash
# Source this to enable v++ sw_emu / hw_emu flow on Vitis 2023.2 + xcu280.
# Usage:  source scripts/setup_emu_env.sh
#
# Default (team): hardcoded paths for the team development server.
# PC2: set C2HLS_SITE=pc2 (or pass --pc2 to the parent script) and configure
#      local.env — see local.env.example.

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/source_local_env.sh"

if [[ "${C2HLS_SITE:-team}" == "pc2" ]]; then
  # shellcheck disable=SC1091
  source "$(dirname "${BASH_SOURCE[0]}")/pc2/setup_vitis_env.sh"
  pc2_setup_vitis_env
else
  # Vitis HLS + Vivado tooling
  # shellcheck disable=SC1091
  source /mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh > /dev/null 2>&1

  # XRT runtime (user-mode install)
  # shellcheck disable=SC1091
  source /mnt/data/luo00466/XRT_2023.2/opt/xilinx/xrt/setup.sh > /dev/null 2>&1

  # U280 deployment platform — points v++ at the xpfm + hw_emu.xsa
  export PLATFORM_REPO_PATHS=/mnt/data/luo00466/U280_PLATFORM/opt/xilinx/platforms

  # Khronos OpenCL headers (xcl2.hpp / cl.h / opencl.h)
  export CPLUS_INCLUDE_PATH=/mnt/data/luo00466/opencl_headers:/usr/include/x86_64-linux-gnu${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}
  export C_INCLUDE_PATH=/mnt/data/luo00466/opencl_headers:/usr/include/x86_64-linux-gnu${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
fi

export C2HLS_DEVICE_PLATFORM="${C2HLS_DEVICE_PLATFORM:-xilinx_u280_gen3x16_xdma_1_202211_1}"
export PLATFORM="${PLATFORM:-${C2HLS_DEVICE_PLATFORM}}"

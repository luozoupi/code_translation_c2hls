#!/bin/bash
# Source this to enable v++ sw_emu / hw_emu flow on Vitis 2023.2 + xcu280.
# Usage:  source scripts/setup_emu_env.sh

# Vitis HLS + Vivado tooling
source /mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh > /dev/null 2>&1

# XRT runtime (user-mode install)
source /mnt/data/luo00466/XRT_2023.2/opt/xilinx/xrt/setup.sh > /dev/null 2>&1

# U280 deployment platform — points v++ at the xpfm + hw_emu.xsa
export PLATFORM_REPO_PATHS=/mnt/data/luo00466/U280_PLATFORM/opt/xilinx/platforms

# Khronos OpenCL headers (xcl2.hpp / cl.h / opencl.h) — XRT only ships
# extension headers; the base ones come from Khronos.
# Plus x86_64-linux-gnu/bits/* needed by glibc on Ubuntu 22.04 multi-arch.
export CPLUS_INCLUDE_PATH=/mnt/data/luo00466/opencl_headers:/usr/include/x86_64-linux-gnu${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}
export C_INCLUDE_PATH=/mnt/data/luo00466/opencl_headers:/usr/include/x86_64-linux-gnu${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}

export C2HLS_DEVICE_PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1

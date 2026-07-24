#!/usr/bin/env bash
# Quick csynth + csim smoke inside standalone Vitis SIF (no LLM, no cosim).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${C2HLS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BENCH="${ROOT}/benchmarks/hlsfactory_gemm"
WORKDIR="${C2HLS_TMP_ROOT:-/home/asa582/scratch/asa582/tmp/c2hls}/smoke_container_gemm"

export C2HLS_SITE=nibi
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/nibi_container_env.sh"

if [[ ! -f "${XILINX_SIF}" ]]; then
  echo "MISS SIF: ${XILINX_SIF}" >&2
  exit 1
fi

rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"
cp "${BENCH}/hls_baseline.cpp" "${WORKDIR}/kernel.cpp"
cp "${BENCH}/gemm.h" "${BENCH}/testbench.cpp" "${WORKDIR}/"

cat > "${WORKDIR}/run_csynth.tcl" <<'EOF'
open_project hls_proj
set_top kernel_gemm
add_files kernel.cpp
add_files gemm.h
open_solution sol1 -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
exit
EOF

cat > "${WORKDIR}/run_csim.tcl" <<'EOF'
open_project hls_proj_csim
set_top kernel_gemm
add_files kernel.cpp
add_files gemm.h
add_files -tb testbench.cpp
add_files -tb gemm.h
open_solution sol1 -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csim_design
exit
EOF

VITIS_HOME="${C2HLS_VITIS_USER_HOME:-/home/asa582/scratch/asa582/tmp/vitis_user_home}"
mkdir -p "${VITIS_HOME}"

echo "=== SIF: ${XILINX_SIF} ==="
run_vitis 'vitis-run --version | head -1'

echo "=== CSYNTH ==="
run_vitis "mkdir -p '${VITIS_HOME}' && cd '${WORKDIR}' && vitis-run --tcl --input_file run_csynth.tcl"
test -f "${WORKDIR}/hls_proj/sol1/syn/report/csynth.rpt"

echo "=== CSIM ==="
run_vitis "cd '${WORKDIR}' && vitis-run --tcl --input_file run_csim.tcl"

echo "PASS container HLS smoke (csynth + csim) workdir=${WORKDIR}"

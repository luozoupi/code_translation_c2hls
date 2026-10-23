#!/bin/bash
# Run LightningSim (CLI mode) on one Vitis HLS solution directory, inside the Vitis 2023.2 container.
#   run_lsim.sh SOLUTION_DIR [extra lightningsim args]
# Needs: source /scratch/hpc-prf-llmfpga/asa582/projects/workspaces/dse_v3_claude/env.sh; dse_v3_source_vitis  (gives $XILINX_SIF, apptainer)
# LSIM_AXI_MODEL (v2021|v2023) is passed through to the patched fork.
set -euo pipefail
SOL=$(realpath "${1:?solution dir}"); shift
ENV=/scratch/asa582/tools/lightningsim/env
: "${XILINX_SIF:?run dse_v3_source_vitis first}"
export APPTAINERENV_LSIM_AXI_MODEL="${LSIM_AXI_MODEL:-v2023}"
export APPTAINERENV_TMPDIR="${LSIM_TMPDIR:-${TMPDIR:-/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/lsim/tmp}}"
mkdir -p "$APPTAINERENV_TMPDIR"
# Per-interface m_axi adapter settings from the generated RTL (v2023 model).
CFG="$APPTAINERENV_TMPDIR/axi_cfg.json"
python3 "$(dirname "$(realpath "$0")")/axi_cfg.py" "$SOL" "$CFG" >&2
export APPTAINERENV_LSIM_AXI_IFCFG="$CFG"
[ -n "${LSIM_AXI_CONST:-}" ] && export APPTAINERENV_LSIM_AXI_CONST="$LSIM_AXI_CONST"
# LSIM_IO_OUT=path: dump per-event m_axi times (lsim_io.py) instead of the CLI summary.
export APPTAINERENV_LSIM_IO_OUT="${LSIM_IO_OUT:-}"
exec apptainer exec "$XILINX_SIF" bash -c '
  export XILINX_HLS=/opt/Xilinx/Vitis_HLS/2023.2
  export PATH='"$ENV"'/bin:$PATH
  unset LD_LIBRARY_PATH PYTHONPATH
  if [ -n "$LSIM_IO_OUT" ]; then cd / && exec python '"$(dirname "$(realpath "$0")")"'/lsim_io.py "$1" "$LSIM_IO_OUT"; fi
  exec lightningsim --cli "$@"' bash "$SOL" "$@"

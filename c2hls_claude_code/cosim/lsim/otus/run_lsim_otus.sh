#!/bin/bash
# Otus copy of ../run_lsim.sh: LightningSim (CLI mode) on one Vitis HLS solution directory, run natively instead of in
# the Fir container. Needs Vitis 2023.2 settings64.sh sourced (sets XILINX_HLS) and LSIM_T (setup_lsim_otus.sh).
#   run_lsim_otus.sh SOLUTION_DIR [extra lightningsim args]
# LSIM_AXI_MODEL (v2021|v2023) is passed through to the patched fork.
set -euo pipefail
SOL=$(realpath "${1:?solution dir}"); shift
LS=$(dirname "$(dirname "$(realpath "$0")")")     # cosim/lsim: axi_cfg.py, lsim_io.py
ENV=${LSIM_T:?set LSIM_T to the setup_lsim_otus.sh install dir}/env
: "${XILINX_HLS:?source Vitis settings64.sh first}"
export LSIM_AXI_MODEL="${LSIM_AXI_MODEL:-v2023}"
export TMPDIR="${LSIM_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "$TMPDIR"
# Per-interface m_axi adapter settings from the generated RTL (v2023 model).
CFG="$TMPDIR/axi_cfg.json"
python3 "$LS/axi_cfg.py" "$SOL" "$CFG" >&2
export LSIM_AXI_IFCFG="$CFG"
# LSIM_IO_OUT=path: dump per-event m_axi times (lsim_io.py) instead of the CLI summary.
export LSIM_IO_OUT="${LSIM_IO_OUT:-}"
export PATH="$ENV/bin:$PATH"
# LightningSim drops LD_LIBRARY_PATH; the testbench it links needs the env's libstdc++ (GLIBCXX_3.4.30+, newer than
# Otus' /lib64 one; Fir's container had it), so it goes in as an rpath. Without it the testbench exits at load:
# "kernel did not run".
unset LD_LIBRARY_PATH PYTHONPATH; export LDFLAGS="-Wl,-rpath,$ENV/lib${LDFLAGS:+ $LDFLAGS}"
if [ -n "$LSIM_IO_OUT" ]; then cd / && exec python "$LS/lsim_io.py" "$SOL" "$LSIM_IO_OUT"; fi
exec lightningsim --cli "$SOL" "$@"

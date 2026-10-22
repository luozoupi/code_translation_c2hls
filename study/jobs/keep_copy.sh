#!/bin/bash
# keep_copy.sh SRC DST : copy only the files worth keeping from a node-local Vitis HLS work tree to /scratch.
# Kept: syn/report (.rpt/.xml), sources (.cpp/.c/.h/.hpp), .json, .log, .tcl, .txt, .xml, .csv, .dat (csim/csynth results).
# Dropped: .autopilot, generated RTL (verilog/vhdl/systemc), impl/, csim build objects, and everything else.
# Used because /scratch has a 1M file-count quota and a Vitis project writes ~600-40,000 files.
SRC=${1:?src}; DST=${2:?dst}
[ -d "$SRC" ] || exit 0
mkdir -p "$DST"
rsync -a --prune-empty-dirs \
  --exclude='.autopilot/' --exclude='verilog/' --exclude='vhdl/' --exclude='systemc/' --exclude='impl/' \
  --exclude='build/' --exclude='obj/' \
  --include='*/' \
  --include='*.rpt' --include='*.xml' --include='*.cpp' --include='*.c' --include='*.h' --include='*.hpp' \
  --include='*.json' --include='*.log' --include='*.tcl' --include='*.txt' --include='*.csv' --include='*.dat' \
  --exclude='*' \
  "$SRC"/ "$DST"/

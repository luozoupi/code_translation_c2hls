#!/bin/bash
# slim_workdir.sh [--dry] DIR...  : shrink finished Vitis HLS work dirs in place (file-count quota).
# Keeps: syn/report (.rpt/.xml), sources (.cpp/.h/.hpp), .json, .log, .tcl, .txt, kernel.xml, csim results.
# Removes: .autopilot, generated RTL (syn/verilog|vhdl|systemc), impl/, csim/build. Never touches cosim dirs.
# Skips any dir modified in the last 90 minutes (a job may still be using it) and every cosim dir.
DRY=0; [ "$1" = --dry ] && { DRY=1; shift; }
for top in "$@"; do
  case "$(basename "$top")" in cosim_*|fastcosim_*|hls_cosim__*|lsim) echo "skip cosim: $top"; continue;; esac
  [ -d "$top" ] || continue
  if [ -n "$(find "$top" -mmin -90 -print -quit 2>/dev/null)" ]; then echo "skip recent: $top"; continue; fi
  find "$top" -type d \( -name .autopilot -o -path '*/syn/verilog' -o -path '*/syn/vhdl' -o -path '*/syn/systemc' \
       -o -path '*/solution*/impl' -o -path '*/sol*/impl' -o -path '*/csim/build' \) -prune -print 2>/dev/null |
  while read -r d; do
    if [ $DRY = 1 ]; then echo "would remove $d ($(find "$d" -type f | wc -l) files)"; else rm -rf "$d"; fi
  done
done

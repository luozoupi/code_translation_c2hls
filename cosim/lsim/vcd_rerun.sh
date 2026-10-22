#!/bin/bash
# Re-run a finished cosim xsim snapshot (needs cosim_design -trace_level port, i.e. xelab -debug all) and dump a
# VCD of the DUT's ports plus the user-side and AXI-side ports of every m_axi adapter.  vcd_rerun.sh CONFIG_DIR
# Writes CONFIG_DIR/axi.vcd.gz.
set -uo pipefail
d=$(realpath "$1")
v="$d/hls_proj/sol1/sim/verilog"
top=$(sed -n 's/^`define AUTOTB_DUT_INST //p' "$v"/*.autotb.v | head -1)
snap=$(grep "xelab" "$v/run_xsim.sh" | sed -n "s/.* -s \([A-Za-z0-9_]*\).*/\1/p" | head -1)
tb=$(ls "$v"/*.tcl | grep -v -e check_sim -e dataflow_monitor -e run_sim -e vcd | head -1)
inst="/apatb_${snap}_top/${top}"
# Kernel-side (user) ports of every m_axi adapter instance, e.g. gmem0_m_axi_U/I_ARVALID.
adapters=$(grep -o "\b[A-Za-z0-9_]*_m_axi_U\b" "$v/${snap}.v" | sort -u)
logs=""
for a in $adapters; do
  for s in I_ARVALID I_ARREADY I_ARLEN I_RVALID I_RREADY I_AWVALID I_AWREADY I_AWLEN I_WVALID I_WREADY I_BVALID I_BREADY; do
    logs="$logs ${inst}/${a}/${s}"
  done
done
sed "s#^run all\$#open_vcd axi.vcd\nlog_vcd [get_objects -filter {type == in_port || type == out_port} ${inst}/*]\nlog_vcd ${logs}\nlog_vcd ${inst}/ap_start ${inst}/ap_done ${inst}/ap_idle\nrun all\nclose_vcd#" "$tb" > "$v/vcd.tcl"
apptainer exec "$XILINX_SIF" bash -lc "cd '$v' && xsim --noieeewarnings $snap -tclbatch vcd.tcl" > "$v/vcd_xsim.log" 2>&1
tail -n 5 "$v/vcd_xsim.log"
[ -s "$v/axi.vcd" ] && gzip -c "$v/axi.vcd" > "$d/axi.vcd.gz"

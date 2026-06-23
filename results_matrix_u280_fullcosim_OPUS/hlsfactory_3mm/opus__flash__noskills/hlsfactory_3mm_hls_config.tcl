# Auto-generated hls_config (TCL) for c2hls cell
#   bench: hlsfactory_3mm
#   setup: opus__flash__noskills
#   source: hlsfactory_3mm_generated.cpp
#   target: xcu280-fsvh2892-2L-e @ 3.33 ns
# To reproduce: `cd` into this cell dir and run
#   vitis-run --tcl --input_file hlsfactory_3mm_hls_config.tcl

open_project hls_proj
set_top kernel_3mm
add_files hlsfactory_3mm_generated.cpp
add_files ../../../benchmarks/hlsfactory_3mm/3mm.h
add_files -tb ../../../benchmarks/hlsfactory_3mm/testbench_cosim.cpp
open_solution "sol1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
cosim_design
exit

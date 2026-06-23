# Auto-generated hls_config (TCL) for c2hls cell
#   bench: hlsfactory_mvt
#   setup: opus__flash__skills
#   source: hlsfactory_mvt_generated.cpp
#   target: xcu280-fsvh2892-2L-e @ 3.33 ns
# To reproduce: `cd` into this cell dir and run
#   vitis-run --tcl --input_file hlsfactory_mvt_hls_config.tcl

open_project hls_proj
set_top kernel_mvt
add_files hlsfactory_mvt_generated.cpp
add_files ../../../benchmarks/hlsfactory_mvt/mvt.h
add_files -tb ../../../benchmarks/hlsfactory_mvt/testbench_cosim.cpp
open_solution "sol1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
cosim_design
exit

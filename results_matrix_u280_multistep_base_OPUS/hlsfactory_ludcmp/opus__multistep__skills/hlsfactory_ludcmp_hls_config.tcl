# Auto-generated hls_config (TCL) for c2hls cell
#   bench: hlsfactory_ludcmp
#   setup: opus__multistep__skills
#   source: hlsfactory_ludcmp_final.cpp
#   target: xcu280-fsvh2892-2L-e @ 3.33 ns
# To reproduce: `cd` into this cell dir and run
#   vitis-run --tcl --input_file hlsfactory_ludcmp_hls_config.tcl

open_project hls_proj
set_top kernel_ludcmp
add_files hlsfactory_ludcmp_final.cpp
add_files ../../../benchmarks/hlsfactory_ludcmp/ludcmp.h
add_files -tb ../../../benchmarks/hlsfactory_ludcmp/testbench_cosim.cpp
open_solution "sol1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
cosim_design
exit

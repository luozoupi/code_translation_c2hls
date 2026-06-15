# Auto-generated hls_config (TCL) for c2hls cell
#   bench: hlsfactory_fdtd-2d
#   setup: sonnet__flash__skills
#   source: hlsfactory_fdtd-2d_generated.cpp
#   target: xcu280-fsvh2892-2L-e @ 3.33 ns
# To reproduce: `cd` into this cell dir and run
#   vitis-run --tcl --input_file hlsfactory_fdtd-2d_hls_config.tcl

open_project hls_proj
set_top kernel_fdtd_2d
add_files hlsfactory_fdtd-2d_generated.cpp
add_files ../../../benchmarks/hlsfactory_fdtd-2d/fdtd-2d.h
add_files -tb ../../../benchmarks/hlsfactory_fdtd-2d/testbench_cosim.cpp
open_solution "sol1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
cosim_design
exit

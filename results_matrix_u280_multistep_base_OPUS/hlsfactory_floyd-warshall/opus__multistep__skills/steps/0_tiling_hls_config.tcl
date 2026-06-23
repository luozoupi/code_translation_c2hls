# Auto-generated hls_config (TCL) for c2hls multistep step
#   bench: hlsfactory_floyd-warshall
#   setup: opus__multistep__skills
#   step:   0_tiling
#   source: 0_tiling.cpp
#   target: xcu280-fsvh2892-2L-e @ 3.33 ns

open_project hls_proj
set_top kernel_floyd_warshall
add_files 0_tiling.cpp
add_files ../../../../benchmarks/hlsfactory_floyd-warshall/floyd-warshall.h
add_files -tb ../../../../benchmarks/hlsfactory_floyd-warshall/testbench_cosim.cpp
open_solution "sol1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
cosim_design
exit

# Auto-generated hls_config (TCL) for c2hls multistep step
#   bench: hlsfactory_ludcmp
#   setup: sonnet__multistep__skills
#   step:   2_unroll
#   source: 2_unroll.cpp
#   target: xcu280-fsvh2892-2L-e @ 3.33 ns

open_project hls_proj
set_top kernel_ludcmp
add_files 2_unroll.cpp
add_files ../../../../benchmarks/hlsfactory_ludcmp/ludcmp.h
add_files -tb ../../../../benchmarks/hlsfactory_ludcmp/testbench_cosim.cpp
open_solution "sol1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
cosim_design
exit

# Auto-generated hls_config (TCL) for c2hls multistep step
#   bench: hlsfactory_2mm
#   setup: opus__multistep__skills
#   step:   4_coalescing
#   source: 4_coalescing.cpp
#   target: xcu280-fsvh2892-2L-e @ 3.33 ns

open_project hls_proj
set_top kernel_2mm
add_files 4_coalescing.cpp
add_files ../../../../benchmarks/hlsfactory_2mm/2mm.h
add_files -tb ../../../../benchmarks/hlsfactory_2mm/testbench_cosim.cpp
open_solution "sol1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
cosim_design
exit

# Auto-generated hls_config (TCL) for c2hls multistep step
#   bench: hlsfactory_lu
#   setup: opus__multistep__skills
#   step:   3_doublebuffer
#   source: 3_doublebuffer.cpp
#   target: xcu280-fsvh2892-2L-e @ 3.33 ns

open_project hls_proj
set_top kernel_lu
add_files 3_doublebuffer.cpp
add_files ../../../../benchmarks/hlsfactory_lu/lu.h
add_files -tb ../../../../benchmarks/hlsfactory_lu/testbench_cosim.cpp
open_solution "sol1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
cosim_design
exit

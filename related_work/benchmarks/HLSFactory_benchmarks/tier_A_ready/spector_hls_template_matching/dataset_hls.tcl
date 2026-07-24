open_project -reset spector_hls_template_matching_prj
set_top SAD_MATCH
add_files hls_baseline.cpp
add_files fpga_temp_matching.h
add_files params.h
add_files -tb testbench.cpp
add_files -tb fpga_temp_matching.h
add_files -tb params.h
open_solution sol1 -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
exit

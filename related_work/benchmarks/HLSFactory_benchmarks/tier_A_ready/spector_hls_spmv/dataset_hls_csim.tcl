open_project -reset spector_hls_spmv_prj
set_top spmv
add_files hls_baseline.cpp
add_files params.h
add_files spmv.h
add_files -tb testbench.cpp
add_files -tb params.h
add_files -tb spmv.h
open_solution sol1 -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csim_design
exit

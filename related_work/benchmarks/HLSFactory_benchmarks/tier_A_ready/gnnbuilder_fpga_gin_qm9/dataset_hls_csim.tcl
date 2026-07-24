open_project -reset gnnbuilder_fpga_gin_qm9_prj
set_top fpga_gin_qm9_top
add_files hls_baseline.cpp
add_files model.h
add_files gnn_builder_lib.h
add_files -tb testbench.cpp
add_files -tb model.h
open_solution sol1 -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csim_design
exit

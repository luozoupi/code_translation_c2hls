open_project -reset machsuite_sort_merge_prj
set_top ms_mergesort
add_files hls_baseline.cpp
add_files sort.h
add_files support.h
add_files -tb testbench.cpp
add_files -tb sort.h
add_files -tb support.h
add_files -tb support.c
add_files -tb local_support.c
add_files -tb input.data
add_files -tb check.data
add_files -tb input.data
add_files -tb check.data
open_solution sol1 -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csim_design
exit

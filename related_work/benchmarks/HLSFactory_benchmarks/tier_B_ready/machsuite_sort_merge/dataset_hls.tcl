open_project -reset machsuite_sort_merge_prj
set_top ms_mergesort
add_files hls_baseline.cpp
add_files sort.h
add_files support.h
open_solution sol1 -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
exit

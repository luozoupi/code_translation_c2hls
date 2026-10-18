open_project hls_proj
set_top kernel_nussinov
add_files /mnt/data1/luo00466/c2hls_runs_20261018/scratch/saved-designs/commercial__021c7a934eb73d8f31761f9d0b5e5da987e9cb171d68a17348220903813ccb44__frozen/rtl/kernel.cpp
add_files /mnt/data1/luo00466/c2hls_runs_20261018/scratch/saved-designs/commercial__021c7a934eb73d8f31761f9d0b5e5da987e9cb171d68a17348220903813ccb44__frozen/rtl/nussinov.h
add_files -tb [list {testbench.cpp}]
open_solution "sol1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default

csynth_design
if {[info exists ::env(LIBRARY_PATH)]} { unset ::env(LIBRARY_PATH) }
cosim_design -trace_level none
exit

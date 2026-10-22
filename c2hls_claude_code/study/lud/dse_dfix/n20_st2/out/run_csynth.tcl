open_project hls_prj_csynth
set_top kernel0
add_files src/kernel_kernel.h
add_files src/kernel_kernel.cpp
add_files -tb src/kernel_host.cpp
add_files -tb src/kernel.h
open_solution "solution1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
config_compile -name_max_length 50
csynth_design
exit

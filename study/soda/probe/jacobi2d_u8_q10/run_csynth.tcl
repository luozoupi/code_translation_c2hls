open_project hls_prj_csynth
set_top jacobi2d_kernel
add_files src/kernel.cpp -cflags "-std=c++14"
open_solution "solution1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
config_compile -name_max_length 80
csynth_design
exit

# Run from <outdir>/out :  vitis-run --tcl --input_file hls_csynth_u280.tcl   (Vitis HLS 2023.2, inside the Fir container)
# Same flow/part/clock as c2hls scripts/autosa_dse_hls_common.py (flow_target vitis, 3.33 ns).
open_project hls_prj
set_top kernel0
add_files src/kernel_kernel.h
add_files src/kernel_kernel.cpp
add_files -tb src/kernel_host.cpp
add_files -tb src/kernel.h
open_solution "solution1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
config_compile -name_max_length 50
#csim_design        ;# enable for functional check (host prints "Passed!")
csynth_design
#cosim_design       ;# paper's latency numbers are RTL sim; off by project policy unless asked
exit

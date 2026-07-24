open_project -reset hp_fft_n256__UF32_prj
set_top FFT_TOP
add_files hls_baseline.cpp
add_files FFT.h
add_files -tb testbench.cpp
add_files -tb FFT.h
open_solution sol1 -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
exit

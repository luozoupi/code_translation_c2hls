open_project -reset forgebench_conv_B_prj
set_top top
add_files hls_baseline.cpp
add_files top.h
add_files support/DRAM_conv_bias.txt
add_files support/DRAM_conv_weight.txt
add_files support/DRAM_image_input.txt
add_files support/DRAM_image_output.txt
add_files -tb testbench.cpp
add_files -tb top.h
add_files -tb support/DRAM_conv_bias.txt
add_files -tb support/DRAM_conv_weight.txt
add_files -tb support/DRAM_image_input.txt
add_files -tb support/DRAM_image_output.txt
open_solution sol1 -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
exit

open_project -reset forgebench_testing_unroll_prj
set_top top
add_files hls_baseline.cpp
add_files top.h
add_files support/DRAM_1.txt
add_files support/DRAM_10.txt
add_files support/DRAM_11.txt
add_files support/DRAM_12.txt
add_files support/DRAM_2.txt
add_files support/DRAM_3.txt
add_files support/DRAM_4.txt
add_files support/DRAM_5.txt
add_files -tb testbench.cpp
add_files -tb top.h
add_files -tb support/DRAM_1.txt
add_files -tb support/DRAM_10.txt
add_files -tb support/DRAM_11.txt
add_files -tb support/DRAM_12.txt
add_files -tb support/DRAM_2.txt
add_files -tb support/DRAM_3.txt
add_files -tb support/DRAM_4.txt
add_files -tb support/DRAM_5.txt
open_solution sol1 -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
exit

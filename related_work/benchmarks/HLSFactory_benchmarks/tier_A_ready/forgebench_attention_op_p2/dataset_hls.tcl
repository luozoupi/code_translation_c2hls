open_project -reset forgebench_attention_op_p2_prj
set_top top
add_files hls_baseline.cpp
add_files top.h
add_files support/DRAM_attn_input.txt
add_files support/DRAM_output.txt
add_files support/DRAM_weights_k.txt
add_files support/DRAM_weights_q.txt
add_files support/DRAM_weights_v.txt
add_files -tb testbench.cpp
add_files -tb top.h
add_files -tb support/DRAM_attn_input.txt
add_files -tb support/DRAM_output.txt
add_files -tb support/DRAM_weights_k.txt
add_files -tb support/DRAM_weights_q.txt
add_files -tb support/DRAM_weights_v.txt
open_solution sol1 -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
exit

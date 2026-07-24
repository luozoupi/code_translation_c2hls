open_project -reset proj
add_files hls_baseline.cpp
set_top mvt
open_solution -reset solution
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 3.33 -name default
csynth_design
exit

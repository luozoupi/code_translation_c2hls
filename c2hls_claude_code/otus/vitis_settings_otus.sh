# vitis_settings_otus.sh : source instead of Vitis 2023.2's settings64.sh on Otus. Compute nodes lack libtinfo.so.5,
# which vitis_hls needs ("couldn't load file libxv_commontasks.so: libtinfo.so.5"); PC2 ships it in vitis_vivado_libs,
# the folder its xilinx/vitis/23.2 module puts on LD_LIBRARY_PATH. This does the same without the module.
source /opt/software/FPGA/Xilinx/Vitis/2023.2/settings64.sh
export LD_LIBRARY_PATH=/opt/software/FPGA/Xilinx/vitis_vivado_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# A loaded EasyBuild module (e.g. Python/3.11.5-GCCcore-13.2.0) puts GCC 13's libstdc++ on LD_LIBRARY_PATH and
# LIBRARY_PATH. Vitis' g++ 8.3 then links the cosim/csim testbench against it (GLIBCXX_3.4.30), but runs it with
# Vitis' own older libstdc++: "C TB simulation failed, nonzero return value '1'". Drop the GCCcore dirs.
_otus_no_gcccore() { printf %s "$1" | awk -v RS=: -v ORS=: '$0 !~ /\/EB-SW\/software\/GCCcore\// && NF' | sed 's/:$//'; }
export LD_LIBRARY_PATH=$(_otus_no_gcccore "$LD_LIBRARY_PATH")
[ -n "${LIBRARY_PATH:-}" ] && export LIBRARY_PATH=$(_otus_no_gcccore "$LIBRARY_PATH")
unset -f _otus_no_gcccore

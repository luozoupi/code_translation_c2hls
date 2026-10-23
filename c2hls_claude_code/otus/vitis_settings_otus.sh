# vitis_settings_otus.sh : source instead of Vitis 2023.2's settings64.sh on Otus. Compute nodes lack libtinfo.so.5,
# which vitis_hls needs ("couldn't load file libxv_commontasks.so: libtinfo.so.5"); PC2 ships it in vitis_vivado_libs,
# the folder its xilinx/vitis/23.2 module puts on LD_LIBRARY_PATH. This does the same without the module.
source /opt/software/FPGA/Xilinx/Vitis/2023.2/settings64.sh
export LD_LIBRARY_PATH=/opt/software/FPGA/Xilinx/vitis_vivado_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

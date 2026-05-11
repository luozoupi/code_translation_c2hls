# Requested Direct hw_emu Mismatch Rerun

Vitis 2023.2 / xilinx_u280_gen3x16_xdma_1_202211_1

| problem | variant | previous | reference | rerun | cycles | log |
|---|---:|:---:|:---:|:---:|---:|---|
| cfd/cfd_step_factor | 5 coalescing | pass | fail | pass | 844 | /tmp/emu_banwwd39/cfd/cfd_step_factor/cfd_step_factor_5_coalescing/c2hls_hw_emu_make_check.log |
| nw | 2 pipeline | fail | pass | fail | - | /tmp/emu_1lmxpx3v/nw/nw_2_pipeline/c2hls_hw_emu_make_check.log |
| nw | 3 unroll | fail | pass | fail | - | /tmp/emu_4lkyamu4/nw/nw_3_unroll/c2hls_hw_emu_make_check.log |

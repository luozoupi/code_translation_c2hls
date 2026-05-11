# Requested Direct hw_emu Matrix

Vitis 2023.2 / xilinx_u280_gen3x16_xdma_1_202211_1

| problem | variant | status | ref_status | cycles | ref_cycles | delta_cycles | clock_mhz |
|---|---:|:---:|:---:|---:|---:|---:|---:|
| leukocyte/lc_dilate | 7 multiddr | fail | fail | - | - | - | 300.0 |
| pathfinder | 5 coalescing | pass | pass | 21929 | 21929 | 0 | 300.0 |
| cfd/cfd_step_factor | 0 baseline | pass | pass | 5362 | 5362 | 0 | 300.0 |
| cfd/cfd_step_factor | 1 tiling | pass | pass | 7414 | 7414 | 0 | 300.0 |
| cfd/cfd_step_factor | 2 pipeline | pass | pass | 7423 | 7423 | 0 | 300.0 |
| cfd/cfd_step_factor | 5 coalescing | pass | fail | 844 | - | - | 300.0 |
| nw | 5 coalescing | fail | fail | - | - | - | 300.0 |
| leukocyte/lc_dilate | 6 coalescing | pass | pass | 30610 | 30610 | 0 | 300.0 |
| pathfinder | 0 baseline | pass | pass | 2110069 | 2110069 | 0 | 300.0 |
| pathfinder | 4 doublebuffer | pass | pass | 1056326 | 1056326 | 0 | 300.0 |
| leukocyte/lc_dilate | 1 tiling | pass | pass | 873111 | 873111 | 0 | 300.0 |
| pathfinder | 3 unroll | pass | pass | 2122628 | 2122628 | 0 | 300.0 |
| leukocyte/lc_dilate | 3 pipeline | pass | pass | 1236830 | 1236830 | 0 | 300.0 |
| pathfinder | 1 tiling | pass | pass | 3156050 | 3156050 | 0 | 300.0 |
| leukocyte/lc_dilate | 5 doublebuffer | pass | pass | 1202543 | 1202543 | 0 | 300.0 |
| leukocyte/lc_dilate | 2 pipeline | pass | pass | 603947 | 603947 | 0 | 300.0 |
| cfd/cfd_step_factor | 4 doublebuffer | pass | pass | 7561 | 7561 | 0 | 300.0 |
| nw | 3 unroll | fail | pass | - | 1345504 | - | 300.0 |
| cfd/cfd_step_factor | 3 unroll | pass | pass | 6454 | 6454 | 0 | 300.0 |
| knn | 0 baseline | pass | pass | 1049027 | 1049027 | 0 | 300.0 |
| leukocyte/lc_dilate | 0 baseline | pass | pass | 3621747 | 3621747 | 0 | 300.0 |
| knn | 5 coalescing | pass | pass | 502008 | 502008 | 0 | 300.0 |
| knn | 1 tiling | pass | pass | 4264936 | 4264936 | 0 | 300.0 |
| nw | 4 doublebuffer | pass | pass | 929932 | 929932 | 0 | 300.0 |
| nw | 2 pipeline | fail | pass | - | 1060871 | - | 300.0 |
| knn | 2 pipeline | pass | pass | 4264936 | 4264936 | 0 | 300.0 |
| knn | 3 unroll | pass | pass | 4033397 | 4033397 | 0 | 300.0 |
| knn | 4 doublebuffer | pass | pass | 3453991 | 3453991 | 0 | 300.0 |
| nw | 1 tiling | pass | pass | 68608558 | 68608558 | 0 | 300.0 |
| nw | 0 baseline | pass | pass | 29105854 | 29105854 | 0 | 300.0 |
| lud | 1 tiling | pass | pass | 7378741 | 7378741 | 0 | 300.0 |
| lud | 0 baseline | pass | pass | 56253426 | 56253426 | 0 | 300.0 |
| lud | 2 coalescing | pass | pass | 1163876 | 1163876 | 0 | 300.0 |
| lud | 3 unroll | pass | - | 1474519 | - | - | 300.0 |
| leukocyte/lc_dilate | 4 unroll | pass | - | 1239143 | - | - | 300.0 |

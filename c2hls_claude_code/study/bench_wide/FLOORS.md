# Per-kernel DSP floor and balance check (proposal)

floor = DSP for compute to keep pace with streaming the kernel's data once through a 512-bit port (fp32 mul 3 / add 2 DSP; fp64 mul 11 / add 3), capped at 8,121 (90%). Window = floor..8,121.
Balance check: latency <= 1.5 x (DSP-weighted work / DSP). No floor and no check for integer/control kernels.

| bench | dtype | floor | balance check | note |
|---|---|---|---|---|
| hlsfactory_heat-3d | double | 6317 | on | floor lowered from 8121 so the window is 6317–8121 |
| hotspot | float | 6317 | on | floor lowered from 8121 so the window is 6317–8121 |
| lavaMD | float | 6317 | on | floor lowered from 8121 so the window is 6317–8121 |
| lc_mgvf | float | 6317 | on | floor lowered from 8121 so the window is 6317–8121 |
| machsuite_backprop | double | 6317 | on | floor lowered from 8121 so the window is 6317–8121 |
| hlsfactory_jacobi-2d | double | 4691 | on |  |
| hlsfactory_syr2k | double | 3692 | on |  |
| hlsfactory_gemm | double | 3594 | on |  |
| lud | float | 3393 | on |  |
| viterbi | double | 3322 | on |  |
| machsuite_md_grid | double | 3218 | on |  |
| hlsfactory_fdtd-2d | double | 2974 | on |  |
| lc_gicov | float | 2858 | on |  |
| hlsfactory_3mm | double | 2596 | on |  |
| hlsfactory_symm | double | 2561 | on |  |
| hlsfactory_2mm | double | 2470 | on |  |
| gemm_ncubed | double | 2389 | on |  |
| hlsfactory_ludcmp | double | 2240 | on |  |
| hlsfactory_syrk | double | 2225 | on |  |
| hlsfactory_lu | double | 2212 | on |  |
| hlsfactory_gramschmidt | double | 2068 | on |  |
| hlsfactory_jacobi-1d | double | 2040 | off | 120 points per time step, 40 steps in sequence |
| machsuite_gemm_blocked | double | 1792 | on |  |
| hlsfactory_durbin | double | 1680 | off | sequential recursion over k (N=120): at most ~120 mul+add per step |
| hlsfactory_correlation | double | 1674 | on |  |
| hlsfactory_covariance | double | 1631 | on |  |
| hlsfactory_doitgen | double | 1629 | on |  |
| hlsfactory_seidel-2d | double | 1440 | off | in-place Gauss-Seidel: only a ~60-point wavefront is independent |
| md_knn | double | 1353 | on |  |
| hlsfactory_trmm | double | 1233 | on |  |
| hlsfactory_cholesky | double | 1120 | on |  |
| srad | float | 907 | on |  |
| fft | double | 580 | on |  |
| kmeans | float | 544 | on |  |
| cfd_flux | float | 355 | on |  |
| hlsfactory_gemver | double | 300 | on |  |
| hlsfactory_atax | double | 218 | on |  |
| hlsfactory_bicg | double | 217 | on |  |
| hlsfactory_mvt | double | 213 | on |  |
| hlsfactory_gesummv | double | 111 | on |  |
| StreamCluster | float | 110 | on |  |
| knn | float | 75 | on |  |
| cfd_step_factor | float | 73 | on |  |
| machsuite_spmv_ellpack | double | 62 | on |  |
| hlsfactory_trisolv | double | 55 | off | forward substitution: row i needs x[0..i-1] |
| spmv_crs | double | 50 | on |  |
| aes | int | none | off |  |
| hlsfactory_floyd-warshall | int | none | off |  |
| hlsfactory_nussinov | int | none | off |  |
| lc_dilate | float | none | off |  |
| machsuite_bfs_bulk | int | none | off |  |
| machsuite_bfs_queue | int | none | off |  |
| machsuite_sort_radix | int | none | off |  |
| machsuite_stencil3D | int | none | off |  |
| nw | int | none | off |  |
| pathfinder | int | none | off |  |
| sort_merge | int | none | off |  |
| stencil2D | int | none | off |  |

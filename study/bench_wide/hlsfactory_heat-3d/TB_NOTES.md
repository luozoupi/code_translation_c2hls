# TB_NOTES: hlsfactory_heat-3d
- testbench.cpp: strict (bench_wide SPEC); original harness kept as testbench.orig.cpp. Kernel call `kernel_heat_3d` unchanged.
- Pure outputs (poisoned with 12345.0): none. In-out (real init_array data, identical copies for golden and kernel): A B. Inputs only: none.
- Compared (exactly the arrays the original harness passed to print_array): A. Everything else (scratch / modified-but-not-printed arrays) is NOT compared, but pure-output ones are still poisoned. NaN in a compared output is an error.
- Golden: static `kernel_heat_3d_ref`, a verbatim copy of plain.cpp's top function, run on identical copies of all arrays.
- Type/tolerance: double, |d-r| <= 1e-6*max(1,|r|).
- Prints at most 10 mismatches, then `Failed with N errors!` (return 1) or `Passed!` (return 0). Input perturbed (init_array field is harmonic, stencil is a no-op on it).
- Control 1 (plain.cpp): rc=0 Passed!
- Control 2 (top returns immediately): rc=1 Failed with 5831 errors!
- Control 3 (plausible bug): Control 3 bug: t <= 39. -> rc=1 Failed with 5 errors!
- Control 4 (scratch/unprinted arrays B made local, real outputs still correct; must PASS): rc=0 Passed!

- hls_baseline.cpp as kernel: rc=0 Passed!
- Validated with g++ -O1 in scratch dir (removed afterwards).

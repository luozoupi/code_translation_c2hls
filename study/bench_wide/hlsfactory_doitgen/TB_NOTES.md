# TB_NOTES: hlsfactory_doitgen
- testbench.cpp: strict (bench_wide SPEC); original harness kept as testbench.orig.cpp. Kernel call `kernel_doitgen` unchanged.
- Pure outputs (poisoned with 12345.0): sum. In-out (real init_array data, identical copies for golden and kernel): A. Inputs only: C4.
- Compared (exactly the arrays the original harness passed to print_array): A. Everything else (scratch / modified-but-not-printed arrays) is NOT compared, but pure-output ones are still poisoned. NaN in a compared output is an error.
- Golden: static `kernel_doitgen_ref`, a verbatim copy of plain.cpp's top function, run on identical copies of all arrays.
- Type/tolerance: double, |d-r| <= 1e-6*max(1,|r|).
- Prints at most 10 mismatches, then `Failed with N errors!` (return 1) or `Passed!` (return 0).
- Control 1 (plain.cpp): rc=0 Passed!
- Control 2 (top returns immediately): rc=1 Failed with 14911 errors!
- Control 3 (plausible bug): Control 3 bug: first loop bound reduced by 1. -> rc=1 Failed with 596 errors!
- Control 4 (scratch/unprinted arrays sum made local, real outputs still correct; must PASS): rc=0 Passed!

- hls_baseline.cpp as kernel: rc=0 Passed!
- Validated with g++ -O1 in scratch dir (removed afterwards).

## 2026-10-17 baseline fix (user-approved exception to SPEC "hls_baseline.cpp stays as copied")
- The HLSFactory baseline fails Vitis 2023.2 csynth: `SYNCHK 200-43 use or assignment of a non-static pointer ''` (c2hls stopped in its reference step). Original kept as `hls_baseline.orig.cpp`.
- Minimal fix: accumulate into a local `double sum_l[NP]`, then write `A[r][q][p]` and the `sum` port from it. Same arithmetic and outputs. Tried and rejected (same error): dropping the top pragma, scalar accumulator, loop vars in the for statements (`_doitgen_fix/`).
- Fixed baseline: csynth OK (Fmax 346.5 MHz); strict testbench with g++: Passed!, rc=0.

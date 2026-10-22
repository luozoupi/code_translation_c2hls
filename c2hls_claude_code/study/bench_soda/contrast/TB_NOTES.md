# TB_NOTES: contrast (SODA)
- Semantics: out[y][x]=sum_{dy,dx in 0..16} wt[dy][dx]*in[y+dy][x+dx] (197 taps, min-corner anchored). Sizes 1024x1024, iterate 1. Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx][...] (first SODA index fastest).
- Arguments: inputs input (read-only), output output (pure output, never read before written; poisoned with 12345.0f).
- Golden: plain.cpp's workload copied into testbench.cpp as workload_ref (static), run on identical srand(42) data. Plain.cpp was also cross-checked
  against the SODA-generated kernel (sodac --xocl-kernel run as C++, its own golden passing) on the same random inputs, see report.
- Compared region: valid interior x,y in [0, size-16) only, every element. Border of the output is undefined in SODA (plain.cpp and the DUT
  may write anything there; plain copies the input); not checked.
- Tolerance: float 1e-3*fmax(1,|ref|), NaN is an error. Data: floats on a 1/16 grid in [0,16) so every sum/reassociation is exact (contrast sums of 197 weighted taps would otherwise exceed 1e-3 abs by rounding).
- Report: first 10 mismatches printed, then exactly `Passed!` (return 0) or `Failed with N errors!` (return 1).
- Controls (g++ -O2):
- Control 1 (plain.cpp as kernel, must pass): Passed!
- Control 2 (workload returns immediately, must fail): Failed with 1016028 errors!
- Control 3 (one bug, must fail): Failed with 1008 errors!
- hls_baseline.cpp as kernel: Passed!
- Controls rerun 2026-10-19 by check_bench.py at the LAAFD sizes; SODA cross-check: plain.cpp semantics are unchanged from the 2026-10-16 bench (0 mismatches vs the sodac kernel there); the sodac kernel at this size passes its own golden (study/soda/dse/designs/contrast_u4_b128*_laafd/gpp_check.log).

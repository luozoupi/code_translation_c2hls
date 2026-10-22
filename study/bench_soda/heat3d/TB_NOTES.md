# TB_NOTES: heat3d (SODA)
- Semantics: 7-point heat stencil (.125f faces, .25f centre), 5 chained applications. Sizes 256x256x256, iterate 5. Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx][...] (first SODA index fastest).
- Arguments: inputs in (read-only), output out (pure output, never read before written; poisoned with 12345.0f).
- Golden: plain.cpp's workload copied into testbench.cpp as workload_ref (static), run on identical srand(42) data. Plain.cpp was also cross-checked
  against the SODA-generated kernel (sodac --xocl-kernel run as C++, its own golden passing) on the same random inputs, see report.
- Compared region: valid interior x,y,z in [5, size-5) only, every element. Border of the output is undefined in SODA (plain.cpp and the DUT
  may write anything there; plain copies the input); not checked.
- Tolerance: float 1e-3*fmax(1,|ref|), NaN is an error. Data: uniform [0,1) floats.
- Report: first 10 mismatches printed, then exactly `Passed!` (return 0) or `Failed with N errors!` (return 1).
- Controls (g++ -O2):
- Control 1 (plain.cpp as kernel, must pass): Passed!
- Control 2 (workload returns immediately, must fail): Failed with 14886936 errors!
- Control 3 (one bug, must fail): Failed with 13558910 errors!
- hls_baseline.cpp as kernel: Passed!
- Controls rerun 2026-10-19 by check_bench.py at the LAAFD sizes; SODA cross-check: plain.cpp semantics are unchanged from the 2026-10-16 bench (0 mismatches vs the sodac kernel there); the sodac kernel at this size passes its own golden (study/soda/dse/designs/heat3d_u4_b128*_laafd/gpp_check.log).

# TB_NOTES: seidel2d (SODA)
- Semantics: 3x3 box average * .1111111f (Jacobi-style, double-buffered), 9 chained applications. Sizes 1024x1024, iterate 9. Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx][...] (first SODA index fastest).
- Arguments: inputs input (read-only), output output (pure output, never read before written; poisoned with 12345.0f).
- Golden: plain.cpp's workload copied into testbench.cpp as workload_ref (static), run on identical srand(42) data. Plain.cpp was also cross-checked
  against the SODA-generated kernel (sodac --xocl-kernel run as C++, its own golden passing) on the same random inputs, see report.
- Compared region: valid interior x,y in [9, size-9) only, every element. Border of the output is undefined in SODA (plain.cpp and the DUT
  may write anything there; plain copies the input); not checked.
- Tolerance: float 1e-3*fmax(1,|ref|), NaN is an error. Data: uniform [0,1) floats.
- Report: first 10 mismatches printed, then exactly `Passed!` (return 0) or `Failed with N errors!` (return 1).
- Control 1 (plain.cpp as kernel, must pass): Passed!
- Control 2 (workload returns immediately, must fail): Failed with 1012036 errors!
- Control 3 (one bug, must fail): Failed with 726668 errors! (one fewer chained iteration (it < ITERATE-1))
- hls_baseline.cpp as kernel: Passed!
- SODA cross-check: sodac kernel (--tile-size = width, --iterate 9) run as C++ on random data; its own golden: INFO: PASS!; plain.cpp vs SODA kernel output on the valid interior: compared 1012036 elements, mismatches 0, max rel err 0.
- Build: g++ -O2 -I<dir> -I/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/vitis_hls_2023.2_include testbench.cpp <kernel> (scratch under /home/asa582/.claude/jobs/72130906/tmp/bench_soda_check/seidel2d/).

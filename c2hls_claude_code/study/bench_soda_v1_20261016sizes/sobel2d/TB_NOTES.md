# TB_NOTES: sobel2d (SODA)
- Semantics: mx=(int16)(sobel x), my=(uint16)(sobel y), mag=(uint16)(65535-(mx*mx+my*my)). Sizes 1024x1024, iterate 1. Layout: SODA name(dx,dy[,dz]) == name[y+dy][x+dx][...] (first SODA index fastest).
- Arguments: inputs img (read-only), output mag (pure output, never read before written; poisoned with (uint16_t)0x5A5A).
- Golden: plain.cpp's workload copied into testbench.cpp as workload_ref (static), run on identical srand(42) data. Plain.cpp was also cross-checked
  against the SODA-generated kernel (sodac --xocl-kernel run as C++, its own golden passing) on the same random inputs, see report.
- Compared region: valid interior x,y in [1, size-1) only, every element. Border of the output is undefined in SODA (plain.cpp and the DUT
  may write anything there; plain copies the input); not checked.
- Tolerance: integers exact. Data: full range of the integer type (rand()&0xFFFF).
- Report: first 10 mismatches printed, then exactly `Passed!` (return 0) or `Failed with N errors!` (return 1).
- Control 1 (plain.cpp as kernel, must pass): Passed!
- Control 2 (workload returns immediately, must fail): Failed with 1044444 errors!
- Control 3 (one bug, must fail): Failed with 1022 errors! (valid-region bound shrunk by one)
- hls_baseline.cpp as kernel: Passed!
- SODA cross-check: sodac kernel (--tile-size = width, --iterate 1) run as C++ on random data; its own golden: INFO: PASS!; plain.cpp vs SODA kernel output on the valid interior: compared 1044484 elements, mismatches 0, max rel err 0.
- Build: g++ -O2 -I<dir> -I/scratch/hpc-prf-llmfpga/asa582/projects/tmp/c2hls/vitis_hls_2023.2_include testbench.cpp <kernel> (scratch under /home/asa582/.claude/jobs/72130906/tmp/bench_soda_check/sobel2d/).

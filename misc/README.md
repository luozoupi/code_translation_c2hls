# misc — ad-hoc comparison artifacts

Team imports live under **`misc/teams/`** (see `misc/teams/README.md`).

## `ours_gemm_hls_baseline/`

Copy of our in-repo `benchmarks/hlsfactory_gemm/hls_baseline.cpp` (naive +
`#pragma HLS top` only).

## Csynth compare

```bash
module load fpga xilinx/xrt/2.16   # PC2
python3 misc/run_gemm_baseline_csynth.py
```

Results: `misc/gemm_baseline_csynth_results.json`

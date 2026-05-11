# Requested Agentic Multistep + hw_emu Smoke

model: `claude-haiku-4-5-20251001`
steps: `tiling,pipeline,unroll,doublebuffer,coalescing`
workflow: `c2hls.run_benchmark_multistep` via `C2HLSOrchestrator`
jsonl records: `24`

| bench | steps | hw_emu | variant | cycles | clock | note |
|---|---:|:---:|---|---:|---:|---|
| knn | 3/5 | pass | knn_4_doublebuffer | 1049027 | 300.0 |  |
| lud | 4/5 | fail | lud_2_coalescing | - | 300.0 | hw_emu timed out after 7200s |
| pathfinder | 4/5 | fail | pathfinder_4_doublebuffer | 674131 | 300.0 | testbench check failed |
| cfd_step_factor | 3/5 | fail | cfd_step_factor_4_doublebuffer | 0 | 300.0 | Error: Failed to create compute kernel! |
| lc_dilate | 0/0 | skip | - | - | - | no final HLS code available for hw_emu |
| nw | 0/0 | skip | - | - | - | no final/selected variant step was provided for nw |

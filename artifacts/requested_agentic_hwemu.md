# Requested Agentic Multistep + hw_emu Smoke

model: `claude-haiku-4-5-20251001`
steps: `tiling,pipeline,unroll,doublebuffer,coalescing`
workflow: `c2hls.run_benchmark_multistep` via `C2HLSOrchestrator`
jsonl records: `16`

| bench | steps | hw_emu | variant | cycles | clock | note |
|---|---:|:---:|---|---:|---:|---|
| knn | 5/5 | pass | knn_5_coalescing | 1049032 | 300.0 |  |
| lud | 4/5 | skip | - | - | - | requested hw_emu variant step 'doublebuffer' for lud does not match any upstream variant; available: lud_0_baseline, lud |
| pathfinder | 0/0 | skip | - | - | - | agentic exception: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credi |
| cfd_step_factor | 0/0 | skip | - | - | - | agentic exception: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credi |
| lc_dilate | 0/0 | skip | - | - | - | agentic exception: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credi |
| nw | 0/0 | skip | - | - | - | agentic exception: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credi |

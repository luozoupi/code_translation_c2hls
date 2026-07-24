# c2hlsc flash → streaming dataflow (c2hls)

**Date:** 2026-07-15  
**Choice:** Approach 1 — ingest Lucaz97/c2hlsc Option-A benches as `c2hlsc_*` corpus + dedicated workflow (mirror ChatHLS).

## Source

- Repo: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hlsc` ([Lucaz97/c2hlsc](https://github.com/Lucaz97/c2hlsc))
- Scope: yaml benches whose `orig_code` is `.c`/`.cpp` (**19 benches**). Skip `.txt`-only AES-fragment benches.

## Campaign

- **Prepare:** `scripts/prepare_c2hlsc_ready.py` → `related_work/benchmarks/HLSFactory_benchmarks/c2hlsc_ready/`
- **Start:** `./scripts/pc2/start_c2hlsc_flash_dataflow_batch_parallel.sh`
- **Config:** `scripts/pc2/batch_parallel_c2hlsc_flash_dataflow.json`
- **Workflow / variant:** `c2hlsc_flash` / `c2hlsc_aav_n`

## Packaging

| Artifact | Rule |
|----------|------|
| `hls_baseline.cpp` / `gold_hls_source.cpp` | Merge includes preamble + `orig_code` (+ ascon helpers); strip HLS pragmas for `plain.cpp` |
| `testbench.cpp` | Curate from upstream `test_code` (`.txt`/`.c`/`.cpp`); smoke TB only if needed |
| Top | From yaml `top_function`; wrap in `extern "C"` for csim link |
| Part / clock | `xcu280-fsvh2892-2L-e` / `3.33` ns |

## Policies (match ChatHLS)

| Knob | Value |
|------|--------|
| Flash skills | `skills_ii_target_miss_solutions_added(90skills).json` (**no** no_RMW overlay) |
| Dataflow skills | `flash_no_RMW_m_axi_skill_entries.json` |
| Cosim | on-if-possible (`C2HLS_COSIM_REQUIRED=0`) |
| GPU | borrow OFF, batch_park ON, `park_grace_s=5400` |
| Nodes | synth/cosim ≈ #benches (19), 1 worker/node |
| Handoff | streaming `wait_c2hlsc_flash_stream_dataflow.sh` |

## Benches

`ascon, block, cusums, des, filter, four_parallel, four_sequential, kmp, monobit, nw, overlapping, present, quicksort, repeated_four_p, repeated_four_s, runs, sha256, two_parallel, two_sequential`

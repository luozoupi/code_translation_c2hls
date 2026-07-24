# ChatHLS flash → streaming dataflow (c2hls)

**Date:** 2026-07-12  
**Choice:** A — ingest ChatHLS `benchmark_optimization` sources as `chathls_*` benches.

## Campaign

- **Start:** `./scripts/pc2/start_chathls_flash_dataflow_batch_parallel.sh`
- **Config:** `scripts/pc2/batch_parallel_chathls_flash_dataflow.json`
- **Corpus:** `related_work/benchmarks/HLSFactory_benchmarks/chathls_ready/` (via `scripts/prepare_chathls_ready.py`)
- **Workflow:** `chathls_flash` / variant `chathls_aav_n`

## Policies

| Knob | Value |
|------|--------|
| GPU | 1× H100, **borrow OFF**, **batch_park ON**, `park_grace_s=5400` |
| Synth/csim nodes | **16** (= #benches), 1 worker/node |
| Cosim nodes | **16**, 1 worker/node |
| Flash skills | `skills_ii_target_miss_solutions_added(90skills).json` (**no** no_RMW overlay) |
| Dataflow skills | `flash_no_RMW_m_axi_skill_entries.json` |
| Cosim | `C2HLS_RUN_COSIM=1`, `C2HLS_COSIM_REQUIRED=0` |
| Handoff | **Streaming:** `wait_chathls_flash_stream_dataflow.sh` starts dataflow per bench as soon as flash is selected (up to 16 parallel) |

## Notes

- Most ChatHLS kernels ship without TBs → smoke TBs generated for csim gate.
- Mobilenet upstream TB has host paths → smoke TB used.
- Transformer keeps native `tb_transformer.cpp` + DRAM inputs.

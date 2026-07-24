# c2hlsc flash+dataflow Implementation Plan

> **For agentic workers:** execute task-by-task. Steps use checkbox syntax.

**Goal:** Package c2hlsc Option-A kernels as naive no-pragma HLS baselines with curated TBs, then launch batch_parallel flash (90 skills) + streaming dataflow in c2hls.

**Architecture:** Mirror ChatHLS campaign: prepare corpus → flash lib/dispatch hooks → start script + streaming watcher.

**Tech Stack:** Python 3, bash/Slurm, Vitis HLS via existing c2hls PC2 scripts.

---

### Task 1: prepare_c2hlsc_ready.py

- Create: `scripts/prepare_c2hlsc_ready.py`
- Output: `related_work/benchmarks/HLSFactory_benchmarks/c2hlsc_ready/c2hlsc_*/`

- [ ] Merge includes + orig_code; ascon special-case helpers
- [ ] Curate testbench from test_code
- [ ] Emit metadata + plain/hls_baseline/gold

### Task 2: campaign libs + config

- Create: `scripts/pc2/c2hlsc_flash_lib.py`, `batch_parallel_c2hlsc_lib.py`, `batch_parallel_c2hlsc_flash_dataflow.json`
- Modify: `batch_parallel_dispatch.py`, `batch_parallel_config.py`

### Task 3: start + stream watcher

- Create: `start_c2hlsc_flash_dataflow_batch_parallel.sh`, `wait_c2hlsc_flash_stream_dataflow.sh`

### Task 4: prepare corpus, dry-run, launch

- [ ] Run prepare
- [ ] Dry-run start script
- [ ] Launch campaign

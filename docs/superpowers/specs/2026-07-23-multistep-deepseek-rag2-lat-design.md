# Design: Multistep DeepSeek + RAG2 + latency-opt (triple corpus)

**Date:** 2026-07-23  
**Status:** approved  
**Goal:** Batch-parallel multistep (not flash/dataflow) on `chathls_ready`, `tier_A_ready`, and `tier_B_ready` with DeepSeek, RAG2, gemm_flatten_v1 (99), per-step latency-opt, and final-selected cosim.

## Settings

| Knob | Value |
|------|-------|
| Workflow | `chathls_multistep` / `tier_a_multistep` / `tier_b_multistep` |
| Opt order | tiling → pipeline → unroll → coalescing → doublebuffer |
| Model | deepseek-chat via **3 dedicated** login proxies (`workers=1`) |
| RAG | RAG2 on |
| Skills | `…gemm_flatten_v1.json` (99), all+avoids |
| Latency-opt | After every successful step; N=3, R=3 |
| Cosim | Only on **final best-selected** kernel |
| Part / clock | U280 / 3.33 ns |
| Nodes | 16 / 54 / 18 combined HLS (1 worker/node) |
| CPUs/mem | 16 / 64 GB |
| Wall | 72 h |
| Peak pause | **OFF** (`C2HLS_DEEPSEEK_SKIP_PEAK=1`) |
| Timeouts | csim 600s, synth 7200s, LLM 172800s, cosim 604800s |

## Architecture

```
proxy_chathls:18094 ── drain_chathls (codegen)
proxy_tier_a:18095 ── drain_tier_a
proxy_tier_b:18096 ── drain_tier_b
        │
   SQLite: codegen ↔ synth per step; cosim/selected once
        │
   combined-HLS Slurm nodes claim synth+cosim
```

## Per-bench flow

1. codegen/phase_b → synth (csim+csynth) → latency-opt×3  
2. For each opt step: codegen → synth → latency-opt×3  
3. Promote best latency among phase_b / steps / lat-opt winners  
4. cosim/selected → finalize  

## Ports

| Corpus | Proxy port |
|--------|------------|
| chathls | 18094 |
| tier_A | 18095 |
| tier_B | 18096 |

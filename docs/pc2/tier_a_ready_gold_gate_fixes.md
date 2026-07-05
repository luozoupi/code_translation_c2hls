# tier_A_ready Gold Gate Fixes (25-bench pilot)

Corpus: `related_work/benchmarks/HLSFactory_benchmarks/tier_A_ready/`  
Materializer: `scripts/prepare_tier_a_ready.py`  
Gold gate: `scripts/pc2/validate_tier_a_gold_gates.py` (synth + csim when `supports_csim: true`)

Signed-off verification matrix (25/25 gold pass, Slurm job `1936154`):

`artifacts/pc2/tier_a_gold_verify_20260702_tier_a_25_signed_off/matrix.json`

---

## Materializer patches

Patches apply to **gold** outputs (`hls_baseline.cpp`, `gold_hls_source.cpp`) and testbenches where noted. `plain.cpp` stays unpatched as the LLM starting point.

| Bench family | Symptom | Root cause | Fix (`prepare_tier_a_ready.py`) |
|--------------|---------|------------|----------------------------------|
| **forgebench attention / tiled_attn** | csim `scale=0`, SIGFPE on `1.0/scale` | `hls::sqrt((data_t)N)` returns 0 in csim for some `ap_fixed` inputs | `_patch_forgebench_ap_fixed_sqrt()` — `hls::sqrt` → `std::sqrt((double)…)`; attention scale uses `1.0/std::sqrt((double)head_dim)` |
| **forgebench llama_transformer_p2** | csim FPE in `rms_norm` | Tiny `rms` from `std::sqrt` cast to `ap_fixed<16,5>` → 0; division by zero | `_patch_forgebench_rms_norm_division()` — keep `rms` in `double`, divide in double before cast |
| **spector_hls normals** | csim SIGFPE in `normalized()` | Constant/collinear vmap → zero cross product → `sqrt(1/0)` | TB: `_patch_normals_testbench_vmap()` (spatial vmap); gold: `_patch_normals_gold_normalized()` (`if (mag_sq == 0) return`) |
| **spector_hls mergesort** | SYNCHK 200-43 on copy loop | `in[k]=out[k]` with `#pragma HLS array_partition` on `in`/`out` | `_patch_mergesort_gold_copy_loop()` — strip partitions on `in`/`out`; direct copy loop |
| **forgebench tiled_attn fopen** | Stale failure reports | Resolved by current TB (`DRAM_*.txt` bare names) + `hls_eval` staging `add_files -tb support/DRAM_*.txt` | No additional corpus patch |
| **forgebench mult_op / mlp** | Synth timeout in gold gate | Long synth for large designs | `BENCH_SYNTH_TIMEOUT_S=3600` in bench metadata |

### Other materializer behavior (no csim bug)

- `_patch_testbench_includes`, `_patch_testbench_data_paths`, `_patch_testbench_static_arrays` — hp_fft / spector TB compatibility with Vitis csim layout
- U280 part, `dataset_hls_csim.tcl`, and `hls_baseline.cpp` materialization for flash corpus

---

## Verification infrastructure

| Path | Role |
|------|------|
| `scripts/pc2/tier_a_25_benches.txt` | 25-bench pilot list |
| `scripts/pc2/run_tier_a_gold_verify_array.sbatch.sh` | Slurm array gold gate |
| `scripts/pc2/merge_tier_a_gold_matrix.py` | Merge per-bench JSON → `matrix.json` |
| `tests/test_hls_eval_csim_tcl.py` | Unit tests for csim TCL staging and patch regressions |

Headless HLS on PC2: use `vitis-run --tcl --input_file …` or `vitis_hls -f …` (not `vitis -f`, which needs DISPLAY).

---

## 24 csim-pass benches

All **25** benches pass gold gate. **24** run csim successfully; one is synth-only.

### forgebench (10)

1. `forgebench_attention_op_p1`
2. `forgebench_conv_A`
3. `forgebench_diff_dims_p1`
4. `forgebench_diff_orders_p1`
5. `forgebench_gpt_transformer_p1`
6. `forgebench_llama_transformer_p2`
7. `forgebench_mlp`
8. `forgebench_mult_op_p1`
9. `forgebench_tiled_attn_p1`
10. `forgebench_vec_mtx_p1`

### hp_fft (4)

11. `hp_fft_n256__UF1`
12. `hp_fft_n1024__UF1`
13. `hp_fft_n256__original_C_style`
14. `hp_fft_n1024__original_C_style`

### spector_hls (10)

15. `spector_hls_dct`
16. `spector_hls_fir`
17. `spector_hls_histogram`
18. `spector_hls_matrix_multiplication`
19. `spector_hls_mergesort`
20. `spector_hls_normals`
21. `spector_hls_sobel_filter_x`
22. `spector_hls_sobel_filter_y`
23. `spector_hls_spmv`
24. `spector_hls_template_matching`

### Synth-only (gold pass, no csim)

- `gnnbuilder_fpga_gcn_qm9` — `supports_csim: false`, matrix `csim: not_supported`

---

## batch_parallel flash (complex tier A)

Campaign config for the 24 csim-verified benches:

`scripts/pc2/batch_parallel_tier_a_24_csim.json`

Launch:

```bash
BATCH_PARALLEL_CONFIG=scripts/pc2/batch_parallel_tier_a_24_csim.json \
  ./scripts/pc2/start_tier_a_batch_parallel.sh --stamp <stamp>
```

Slurm job names use prefix `bpcplx-*` (GPU: `bpcplx-gpu-<stamp>`, synth: `bpcplx-synth-n<N>-<stamp>`).

Phase A compile checks use Vitis HLS include paths (`XILINX_HLS/include` or `C2HLS_COMPILE_INCLUDE_PATHS`).
The Phase A prompt instructs the LLM to include `ap_fixed.h` when fixed-point types are used.

Artifacts: `artifacts/pc2/batch_parallel_complex_<stamp>/`

Variant: `tier_a_90` (90 skills, synth-only flash — `cosim_nodes_per_variant: 0`).
RTL cosim is disabled; phase validation uses **csynth + csim** (same as the 24-bench pilot).

---

## Remaining 30 benches (not in bpcplx_r2)

The other **30** materialized `tier_A_ready` benches are prepared with the same materializer
patches, metadata, and static audit as the 24-bench pilot. They are listed in:

- `scripts/pc2/tier_a_30_remaining_benches.txt`
- `scripts/pc2/batch_parallel_tier_a_30_remaining.json`

**forgebench (14):** attention p2/p3, conv B/C, diff_dims p2/p3, diff_orders p2/p3,
mult_op p2/p3, testing_impl, testing_unroll, tiled_attn p2, vec_mtx p2

**hp_fft (12):** n256/n1024 UF2/4/8/16/32 and no_StagePipeline variants

**gnnbuilder (4):** gcn, gin, pna, sage (gcn is synth-only gold; others `supports_csim: false`)

Refresh lists from disk:

```bash
python3 scripts/pc2/generate_tier_a_bench_lists.py
```

Launch batch_parallel on the remaining 30:

```bash
BATCH_PARALLEL_CONFIG=scripts/pc2/batch_parallel_tier_a_30_remaining.json \
  ./scripts/pc2/start_tier_a_batch_parallel.sh --stamp <stamp>
```

Gold-gate array (run before flash):

```bash
./scripts/pc2/submit_tier_a_gold_verify_30_remaining.sh
# when complete:
python3 scripts/pc2/merge_tier_a_gold_matrix.py artifacts/pc2/tier_a_gold_verify_<stamp>
```

Re-materialize full corpus after materializer changes:

```bash
python3 scripts/prepare_tier_a_ready.py
```

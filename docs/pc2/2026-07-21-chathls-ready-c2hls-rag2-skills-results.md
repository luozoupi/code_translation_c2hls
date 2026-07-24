# c2hls on `chathls_ready` — DeepSeek RAG2+skills results

Generated: 2026-07-21 19:32 UTC

**Campaign:** `batch_parallel_chathls_fd_ds_rag2_20260721_123449`

- Flavor: `rag2_skills` (DeepSeek)
- Corpus: `chathls_ready` (ChatHLS `benchmark_optimization` sizes; gemm Mini **20×25×30**)
- Flow: flash → streaming dataflow (+ cosim attempted)
- Status: `CAMPAIGN_COMPLETE`
- Flash csynth: **15/16**
- Dataflow pass (stream state): **8/16** (with csynth XML: **8**)
- Measured cosim on final: **10/16**
- Geomean speedup vs gold: **13.58×** (n=15)
- CSV: [`2026-07-21-chathls-ready-c2hls-rag2-skills-results.csv`](2026-07-21-chathls-ready-c2hls-rag2-skills-results.csv)

## Final latency (prefer dataflow if pass and ≤ flash)

| Bench | size | flash | dataflow | df status | **final** | cosim | gold | speedup | stage |
|-------|------|------:|---------:|-----------|----------:|------:|-----:|--------:|-------|
| 2mm | chathls_imported_hlsfactory_sized | 37,645 | 40,922 | pass | **37,645** | — | 25,296,077 | 671.96× | flash |
| 3mm | chathls_imported_hlsfactory_sized | 93,850 | — | fail | **93,850** | 159,298 | 45,441,119 | 484.19× | flash |
| atax | chathls_imported_mini | 1,098 | 1,477 | pass | **1,098** | — | 236,250 | 215.16× | flash |
| bicg | chathls_imported_mini | 1,893 | 1,987 | pass | **1,893** | — | 234,516 | 123.89× | flash |
| gemm | chathls_imported_mini | 32,930 | — | fail | **32,930** | — | 2,871,462 | 87.20× | flash |
| covariance | chathls_imported_mini | 8,227 | 6,504 | pass | **6,504** | 1,081 | 233,415 | 35.89× | dataflow |
| symm | chathls_imported_hlsfactory_sized | 2,846,553 | — | fail | **2,846,553** | — | 37,430,401 | 13.15× | flash |
| gemm_ncubed | machsuite_64 | 34,135 | 43,215 | pass | **34,135** | 4,359 | 295,512 | 8.66× | flash |
| matmul | chathls_imported_mini | 6,301 | 6,298 | pass | **6,298** | 515 | 34,963 | 5.55× | dataflow |
| syr2k | chathls_imported_hlsfactory_sized | 407,673 | — | fail | **407,673** | — | 1,952,641 | 4.79× | flash |
| mvt | chathls_imported_mini | 5,007 | — | fail | **5,007** | 66 | 14,327 | 2.86× | flash |
| syrk | chathls_imported_hlsfactory_sized | 979,273 | — | fail | **979,273** | 10,404 | 1,183,841 | 1.21× | flash |
| transformer | unknown | 94,209 | — | fail | **94,209** | 246 | 73,953 | 0.78× | flash |
| gemm_blocked | machsuite_64 | 1,319,078 | 1,843,291 | pass | **1,319,078** | 3,955 | 901,125 | 0.68× | flash |
| gesummv | chathls_imported_mini | 2,314 | 2,317 | pass | **2,314** | 20 | 1,499 | 0.65× | flash |
| mobilenet | unknown | — | — | fail | **—** | 19,818 | — | — | none |

## Best / avg / worst (final)

| Bench | best | avg | worst | cosim | LUT | DSP |
|-------|-----:|----:|------:|------:|----:|----:|
| 2mm | 37,645 | 37,645 | 37,645 | — | 88,103 | 771 |
| 3mm | 93,850 | 93,850 | 93,850 | 159,298 | 127,188 | 882 |
| atax | 1,098 | 1,098 | 1,098 | — | 19,081 | 168 |
| bicg | 1,893 | 1,893 | 1,893 | — | 20,080 | 304 |
| covariance | 5,412 | 6,504 | 7,680 | 1,081 | 31,216 | 128 |
| gemm | 32,930 | 32,930 | 32,930 | — | 7,153 | 8 |
| gemm_blocked | 1,319,078 | 1,319,078 | 1,319,078 | 3,955 | 15,321 | 8 |
| gemm_ncubed | 34,135 | 34,135 | 34,135 | 4,359 | 26,361 | 104 |
| gesummv | 2,314 | 2,314 | 2,314 | 20 | 10,508 | 8 |
| matmul | 6,298 | 6,298 | 6,298 | 515 | 93,541 | 24 |
| mobilenet | — | — | — | 19,818 | — | — |
| mvt | 5,007 | 5,007 | 5,007 | 66 | 13,186 | 4 |
| symm | 201,753 | 2,846,553 | 5,582,553 | — | 13,143 | 11 |
| syr2k | 407,673 | 407,673 | 407,673 | — | 12,955 | 38 |
| syrk | 40,153 | 979,273 | 1,942,473 | 10,404 | 9,815 | 8 |
| transformer | 94,209 | 94,209 | 94,209 | 246 | 73,438 | 123 |

## Dataflow failures

- **gemm**: DATAFLOW contract check failed (3 breach(es)): - [m_axi-port-concurrent-rw] Port `C` has concurrent reader(s) ['load_all_inputs'] and writer(s) ['store_outputs'
- **3mm**: /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2hls_tmp/batch_parallel_chathls_fd_ds_rag2_20260721_123449/chathls_kernel_3mm/c2hls_compile__run_compile_006/ker
- **symm**: DATAFLOW contract check failed (2 breach(es)): - [local-buffer-multi-writer] Local `local_C` has multiple writer tasks ['compute_task', 'load_C_task'] — use fus
- **syr2k**: failed
- **syrk**: DATAFLOW contract check failed (5 breach(es)): - [m_axi-port-concurrent-rw] Port `C` has concurrent reader(s) ['load_C_task'] and writer(s) ['store_C_task'] — s
- **mobilenet**: ERROR: [HLS 214-194] in function 'compute_conv1_task(signed char*, signed char*)': Undefined function conv2d (/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/c2h
- **mvt**: failed
- **transformer**: DATAFLOW contract check failed (2 breach(es)): - [RULE_1] The DATAFLOW region contains only 3 sequential stages (load_all_inputs_task, compute_fused_task, store

## Notes

- Dataflow csynth recovered from `c2hls_tmp/.../hls_synth__dataflow*_synth` top XML (`dataflow_selected` bundle was empty).
- Gold = `reference_validation.report.latency_cycles` from flash multistep.
- Only **covariance** and **matmul** beat flash after dataflow; other dataflow passes were slower → final stays flash.

Path: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_20260721_123449`

# U280 inventory: c2hls + ChatHLS campaigns (stage latency & resources)

Generated: 2026-07-21 10:12 UTC (updated: gold baselines 2026-07-21 11:12 UTC)

**Update:** AutoSA campaigns were initially omitted from discovery; they are appended below (section *AutoSA campaigns*). Early `autosa_nav_n` runs used **4.0 ns** clock; `autosa_dse_fd` U280 runs use **3.33 ns**.

Target FPGA: `xcu280-fsvh2892-2L-e` @ 3.33 ns (where configured).

## Stage definitions

| Stage | Meaning | Typical source |
|-------|---------|----------------|
| **baseline** | Gold HLS (`hls_baseline` / reference gate) | `reference_validation.report` (NOT `phase_b_report`) |
| **flash_selected** | Best flash-opt kernel | `flash_opt_report` / `flash_selected/.../synth_report.json` / multistep final |
| **dataflow** | Post-flash dataflow rewrite | `*_dataflow_report.json` / `*_dataflow_result.json` |
| **final** | Pipeline-selected kernel | `*_selected_report.json` (ChatHLS: final CSV / summary.json) |

Latency: **csynth** = HLS estimated cycles; **cosim** = measured runtime cycles when available.
Resources: LUT / DSP / BRAM / FF from the same report as that stage.
**Over-device (`**`):** resource number exceeds full U280 device totals (LUT=1,303,680, FF=2,607,360, DSP=9,024, BRAM_18K=4,032). Those designs are excluded from speedup/geomean tables.
`—` = missing / stage not run.

**Cosim note:** most c2hls batch campaigns used `cosim_nodes_per_variant=0`. Flash-selected cosim cycles are attached from `artifacts/pc2/u280_compare_cosim_20260719_072401/flash_cosim/` when present (DS RAG2+skills, GLM RAG2+skills, GLM skills, ChatHLS native).

Companion CSV: [`2026-07-20-u280-all-campaigns-stage-inventory.csv`](2026-07-20-u280-all-campaigns-stage-inventory.csv).

## Completed: ChatHLS U280 DeepSeek machsuite + Tier-A

Finished aggregate as of **2026-07-21 10:12 UTC** (DeepSeek hybrid, U280 @ 3.33 ns):

- **Session:** `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-machsuite-tierA-20260721-012940`
- **Results:** **64 / 68** benches with `final_latency_csynth.csv` / `final_resources_csynth.csv`
- **Missing (4):** `machsuite_bfs_queue`, `machsuite_nw`, `machsuite_sort_merge`, `machsuite_spmv_crs`
- **Suites present:** 24 forgebench + 16 hp_fft + 14 machsuite + 10 spector_hls
- **`passed_optimization=True`:** 15 / 64
- **Cosim:** CSV marks `cosim_status=passed` for all 64, but **no `cosim_latency_cycles`** recorded (same pattern as native session in-session skip of measured cycles)
- **Over-device resources:** none
- **Slurm:** GPU `2030684`, gate `2030685`, array `2030686`
- **Spector/ChatHLS fix (2026-07-21 11:36 UTC):** cleared unbound `spector_hls_dct` and stub `spector_hls_template_matching`. Speedups prefer **cosim over csynth** when available.
- **Baseline = gold HLS (2026-07-21 11:12 UTC):** `baseline_*` from c2hls `reference_validation.json` (`report.latency_cycles` / resources), scanned across artifacts. **`phase_b` is translator output, not gold.** ChatHLS shared-bench baselines use the same gold.
- **Re-aggregate:** `python3 scripts/pc2/aggregate_chathls_u280_session_csvs.py --session-dir <session>`


## Quick map (architectures × models with data)

| Model / Architecture | Campaigns with data | Total bench-rows |
|----------------------|---------------------|------------------|
| DeepSeek / ChatHLS hybrid agent | 1 | 16 |
| DeepSeek / ChatHLS machsuite+Tier-A | 1 | 64 |
| DeepSeek / ChatHLS on c2hls-port benches | 1 | 42 |
| DeepSeek / RAG2+noskills | 1 | 14 |
| DeepSeek / RAG2+skills | 6 | 68 |
| DeepSeek / scrape/RAG1+noskills | 1 | 11 |
| DeepSeek / scrape/RAG1+skills | 1 | 13 |
| DeepSeek / skills | 1 | 15 |
| Devstral / RAG2+noskills | 1 | 14 |
| Devstral / RAG2+skills | 1 | 14 |
| Devstral / base/unknown | 2 | 24 |
| Devstral / scrape/RAG1+noskills | 2 | 23 |
| Devstral / scrape/RAG1+skills | 1 | 14 |
| GLM-4.7 / RAG2+noskills | 1 | 13 |
| GLM-4.7 / RAG2+skills | 1 | 14 |
| GLM-4.7 / scrape/RAG1+skills | 1 | 15 |
| GLM-4.7 / skills | 1 | 15 |

## Campaign index

| # | System | Model | Architecture | Campaign | Complete | #benches | Path |
|---|--------|-------|--------------|----------|----------|----------|------|
| 1 | c2hls-chathls-benches | Devstral | base/unknown | `batch_parallel_chathls_fd_20260713_chathls_fd_externc` | yes | 14 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_20260713_chathls_fd_externc` |
| 2 | c2hls-chathls-benches | Devstral | scrape/RAG1+skills | `batch_parallel_chathls_fd_20260714_chathls_rag` | yes | 14 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_20260714_chathls_rag` |
| 3 | c2hls-chathls-benches | DeepSeek | RAG2+skills | `batch_parallel_chathls_fd_ds_rag2_20260717_195140_rag2_skills` | yes | 14 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_20260717_195140_rag2_skills` |
| 4 | c2hls-chathls-benches | DeepSeek | RAG2+noskills | `batch_parallel_chathls_fd_ds_rag2_ns_20260717_215344_rag2_ns` | yes | 14 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_ns_20260717_215344_rag2_ns` |
| 5 | c2hls-chathls-benches | DeepSeek | scrape/RAG1+skills | `batch_parallel_chathls_fd_ds_rag_20260717_195138_rag_skills` | yes | 13 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag_20260717_195138_rag_skills` |
| 6 | c2hls-chathls-benches | DeepSeek | scrape/RAG1+noskills | `batch_parallel_chathls_fd_ds_rag_ns_20260718_040145_rag_ns` | yes | 11 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag_ns_20260718_040145_rag_ns` |
| 7 | c2hls-chathls-benches | DeepSeek | skills | `batch_parallel_chathls_fd_ds_skills_20260717_232141_skills` | yes | 15 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_skills_20260717_232141_skills` |
| 8 | c2hls-chathls-benches | GLM-4.7 | RAG2+skills | `batch_parallel_chathls_fd_glm_rag2_20260718_023346_rag2_skills` | yes | 14 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_rag2_20260718_023346_rag2_skills` |
| 9 | c2hls-chathls-benches | GLM-4.7 | RAG2+noskills | `batch_parallel_chathls_fd_glm_rag2_ns_20260718_131155_rag2_ns` | yes | 13 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_rag2_ns_20260718_131155_rag2_ns` |
| 10 | c2hls-chathls-benches | GLM-4.7 | scrape/RAG1+skills | `batch_parallel_chathls_fd_glm_rag_20260718_171759_rag_skills` | yes | 15 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_rag_20260718_171759_rag_skills` |
| 11 | c2hls-chathls-benches | GLM-4.7 | skills | `batch_parallel_chathls_fd_glm_skills_20260718_205203_skills` | yes | 15 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_skills_20260718_205203_skills` |
| 12 | c2hls-chathls-benches | Devstral | RAG2+skills | `batch_parallel_chathls_fd_rag2_20260717_091157_rag2_skills` | yes | 14 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag2_20260717_091157_rag2_skills` |
| 13 | c2hls-chathls-benches | Devstral | RAG2+noskills | `batch_parallel_chathls_fd_rag2_ns_20260717_091157_rag2_ns` | yes | 14 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag2_ns_20260717_091157_rag2_ns` |
| 14 | c2hls-chathls-benches | Devstral | scrape/RAG1+noskills | `batch_parallel_chathls_fd_rag_ns_20260716_chathls_rag_ns` | no | 9 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag_ns_20260716_chathls_rag_ns` |
| 15 | c2hls-chathls-benches | Devstral | scrape/RAG1+noskills | `batch_parallel_chathls_fd_rag_ns_20260716_chathls_rag_ns2` | yes | 14 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag_ns_20260716_chathls_rag_ns2` |
| 16 | c2hls-control | DeepSeek | RAG2+skills | `batch_parallel_chathls_fd_ds_rag2_20260719_095037_ds_ctrl` | yes | 14 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_20260719_095037_ds_ctrl` |
| 17 | c2hls-hlsfactory-port | DeepSeek | RAG2+skills | `batch_parallel_hlsfactory_ds_rag2_20260718_100518` | yes | 11 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_hlsfactory_ds_rag2_20260718_100518` |
| 18 | c2hls-latency-opt | DeepSeek | RAG2+skills | `batch_parallel_chathls_fd_ds_rag2_lat_20260719_095037_ds_lat` | yes | 16 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_lat_20260719_095037_ds_lat` |
| 19 | c2hls-machsuite-port | DeepSeek | RAG2+skills | `batch_parallel_machsuite_ds_rag2_20260718_100518` | yes | 12 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_machsuite_ds_rag2_20260718_100518` |
| 20 | c2hls-machsuite-port | Devstral | base/unknown | `batch_parallel_machsuite_fd_20260710_machsuite_flash_dataflow` | yes | 10 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_machsuite_fd_20260710_machsuite_flash_dataflow` |
| 21 | c2hls-transformer | DeepSeek | RAG2+skills | `batch_parallel_chathls_fd_ds_rag2_xfmr_20260720_070124` | yes | 1 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_xfmr_20260720_070124` |
| 22 | chathls-c2hlsport | DeepSeek | ChatHLS on c2hls-port benches | `hybrid-u280-c2hlsport-20260719-090238` | yes | 42 | `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-c2hlsport-20260719-090238` |
| 23 | chathls-native | DeepSeek | ChatHLS hybrid agent | `hybrid-u280-split-20260717-001649` | yes | 16 | `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-split-20260717-001649` |
| 23b | chathls-machsuite-tierA | DeepSeek | ChatHLS hybrid agent (machsuite+Tier-A) | `hybrid-u280-machsuite-tierA-20260721-012940` | no (64/68) | 64 | `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-machsuite-tierA-20260721-012940` |
| 24 | c2hls-chathls-benches | Devstral | base/unknown | `batch_parallel_chathls_fd_20260712_chathls_fd` | yes | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_20260712_chathls_fd` |
| 25 | c2hls-chathls-benches | Devstral | base/unknown | `batch_parallel_chathls_fd_20260713_chathls_fd_tb2` | yes | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_20260713_chathls_fd_tb2` |
| 26 | c2hls-chathls-benches | Devstral | scrape/RAG1+skills | `batch_parallel_chathls_fd_20260714_rag_dry` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_20260714_rag_dry` |
| 27 | c2hls-chathls-benches | Devstral | base/unknown | `batch_parallel_chathls_fd_dryrun_chathls_fd` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_dryrun_chathls_fd` |
| 28 | c2hls-chathls-benches | DeepSeek | RAG2+skills | `batch_parallel_chathls_fd_ds_rag2_20260717_100113_rag2_skills` | yes | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_20260717_100113_rag2_skills` |
| 29 | c2hls-chathls-benches | DeepSeek | RAG2+skills | `batch_parallel_chathls_fd_ds_rag2_20260720_070037` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_20260720_070037` |
| 30 | c2hls-chathls-benches | DeepSeek | RAG2+noskills | `batch_parallel_chathls_fd_ds_rag2_ns_20260717_120315_rag2_ns` | yes | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_ns_20260717_120315_rag2_ns` |
| 31 | c2hls-chathls-benches | DeepSeek | scrape/RAG1+skills | `batch_parallel_chathls_fd_ds_rag_20260717_100332_rag_skills` | yes | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag_20260717_100332_rag_skills` |
| 32 | c2hls-chathls-benches | DeepSeek | scrape/RAG1+noskills | `batch_parallel_chathls_fd_ds_rag_ns_20260717_140736_rag_ns` | yes | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag_ns_20260717_140736_rag_ns` |
| 33 | c2hls-chathls-benches | DeepSeek | skills | `batch_parallel_chathls_fd_ds_skills_20260717_120534_skills` | yes | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_skills_20260717_120534_skills` |
| 34 | c2hls-chathls-benches | GLM-4.7 | RAG2+skills | `batch_parallel_chathls_fd_glm_rag2_20260718_021557_rag2_skills` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_rag2_20260718_021557_rag2_skills` |
| 35 | c2hls-chathls-benches | GLM-4.7 | RAG2+noskills | `batch_parallel_chathls_fd_glm_rag2_ns_20260718_021557_rag2_ns` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_rag2_ns_20260718_021557_rag2_ns` |
| 36 | c2hls-chathls-benches | GLM-4.7 | scrape/RAG1+skills | `batch_parallel_chathls_fd_glm_rag_20260718_021558_rag_skills` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_rag_20260718_021558_rag_skills` |
| 37 | c2hls-chathls-benches | GLM-4.7 | skills | `batch_parallel_chathls_fd_glm_skills_20260718_021558_skills` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_skills_20260718_021558_skills` |
| 38 | c2hls-chathls-benches | Devstral | scrape/RAG1+noskills | `batch_parallel_chathls_fd_rag_ns_20260716_rag_ns_dry` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag_ns_20260716_rag_ns_dry` |
| 39 | c2hls-chathls-benches | unknown | base/unknown | `latency_opt_ab_20260719_094947` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/latency_opt_ab_20260719_094947` |
| 40 | c2hls-chathls-benches | unknown | base/unknown | `latency_opt_ab_20260719_095016` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/latency_opt_ab_20260719_095016` |
| 41 | c2hls-chathls-benches | unknown | base/unknown | `latency_opt_ab_20260719_095037` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/latency_opt_ab_20260719_095037` |
| 42 | c2hls-control | DeepSeek | RAG2+skills | `batch_parallel_chathls_fd_ds_rag2_20260719_094947_ds_ctrl` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_20260719_094947_ds_ctrl` |
| 43 | c2hls-control | DeepSeek | RAG2+skills | `batch_parallel_chathls_fd_ds_rag2_20260719_095016_ds_ctrl` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_20260719_095016_ds_ctrl` |
| 44 | c2hls-control | GLM-4.7 | RAG2+skills | `batch_parallel_chathls_fd_glm_rag2_20260719_094947_glm_ctrl` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_rag2_20260719_094947_glm_ctrl` |
| 45 | c2hls-control | Devstral | RAG2+skills | `batch_parallel_chathls_fd_rag2_20260719_094947_dv_ctrl` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag2_20260719_094947_dv_ctrl` |
| 46 | c2hls-control | Devstral | RAG2+skills | `batch_parallel_chathls_fd_rag2_20260719_095016_dv_ctrl` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag2_20260719_095016_dv_ctrl` |
| 47 | c2hls-control | Devstral | RAG2+skills | `batch_parallel_chathls_fd_rag2_20260719_095037_dv_ctrl` | yes | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag2_20260719_095037_dv_ctrl` |
| 48 | c2hls-hlsfactory-port | DeepSeek | RAG2+skills | `batch_parallel_hlsfactory_ds_rag2_20260718_095945` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_hlsfactory_ds_rag2_20260718_095945` |
| 49 | c2hls-hlsfactory-port | DeepSeek | RAG2+skills | `batch_parallel_hlsfactory_ds_rag2_20260718_100006` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_hlsfactory_ds_rag2_20260718_100006` |
| 50 | c2hls-hlsfactory-port | DeepSeek | RAG2+skills | `batch_parallel_hlsfactory_ds_rag2_20260718_100010` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_hlsfactory_ds_rag2_20260718_100010` |
| 51 | c2hls-latency-opt | DeepSeek | RAG2+skills | `batch_parallel_chathls_fd_ds_rag2_lat_20260719_094947_ds_lat` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_lat_20260719_094947_ds_lat` |
| 52 | c2hls-latency-opt | DeepSeek | RAG2+skills | `batch_parallel_chathls_fd_ds_rag2_lat_dry_ds_lat` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_lat_dry_ds_lat` |
| 53 | c2hls-latency-opt | GLM-4.7 | RAG2+skills | `batch_parallel_chathls_fd_glm_rag2_lat_20260719_094947_glm_lat` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_rag2_lat_20260719_094947_glm_lat` |
| 54 | c2hls-latency-opt | Devstral | RAG2+skills | `batch_parallel_chathls_fd_rag2_lat_20260719_094947_dv_lat` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag2_lat_20260719_094947_dv_lat` |
| 55 | c2hls-latency-opt | Devstral | RAG2+skills | `batch_parallel_chathls_fd_rag2_lat_20260719_095037_dv_lat` | yes | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag2_lat_20260719_095037_dv_lat` |
| 56 | c2hls-latency-opt | Devstral | RAG2+skills | `batch_parallel_chathls_fd_rag2_lat_dry_lat_test` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag2_lat_dry_lat_test` |
| 57 | c2hls-machsuite-port | DeepSeek | RAG2+skills | `batch_parallel_machsuite_ds_rag2_20260718_095815` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_machsuite_ds_rag2_20260718_095815` |
| 58 | c2hls-seq-launcher | DeepSeek | RAG2+skills | `deepseek_u280_rag2_seq_20260717_085610` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/deepseek_u280_rag2_seq_20260717_085610` |
| 59 | c2hls-seq-launcher | DeepSeek | RAG2+skills | `deepseek_u280_rag2_seq_20260717_195138` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/deepseek_u280_rag2_seq_20260717_195138` |
| 60 | c2hls-seq-launcher | DeepSeek | base/unknown | `deepseek_u280_seq_20260717_063829` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/deepseek_u280_seq_20260717_063829` |
| 61 | c2hls-seq-launcher | DeepSeek | base/unknown | `deepseek_u280_seq_20260717_195136` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/deepseek_u280_seq_20260717_195136` |
| 62 | c2hls-seq-launcher | unknown | base/unknown | `dual_track_u280_20260718_100202` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/dual_track_u280_20260718_100202` |
| 63 | c2hls-seq-launcher | DeepSeek | base/unknown | `dual_track_u280_20260718_100516` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/dual_track_u280_20260718_100516` |
| 64 | c2hls-seq-launcher | GLM-4.7 | base/unknown | `glm_u280_seq_20260718_021556` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/glm_u280_seq_20260718_021556` |
| 65 | c2hls-seq-launcher | GLM-4.7 | base/unknown | `glm_u280_seq_20260718_021707` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/glm_u280_seq_20260718_021707` |
| 66 | c2hls-seq-launcher | GLM-4.7 | base/unknown | `glm_u280_seq_20260718_021751` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/glm_u280_seq_20260718_021751` |
| 67 | c2hls-seq-launcher | GLM-4.7 | base/unknown | `glm_u280_seq_20260718_021859` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/glm_u280_seq_20260718_021859` |
| 68 | c2hls-seq-launcher | GLM-4.7 | base/unknown | `glm_u280_seq_20260718_021929` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/glm_u280_seq_20260718_021929` |
| 69 | chathls-c2hlsport | DeepSeek | ChatHLS on c2hls-port benches | `hybrid-u280-c2hlsport-20260718-120516` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-c2hlsport-20260718-120516` |
| 70 | chathls-session-empty | ChatHLS | ChatHLS | `hybrid-u280-batch-20260716-103317` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-batch-20260716-103317` |
| 71 | chathls-session-empty | ChatHLS | ChatHLS | `hybrid-u280-batch-20260716-104537` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-batch-20260716-104537` |
| 72 | chathls-session-empty | ChatHLS | ChatHLS | `hybrid-u280-split-20260716-113938` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-split-20260716-113938` |
| 73 | chathls-session-empty | ChatHLS | ChatHLS | `hybrid-u280-split-20260716-234659` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-split-20260716-234659` |
| 74 | cosim-compare-batch | unknown | base/unknown | `u280_compare_cosim_20260719_072352` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/u280_compare_cosim_20260719_072352` |
| 75 | cosim-compare-batch | unknown | base/unknown | `u280_compare_cosim_20260719_072401` | no | 0 | `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/u280_compare_cosim_20260719_072401` |

## Campaigns / launchers with no extractable stage reports

These exist on disk (often aborted relaunches, dry-runs, or seq launchers that only pointer to batch campaigns) but have no `*_report.json` / CSV metrics.

- `batch_parallel_chathls_fd_20260712_chathls_fd` — c2hls-chathls-benches, model=Devstral, arch=base/unknown, complete=True, flash_link=True
- `batch_parallel_chathls_fd_20260713_chathls_fd_tb2` — c2hls-chathls-benches, model=Devstral, arch=base/unknown, complete=True, flash_link=True
- `batch_parallel_chathls_fd_20260714_rag_dry` — c2hls-chathls-benches, model=Devstral, arch=scrape/RAG1+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_dryrun_chathls_fd` — c2hls-chathls-benches, model=Devstral, arch=base/unknown, complete=False, flash_link=False
- `batch_parallel_chathls_fd_ds_rag2_20260717_100113_rag2_skills` — c2hls-chathls-benches, model=DeepSeek, arch=RAG2+skills, complete=True, flash_link=True
- `batch_parallel_chathls_fd_ds_rag2_20260720_070037` — c2hls-chathls-benches, model=DeepSeek, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_ds_rag2_ns_20260717_120315_rag2_ns` — c2hls-chathls-benches, model=DeepSeek, arch=RAG2+noskills, complete=True, flash_link=True
- `batch_parallel_chathls_fd_ds_rag_20260717_100332_rag_skills` — c2hls-chathls-benches, model=DeepSeek, arch=scrape/RAG1+skills, complete=True, flash_link=True
- `batch_parallel_chathls_fd_ds_rag_ns_20260717_140736_rag_ns` — c2hls-chathls-benches, model=DeepSeek, arch=scrape/RAG1+noskills, complete=True, flash_link=True
- `batch_parallel_chathls_fd_ds_skills_20260717_120534_skills` — c2hls-chathls-benches, model=DeepSeek, arch=skills, complete=True, flash_link=True
- `batch_parallel_chathls_fd_glm_rag2_20260718_021557_rag2_skills` — c2hls-chathls-benches, model=GLM-4.7, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_glm_rag2_ns_20260718_021557_rag2_ns` — c2hls-chathls-benches, model=GLM-4.7, arch=RAG2+noskills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_glm_rag_20260718_021558_rag_skills` — c2hls-chathls-benches, model=GLM-4.7, arch=scrape/RAG1+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_glm_skills_20260718_021558_skills` — c2hls-chathls-benches, model=GLM-4.7, arch=skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_rag_ns_20260716_rag_ns_dry` — c2hls-chathls-benches, model=Devstral, arch=scrape/RAG1+noskills, complete=False, flash_link=False
- `latency_opt_ab_20260719_094947` — c2hls-chathls-benches, model=unknown, arch=base/unknown, complete=False, flash_link=False
- `latency_opt_ab_20260719_095016` — c2hls-chathls-benches, model=unknown, arch=base/unknown, complete=False, flash_link=False
- `latency_opt_ab_20260719_095037` — c2hls-chathls-benches, model=unknown, arch=base/unknown, complete=False, flash_link=False
- `batch_parallel_chathls_fd_ds_rag2_20260719_094947_ds_ctrl` — c2hls-control, model=DeepSeek, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_ds_rag2_20260719_095016_ds_ctrl` — c2hls-control, model=DeepSeek, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_glm_rag2_20260719_094947_glm_ctrl` — c2hls-control, model=GLM-4.7, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_rag2_20260719_094947_dv_ctrl` — c2hls-control, model=Devstral, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_rag2_20260719_095016_dv_ctrl` — c2hls-control, model=Devstral, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_rag2_20260719_095037_dv_ctrl` — c2hls-control, model=Devstral, arch=RAG2+skills, complete=True, flash_link=True
- `batch_parallel_hlsfactory_ds_rag2_20260718_095945` — c2hls-hlsfactory-port, model=DeepSeek, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_hlsfactory_ds_rag2_20260718_100006` — c2hls-hlsfactory-port, model=DeepSeek, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_hlsfactory_ds_rag2_20260718_100010` — c2hls-hlsfactory-port, model=DeepSeek, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_ds_rag2_lat_20260719_094947_ds_lat` — c2hls-latency-opt, model=DeepSeek, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_ds_rag2_lat_dry_ds_lat` — c2hls-latency-opt, model=DeepSeek, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_glm_rag2_lat_20260719_094947_glm_lat` — c2hls-latency-opt, model=GLM-4.7, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_rag2_lat_20260719_094947_dv_lat` — c2hls-latency-opt, model=Devstral, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_chathls_fd_rag2_lat_20260719_095037_dv_lat` — c2hls-latency-opt, model=Devstral, arch=RAG2+skills, complete=True, flash_link=True
- `batch_parallel_chathls_fd_rag2_lat_dry_lat_test` — c2hls-latency-opt, model=Devstral, arch=RAG2+skills, complete=False, flash_link=False
- `batch_parallel_machsuite_ds_rag2_20260718_095815` — c2hls-machsuite-port, model=DeepSeek, arch=RAG2+skills, complete=False, flash_link=False
- `deepseek_u280_rag2_seq_20260717_085610` — c2hls-seq-launcher, model=DeepSeek, arch=RAG2+skills, complete=False, flash_link=False
- `deepseek_u280_rag2_seq_20260717_195138` — c2hls-seq-launcher, model=DeepSeek, arch=RAG2+skills, complete=False, flash_link=False
- `deepseek_u280_seq_20260717_063829` — c2hls-seq-launcher, model=DeepSeek, arch=base/unknown, complete=False, flash_link=False
- `deepseek_u280_seq_20260717_195136` — c2hls-seq-launcher, model=DeepSeek, arch=base/unknown, complete=False, flash_link=False
- `dual_track_u280_20260718_100202` — c2hls-seq-launcher, model=unknown, arch=base/unknown, complete=False, flash_link=False
- `dual_track_u280_20260718_100516` — c2hls-seq-launcher, model=DeepSeek, arch=base/unknown, complete=False, flash_link=False
- `glm_u280_seq_20260718_021556` — c2hls-seq-launcher, model=GLM-4.7, arch=base/unknown, complete=False, flash_link=False
- `glm_u280_seq_20260718_021707` — c2hls-seq-launcher, model=GLM-4.7, arch=base/unknown, complete=False, flash_link=False
- `glm_u280_seq_20260718_021751` — c2hls-seq-launcher, model=GLM-4.7, arch=base/unknown, complete=False, flash_link=False
- `glm_u280_seq_20260718_021859` — c2hls-seq-launcher, model=GLM-4.7, arch=base/unknown, complete=False, flash_link=False
- `glm_u280_seq_20260718_021929` — c2hls-seq-launcher, model=GLM-4.7, arch=base/unknown, complete=False, flash_link=False
- `hybrid-u280-c2hlsport-20260718-120516` — chathls-c2hlsport, model=DeepSeek, arch=ChatHLS on c2hls-port benches, complete=False, flash_link=False
- `hybrid-u280-batch-20260716-103317` — chathls-session-empty, model=ChatHLS, arch=ChatHLS, complete=False, flash_link=False
- `hybrid-u280-batch-20260716-104537` — chathls-session-empty, model=ChatHLS, arch=ChatHLS, complete=False, flash_link=False
- `hybrid-u280-split-20260716-113938` — chathls-session-empty, model=ChatHLS, arch=ChatHLS, complete=False, flash_link=False
- `hybrid-u280-split-20260716-234659` — chathls-session-empty, model=ChatHLS, arch=ChatHLS, complete=False, flash_link=False
- `u280_compare_cosim_20260719_072352` — cosim-compare-batch, model=unknown, arch=base/unknown, complete=False, flash_link=False
- `u280_compare_cosim_20260719_072401` — cosim-compare-batch, model=unknown, arch=base/unknown, complete=False, flash_link=False

## `batch_parallel_chathls_fd_20260713_chathls_fd_externc`

- **System**: c2hls-chathls-benches
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: base/unknown (flavor=``)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_20260713_chathls_fd_externc`
- **Created**: 2026-07-13T22:33:25.561902+00:00; **Completed**: 2026-07-14T10:34:15.330580+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 73,875 | — | 41,967/168/84/73,545 | — | — | —/—/—/— | False | 73,875 | — | 41,967/168/84/73,545 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 100,086 | — | 84,207/663/566/144,735 | — | — | —/—/—/— | False | 100,086 | — | 84,207/663/566/144,735 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 1,099 | — | 20,847/336/16/37,582 | 1,139 | — | 20,755/336/16/38,595 | True | 1,099 | — | 20,847/336/16/37,582 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 1,056 | — | 19,730/304/20/34,081 | 1,053 | — | 29,742/304/20/54,271 | True | 1,056 | — | 19,730/304/20/34,081 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 42,690 | — | 9,621/4/32/10,810 | — | — | —/—/—/— | False | 42,690 | — | 9,621/4/32/10,810 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 2,281 | — | 18,636/183/12/21,983 | 2,283 | — | 17,906/183/12/22,572 | True | 2,281 | — | 18,636/183/12/21,983 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 612,682 | — | 21,029/22/90/35,707 | — | — | —/—/—/— | False | 612,682 | — | 21,029/22/90/35,707 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 42,449 | — | 88,694/704/106/116,583 | 46,549 | — | 82,578/704/122/118,242 | True | 42,449 | — | 88,694/704/106/116,583 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 698 | — | 18,484/248/20/27,375 | 696 | — | 16,711/248/20/28,587 | True | 698 | — | 18,484/248/20/27,375 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 34,974 | — | 12,477/3/96/14,903 | 34,970 | — | 11,794/3/102/15,606 | True | 34,974 | — | 12,477/3/96/14,903 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 474 | — | 37,333/320/80/97,298 | — | — | —/—/—/— | False | 474 | — | 37,333/320/80/97,298 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 24,552,001 | — | 29,209/19/12/57,775 | — | — | —/—/—/— | False | 24,552,001 | — | 29,209/19/12/57,775 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | — | — | 16,669/37/94/18,685 | — | — | —/—/—/— | — | — | — | 16,669/37/94/18,685 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 118,874 | — | 13,060/52/46/16,363 | 110,796 | — | 11,594/52/46/15,648 | True | 118,874 | — | 13,060/52/46/16,363 |

## `batch_parallel_chathls_fd_20260714_chathls_rag`

- **System**: c2hls-chathls-benches
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: scrape/RAG1+skills (flavor=``)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_20260714_chathls_rag`
- **Created**: 2026-07-14T12:07:00.914686+00:00; **Completed**: 2026-07-15T00:17:18.201200+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 55,034 | — | 84,678/780/184/129,707 | 55,035 | — | 77,768/780/388/125,787 | True | 55,034 | — | 84,678/780/184/129,707 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 90,346 | — | 98,915/663/366/152,845 | 67,093 | — | 167,482/2,091/498/273,191 | True | 90,346 | — | 98,915/663/366/152,845 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 1,098 | — | 19,423/168/16/27,264 | 1,098 | — | 31,647/168/16/48,717 | True | 1,098 | — | 19,423/168/16/27,264 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 1,056 | — | 19,730/304/20/34,081 | 1,097 | — | 24,457/304/20/44,344 | True | 1,056 | — | 19,730/304/20/34,081 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 35,994 | — | 12,662/4/22/16,649 | 41,794 | — | 10,792/4/36/12,945 | True | 35,994 | — | 12,662/4/22/16,649 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 2,301 | — | 18,946/183/10/21,974 | 2,303 | — | 18,256/183/12/22,559 | True | 2,301 | — | 18,946/183/10/21,974 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 746,497 | — | 17,815/8/90/31,050 | 1,319,007 | — | 11,398/8/186/16,245 | True | 746,497 | — | 17,815/8/90/31,050 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 42,449 | — | 88,688/704/106/116,574 | 46,549 | — | 82,572/704/122/118,233 | True | 42,449 | — | 88,688/704/106/116,574 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 698 | — | 18,789/248/20/27,617 | 625 | — | 17,132/248/20/28,893 | True | 698 | — | 18,789/248/20/27,617 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 18,577 | — | 12,381/12/98/15,383 | 19,606 | — | 12,438/12/102/15,774 | True | 18,577 | — | 12,381/12/98/15,383 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 2,603 | — | 14,407/16/88/15,652 | 2,604 | — | 14,458/16/96/14,667 | True | 2,603 | — | 14,407/16/88/15,652 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 66,528,001 | — | 10,503/8/12/8,182 | 1,723,521 | — | 13,850/19/222/18,214 | True | 66,528,001 | — | 10,503/8/12/8,182 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 407,673 | — | 12,935/38/158/15,760 | 396,929 | — | 12,712/38/304/17,243 | True | 407,673 | — | 12,935/38/158/15,760 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 118,874 | — | 13,501/76/46/17,911 | 110,796 | — | 12,035/76/46/17,196 | True | 118,874 | — | 13,501/76/46/17,911 |

## `batch_parallel_chathls_fd_ds_rag2_20260717_195140_rag2_skills`

- **System**: c2hls-chathls-benches
- **Model**: DeepSeek (`deepseek-chat`)
- **Architecture**: RAG2+skills (flavor=`rag2_skills`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_20260717_195140_rag2_skills`
- **Created**: 2026-07-17T19:51:40.797179+00:00; **Completed**: 2026-07-17T21:53:22.016536+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 54,794 | 54,763 | 85,846/772/184/133,578 | — | — | —/—/—/— | False | 54,794 | 54,763 | 85,846/772/184/133,578 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 90,644 | 90,539 | 100,024/882/346/158,198 | — | — | —/—/—/— | False | 90,644 | 90,539 | 100,024/882/346/158,198 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 1,921 | 2,015 | 19,561/337/54/38,331 | 2,080 | — | 20,700/57/24/27,599 | True | 1,921 | 2,015 | 19,561/337/54/38,331 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 1,934 | 2,034 | 18,253/153/56/25,191 | 2,305 | — | 15,622/33/28/15,393 | True | 1,934 | 2,034 | 18,253/153/56/25,191 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 8,674 | 8,870 | 42,815/128/28/38,718 | — | — | —/—/—/— | False | 8,674 | 8,870 | 42,815/128/28/38,718 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 1,319,078 | 1,319,202 | 15,315/8/138/18,048 | 1,843,291 | — | 13,667/8/186/16,475 | True | 1,319,078 | 1,319,202 | 15,315/8/138/18,048 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 1,319,080 | 1,319,211 | 13,430/8/90/16,414 | 1,319,075 | — | 12,786/8/122/16,782 | True | 1,319,080 | 1,319,211 | 13,430/8/90/16,414 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 2,344 | 3,497 | 9,127/8/24/11,780 | 1,686 | — | 15,627/32/28/18,234 | True | 2,344 | 3,497 | 9,127/8/24/11,780 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 3,767 | 3,834 | 20,729/96/94/25,083 | 3,765 | — | 18,523/96/98/25,792 | True | 3,767 | 3,834 | 20,729/96/94/25,083 |
| mobilenet | — | — | 159,102/712/280/77,355 | 3,742,647 | 3,743,427 | 79,041/571/606/50,331 | — | — | —/—/—/— | False | 3,742,647 | 3,743,427 | 79,041/571/606/50,331 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 5,007 | 5,103 | 13,186/4/84/15,429 | — | — | —/—/—/— | False | 5,007 | 5,103 | 13,186/4/84/15,429 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 50,432 | 50,423 | 116,856/668/76/617,062 | — | — | —/—/—/— | False | 50,432 | 50,423 | 116,856/668/76/617,062 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 392,034 | 384,713 | 16,186/22/158/26,210 | — | — | —/—/—/— | False | 392,034 | 384,713 | 16,186/22/158/26,210 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 363,295 | 355,965 | 11,035/11/46/12,039 | — | — | —/—/—/— | False | 363,295 | 355,965 | 11,035/11/46/12,039 |

## `batch_parallel_chathls_fd_ds_rag2_ns_20260717_215344_rag2_ns`

- **System**: c2hls-chathls-benches
- **Model**: DeepSeek (`deepseek-chat`)
- **Architecture**: RAG2+noskills (flavor=`rag2_ns`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_ns_20260717_215344_rag2_ns`
- **Created**: 2026-07-17T21:53:44.376860+00:00; **Completed**: 2026-07-18T04:18:27.396556+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 25,322 | — | 207,753/1,330/190/188,811 | 25,442 | — | 207,806/1,330/308/201,613 | True | 25,322 | — | 207,753/1,330/190/188,811 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 90,518 | — | 172,059/881/150/203,856 | 86,775 | — | 152,136/880/150/200,999 | True | 90,518 | — | 172,059/881/150/203,856 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 1,859 | — | 44,974/336/16/94,042 | 1,903 | — | 54,386/336/16/115,082 | True | 1,859 | — | 44,974/336/16/94,042 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 982 | — | 12,502/16/20/13,778 | — | — | —/—/—/— | False | 982 | — | 12,502/16/20/13,778 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 5,114 | — | 101,262/44/12/171,408 | — | — | —/—/—/— | False | 5,114 | — | 101,262/44/12/171,408 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 1,419 | — | 1,447,466**/9,160**/8/927,627 | — | — | —/—/—/— | False | 1,419 | — | 1,447,466**/9,160**/8/927,627 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 593,986 | — | 22,743/77/90/32,444 | — | — | —/—/—/— | False | 593,986 | — | 22,743/77/90/32,444 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 307,404 | — | 25,160/11/90/168,553 | 34,463 | — | 29,000/88/474/46,974 | True | 307,404 | — | 25,160/11/90/168,553 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 878 | — | 14,895/64/28/15,899 | — | — | —/—/—/— | False | 878 | — | 14,895/64/28/15,899 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 53,217 | — | 9,369/24/34/17,225 | 14,310 | — | 17,252/24/90/21,397 | True | 53,217 | — | 9,369/24/34/17,225 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 2,149 | — | 25,725/32/80/76,204 | — | — | —/—/—/— | False | 2,149 | — | 25,725/32/80/76,204 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 21,731,057 | — | 28,540/19/98/49,097 | — | — | —/—/—/— | False | 21,731,057 | — | 28,540/19/98/49,097 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 1,990,797 | — | 25,139/11/302/38,715 | 19,858 | — | 119,133/2,288/642/227,862 | True | 1,990,797 | — | 25,139/11/302/38,715 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 1,351,991 | — | 279,995/8/46/68,333 | — | — | —/—/—/— | False | 1,351,991 | — | 279,995/8/46/68,333 |

## `batch_parallel_chathls_fd_ds_rag_20260717_195138_rag_skills`

- **System**: c2hls-chathls-benches
- **Model**: DeepSeek (`deepseek-chat`)
- **Architecture**: scrape/RAG1+skills (flavor=`rag_skills`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag_20260717_195138_rag_skills`
- **Created**: 2026-07-17T19:51:38.804078+00:00; **Completed**: 2026-07-17T23:21:15.541746+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 70,402 | — | 49,869/616/120/98,759 | — | — | —/—/—/— | False | 70,402 | — | 49,869/616/120/98,759 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 90,644 | — | 99,969/882/346/158,172 | — | — | —/—/—/— | False | 90,644 | — | 99,969/882/346/158,172 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 1,897 | — | 20,885/336/16/37,034 | 2,121 | — | 15,407/57/24/19,633 | True | 1,897 | — | 20,885/336/16/37,034 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 1,135 | — | 19,624/152/20/27,782 | 2,712 | — | 15,837/32/28/15,281 | True | 1,135 | — | 19,624/152/20/27,782 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 7,133 | — | 41,848/128/28/32,037 | — | — | —/—/—/— | False | 7,133 | — | 41,848/128/28/32,037 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 25,405 | — | 6,673/20/20/5,183 | — | — | —/—/—/— | False | 25,405 | — | 6,673/20/20/5,183 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 256,425 | — | 13,981/16/94/18,514 | — | — | —/—/—/— | False | 256,425 | — | 13,981/16/94/18,514 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 1,319,077 | — | 11,744/8/90/15,310 | — | — | —/—/—/— | False | 1,319,077 | — | 11,744/8/90/15,310 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 1,115 | — | 16,461/248/76/22,737 | — | — | —/—/—/— | False | 1,115 | — | 16,461/248/76/22,737 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 333 | — | 106,004/3,072/90/257,124 | 18,587 | — | 12,060/6/102/15,986 | True | 333 | — | 106,004/3,072/90/257,124 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 2,605 | — | 14,692/16/88/19,225 | — | — | —/—/—/— | False | 2,605 | — | 14,692/16/88/19,225 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 24,859,201 | — | 22,874/22/12/46,923 | — | — | —/—/—/— | False | 24,859,201 | — | 22,874/22/12/46,923 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 800,481 | — | 8,081/35/12/8,524 | — | — | —/—/—/— | False | 800,481 | — | 8,081/35/12/8,524 |

## `batch_parallel_chathls_fd_ds_rag_ns_20260718_040145_rag_ns`

- **System**: c2hls-chathls-benches
- **Model**: DeepSeek (`deepseek-chat`)
- **Architecture**: scrape/RAG1+noskills (flavor=`rag_ns`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag_ns_20260718_040145_rag_ns`
- **Created**: 2026-07-18T04:01:46.091598+00:00; **Completed**: 2026-07-18T06:50:03.812069+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 307,576 | — | 96,031/553/186/208,612 | 67,025 | — | 164,882/2,091/482/269,446 | True | 307,576 | — | 96,031/553/186/208,612 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 1,288 | — | 59,807/168/16/86,016 | 2,018 | — | 75,277/168/14/99,820 | True | 1,288 | — | 59,807/168/16/86,016 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 1,708 | — | 13,241/304/4/25,841 | 1,753 | — | 22,598/304/4/38,433 | True | 1,708 | — | 13,241/304/4/25,841 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 5,824 | — | 34,917/64/10/40,788 | — | — | —/—/—/— | False | 5,824 | — | 34,917/64/10/40,788 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 3,251 | — | 58,843/40/8/100,290 | 2,621 | — | 9,699/107/16/13,654 | True | 3,251 | — | 58,843/40/8/100,290 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 2,719,783 | — | 32,629/—/90/32,681 | — | — | —/—/—/— | False | 2,719,783 | — | 32,629/—/90/32,681 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 306,388 | — | 26,608/11/106/172,855 | 46,286 | — | 88,555/704/96/121,578 | True | 306,388 | — | 26,608/11/106/172,855 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 245 | — | 12,672/16/10/15,198 | 700 | — | 22,449/240/20/37,078 | True | 245 | — | 12,672/16/10/15,198 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 3,230 | — | 21,507/96/90/27,154 | 3,226 | — | 18,548/96/90/27,880 | True | 3,230 | — | 21,507/96/90/27,154 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 499 | — | 31,770/320/52/88,940 | 1,877 | — | 140,695/320/80/261,105 | True | 499 | — | 31,770/320/52/88,940 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 1,099,272 | — | 9,698/11/16/12,456 | — | — | —/—/—/— | False | 1,099,272 | — | 9,698/11/16/12,456 |

## `batch_parallel_chathls_fd_ds_skills_20260717_232141_skills`

- **System**: c2hls-chathls-benches
- **Model**: DeepSeek (`deepseek-chat`)
- **Architecture**: skills (flavor=`skills`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_skills_20260717_232141_skills`
- **Created**: 2026-07-17T23:21:41.993712+00:00; **Completed**: 2026-07-18T04:01:28.850880+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 53,914 | — | 106,963/771/84/143,338 | — | — | —/—/—/— | — | 53,914 | — | 106,963/771/84/143,338 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 92,640 | — | 122,217/882/254/168,945 | — | — | —/—/—/— | — | 92,640 | — | 122,217/882/254/168,945 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 3,318 | — | 10,197/33/20/16,561 | — | — | —/—/—/— | — | 3,318 | — | 10,197/33/20/16,561 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 2,596 | — | 9,922/9/24/11,376 | — | — | —/—/—/— | — | 2,596 | — | 9,922/9/24/11,376 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 6,234 | — | 42,279/128/28/32,661 | — | — | —/—/—/— | — | 6,234 | — | 42,279/128/28/32,661 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 31,419 | — | 6,230/21/14/5,168 | — | — | —/—/—/— | — | 31,419 | — | 6,230/21/14/5,168 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 1,695,911 | — | 15,331/11/138/19,205 | — | — | —/—/—/— | — | 1,695,911 | — | 15,331/11/138/19,205 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 1,319,077 | — | 11,744/8/90/15,310 | — | — | —/—/—/— | — | 1,319,077 | — | 11,744/8/90/15,310 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 2,315 | — | 10,590/8/24/12,721 | — | — | —/—/—/— | — | 2,315 | — | 10,590/8/24/12,721 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 3,640 | — | 25,009/96/98/27,279 | — | — | —/—/—/— | — | 3,640 | — | 25,009/96/98/27,279 |
| mobilenet | — | — | 159,052/712/269/77,062 | 1,396,884 | — | 125,024/559/363/47,291 | — | — | —/—/—/— | — | 1,396,884 | — | 125,024/559/363/47,291 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 5,007 | — | 12,954/4/84/15,372 | — | — | —/—/—/— | — | 5,007 | — | 12,954/4/84/15,372 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 34,190,402 | — | 9,605/22/12/8,323 | — | — | —/—/—/— | — | 34,190,402 | — | 9,605/22/12/8,323 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 1,365,768 | — | 13,774/11/62/14,803 | — | — | —/—/—/— | — | 1,365,768 | — | 13,774/11/62/14,803 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 1,362,874 | — | 10,311/8/46/11,016 | — | — | —/—/—/— | — | 1,362,874 | — | 10,311/8/46/11,016 |

## `batch_parallel_chathls_fd_glm_rag2_20260718_023346_rag2_skills`

- **System**: c2hls-chathls-benches
- **Model**: GLM-4.7 (`GLM-4.7-FP8`)
- **Architecture**: RAG2+skills (flavor=`rag2_skills`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_rag2_20260718_023346_rag2_skills`
- **Created**: 2026-07-18T02:33:46.543379+00:00; **Completed**: 2026-07-18T13:10:17.066537+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 1,511,402 | — | 17,488/17/84/19,330 | — | — | —/—/—/— | False | 1,511,402 | — | 17,488/17/84/19,330 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 852,111 | 852,352 | 19,554/34/94/20,657 | 845,681 | — | 19,784/34/166/20,626 | True | 852,111 | 852,352 | 19,554/34/94/20,657 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 5,370 | 5,395 | 8,285/10/20/9,988 | 5,337 | — | 20,613/10/16/22,244 | True | 5,370 | 5,395 | 8,285/10/20/9,988 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 296 | 395 | 17,030/152/16/30,857 | 3,088 | — | 17,255/32/28/30,989 | True | 296 | 395 | 17,030/152/16/30,857 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 41,879 | 42,470 | 14,432/4/28/13,995 | 33,535 | — | 15,687/8/28/15,244 | True | 41,879 | 42,470 | 14,432/4/28/13,995 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 26,062 | 23,804 | 11,773/16/10/10,041 | — | — | —/—/—/— | False | 26,062 | 23,804 | 11,773/16/10/10,041 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 183,657 | 246,393 | 37,618/16/90/129,259 | — | — | —/—/—/— | False | 183,657 | 246,393 | 37,618/16/90/129,259 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 30,226 | 30,105 | 30,727/104/90/42,703 | — | — | —/—/—/— | False | 30,226 | 30,105 | 30,727/104/90/42,703 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 687 | 803 | 18,662/248/20/27,460 | — | — | —/—/—/— | False | 687 | 803 | 18,662/248/20/27,460 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 18,591 | 18,720 | 17,325/6/92/17,353 | 18,587 | — | 15,186/6/94/18,121 | True | 18,591 | 18,720 | 17,325/6/92/17,353 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 1,035 | 1,115 | 11,518/8/64/15,452 | 3,230 | — | 13,844/16/96/16,707 | True | 1,035 | 1,115 | 11,518/8/64/15,452 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 969,933 | 972,443 | 17,976/19/172/17,874 | — | — | —/—/—/— | False | 969,933 | 972,443 | 17,976/19/172/17,874 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 1,363,673 | 1,380,301 | 14,464/11/158/14,799 | 407,756 | — | 17,294/38/638/18,585 | True | 1,363,673 | 1,380,301 | 14,464/11/158/14,799 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 979,274 | 991,325 | 10,061/8/78/10,787 | — | — | —/—/—/— | False | 979,274 | 991,325 | 10,061/8/78/10,787 |

## `batch_parallel_chathls_fd_glm_rag2_ns_20260718_131155_rag2_ns`

- **System**: c2hls-chathls-benches
- **Model**: GLM-4.7 (`GLM-4.7-FP8`)
- **Architecture**: RAG2+noskills (flavor=`rag2_ns`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_rag2_ns_20260718_131155_rag2_ns`
- **Created**: 2026-07-18T13:11:55.849848+00:00; **Completed**: 2026-07-18T17:16:50.548049+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 297,273 | — | 23,115/30/94/53,045 | — | — | —/—/—/— | False | 297,273 | — | 23,115/30/94/53,045 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 2,436 | — | 8,930/16/20/7,718 | 3,665 | — | 7,256/10/24/7,297 | True | 2,436 | — | 8,930/16/20/7,718 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 232,877 | — | 8,564/4/18/6,626 | — | — | —/—/—/— | False | 232,877 | — | 8,564/4/18/6,626 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 36,778 | — | 22,854/4/26/25,411 | — | — | —/—/—/— | False | 36,778 | — | 22,854/4/26/25,411 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 5,610 | — | 6,511/20/20/5,294 | — | — | —/—/—/— | False | 5,610 | — | 6,511/20/20/5,294 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 49,385 | — | 18,671/88/94/26,581 | — | — | —/—/—/— | False | 49,385 | — | 18,671/88/94/26,581 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 1,590,305 | — | 18,785/8/90/23,925 | 1,592,998 | — | 18,130/8/186/21,717 | True | 1,590,305 | — | 18,785/8/90/23,925 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 1,435 | — | 7,498/8/20/6,676 | — | — | —/—/—/— | False | 1,435 | — | 7,498/8/20/6,676 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 3,640 | — | 15,458/96/124/23,046 | 6,677 | — | 13,853/24/110/18,689 | True | 3,640 | — | 15,458/96/124/23,046 |
| mobilenet | — | — | 159,052/712/269/77,062 | 24,626,516 | — | 205,663/490/277/238,941 | — | — | —/—/—/— | False | 24,626,516 | — | 205,663/490/277/238,941 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 5,166 | — | 9,932/4/56/11,722 | — | — | —/—/—/— | False | 5,166 | — | 9,932/4/56/11,722 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 407,262 | — | 18,094/41/76/21,589 | — | — | —/—/—/— | False | 407,262 | — | 18,094/41/76/21,589 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 417,437 | — | 15,144/38/62/16,382 | — | — | —/—/—/— | False | 417,437 | — | 15,144/38/62/16,382 |

## `batch_parallel_chathls_fd_glm_rag_20260718_171759_rag_skills`

- **System**: c2hls-chathls-benches
- **Model**: GLM-4.7 (`GLM-4.7-FP8`)
- **Architecture**: scrape/RAG1+skills (flavor=`rag_skills`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_rag_20260718_171759_rag_skills`
- **Created**: 2026-07-18T17:17:59.997393+00:00; **Completed**: 2026-07-18T20:51:30.671513+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 192,156 | — | 21,498/48/100/25,223 | — | — | —/—/—/— | False | 192,156 | — | 21,498/48/100/25,223 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 66,069 | — | 44,834/117/94/81,223 | — | — | —/—/—/— | False | 66,069 | — | 44,834/117/94/81,223 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 3,351 | — | 10,654/32/16/17,044 | 7,271 | — | 9,166/21/24/7,393 | True | 3,351 | — | 10,654/32/16/17,044 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 1,840 | — | 18,114/304/54/34,688 | — | — | —/—/—/— | False | 1,840 | — | 18,114/304/54/34,688 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 19,122 | — | 8,846/8/28/9,290 | — | — | —/—/—/— | False | 19,122 | — | 8,846/8/28/9,290 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 26,005 | — | 6,123/11/14/4,713 | 17,707 | — | 5,854/11/22/6,377 | True | 26,005 | — | 6,123/11/14/4,713 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 172,201 | — | 16,257/16/138/20,287 | 33,905 | — | 19,914/6/48/32,216 | True | 172,201 | — | 16,257/16/138/20,287 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 21,209 | — | 64,060/704/218/103,384 | — | — | —/—/—/— | False | 21,209 | — | 64,060/704/218/103,384 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 1,540 | — | 9,002/24/24/7,968 | 1,538 | — | 8,623/24/28/9,433 | True | 1,540 | — | 9,002/24/24/7,968 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 3,231 | — | 19,160/96/92/24,996 | 3,227 | — | 16,974/96/94/25,759 | True | 3,231 | — | 19,160/96/92/24,996 |
| mobilenet | — | — | 159,052/712/269/77,062 | 12,651,459 | — | 52,857/163/324/42,977 | — | — | —/—/—/— | False | 12,651,459 | — | 52,857/163/324/42,977 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 5,007 | — | 12,950/4/84/15,337 | — | — | —/—/—/— | False | 5,007 | — | 12,950/4/84/15,337 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 63,753 | — | 37,296/205/348/85,350 | — | — | —/—/—/— | False | 63,753 | — | 37,296/205/348/85,350 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 979,993 | — | 12,686/11/158/14,281 | — | — | —/—/—/— | False | 979,993 | — | 12,686/11/158/14,281 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 1,365,680 | — | 11,723/8/46/10,862 | — | — | —/—/—/— | False | 1,365,680 | — | 11,723/8/46/10,862 |

## `batch_parallel_chathls_fd_glm_skills_20260718_205203_skills`

- **System**: c2hls-chathls-benches
- **Model**: GLM-4.7 (`GLM-4.7-FP8`)
- **Architecture**: skills (flavor=`skills`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_glm_skills_20260718_205203_skills`
- **Created**: 2026-07-18T20:52:03.598577+00:00; **Completed**: 2026-07-18T23:49:50.470300+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 2,123,355 | 2,113,092 | 18,444/16/148/19,514 | — | — | —/—/—/— | — | 2,123,355 | 2,113,092 | 18,444/16/148/19,514 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 1,131,056 | 1,106,320 | 20,773/17/182/22,198 | — | — | —/—/—/— | — | 1,131,056 | 1,106,320 | 20,773/17/182/22,198 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 3,994 | 3,935 | 7,534/9/20/6,232 | — | — | —/—/—/— | — | 3,994 | 3,935 | 7,534/9/20/6,232 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 3,394 | 3,508 | 8,428/10/24/7,093 | — | — | —/—/—/— | — | 3,394 | 3,508 | 8,428/10/24/7,093 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 38,065 | 38,740 | 47,893/4/26/73,534 | — | — | —/—/—/— | — | 38,065 | 38,740 | 47,893/4/26/73,534 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 2,517 | 2,673 | 16,129/104/10/23,706 | — | — | —/—/—/— | — | 2,517 | 2,673 | 16,129/104/10/23,706 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 671,688 | 947,592 | 14,324/16/90/21,359 | — | — | —/—/—/— | — | 671,688 | 947,592 | 14,324/16/90/21,359 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 355,607 | 353,691 | 11,714/8/90/15,658 | — | — | —/—/—/— | — | 355,607 | 353,691 | 11,714/8/90/15,658 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 1,435 | 1,492 | 7,743/8/20/6,789 | — | — | —/—/—/— | — | 1,435 | 1,492 | 7,743/8/20/6,789 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 3,576 | 3,645 | 16,684/96/92/23,552 | — | — | —/—/—/— | — | 3,576 | 3,645 | 16,684/96/92/23,552 |
| mobilenet | — | — | 159,052/712/269/77,062 | 22,593,315 | 20,647,995 | 19,093/36/266/12,849 | — | — | —/—/—/— | — | 22,593,315 | 20,647,995 | 19,093/36/266/12,849 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 5,005 | 5,103 | 16,422/4/80/16,525 | — | — | —/—/—/— | — | 5,005 | 5,103 | 16,422/4/80/16,525 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 968,974 | 971,354 | 15,343/19/76/21,139 | — | — | —/—/—/— | — | 968,974 | 971,354 | 15,343/19/76/21,139 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 712,801 | 764,048 | 15,603/35/64/19,342 | — | — | —/—/—/— | — | 712,801 | 764,048 | 15,603/35/64/19,342 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 365,600 | 413,755 | 11,797/11/46/11,356 | — | — | —/—/—/— | — | 365,600 | 413,755 | 11,797/11/46/11,356 |

## `batch_parallel_chathls_fd_rag2_20260717_091157_rag2_skills`

- **System**: c2hls-chathls-benches
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: RAG2+skills (flavor=`rag2_skills`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag2_20260717_091157_rag2_skills`
- **Created**: 2026-07-17T09:11:57.821479+00:00; **Completed**: 2026-07-17T12:27:14.625722+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 54,794 | — | 85,846/772/184/133,578 | — | — | —/—/—/— | False | 54,794 | — | 85,846/772/184/133,578 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 91,045 | — | 103,740/663/374/154,508 | 132,646 | — | 93,755/553/214/142,510 | True | 91,045 | — | 103,740/663/374/154,508 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 5,370 | — | 9,248/10/20/13,782 | 5,333 | — | 23,620/10/24/35,282 | True | 5,370 | — | 9,248/10/20/13,782 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 1,892 | — | 20,727/304/20/34,887 | 1,932 | — | 20,605/304/20/36,380 | True | 1,892 | — | 20,727/304/20/34,887 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 34,131 | — | 12,110/4/22/16,474 | 19,736 | — | 15,793/4/36/16,931 | True | 34,131 | — | 12,110/4/22/16,474 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 2,764 | — | 19,069/336/11/29,380 | — | — | —/—/—/— | False | 2,764 | — | 19,069/336/11/29,380 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 603,722 | — | 19,542/22/90/39,136 | — | — | —/—/—/— | False | 603,722 | — | 19,542/22/90/39,136 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 48,090 | — | 68,858/704/90/105,075 | 52,186 | — | 62,507/704/250/101,684 | True | 48,090 | — | 68,858/704/90/105,075 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 682 | — | 18,351/248/20/29,337 | 682 | — | 21,993/248/20/41,608 | True | 682 | — | 18,351/248/20/29,337 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 2,201 | — | 18,945/96/90/25,019 | 3,227 | — | 16,968/96/94/25,750 | True | 2,201 | — | 18,945/96/90/25,019 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 5,007 | — | 13,309/4/84/18,133 | 5,008 | — | 14,060/4/84/18,587 | True | 5,007 | — | 13,309/4/84/18,133 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 24,552,001 | — | 29,209/19/12/57,775 | — | — | —/—/—/— | False | 24,552,001 | — | 29,209/19/12/57,775 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 407,673 | — | 12,935/38/158/15,760 | — | — | —/—/—/— | False | 407,673 | — | 12,935/38/158/15,760 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 407,033 | — | 9,751/19/110/11,359 | — | — | —/—/—/— | False | 407,033 | — | 9,751/19/110/11,359 |

## `batch_parallel_chathls_fd_rag2_ns_20260717_091157_rag2_ns`

- **System**: c2hls-chathls-benches
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: RAG2+noskills (flavor=`rag2_ns`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag2_ns_20260717_091157_rag2_ns`
- **Created**: 2026-07-17T09:11:57.828175+00:00; **Completed**: 2026-07-17T11:38:37.375824+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 114,860 | — | 84,644/771/184/133,244 | 54,292 | — | 111,544/1,329/380/183,072 | True | 114,860 | — | 84,644/771/184/133,244 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 2,100,309 | — | 17,559/20/70/31,317 | 2,718,093 | — | 20,995/20/198/20,845 | True | 2,100,309 | — | 17,559/20/70/31,317 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 317,725 | — | 15,635/20/16/12,531 | 3,164 | — | 15,211/16/16/13,577 | True | 317,725 | — | 15,635/20/16/12,531 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 234,607 | — | 8,889/4/20/6,152 | 3,321 | — | 8,192/9/20/8,768 | True | 234,607 | — | 8,889/4/20/6,152 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 6,234 | — | 23,859/128/26/28,437 | — | — | —/—/—/— | False | 6,234 | — | 23,859/128/26/28,437 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 16,417 | — | 5,647/20/13/4,674 | 16,995 | — | 6,038/20/17/5,513 | True | 16,417 | — | 5,647/20/13/4,674 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 1,224,705 | — | 12,888/16/90/17,818 | — | — | —/—/—/— | False | 1,224,705 | — | 12,888/16/90/17,818 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 1,950,220 | — | 9,599/8/64/11,667 | — | — | —/—/—/— | False | 1,950,220 | — | 9,599/8/64/11,667 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 4,411 | — | 8,686/16/18/7,275 | — | — | —/—/—/— | False | 4,411 | — | 8,686/16/18/7,275 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 34,974 | — | 12,483/3/96/14,912 | 34,970 | — | 11,800/3/102/15,615 | True | 34,974 | — | 12,483/3/96/14,912 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 8,868 | — | 18,160/12/53/21,948 | 907 | — | 123,293/160/80/200,276 | True | 8,868 | — | 18,160/12/53/21,948 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 24,552,001 | — | 29,241/19/12/57,787 | — | — | —/—/—/— | False | 24,552,001 | — | 29,241/19/12/57,787 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 56,952 | — | 67,133/404/222/94,348 | — | — | —/—/—/— | False | 56,952 | — | 67,133/404/222/94,348 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 407,033 | — | 9,751/19/110/11,359 | — | — | —/—/—/— | False | 407,033 | — | 9,751/19/110/11,359 |

## `batch_parallel_chathls_fd_rag_ns_20260716_chathls_rag_ns`

- **System**: c2hls-chathls-benches
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: scrape/RAG1+noskills (flavor=``)
- **Complete**: False; flash_selected=False; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag_ns_20260716_chathls_rag_ns`
- **Created**: 2026-07-16T05:30:08.116177+00:00; **Completed**: 

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| atax | 236,250 | — | 25,208/8/4/26,830 | 1,854 | — | 19,788/336/16/35,587 | — | — | —/—/—/— | — | 1,854 | — | 19,788/336/16/35,587 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 1,957 | — | 20,767/304/20/36,246 | — | — | —/—/—/— | — | 1,957 | — | 20,767/304/20/36,246 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 5,269 | — | 51,547/64/18/81,787 | — | — | —/—/—/— | — | 5,269 | — | 51,547/64/18/81,787 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 2,026 | — | 28,760/183/8/55,096 | — | — | —/—/—/— | — | 2,026 | — | 28,760/183/8/55,096 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 4,895,873 | — | 26,006/536/64/52,243 | — | — | —/—/—/— | — | 4,895,873 | — | 26,006/536/64/52,243 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 262,887 | — | 49,788/11/90/171,623 | — | — | —/—/—/— | — | 262,887 | — | 49,788/11/90/171,623 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 175,441 | — | 14,636/12/12/13,844 | — | — | —/—/—/— | — | 175,441 | — | 14,636/12/12/13,844 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 2,201 | — | 18,945/96/90/25,019 | — | — | —/—/—/— | — | 2,201 | — | 18,945/96/90/25,019 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 120,058 | — | 13,496/4/52/12,480 | — | — | —/—/—/— | — | 120,058 | — | 13,496/4/52/12,480 |

## `batch_parallel_chathls_fd_rag_ns_20260716_chathls_rag_ns2`

- **System**: c2hls-chathls-benches
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: scrape/RAG1+noskills (flavor=``)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_rag_ns_20260716_chathls_rag_ns2`
- **Created**: 2026-07-16T21:55:37.540962+00:00; **Completed**: 2026-07-17T00:27:44.261416+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 2,970,002 | — | 328,403/8/1,280/337,466 | 1,506,901 | — | 374,166/22/2,668/450,942 | True | 2,970,002 | — | 328,403/8/1,280/337,466 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 2,100,309 | — | 17,499/20/70/31,291 | — | — | —/—/—/— | False | 2,100,309 | — | 17,499/20/70/31,291 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 5,552 | — | 7,022/5/20/5,717 | 5,591 | — | 6,533/5/16/6,607 | True | 5,552 | — | 7,022/5/20/5,717 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 236,162 | — | 10,277/9/24/7,490 | 3,321 | — | 8,088/9/20/8,613 | True | 236,162 | — | 10,277/9/24/7,490 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 3,491 | — | 64,246/64/24/85,368 | 3,445 | — | 309,246/64/24/384,159 | True | 3,491 | — | 64,246/64/24/85,368 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 16,407 | — | 9,107/25/13/6,751 | — | — | —/—/—/— | False | 16,407 | — | 9,107/25/13/6,751 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 848,833 | — | 17,111/32/90/32,723 | — | — | —/—/—/— | False | 848,833 | — | 17,111/32/90/32,723 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 656,012 | — | 9,529/11/64/11,797 | — | — | —/—/—/— | False | 656,012 | — | 9,529/11/64/11,797 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 9,835 | — | 8,388/8/16/7,020 | 4,775 | — | 7,861/16/28/8,264 | True | 9,835 | — | 8,388/8/16/7,020 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 34,974 | — | 12,477/3/96/14,903 | 34,970 | — | 11,794/3/102/15,606 | True | 34,974 | — | 12,477/3/96/14,903 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 5,166 | — | 13,574/4/52/15,664 | 5,630 | — | 28,121/4/80/44,140 | True | 5,166 | — | 13,574/4/52/15,664 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 66,528,001 | — | 10,503/8/12/8,182 | — | — | —/—/—/— | False | 66,528,001 | — | 10,503/8/12/8,182 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 56,953 | — | 156,429/404/222/713,167 | — | — | —/—/—/— | False | 56,953 | — | 156,429/404/222/713,167 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 407,033 | — | 9,751/19/110/11,359 | — | — | —/—/—/— | False | 407,033 | — | 9,751/19/110/11,359 |

## `batch_parallel_chathls_fd_ds_rag2_20260719_095037_ds_ctrl`

- **System**: c2hls-control
- **Model**: DeepSeek (`deepseek-chat`)
- **Architecture**: RAG2+skills (flavor=`rag2_skills`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_20260719_095037_ds_ctrl`
- **Created**: 2026-07-19T09:50:39.915485+00:00; **Completed**: 2026-07-20T09:40:57.653385+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 53,034 | — | 127,984/771/132/160,401 | — | — | —/—/—/— | False | 53,034 | — | 127,984/771/132/160,401 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 87,544 | — | 172,129/881/150/203,717 | — | — | —/—/—/— | False | 87,544 | — | 172,129/881/150/203,717 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 5,552 | — | 7,164/5/20/5,726 | 5,599 | — | 7,643/5/24/5,654 | True | 5,552 | — | 7,164/5/20/5,726 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 3,393 | — | 9,492/9/24/11,155 | — | — | —/—/—/— | False | 3,393 | — | 9,492/9/24/11,155 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 6,345 | — | 34,383/128/26/31,186 | 5,334 | — | 33,454/128/28/36,775 | True | 6,345 | — | 34,383/128/26/31,186 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 16,414 | — | 5,392/13/13/4,108 | 16,342 | — | 5,782/16/20/4,946 | True | 16,414 | — | 5,392/13/13/4,108 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 1,319,077 | — | 12,004/8/90/15,351 | 1,847,460 | — | 12,675/8/170/16,840 | True | 1,319,077 | — | 12,004/8/90/15,351 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 46,296 | — | 117,733/704/122/133,911 | 46,293 | — | 104,540/704/154/135,078 | True | 46,296 | — | 117,733/704/122/133,911 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 1,148 | — | 16,922/248/76/27,892 | 1,149 | — | 17,730/248/20/26,823 | True | 1,148 | — | 16,922/248/76/27,892 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 1,309 | — | 22,934/96/90/58,377 | 6,300 | — | 13,428/24/110/17,850 | True | 1,309 | — | 22,934/96/90/58,377 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 1,876 | — | 52,246/320/80/100,683 | — | — | —/—/—/— | False | 1,876 | — | 52,246/320/80/100,683 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 66,528,001 | — | 10,503/8/12/8,182 | — | — | —/—/—/— | False | 66,528,001 | — | 10,503/8/12/8,182 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 1,363,675 | — | 13,706/11/62/14,750 | — | — | —/—/—/— | False | 1,363,675 | — | 13,706/11/62/14,750 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 1,362,955 | — | 10,569/8/46/11,219 | — | — | —/—/—/— | False | 1,362,955 | — | 10,569/8/46/11,219 |

## `batch_parallel_hlsfactory_ds_rag2_20260718_100518`

- **System**: c2hls-hlsfactory-port
- **Model**: DeepSeek (`deepseek-chat`)
- **Architecture**: RAG2+skills (flavor=`rag2_skills`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_hlsfactory_ds_rag2_20260718_100518`
- **Created**: 2026-07-18T10:05:19.050979+00:00; **Completed**: 2026-07-18T10:12:10.319925+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| covariance | 233,415 | 64,703,715 | 69,911/4/8/40,998 | 599,039 | — | 21,426/11/122/23,527 | — | — | —/—/—/— | False | 599,039 | — | 21,426/11/122/23,527 |
| doitgen | 318,477 | 316,890 | 33,317/331/24/50,081 | 2,014,153 | — | 7,031/11/84/8,967 | — | — | —/—/—/— | False | 2,014,153 | — | 7,031/11/84/8,967 |
| durbin | 123,546 | 174,651 | 7,269/11/10/5,640 | 78,207 | — | 7,303/11/8/5,968 | 43,871 | — | 10,381/14/68/13,769 | True | 78,207 | — | 7,303/11/8/5,968 |
| floyd-warshall | 833,976,005 | 921,456,043 | 2,496/2/2/1,512 | 6,156,515 | — | 3,868/2/136/2,960 | — | — | —/—/—/— | False | 6,156,515 | — | 3,868/2/136/2,960 |
| gemm | 2,871,462 | 57,562,592 | 30,788/12/4/21,949 | 96,373 | — | 13,129/52/46/14,431 | 96,320 | — | 12,451/52/46/16,715 | True | 96,373 | — | 13,129/52/46/14,431 |
| gemver | 2,547,303 | 2,763,356 | 47,979/11/244/60,862 | 158,742 | — | 34,618/22/342/48,918 | — | — | —/—/—/— | False | 158,742 | — | 34,618/22/342/48,918 |
| lu | 134,372,041 | 96,883,182 | 6,310/11/4/4,377 | 3,925,833 | — | 8,094/11/30/8,278 | — | — | —/—/—/— | False | 3,925,833 | — | 8,094/11/30/8,278 |
| ludcmp | 10,039,394 | 9,255,625 | 13,114/11/42/13,671 | — | — | 30,335/11/184/74,072 | — | — | —/—/—/— | False | — | — | 30,335/11/184/74,072 |
| syrk | 1,183,841 | 33,658,995 | 4,349/11/4/4,252 | 408,020 | — | 7,432/19/52/6,571 | — | — | —/—/—/— | False | 408,020 | — | 7,432/19/52/6,571 |
| trisolv | 89,232 | 1,240,263 | 8,352/11/38/9,524 | 110,911 | — | 15,081/11/90/45,863 | 146,310 | — | 39,025/11/218/79,551 | True | 110,911 | — | 15,081/11/90/45,863 |
| trmm | 22,593,601 | 24,424,866 | 6,023/11/8/4,596 | 1,872,048 | — | 5,637/11/96/4,606 | — | — | —/—/—/— | False | 1,872,048 | — | 5,637/11/96/4,606 |

## `batch_parallel_chathls_fd_ds_rag2_lat_20260719_095037_ds_lat`

- **System**: c2hls-latency-opt
- **Model**: DeepSeek (`deepseek-chat`)
- **Architecture**: RAG2+skills (flavor=`rag2_skills`)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_lat_20260719_095037_ds_lat`
- **Created**: 2026-07-20T09:42:55.863189+00:00; **Completed**: 2026-07-20T11:46:30.953604+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | 53,034 | — | 128,032/771/132/160,428 | — | — | —/—/—/— | False | 53,034 | — | 128,032/771/132/160,428 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | 2,109,291 | — | 19,247/30/118/20,639 | — | — | —/—/—/— | False | 2,109,291 | — | 19,247/30/118/20,639 |
| atax | 236,250 | — | 25,208/8/4/26,830 | 1,868 | — | 20,663/337/54/38,665 | — | — | —/—/—/— | False | 1,868 | — | 20,663/337/54/38,665 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | 3,893 | — | 11,274/9/24/14,917 | 3,890 | — | 18,370/9/28/36,344 | True | 3,893 | — | 11,274/9/24/14,917 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | 43,620 | — | 14,288/4/32/14,479 | 42,805 | — | 18,323/4/36/21,171 | True | 43,620 | — | 14,288/4/32/14,479 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | 2,774 | — | 29,616/334/10/34,424 | — | — | —/—/—/— | False | 2,774 | — | 29,616/334/10/34,424 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | 172,201 | — | 15,877/16/138/19,261 | 172,129 | — | 14,927/16/186/19,990 | True | 172,201 | — | 15,877/16/138/19,261 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 46,296 | — | 117,733/704/122/133,911 | 544,920 | — | 14,656/11/218/17,893 | True | 46,296 | — | 117,733/704/122/133,911 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | 1,148 | — | 16,898/248/76/27,883 | 1,356 | — | 12,996/32/28/14,727 | True | 1,148 | — | 16,898/248/76/27,883 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | 3,767 | — | 20,723/96/94/25,074 | 3,765 | — | 18,517/96/98/25,783 | True | 3,767 | — | 20,723/96/94/25,074 |
| mobilenet | — | — | 159,052/712/269/77,062 | 1,555,848 | — | 60,320/318/37,776**/27,071 | — | — | —/—/—/— | False | 1,555,848 | — | 60,320/318/37,776**/27,071 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | 5,007 | — | 12,999/4/84/15,427 | — | — | —/—/—/— | False | 5,007 | — | 12,999/4/84/15,427 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | 60,272 | — | 145,986/450/76/488,176 | — | — | —/—/—/— | False | 60,272 | — | 145,986/450/76/488,176 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | 979,993 | — | 12,732/11/158/14,360 | — | — | —/—/—/— | False | 979,993 | — | 12,732/11/158/14,360 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | 979,354 | — | 9,889/8/110/10,772 | — | — | —/—/—/— | False | 979,354 | — | 9,889/8/110/10,772 |
| transformer | 73,953 | — | 155,807/302/67/183,025 | 26,587 | — | 155,160/282/283/208,059 | 26,584 | — | 152,640/282/292/210,235 | True | 26,587 | — | 155,160/282/283/208,059 |

## `batch_parallel_machsuite_ds_rag2_20260718_100518`

- **System**: c2hls-machsuite-port
- **Model**: DeepSeek (`deepseek-chat`)
- **Architecture**: RAG2+skills (flavor=`rag2_skills`)
- **Complete**: True; flash_selected=False; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_machsuite_ds_rag2_20260718_100518`
- **Created**: 2026-07-18T10:05:19.073451+00:00; **Completed**: 2026-07-18T22:10:44.505887+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| aes_tableless | 146,890 | — | 105,752/0/16/34,707 | 13,041 | — | 12,224/—/24/7,841 | — | — | —/—/—/— | False | 13,041 | — | 12,224/—/24/7,841 |
| backprop | 279,873,283 | — | 196,108/562/32/168,384 | 3,986,195 | — | 143,724/753/260/212,010 | 3,984,803 | — | 142,868/738/294/209,616 | True | 3,986,195 | — | 143,724/753/260/212,010 |
| bfs_bulk | 14,652,591 | — | 7,661/0/8/4,011 | 115,040 | — | 38,742/—/96/32,325 | 115,905 | — | 50,488/—/118/37,734 | True | 115,040 | — | 38,742/—/96/32,325 |
| bfs_queue | 4,406,344 | — | 8,130/0/10/4,274 | 71,960 | — | 12,478/—/76/13,510 | — | — | —/—/—/— | False | 71,960 | — | 12,478/—/76/13,510 |
| fft_transpose | 26,433 | — | 61,297/318/46/84,068 | 2,092 | — | 199,933/1,906/60/310,922 | — | — | —/—/—/— | False | 2,092 | — | 199,933/1,906/60/310,922 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | 10,571,928 | — | 14,974/8/138/16,778 | 10,571,925 | — | 14,135/8/90/17,429 | True | 10,571,928 | — | 14,974/8/138/16,778 |
| md_knn | 13,748 | — | 126,733/41/30/60,210 | 58,257 | — | 26,850/30/216/36,629 | 25,108 | — | 26,480/30/238/37,980 | True | 58,257 | — | 26,850/30/216/36,629 |
| sort_merge | 48,705,559 | — | 4,270/—/6/2,393 | 34,810,029 | — | 6,049/—/38/6,389 | — | — | —/—/—/— | False | 34,810,029 | — | 6,049/—/38/6,389 |
| spmv_ellpack | 270,219 | — | 14,335/11/8/8,703 | 6,172 | — | 16,267/110/420/22,661 | 98,878 | — | 8,803/8/120/10,846 | True | 6,172 | — | 16,267/110/420/22,661 |
| stencil2D | 140,762 | — | 7,706/3/4/6,501 | 21,338 | — | 12,715/84/66/15,173 | 21,327 | — | 12,700/84/144/16,512 | True | 21,338 | — | 12,715/84/66/15,173 |
| stencil3D | 90,919 | — | 32,316/3/30/15,346 | 47,478 | — | 17,535/10/100/14,978 | 47,465 | — | 17,096/10/136/15,620 | True | 47,478 | — | 17,535/10/100/14,978 |
| viterbi | 636,989 | — | 348,625/21/240/141,196 | 881,909 | — | 87,216/67/222/62,846 | — | — | —/—/—/— | False | 881,909 | — | 87,216/67/222/62,846 |

## `batch_parallel_machsuite_fd_20260710_machsuite_flash_dataflow`

- **System**: c2hls-machsuite-port
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: base/unknown (flavor=``)
- **Complete**: True; flash_selected=True; dataflow_selected=True
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_machsuite_fd_20260710_machsuite_flash_dataflow`
- **Created**: 2026-07-10T21:23:56.671099+00:00; **Completed**: 2026-07-13T22:32:33.419647+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| aes_tableless | 146,890 | 171,630 | 105,752/0/16/34,707 | 23,048 | — | 82,900/—/40/40,720 | — | — | —/—/—/— | False | 23,048 | — | 82,900/—/40/40,720 |
| bfs_bulk | 14,652,591 | 5,401,087 | 7,661/0/8/4,011 | 214,782 | — | 12,851/—/100/15,792 | 213,164 | — | 12,348/—/107/16,730 | True | 214,782 | — | 12,851/—/100/15,792 |
| bfs_queue | 4,406,344 | 8,052,263 | 8,130/0/10/4,274 | 66,072 | — | 13,318/—/104/15,690 | 66,128 | — | 12,841/—/107/16,758 | True | 66,072 | — | 13,318/—/104/15,690 |
| fft_transpose | 26,433 | 19,819 | 61,297/318/46/84,068 | 21,918 | — | 66,642/342/46/82,515 | — | — | —/—/—/— | False | 21,918 | — | 66,642/342/46/82,515 |
| gemm_ncubed | 295,512 | 1,309,104 | 152,198/11/30/60,410 | 35,672 | — | 72,060/704/90/88,764 | 48,087 | — | 62,256/704/250/101,640 | True | 35,672 | — | 72,060/704/90/88,764 |
| md_knn | 13,748 | 59,495 | 126,733/41/30/60,210 | 4,675 | — | 149,247/1,840/336/195,873 | 7,059 | — | 43,520/208/232/83,385 | True | 4,675 | — | 149,247/1,840/336/195,873 |
| sort_radix | 19,274,417 | 20,745,785 | 258,439/0/30/85,276 | 18,557,825 | — | 268,129/—/120/95,291 | — | — | —/—/—/— | False | 18,557,825 | — | 268,129/—/120/95,291 |
| spmv_ellpack | 270,219 | 281,165 | 14,335/11/8/8,703 | 12,295 | — | 17,263/110/396/17,986 | 8,073 | — | 9,519/22/128/14,105 | True | 12,295 | — | 17,263/110/396/17,986 |
| stencil2D | 140,762 | 624,186 | 7,706/3/4/6,501 | 37,078 | — | 8,734/27/36/9,874 | — | — | —/—/—/— | False | 37,078 | — | 8,734/27/36/9,874 |
| stencil3D | 90,919 | 401,310 | 32,316/3/30/15,346 | 47,472 | — | 16,729/10/150/14,691 | 47,464 | — | 15,994/10/236/15,403 | True | 47,472 | — | 16,729/10/150/14,691 |

## `batch_parallel_chathls_fd_ds_rag2_xfmr_20260720_070124`

- **System**: c2hls-transformer
- **Model**: DeepSeek (`deepseek-chat`)
- **Architecture**: RAG2+skills (flavor=``)
- **Complete**: True; flash_selected=False; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_chathls_fd_ds_rag2_xfmr_20260720_070124`
- **Created**: 2026-07-20T07:01:24.377837+00:00; **Completed**: 2026-07-20T11:20:52.043288+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| transformer | 73,953 | — | 155,807/302/67/183,025 | 9,702 | — | 1,050,105/12,544**/281/1,421,457 | — | — | —/—/—/— | — | 9,702 | — | 1,050,105/12,544**/281/1,421,457 |

## `hybrid-u280-c2hlsport-20260719-090238`

- **System**: chathls-c2hlsport
- **Model**: DeepSeek (`deepseek`)
- **Architecture**: ChatHLS on c2hls-port benches (flavor=`c2hlsport-u280`)
- **Complete**: True; flash_selected=False; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-c2hlsport-20260719-090238`

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | — | — | —/—/—/— | — | — | —/—/—/— | — | 3,601 | — | 0/0/0/0 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,820 | — | 0/0/0/0 |
| aes_table | 134,242 | — | 351,304/0/16/124,371 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| aes_tableless | 146,890 | — | 105,752/0/16/34,707 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| atax | 236,250 | — | 25,208/8/4/26,830 | — | — | —/—/—/— | — | — | —/—/—/— | — | 8,372 | — | 0/0/0/0 |
| backprop | 279,873,283 | — | 196,108/562/32/168,384 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| bfs_queue | 4,406,344 | — | 8,130/0/10/4,274 | — | — | —/—/—/— | — | — | —/—/—/— | — | 478 | — | 0/0/0/0 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | — | — | —/—/—/— | — | — | —/—/—/— | — | 8,075 | — | 0/0/0/0 |
| cholesky | — | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | 4,201 | — | 0/0/0/0 |
| correlation | — | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | 123 | — | 0/0/0/0 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | — | — | —/—/—/— | — | — | —/—/—/— | — | 4,787 | — | 0/0/0/0 |
| doitgen | 318,477 | — | 33,317/331/24/50,081 | — | — | —/—/—/— | — | — | —/—/—/— | — | 486 | — | 0/0/0/0 |
| durbin | 123,546 | — | 7,269/11/10/5,640 | — | — | —/—/—/— | — | — | —/—/—/— | — | 20 | — | 0/0/0/0 |
| fdtd-2d | — | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | 82 | — | 0/0/0/0 |
| fft_transpose | 26,433 | — | 61,297/318/46/84,068 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| floyd-warshall | 833,976,005 | — | 2,496/2/2/1,512 | — | — | —/—/—/— | — | — | —/—/—/— | — | 11,664,005 | — | 0/0/0/0 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | — | — | —/—/—/— | — | — | —/—/—/— | — | 5,629 | — | 0/0/0/0 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | — | — | —/—/—/— | — | — | —/—/—/— | — | 2,097,166 | — | 0/0/0/0 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | — | — | —/—/—/— | — | — | —/—/—/— | — | 2,578 | — | 0/0/0/0 |
| gemver | 2,547,303 | — | 47,979/11/244/60,862 | — | — | —/—/—/— | — | — | —/—/—/— | — | 8,124 | — | 0/0/0/0 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | — | — | —/—/—/— | — | — | —/—/—/— | — | 4,801 | — | 0/0/0/0 |
| gramschmidt | — | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | 81 | — | 0/0/0/0 |
| heat-3d | — | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,871,281 | — | 0/0/0/0 |
| jacobi-1d | — | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,286 | — | 0/0/0/0 |
| jacobi-2d | — | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | 15,532 | — | 0/0/0/0 |
| lu | 134,372,041 | — | 6,310/11/4/4,377 | — | — | —/—/—/— | — | — | —/—/—/— | — | 2 | — | 0/0/0/0 |
| ludcmp | 10,039,394 | — | 13,114/11/42/13,671 | — | — | —/—/—/— | — | — | —/—/—/— | — | 2 | — | 0/0/0/0 |
| md_grid | — | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| md_knn | 13,748 | — | 126,733/41/30/60,210 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| mvt | 14,327 | — | 56,806/8/16/25,208 | — | — | —/—/—/— | — | — | —/—/—/— | — | 8,116 | — | 0/0/0/0 |
| nussinov | — | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | 541 | — | 0/0/0/0 |
| seidel-2d | — | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | 45,670,739 | — | 0/0/0/0 |
| sort_radix | 19,274,417 | — | 258,439/0/30/85,276 | — | — | —/—/—/— | — | — | —/—/—/— | — | 66 | — | 0/0/0/0 |
| spmv_ellpack | 270,219 | — | 14,335/11/8/8,703 | — | — | —/—/—/— | — | — | —/—/—/— | — | 2,559 | — | 0/0/0/0 |
| stencil2D | 140,762 | — | 7,706/3/4/6,501 | — | — | —/—/—/— | — | — | —/—/—/— | — | 12,108 | — | 0/0/0/0 |
| stencil3D | 90,919 | — | 32,316/3/30/15,346 | — | — | —/—/—/— | — | — | —/—/—/— | — | 20,371 | — | 0/0/0/0 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | — | — | —/—/—/— | — | — | —/—/—/— | — | 2 | — | 0/0/0/0 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | — | — | —/—/—/— | — | — | —/—/—/— | — | 37 | — | 0/0/0/0 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | — | — | —/—/—/— | — | — | —/—/—/— | — | 149,921 | — | 0/0/0/0 |
| trisolv | 89,232 | — | 8,352/11/38/9,524 | — | — | —/—/—/— | — | — | —/—/—/— | — | 4,441 | — | 0/0/0/0 |
| trmm | 22,593,601 | — | 6,023/11/8/4,596 | — | — | —/—/—/— | — | — | —/—/—/— | — | 62,401 | — | 0/0/0/0 |
| viterbi | 636,989 | — | 348,625/21/240/141,196 | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,686 | — | 0/0/0/0 |

## `hybrid-u280-split-20260717-001649`

- **System**: chathls-native
- **Model**: DeepSeek (`deepseek`)
- **Architecture**: ChatHLS hybrid agent (flavor=`hybrid-u280`)
- **Complete**: True; flash_selected=False; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-split-20260717-001649`

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| 2mm | 25,296,077 | — | 38,403/20/8/26,447 | — | — | —/—/—/— | — | — | —/—/—/— | — | 7,582 | 38,695 | 377,621/1,116/0/2,065,793 |
| 3mm | 45,441,119 | — | 361,516/14/30/125,211 | — | — | —/—/—/— | — | — | —/—/—/— | — | 7,018 | — | 779,385/2,860/16/3,492,774** |
| atax | 236,250 | — | 25,208/8/4/26,830 | — | — | —/—/—/— | — | — | —/—/—/— | — | 915 | 1,265 | 8,090/16/0/11,872 |
| bicg | 234,516 | — | 23,798/12/4/24,382 | — | — | —/—/—/— | — | — | —/—/—/— | — | 843 | 4,113 | 7,104/16/0/5,996 |
| covariance | 233,415 | — | 69,911/4/8/40,998 | — | — | —/—/—/— | — | — | —/—/—/— | — | 4,926 | 74,630 | 20,406/8/0/21,447 |
| gemm | 2,871,462 | — | 30,788/12/4/21,949 | — | — | —/—/—/— | — | — | —/—/—/— | — | 902 | — | 75,738/271/0/114,324 |
| gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | — | — | —/—/—/— | — | — | —/—/—/— | — | 2,097,166 | 1,187,910 | 1,580/11/0/1,747 |
| gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | — | — | —/—/—/— | — | — | —/—/—/— | — | 4,627 | — | 294,192/1,408/0/1,250,964 |
| gesummv | 1,499 | — | 10,272/8/4/11,651 | — | — | —/—/—/— | — | — | —/—/—/— | — | 482 | 4,339 | 5,153/20/0/4,827 |
| matmul | 34,963 | — | 78,416/3/30/28,613 | — | — | —/—/—/— | — | — | —/—/—/— | — | 660 | — | 762,811/19,974**/0/1,440,671 |
| mobilenet | 149 | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | 152,924/712/239/71,688 |
| mvt | 14,327 | — | 56,806/8/16/25,208 | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,234 | 2,979 | 8,778/16/0/9,585 |
| symm | 37,430,401 | — | 24,552/19/4/58,192 | — | — | —/—/—/— | — | — | —/—/—/— | — | 311,596 | — | 37,543/30/0/144,780 |
| syr2k | 1,952,641 | — | 4,588/11/4/4,919 | — | — | —/—/—/— | — | — | —/—/—/— | — | 394,721 | 5,232,710 | 2,566/38/0/4,370 |
| syrk | 1,183,841 | — | 4,349/11/4/4,252 | — | — | —/—/—/— | — | — | —/—/—/— | — | 164,241 | — | 20,211/11/0/29,842 |
| transformer | 73,953 | — | 155,807/302/67/183,025 | — | — | —/—/—/— | — | — | —/—/—/— | — | 80,676 | — | 96,746/238/16/104,772 |

## `batch_parallel_autosa_20260708_autosa_nav_n_r3`

- **System**: c2hls-autosa-flash
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: AutoSA flash (nav_n) (flavor=`autosa_nav_n`)
- **Clock**: 4.0 ns
- **Complete**: True; flash_selected=False; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_20260708_autosa_nav_n_r3`
- **Created**: 2026-07-08T05:53:06.882551+00:00; **Completed**: 2026-07-08T06:01:53.715187+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| autosa_dnn_ops | 347 | — | 11,018/80/30/15,532 | 1,202 | — | 10,794/3/91/14,528 | — | — | —/—/—/— | — | 1,202 | — | 10,794/3/91/14,528 |

## `batch_parallel_autosa_20260708_autosa_nav_n_r4`

- **System**: c2hls-autosa-flash
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: AutoSA flash (nav_n) (flavor=`autosa_nav_n`)
- **Clock**: 4.0 ns
- **Complete**: True; flash_selected=False; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_20260708_autosa_nav_n_r4`
- **Created**: 2026-07-08T07:53:18.879626+00:00; **Completed**: 2026-07-08T08:04:52.049587+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| autosa_cnn | 5,066,753 | — | 14,475/5/30/7,514 | 39,562 | — | 62,309/84/346/85,687 | — | — | —/—/—/— | — | 39,562 | — | 62,309/84/346/85,687 |
| autosa_dnn_ops | 347 | — | 11,018/80/30/15,532 | 568 | — | 15,663/80/90/24,053 | — | — | —/—/—/— | — | 568 | — | 15,663/80/90/24,053 |
| autosa_large_mm | 4,007,657,544 | — | 8,118/3/30/8,480 | 372,705 | — | 39,763/384/4,034**/57,889 | — | — | —/—/—/— | — | 372,705 | — | 39,763/384/4,034**/57,889 |
| autosa_lu | 266,273 | — | 2,997/5/10/2,900 | 190,776 | — | 13,597/3/104/15,107 | — | — | —/—/—/— | — | 190,776 | — | 13,597/3/104/15,107 |

## `batch_parallel_autosa_20260708_autosa_nav_n_r5`

- **System**: c2hls-autosa-flash
- **Model**: Devstral (`devstral2`)
- **Architecture**: AutoSA flash (nav_n) (flavor=`autosa_nav_n`)
- **Clock**: 4.0 ns
- **Complete**: False; flash_selected=False; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_20260708_autosa_nav_n_r5`
- **Created**: 2026-07-08T08:46:58.182510+00:00; **Completed**: 

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| autosa_large_cnn | 73,987,522,712 | — | 3,034/7/2/2,782 | 65,702,461,442 | — | 61,469/52/62/324,809 | — | — | —/—/—/— | — | 65,702,461,442 | — | 61,469/52/62/324,809 |
| autosa_large_mm_block_sparse | 167,503,724,616 | — | 8,396/5/30/8,950 | 115,171,585 | — | 101,306/322/1,114/201,202 | — | — | —/—/—/— | — | 115,171,585 | — | 101,306/322/1,114/201,202 |
| autosa_large_mm_int16 | 157,840,048,200 | — | 8,074/1/30/8,352 | 72,638,753 | — | 54,558/1,024/250/60,374 | — | — | —/—/—/— | — | 72,638,753 | — | 54,558/1,024/250/60,374 |
| autosa_large_mm_int8 | 162,772,549,704 | — | 8,077/1/30/8,278 | 47,064,139 | — | 57,844/512/332/32,095 | — | — | —/—/—/— | — | 47,064,139 | — | 57,844/512/332/32,095 |
| autosa_large_mm_intel | 170,120,970,312 | — | 8,395/5/30/8,950 | 909,572,995 | — | 14,393/8/474/16,530 | — | — | —/—/—/— | — | 909,572,995 | — | 14,393/8/474/16,530 |
| autosa_large_mttkrp | 907,580,276,809 | — | 11,132/7/30/9,766 | 39,460,012,184 | — | 10,020/7/64/22,996 | — | — | —/—/—/— | — | 39,460,012,184 | — | 10,020/7/64/22,996 |
| autosa_large_ttm | 34,931,736,577 | — | 10,460/5/30/18,817 | 198,910,772 | — | 100,073/1,280/8,538**/169,414 | — | — | —/—/—/— | — | 198,910,772 | — | 100,073/1,280/8,538**/169,414 |
| autosa_large_ttmc | — | — | 11,261/5/30/9,678 | 206,640,775,242 | — | 18,190/5/92/28,581 | — | — | —/—/—/— | — | 206,640,775,242 | — | 18,190/5/92/28,581 |
| autosa_mm_block_sparse | 33,357 | — | 12,301/40/30/21,212 | 47,908 | — | 30,701/320/30/48,485 | — | — | —/—/—/— | — | 47,908 | — | 30,701/320/30/48,485 |
| autosa_mm_catapult | 32,916 | — | 9,536/24/30/11,621 | 9,425 | — | 18,152/192/218/30,901 | — | — | —/—/—/— | — | 9,425 | — | 18,152/192/218/30,901 |
| autosa_mm_getting_started | 33,357 | — | 12,301/40/30/21,212 | 43,736 | — | 35,731/320/90/55,454 | — | — | —/—/—/— | — | 43,736 | — | 35,731/320/90/55,454 |
| autosa_mm_hbm | 33,357 | — | 12,301/40/30/21,212 | 43,736 | — | 35,690/320/90/55,346 | — | — | —/—/—/— | — | 43,736 | — | 35,690/320/90/55,346 |
| autosa_mm_hcl | 33,357 | — | 12,301/40/30/21,212 | 42,195 | — | 32,748/320/218/56,517 | — | — | —/—/—/— | — | 42,195 | — | 32,748/320/218/56,517 |
| autosa_mm_hcl_intel | 33,357 | — | 12,301/40/30/21,212 | 43,736 | — | 34,063/320/146/57,210 | — | — | —/—/—/— | — | 43,736 | — | 34,063/320/146/57,210 |
| autosa_mm_int16 | 16,533 | — | 6,940/64/30/7,860 | 15,000 | — | 17,749/64/98/17,337 | — | — | —/—/—/— | — | 15,000 | — | 17,749/64/98/17,337 |
| autosa_mm_intel | 33,357 | — | 12,301/40/30/21,212 | 41,943 | — | 33,553/320/162/57,126 | — | — | —/—/—/— | — | 41,943 | — | 33,553/320/162/57,126 |

## `batch_parallel_autosa_dse_fd_20260712_212400_autosa_dse_fd_nocosim_cont`

- **System**: c2hls-autosa-dse-fd
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: AutoSA-DSE flash+dataflow (flavor=`autosa_dse_aav_n`)
- **Clock**: 3.33 ns
- **Complete**: True; flash_selected=True; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_dse_fd_20260712_212400_autosa_dse_fd_nocosim_cont`
- **Created**: 2026-07-12T21:23:47.423084+00:00; **Completed**: 2026-07-13T02:34:04.225279+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| autosa_cnn_rank1 | 18,554 | — | 72,085/160/190/55,760 | 18,554 | — | 72,085/160/190/55,760 | — | — | —/—/—/— | — | 18,554 | — | 72,085/160/190/55,760 |
| autosa_cnn_rank2 | 18,554 | — | 55,353/160/194/52,094 | 18,554 | — | 55,353/160/194/52,094 | — | — | —/—/—/— | — | 18,554 | — | 55,353/160/194/52,094 |
| autosa_cnn_rank3 | 18,554 | — | 55,083/160/194/52,036 | 18,554 | — | 55,083/160/194/52,036 | — | — | —/—/—/— | — | 18,554 | — | 55,083/160/194/52,036 |
| autosa_large_mm_intel_rank1 | 13,631,621 | — | 90,306/400/578/105,617 | 13,631,621 | — | 90,306/400/578/105,617 | — | — | —/—/—/— | — | 13,631,621 | — | 90,306/400/578/105,617 |
| autosa_large_mm_intel_rank2 | 27,263,098 | — | 85,830/200/578/83,582 | 27,263,098 | — | 89,830/200/578/76,322 | — | — | —/—/—/— | — | 27,263,098 | — | 89,830/200/578/76,322 |
| autosa_large_mm_intel_rank3 | 27,263,100 | — | 57,427/200/252/59,985 | 27,263,100 | — | 57,427/200/252/59,985 | — | — | —/—/—/— | — | 27,263,100 | — | 57,427/200/252/59,985 |
| autosa_large_mm_rank1 | 524,403 | — | 86,570/156/354/75,322 | 524,403 | — | 86,570/156/354/75,322 | — | — | —/—/—/— | — | 524,403 | — | 86,570/156/354/75,322 |
| autosa_large_mm_rank2 | 524,386 | — | 58,995/156/275/56,468 | 524,386 | — | 58,995/156/275/56,468 | — | — | —/—/—/— | — | 524,386 | — | 58,995/156/275/56,468 |
| autosa_large_mm_rank3 | 524,386 | — | 54,923/156/267/55,398 | 524,386 | — | 54,923/156/267/55,398 | — | — | —/—/—/— | — | 524,386 | — | 54,923/156/267/55,398 |
| autosa_mm_catapult_rank1 | 8,286 | — | 36,729/96/226/39,097 | 8,286 | — | 36,729/96/226/39,097 | — | — | —/—/—/— | — | 8,286 | — | 36,729/96/226/39,097 |
| autosa_mm_catapult_rank2 | 5,078 | — | 65,613/192/330/68,455 | 5,078 | — | 65,613/192/330/68,455 | — | — | —/—/—/— | — | 5,078 | — | 65,613/192/330/68,455 |
| autosa_mm_catapult_rank3 | 8,285 | — | 37,137/96/226/37,286 | 8,285 | — | 37,137/96/226/37,286 | — | — | —/—/—/— | — | 8,285 | — | 37,137/96/226/37,286 |
| autosa_mm_getting_started_rank1 | 2,194 | — | 143,218/640/540/152,657 | 2,194 | — | 143,224/640/540/152,666 | — | — | —/—/—/— | — | 2,194 | — | 143,224/640/540/152,666 |
| autosa_mm_getting_started_rank2 | 2,254 | — | 139,143/640/540/152,574 | 2,254 | — | 139,143/640/540/152,574 | — | — | —/—/—/— | — | 2,254 | — | 139,143/640/540/152,574 |
| autosa_mm_getting_started_rank3 | 4,228 | — | 76,519/320/330/91,366 | 4,228 | — | 76,519/320/330/91,366 | — | — | —/—/—/— | — | 4,228 | — | 76,519/320/330/91,366 |
| autosa_mm_hcl_intel_rank1 | 4,226 | — | 78,064/320/330/84,618 | 4,226 | — | 78,064/320/330/84,618 | — | — | —/—/—/— | — | 4,226 | — | 78,064/320/330/84,618 |
| autosa_mm_hcl_intel_rank3 | 4,229 | — | 118,312/320/540/113,274 | 4,229 | — | 118,312/320/540/113,274 | — | — | —/—/—/— | — | 4,229 | — | 118,312/320/540/113,274 |
| autosa_mm_hcl_rank1 | 4,230 | — | 117,365/320/516/111,757 | 4,230 | — | 117,365/320/516/111,757 | — | — | —/—/—/— | — | 4,230 | — | 117,365/320/516/111,757 |
| autosa_mm_hcl_rank2 | 4,226 | — | 75,528/320/306/83,574 | 4,226 | — | 75,528/320/306/83,574 | — | — | —/—/—/— | — | 4,226 | — | 75,528/320/306/83,574 |
| autosa_mm_hcl_rank3 | 8,309 | — | 67,619/160/306/58,825 | 8,309 | — | 67,619/160/306/58,825 | — | — | —/—/—/— | — | 8,309 | — | 67,619/160/306/58,825 |
| autosa_mm_int16_rank1 | 4,219 | — | 93,003/64/384/57,231 | 4,219 | — | 93,003/64/384/57,231 | — | — | —/—/—/— | — | 4,219 | — | 93,003/64/384/57,231 |
| autosa_mm_int16_rank2 | 8,287 | — | 35,066/32/202/31,308 | 8,287 | — | 35,066/32/202/31,308 | — | — | —/—/—/— | — | 8,287 | — | 35,066/32/202/31,308 |
| autosa_mm_int16_rank3 | 8,287 | — | 35,033/32/202/29,816 | 8,287 | — | 35,033/32/202/29,816 | — | — | —/—/—/— | — | 8,287 | — | 35,033/32/202/29,816 |
| autosa_mm_intel_rank1 | 4,178 | — | 118,330/640/346/155,487 | 4,178 | — | 118,330/640/346/155,487 | — | — | —/—/—/— | — | 4,178 | — | 118,330/640/346/155,487 |
| autosa_mm_intel_rank2 | 4,208 | — | 99,934/320/346/105,307 | 4,208 | — | 99,934/320/346/105,307 | — | — | —/—/—/— | — | 4,208 | — | 99,934/320/346/105,307 |
| autosa_mm_intel_rank3 | 8,310 | — | 43,104/160/210/52,414 | 8,310 | — | 43,104/160/210/52,414 | — | — | —/—/—/— | — | 8,310 | — | 43,104/160/210/52,414 |
| autosa_mm_rank1 | 4,228 | — | 77,288/320/330/89,933 | 4,228 | — | 77,288/320/330/89,933 | — | — | —/—/—/— | — | 4,228 | — | 77,288/320/330/89,933 |
| autosa_mm_rank2 | 4,227 | — | 76,528/320/314/88,055 | 4,227 | — | 76,528/320/314/88,055 | — | — | —/—/—/— | — | 4,227 | — | 76,528/320/314/88,055 |
| autosa_mm_rank3 | 8,309 | — | 66,114/160/314/63,632 | 8,309 | — | 66,114/160/314/63,632 | — | — | —/—/—/— | — | 8,309 | — | 66,114/160/314/63,632 |

## `batch_parallel_autosa_dse_fd_20260713_091100_autosa_dse_fd_nocosim_plainseed`

- **System**: c2hls-autosa-dse-fd
- **Model**: Devstral (`devstral2`)
- **Architecture**: AutoSA-DSE flash+dataflow (flavor=`autosa_dse_aav_n`)
- **Clock**: 3.33 ns
- **Complete**: True; flash_selected=True; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_dse_fd_20260713_091100_autosa_dse_fd_nocosim_plainseed`
- **Created**: 2026-07-13T09:10:27.472865+00:00; **Completed**: 2026-07-13T16:54:36.121812+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| autosa_mm_getting_started_rank3 | 4,228 | — | 76,519/320/330/91,366 | 5,282 | — | 77,475/320/138/94,332 | — | — | —/—/—/— | — | 5,282 | — | 77,475/320/138/94,332 |

## `batch_parallel_autosa_dse_fd_20260713_224002_autosa_dse_fd_nocosim_plainseed_stream`

- **System**: c2hls-autosa-dse-fd
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: AutoSA-DSE flash+dataflow (flavor=`autosa_dse_aav_n`)
- **Clock**: 3.33 ns
- **Complete**: True; flash_selected=True; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_autosa_dse_fd_20260713_224002_autosa_dse_fd_nocosim_plainseed_stream`
- **Created**: 2026-07-13T22:40:02.356301+00:00; **Completed**: 2026-07-14T09:13:32.808506+00:00

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| autosa_cnn_rank1 | 18,554 | — | 72,085/160/190/55,760 | 561,242 | — | 20,732/10/34/17,775 | — | — | —/—/—/— | False | 561,242 | — | 20,732/10/34/17,775 |
| autosa_cnn_rank2 | 18,554 | — | 55,353/160/194/52,094 | 285,836 | — | 21,587/20/46/21,711 | — | — | —/—/—/— | False | 285,836 | — | 21,587/20/46/21,711 |
| autosa_cnn_rank3 | 18,554 | — | 55,083/160/194/52,036 | 285,836 | — | 21,526/20/46/21,699 | — | — | —/—/—/— | False | 285,836 | — | 21,526/20/46/21,699 |
| autosa_large_mm_intel_rank2 | 27,263,098 | — | 85,830/200/578/83,582 | 1,012,751,765 | — | 31,535/10/106/35,077 | — | — | —/—/—/— | False | 1,012,751,765 | — | 31,535/10/106/35,077 |
| autosa_large_mm_intel_rank3 | 27,263,100 | — | 57,427/200/252/59,985 | 464,281,318 | — | 28,820/20/123/35,377 | — | — | —/—/—/— | — | 464,281,318 | — | 28,820/20/123/35,377 |
| autosa_large_mm_rank1 | 524,403 | — | 86,570/156/354/75,322 | 25,878,718 | — | 34,351/6/102/36,674 | — | — | —/—/—/— | False | 25,878,718 | — | 34,351/6/102/36,674 |
| autosa_large_mm_rank2 | 524,386 | — | 58,995/156/275/56,468 | 11,778,266 | — | 22,522/12/55/25,265 | — | — | —/—/—/— | False | 11,778,266 | — | 22,522/12/55/25,265 |
| autosa_large_mm_rank3 | 524,386 | — | 54,923/156/267/55,398 | 12,801,207 | — | 22,511/12/47/33,825 | — | — | —/—/—/— | False | 12,801,207 | — | 22,511/12/47/33,825 |
| autosa_mm_catapult_rank1 | 8,286 | — | 36,729/96/226/39,097 | 8,518 | — | 57,500/96/90/228,907 | — | — | —/—/—/— | False | 8,518 | — | 57,500/96/90/228,907 |
| autosa_mm_catapult_rank2 | 5,078 | — | 65,613/192/330/68,455 | 199,399 | — | 26,978/12/30/32,601 | — | — | —/—/—/— | False | 199,399 | — | 26,978/12/30/32,601 |
| autosa_mm_catapult_rank3 | 8,285 | — | 37,137/96/226/37,286 | 152,312 | — | 33,721/12/218/91,384 | — | — | —/—/—/— | False | 152,312 | — | 33,721/12/218/91,384 |
| autosa_mm_getting_started_rank1 | 2,194 | — | 143,218/640/540/152,657 | 209,808 | — | 38,877/20/62/48,421 | — | — | —/—/—/— | False | 209,808 | — | 38,877/20/62/48,421 |
| autosa_mm_getting_started_rank2 | 2,254 | — | 139,143/640/540/152,574 | 211,736 | — | 37,295/20/46/49,686 | — | — | —/—/—/— | False | 211,736 | — | 37,295/20/46/49,686 |
| autosa_mm_getting_started_rank3 | 4,228 | — | 76,519/320/330/91,366 | 178,136 | — | 25,650/20/47/34,614 | — | — | —/—/—/— | False | 178,136 | — | 25,650/20/47/34,614 |
| autosa_mm_hcl_intel_rank1 | 4,226 | — | 78,064/320/330/84,618 | 180,844 | — | 31,979/20/122/37,285 | — | — | —/—/—/— | False | 180,844 | — | 31,979/20/122/37,285 |
| autosa_mm_hcl_intel_rank3 | 4,229 | — | 118,312/320/540/113,274 | 282,732 | — | 33,938/10/62/40,866 | — | — | —/—/—/— | False | 282,732 | — | 33,938/10/62/40,866 |
| autosa_mm_hcl_rank1 | 4,230 | — | 117,365/320/516/111,757 | 257,574 | — | 32,640/10/30/40,852 | — | — | —/—/—/— | False | 257,574 | — | 32,640/10/30/40,852 |
| autosa_mm_hcl_rank2 | 4,226 | — | 75,528/320/306/83,574 | 105,410 | — | 46,624/20/90/146,117 | — | — | —/—/—/— | False | 105,410 | — | 46,624/20/90/146,117 |
| autosa_mm_hcl_rank3 | 8,309 | — | 67,619/160/306/58,825 | 243,664 | — | 28,568/10/90/34,921 | — | — | —/—/—/— | False | 243,664 | — | 28,568/10/90/34,921 |
| autosa_mm_int16_rank1 | 4,219 | — | 93,003/64/384/57,231 | 273,198 | — | 27,206/2/30/28,249 | — | — | —/—/—/— | False | 273,198 | — | 27,206/2/30/28,249 |
| autosa_mm_int16_rank2 | 8,287 | — | 35,066/32/202/31,308 | 192,840 | — | 24,983/4/90/28,679 | — | — | —/—/—/— | False | 192,840 | — | 24,983/4/90/28,679 |
| autosa_mm_int16_rank3 | 8,287 | — | 35,033/32/202/29,816 | 131,285 | — | 22,100/4/91/26,920 | — | — | —/—/—/— | False | 131,285 | — | 22,100/4/91/26,920 |
| autosa_mm_intel_rank1 | 4,178 | — | 118,330/640/346/155,487 | 65,859 | — | 46,507/80/30/69,713 | — | — | —/—/—/— | False | 65,859 | — | 46,507/80/30/69,713 |
| autosa_mm_intel_rank2 | 4,208 | — | 99,934/320/346/105,307 | 95,939 | — | 39,240/40/30/49,379 | — | — | —/—/—/— | False | 95,939 | — | 39,240/40/30/49,379 |
| autosa_mm_intel_rank3 | 8,310 | — | 43,104/160/210/52,414 | 119,338 | — | 31,271/40/30/35,072 | — | — | —/—/—/— | False | 119,338 | — | 31,271/40/30/35,072 |
| autosa_mm_rank1 | 4,228 | — | 77,288/320/330/89,933 | 1,146,551 | — | 31,354/3/154/56,719 | — | — | —/—/—/— | False | 1,146,551 | — | 31,354/3/154/56,719 |
| autosa_mm_rank2 | 4,227 | — | 76,528/320/314/88,055 | 1,164,446 | — | 33,073/3/90/112,334 | — | — | —/—/—/— | False | 1,164,446 | — | 33,073/3/90/112,334 |
| autosa_mm_rank3 | 8,309 | — | 66,114/160/314/63,632 | 244,259 | — | 43,598/10/106/335,690 | — | — | —/—/—/— | False | 244,259 | — | 43,598/10/106/335,690 |

## `batch_parallel_complex_20260701_tier_a_bp_25`

- **System**: c2hls-tierA-complex
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: Tier-A complex flash (flavor=`complex/tier_a`)
- **Clock**: 4.0 ns
- **Suites in this extract**: spector_hls=2
- **Complete**: False; flash_selected=False; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_complex_20260701_tier_a_bp_25`

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| spector_hls_dct | 229,244,929 | — | 24,336/20/2/12,765 | 124,190,721 | — | 14,367/61/16/15,035 | — | — | —/—/—/— | — | 124,190,721 | — | 14,367/61/16/15,035 |
| spector_hls_histogram | 1,762 | — | 5,601/0/10/3,851 | 1,626 | — | 4,378/—/17/2,976 | — | — | —/—/—/— | — | 1,626 | — | 4,378/—/17/2,976 |

## `batch_parallel_complex_20260702_bpcplx_r2`

- **System**: c2hls-tierA-complex
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: Tier-A complex flash (flavor=`complex/tier_a`)
- **Clock**: 4.0 ns
- **Suites in this extract**: forgebench=4, hp_fft=2, spector_hls=7
- **Complete**: True; flash_selected=False; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_complex_20260702_bpcplx_r2`

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| forgebench_conv_A | 133,088 | — | 57,270/513/112/32,215 | 198,186 | — | 20,342/66/118/14,759 | — | — | —/—/—/— | — | 198,186 | — | 20,342/66/118/14,759 |
| forgebench_gpt_transformer_p1 | 73,978 | — | 157,002/302/67/182,146 | 44,201 | — | 163,857/302/92/181,488 | — | — | —/—/—/— | — | 44,201 | — | 163,857/302/92/181,488 |
| forgebench_mlp | 372,981 | — | 425,486/8,240/76/244,029 | 812,372 | — | 13,292/11/162/15,968 | — | — | —/—/—/— | — | 812,372 | — | 13,292/11/162/15,968 |
| forgebench_vec_mtx_p1 | 1,399 | — | 22,196/256/46/35,540 | 424 | — | 8,659/64/30/10,290 | — | — | —/—/—/— | — | 424 | — | 8,659/64/30/10,290 |
| hp_fft_n1024__UF1 | 5,811 | — | 37,210/96/96/34,122 | 4,541 | — | 92,255/96/90/109,990 | — | — | —/—/—/— | — | 4,541 | — | 92,255/96/90/109,990 |
| hp_fft_n256__UF1 | 1,293 | — | 23,261/72/40/26,290 | 1,187 | — | 42,435/96/—/54,863 | — | — | —/—/—/— | — | 1,187 | — | 42,435/96/—/54,863 |
| spector_hls_dct | 229,244,929 | — | 24,336/20/2/12,765 | 106,299,393 | — | 79,241/349/16/59,235 | — | — | —/—/—/— | — | 106,299,393 | — | 79,241/349/16/59,235 |
| spector_hls_fir | 273,177 | — | 19,654/25/16/25,139 | 155,817 | — | 14,005/45/4/14,216 | — | — | —/—/—/— | — | 155,817 | — | 14,005/45/4/14,216 |
| spector_hls_histogram | 1,762 | — | 5,601/0/10/3,851 | 1,761 | — | 13,198/—/2/6,019 | — | — | —/—/—/— | — | 1,761 | — | 13,198/—/2/6,019 |
| spector_hls_normals | 11,770,040 | — | 43,291/14/62/28,092 | 10,331,844 | — | 36,553/12/30/18,464 | — | — | —/—/—/— | — | 10,331,844 | — | 36,553/12/30/18,464 |
| spector_hls_sobel_filter_x | — | — | 4,573/3/2/2,551 | 16,540,979 | — | 2,518/8/2/1,561 | — | — | —/—/—/— | — | 16,540,979 | — | 2,518/8/2/1,561 |
| spector_hls_sobel_filter_y | — | — | 3,981/3/2/1,983 | 2,100,080 | — | 20,070/—/926/5,886 | — | — | —/—/—/— | — | 2,100,080 | — | 20,070/—/926/5,886 |
| spector_hls_template_matching | 44,280,002 | — | 3,466/2/1/1,171 | 40,148 | — | 30,387/—/30/47,859 | — | — | —/—/—/— | — | 40,148 | — | 30,387/—/30/47,859 |

## `batch_parallel_complex_tier_a_30_remaining_20260704_065754`

- **System**: c2hls-tierA-complex
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: Tier-A complex flash (flavor=`complex/tier_a`)
- **Clock**: 4.0 ns
- **Suites in this extract**: forgebench=9, hp_fft=7
- **Complete**: True; flash_selected=False; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_complex_tier_a_30_remaining_20260704_065754`

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| forgebench_attention_op_p2 | 17,057 | — | 44,538/228/65/43,005 | 3,110 | — | 123,650/12,324**/150/591,722 | — | — | —/—/—/— | — | 3,110 | — | 123,650/12,324**/150/591,722 |
| forgebench_attention_op_p3 | 17,057 | — | 44,539/228/65/43,005 | 4,868 | — | 51,186/228/153/48,124 | — | — | —/—/—/— | — | 4,868 | — | 51,186/228/153/48,124 |
| forgebench_conv_B | 454,688 | — | 60,545/513/552/20,623 | 185,307 | — | 44,777/128/300/18,020 | — | — | —/—/—/— | — | 185,307 | — | 44,777/128/300/18,020 |
| forgebench_conv_C | 1,375,058 | — | 58,863/513/640/25,827 | 14,646,516 | — | 9,983/1/230/8,118 | — | — | —/—/—/— | — | 14,646,516 | — | 9,983/1/230/8,118 |
| forgebench_diff_dims_p2 | 1,348,608 | — | 25,611/128/172/19,673 | 83,097 | — | 21,238/64/224/21,852 | — | — | —/—/—/— | — | 83,097 | — | 21,238/64/224/21,852 |
| forgebench_diff_dims_p3 | 1,188,881 | — | 57,987/512/412/27,429 | 36,256 | — | 61,915/192/536/52,261 | — | — | —/—/—/— | — | 36,256 | — | 61,915/192/536/52,261 |
| forgebench_diff_orders_p2 | 49,947 | — | 138,632/2,048/60/47,649 | 443 | — | 111,163/1,024/90/70,942 | — | — | —/—/—/— | — | 443 | — | 111,163/1,024/90/70,942 |
| forgebench_testing_impl | 2,503 | — | 22,126/64/60/13,721 | 336 | — | 22,051/32/122/20,973 | — | — | —/—/—/— | — | 336 | — | 22,051/32/122/20,973 |
| forgebench_vec_mtx_p2 | 391 | — | 12,496/256/46/28,614 | 1,222 | — | 15,813/1/62/10,268 | — | — | —/—/—/— | — | 1,222 | — | 15,813/1/62/10,268 |
| hp_fft_n1024__UF2 | 2,738 | — | 62,644/192/88/58,405 | 5,859 | — | 12,186/28/54/20,053 | — | — | —/—/—/— | — | 5,859 | — | 12,186/28/54/20,053 |
| hp_fft_n1024__UF32 | 486 | — | 708,280/2,283/0/809,481 | 4,418 | — | 71,632/268/30/150,464 | — | — | —/—/—/— | — | 4,418 | — | 71,632/268/30/150,464 |
| hp_fft_n1024__UF4 | 1,575 | — | 115,704/354/112/106,279 | 6,269 | — | 20,204/44/46/25,901 | — | — | —/—/—/— | — | 6,269 | — | 20,204/44/46/25,901 |
| hp_fft_n256__UF2 | 652 | — | 45,661/144/0/47,879 | 1,332 | — | 11,530/28/10/16,541 | — | — | —/—/—/— | — | 1,332 | — | 11,530/28/10/16,541 |
| hp_fft_n256__UF32 | 198 | — | 532,687/1,515/0/549,771 | 972 | — | 57,642/268/10/109,911 | — | — | —/—/—/— | — | 972 | — | 57,642/268/10/109,911 |
| hp_fft_n256__UF4 | 423 | — | 92,270/258/0/85,686 | 1,700 | — | 17,628/44/10/22,636 | — | — | —/—/—/— | — | 1,700 | — | 17,628/44/10/22,636 |
| hp_fft_n256__UF8 | 322 | — | 139,851/423/0/165,546 | 1,044 | — | 23,575/76/10/48,576 | — | — | —/—/—/— | — | 1,044 | — | 23,575/76/10/48,576 |

## `batch_parallel_complex_tier_a_30_remaining_20260704_100656`

- **System**: c2hls-tierA-complex
- **Model**: Devstral (`mistralai/Devstral-2-123B-Instruct-2512`)
- **Architecture**: Tier-A complex flash (flavor=`complex/tier_a`)
- **Clock**: 4.0 ns
- **Suites in this extract**: forgebench=10, hp_fft=7
- **Complete**: True; flash_selected=False; dataflow_selected=False
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/batch_parallel_complex_tier_a_30_remaining_20260704_100656`

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| forgebench_attention_op_p2 | 17,057 | — | 44,538/228/65/43,005 | 12,602 | — | 44,925/207/155/47,522 | — | — | —/—/—/— | — | 12,602 | — | 44,925/207/155/47,522 |
| forgebench_attention_op_p3 | 17,057 | — | 44,539/228/65/43,005 | 1,833 | — | 133,022/12,376**/150/601,781 | — | — | —/—/—/— | — | 1,833 | — | 133,022/12,376**/150/601,781 |
| forgebench_conv_C | 1,375,058 | — | 58,863/513/640/25,827 | 285,660 | — | 50,125/128/445/16,478 | — | — | —/—/—/— | — | 285,660 | — | 50,125/128/445/16,478 |
| forgebench_diff_dims_p2 | 1,348,608 | — | 25,611/128/172/19,673 | 2,146,467 | — | 16,217/1/184/19,346 | — | — | —/—/—/— | — | 2,146,467 | — | 16,217/1/184/19,346 |
| forgebench_diff_dims_p3 | 1,188,881 | — | 57,987/512/412/27,429 | 6,439,075 | — | 16,304/1/272/19,435 | — | — | —/—/—/— | — | 6,439,075 | — | 16,304/1/272/19,435 |
| forgebench_diff_orders_p2 | 49,947 | — | 138,632/2,048/60/47,649 | 443 | — | 111,163/1,024/90/70,942 | — | — | —/—/—/— | — | 443 | — | 111,163/1,024/90/70,942 |
| forgebench_diff_orders_p3 | 280,012 | — | 134,324/2,048/60/64,291 | 1,073,310 | — | 12,740/1/122/14,670 | — | — | —/—/—/— | — | 1,073,310 | — | 12,740/1/122/14,670 |
| forgebench_testing_impl | 2,503 | — | 22,126/64/60/13,721 | 336 | — | 22,051/32/122/20,973 | — | — | —/—/—/— | — | 336 | — | 22,051/32/122/20,973 |
| forgebench_testing_unroll | 17,250 | — | 308,970/592/60/71,579 | 9,576 | — | 110,938/80/120/55,405 | — | — | —/—/—/— | — | 9,576 | — | 110,938/80/120/55,405 |
| forgebench_vec_mtx_p2 | 391 | — | 12,496/256/46/28,614 | 263 | — | 15,363/16/62/10,506 | — | — | —/—/—/— | — | 263 | — | 15,363/16/62/10,506 |
| hp_fft_n1024__UF2 | 2,738 | — | 62,644/192/88/58,405 | 5,859 | — | 12,186/28/54/20,053 | — | — | —/—/—/— | — | 5,859 | — | 12,186/28/54/20,053 |
| hp_fft_n1024__UF32 | 486 | — | 708,280/2,283/0/809,481 | 4,418 | — | 71,632/268/30/150,464 | — | — | —/—/—/— | — | 4,418 | — | 71,632/268/30/150,464 |
| hp_fft_n1024__UF4 | 1,575 | — | 115,704/354/112/106,279 | 6,271 | — | 20,189/44/46/25,903 | — | — | —/—/—/— | — | 6,271 | — | 20,189/44/46/25,903 |
| hp_fft_n256__UF2 | 652 | — | 45,661/144/0/47,879 | 1,332 | — | 11,530/28/10/16,541 | — | — | —/—/—/— | — | 1,332 | — | 11,530/28/10/16,541 |
| hp_fft_n256__UF32 | 198 | — | 532,687/1,515/0/549,771 | 972 | — | 57,642/268/10/109,911 | — | — | —/—/—/— | — | 972 | — | 57,642/268/10/109,911 |
| hp_fft_n256__UF4 | 423 | — | 92,270/258/0/85,686 | 1,700 | — | 17,628/44/10/22,636 | — | — | —/—/—/— | — | 1,700 | — | 17,628/44/10/22,636 |
| hp_fft_n256__UF8 | 322 | — | 139,851/423/0/165,546 | 2,173 | — | 22,194/100/14/35,415 | — | — | —/—/—/— | — | 2,173 | — | 22,194/100/14/35,415 |

## `hybrid-u280-machsuite-tierA-20260721-012940`

- **System**: chathls-machsuite-tierA
- **Model**: DeepSeek (`deepseek`)
- **Architecture**: ChatHLS hybrid agent (machsuite + Tier-A)
- **Complete**: False (64/68)
- **Path**: `/scratch/hpc-prf-llmfpga/asa582/projects/test-chathls/ChatHLS-ACL-26/artifacts/pc2/sessions/hybrid-u280-machsuite-tierA-20260721-012940`
- **Baselines**: aligned to c2hls for shared benches
- **Updated**: 2026-07-21 10:55 UTC

| Bench | base csynth | base cosim | base LUT/DSP/BRAM/FF | flash csynth | flash cosim | flash LUT/DSP/BRAM/FF | df csynth | df cosim | df LUT/DSP/BRAM/FF | df ok | final csynth | final cosim | final LUT/DSP/BRAM/FF |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|
| forgebench_attention_op_p1 | 754,462 | — | 31,189/51/280/38,022 | — | — | —/—/—/— | — | — | —/—/—/— | — | 754,147 | — | 22,883/51/224/26,902 |
| forgebench_attention_op_p2 | 17,057 | — | 44,538/228/65/43,005 | — | — | —/—/—/— | — | — | —/—/—/— | — | 16,742 | — | 36,232/228/9/31,645 |
| forgebench_attention_op_p3 | 17,057 | — | 44,539/228/65/43,005 | — | — | —/—/—/— | — | — | —/—/—/— | — | 16,742 | — | 36,233/228/9/31,645 |
| forgebench_conv_A | 133,088 | — | 57,270/513/112/32,215 | — | — | —/—/—/— | — | — | —/—/—/— | — | 29,860,994 | — | 25,661/0/64/2,860 |
| forgebench_conv_B | 454,688 | — | 60,545/513/552/20,623 | — | — | —/—/—/— | — | — | —/—/—/— | — | 161,604 | — | 51,722/512/524/13,765 |
| forgebench_conv_C | 1,375,058 | — | 58,863/513/640/25,827 | — | — | —/—/—/— | — | — | —/—/—/— | — | 203,501 | — | 49,747/512/612/14,859 |
| forgebench_diff_dims_p1 | 2,060,282 | — | 41,372/384/268/28,235 | — | — | —/—/—/— | — | — | —/—/—/— | — | 2,060,029 | — | 33,725/384/212/17,664 |
| forgebench_diff_dims_p2 | 1,348,608 | — | 25,611/128/172/19,673 | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,348,355 | — | 18,159/128/116/9,600 |
| forgebench_diff_dims_p3 | 1,188,881 | — | 57,987/512/412/27,429 | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,188,628 | — | 50,475/512/356/17,003 |
| forgebench_diff_orders_p1 | 111,468 | — | 129,037/2,048/60/50,939 | — | — | —/—/—/— | — | — | —/—/—/— | — | 111,278 | — | 121,606/2,048/4/39,520 |
| forgebench_diff_orders_p2 | 49,947 | — | 138,632/2,048/60/47,649 | — | — | —/—/—/— | — | — | —/—/—/— | — | 54,718 | — | 127,536/2,048/4/39,652 |
| forgebench_diff_orders_p3 | 280,012 | — | 134,324/2,048/60/64,291 | — | — | —/—/—/— | — | — | —/—/—/— | — | 279,822 | — | 127,876/2,048/4/52,871 |
| forgebench_gpt_transformer_p1 | 73,978 | — | 157,002/302/67/182,146 | — | — | —/—/—/— | — | — | —/—/—/— | — | 80,796 | — | 96,034/238/16/103,068 |
| forgebench_llama_transformer_p2 | 80,828 | — | 175,614/325/73/95,864 | — | — | —/—/—/— | — | — | —/—/—/— | — | 87,518 | — | 103,979/261/22/56,386 |
| forgebench_mlp | 372,981 | — | 425,486/8,240/76/244,029 | — | — | —/—/—/— | — | — | —/—/—/— | — | 378,243 | — | 408,773/8,198/8/217,601 |
| forgebench_mult_op_p1 | 10,642 | — | 410,671/2,048/60/40,710 | — | — | —/—/—/— | — | — | —/—/—/— | — | 12,934 | — | 401,871/2,048/4/32,718 |
| forgebench_mult_op_p2 | 1,401 | — | 22,216/256/46/35,800 | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,209 | — | 15,795/256/4/28,086 |
| forgebench_mult_op_p3 | 419 | — | 6,592/16/32/9,682 | — | — | —/—/—/— | — | — | —/—/—/— | — | 276 | — | 4,732/16/4/5,640 |
| forgebench_testing_impl | 2,503 | — | 22,126/64/60/13,721 | — | — | —/—/—/— | — | — | —/—/—/— | — | 3,481 | — | 11,549/64/4/5,411 |
| forgebench_testing_unroll | 17,250 | — | 308,970/592/60/71,579 | — | — | —/—/—/— | — | — | —/—/—/— | — | 18,350 | — | 305,716/592/4/64,287 |
| forgebench_tiled_attn_p1 | 754,462 | — | 31,189/51/280/38,022 | — | — | —/—/—/— | — | — | —/—/—/— | — | 754,147 | — | 22,883/51/224/26,902 |
| forgebench_tiled_attn_p2 | 190,434 | — | 31,545/51/111/38,293 | — | — | —/—/—/— | — | — | —/—/—/— | — | 190,119 | — | 23,239/51/55/27,173 |
| forgebench_vec_mtx_p1 | 1,399 | — | 22,196/256/46/35,540 | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,208 | — | 15,778/256/4/28,083 |
| forgebench_vec_mtx_p2 | 391 | — | 12,496/256/46/28,614 | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,160 | — | 8,705/256/4/22,949 |
| hp_fft_n1024__UF1 | 5,811 | — | 37,210/96/96/34,122 | — | — | —/—/—/— | — | — | —/—/—/— | — | 5,809 | — | 37,166/96/96/34,081 |
| hp_fft_n1024__UF16 | 681 | — | 361,152/1,173/0/461,003 | — | — | —/—/—/— | — | — | —/—/—/— | — | 680 | — | 361,108/1,173/0/460,962 |
| hp_fft_n1024__UF2 | 2,738 | — | 62,644/192/88/58,405 | — | — | —/—/—/— | — | — | —/—/—/— | — | 0 | — | 0/0/0/0 |
| hp_fft_n1024__UF32 | 486 | — | 708,280/2,283/0/809,481 | — | — | —/—/—/— | — | — | —/—/—/— | — | 485 | — | 708,236/2,283/0/809,440 |
| hp_fft_n1024__UF4 | 1,575 | — | 115,704/354/112/106,279 | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,575 | — | 115,671/354/112/106,246 |
| hp_fft_n1024__UF8 | 948 | — | 214,551/615/0/202,271 | — | — | —/—/—/— | — | — | —/—/—/— | — | 946 | — | 214,522/615/0/203,261 |
| hp_fft_n1024__no_StagePipeline | 5,162 | — | 9,426/48/43/12,904 | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,858 | — | 5,122/48/13/7,481 |
| hp_fft_n1024__original_C_style | 37,281 | — | 24,092/214/41/24,377 | — | — | —/—/—/— | — | — | —/—/—/— | — | 7,329 | — | 19,720/214/11/18,872 |
| hp_fft_n256__UF1 | 1,293 | — | 23,261/72/40/26,290 | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,291 | — | 23,217/72/40/26,249 |
| hp_fft_n256__UF16 | 241 | — | 269,842/789/0/293,552 | — | — | —/—/—/— | — | — | —/—/—/— | — | 0 | — | 0/0/0/0 |
| hp_fft_n256__UF2 | 652 | — | 45,661/144/0/47,879 | — | — | —/—/—/— | — | — | —/—/—/— | — | 650 | — | 45,632/144/0/48,101 |
| hp_fft_n256__UF32 | 198 | — | 532,687/1,515/0/549,771 | — | — | —/—/—/— | — | — | —/—/—/— | — | 197 | — | 532,643/1,515/0/549,730 |
| hp_fft_n256__UF4 | 423 | — | 92,270/258/0/85,686 | — | — | —/—/—/— | — | — | —/—/—/— | — | 423 | — | 92,237/258/0/85,651 |
| hp_fft_n256__UF8 | 322 | — | 139,851/423/0/165,546 | — | — | —/—/—/— | — | — | —/—/—/— | — | 0 | — | 0/0/0/0 |
| hp_fft_n256__no_StagePipeline | 1,684 | — | 9,663/48/34/12,978 | — | — | —/—/—/— | — | — | —/—/—/— | — | 644 | — | 5,359/48/4/7,551 |
| hp_fft_n256__original_C_style | 8,509 | — | 24,131/214/34/24,193 | — | — | —/—/—/— | — | — | —/—/—/— | — | 2,293 | — | 19,759/214/4/18,688 |
| machsuite_aes_table | 134,242 | — | 351,304/0/16/124,371 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| machsuite_aes_tableless | 146,890 | — | 105,752/0/16/34,707 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| machsuite_backprop | 279,873,283 | — | 196,108/562/32/168,384 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| machsuite_bfs_bulk | 14,652,591 | — | 7,661/0/8/4,011 | — | — | —/—/—/— | — | — | —/—/—/— | — | 770 | — | 665/0/0/462 |
| machsuite_fft_transpose | 26,433 | — | 61,297/318/46/84,068 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| machsuite_gemm_blocked | 901,125 | — | 40,482/22/30/23,882 | — | — | —/—/—/— | — | — | —/—/—/— | — | 1,377,281 | — | 3,164/8/0/3,347 |
| machsuite_gemm_ncubed | 295,512 | — | 152,198/11/30/60,410 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| machsuite_md_grid | — | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| machsuite_md_knn | 13,748 | — | 126,733/41/30/60,210 | — | — | —/—/—/— | — | — | —/—/—/— | — | 2,560 | — | 21,357/208/0/48,895 |
| machsuite_sort_radix | 19,274,417 | — | 258,439/0/30/85,276 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| machsuite_spmv_ellpack | 270,219 | — | 14,335/11/8/8,703 | — | — | —/—/—/— | — | — | —/—/—/— | — | 2,559 | — | 3,039/22/0/5,201 |
| machsuite_stencil2D | 140,762 | — | 7,706/3/4/6,501 | — | — | —/—/—/— | — | — | —/—/—/— | — | 12,108 | — | 6,711/18/0/3,032 |
| machsuite_stencil3D | 90,919 | — | 32,316/3/30/15,346 | — | — | —/—/—/— | — | — | —/—/—/— | — | 22,793 | — | 68,485/9/0/20,367 |
| machsuite_viterbi | 636,989 | — | 348,625/21/240/141,196 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| spector_hls_dct | 229,244,929 | — | 24,336/20/2/12,765 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | 27,578/324/0/38,750 |
| spector_hls_fir | 273,177 | — | 19,654/25/16/25,139 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| spector_hls_histogram | 1,762 | — | 5,601/0/10/3,851 | — | — | —/—/—/— | — | — | —/—/—/— | — | 515 | — | 275/0/0/152 |
| spector_hls_matrix_multiplication | 153,592,266,753 | — | 10,080/3/30/8,937 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| spector_hls_mergesort | 21,971 | — | 117,276/19/30/48,096 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| spector_hls_normals | 11,770,040 | — | 43,291/14/62/28,092 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| spector_hls_sobel_filter_x | — | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | 677,927 | — | 5,568/8/0/1,028 |
| spector_hls_sobel_filter_y | — | — | —/—/—/— | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| spector_hls_spmv | 225,793 | — | 17,169/8/4/15,200 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | —/—/—/— |
| spector_hls_template_matching | 44,280,002 | — | 3,466/2/1/1,171 | — | — | —/—/—/— | — | — | —/—/—/— | — | — | — | 6,677/0/0/2,946 |

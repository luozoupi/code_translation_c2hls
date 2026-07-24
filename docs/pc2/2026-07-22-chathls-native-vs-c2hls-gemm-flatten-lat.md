# ChatHLS native vs c2hls gemm_flatten + latency-opt

Generated: 2026-07-22 02:01 UTC

**Fair pair:** same ChatHLS `benchmark_optimization` / `chathls_ready` sizes.

- **ChatHLS:** `hybrid-u280-split-20260717-001649` (+ legal matmul fallback)
- **c2hls:** `batch_parallel_chathls_fd_ds_rag2_lat_20260721_215722_chathls` (`gemm_flatten_v1` + `--latency-opt`)
- **FPGA / clock:** U280, **3.33 ns**
- **c2hls latency:** best legal among `*_selected_report` / flash final / dataflow / latency_opt reports

Note: campaign `reference_validation.json` still points at baselines; numbers below use optimized reports.

## Headline (csynth, legal pairs)

- Legal paired benches: **13**
- c2hls wins: **6**
- ChatHLS wins: **7**
- **Geomean speedup of c2hls over ChatHLS** = **0.986×**

Interpretation: **>1 means c2hls is faster**.

## Per-bench

| Bench | ChatHLS | c2hls | **speedup c2/ch** | winner | c2 stage | vs old c2 |
|-------|--------:|------:|------------------:|--------|----------|----------:|
| syrk | 388,881 | 24,181 | **16.08×** | c2hls | flash | 0.02× |
| gemm_blocked | 2,097,166 | 210,832 | **9.95×** | c2hls | dataflow | 0.16× |
| matmul | 16,398 | 2,456 | **6.68×** | c2hls | dataflow | 0.39× |
| transformer | 80,676 | 20,940 | **3.85×** | c2hls | dataflow | 0.22× |
| covariance | 10,750 | 8,030 | **1.34×** | c2hls | flash | 1.23× |
| gemm | 902 | 824 | **1.09×** | c2hls | dataflow | 0.03× |
| atax | 915 | 1,146 | **0.80×** | chathls | dataflow | 1.04× |
| bicg | 843 | 1,135 | **0.74×** | chathls | flash | 0.60× |
| gesummv | 482 | 659 | **0.73×** | chathls | dataflow | 0.28× |
| syr2k | 394,721 | 995,108 | **0.40×** | chathls | flash | 2.44× |
| gemm_ncubed | 4,627 | 12,959 | **0.36×** | chathls | dataflow | 0.38× |
| mvt | 1,234 | 5,003 | **0.25×** | chathls | flash | 1.00× |
| 2mm | 7,582 | 829,138 | **0.01×** | chathls | flash | 22.03× |

## Excluded

| Bench | reason |
|-------|--------|
| 3mm | CH: illegal_or_missing(ff=133%); C2: ok |
| mobilenet | CH: illegal_or_missing(no_latency); C2: illegal(no_legal_c2) |
| symm | CH: ok; C2: illegal(no_legal_c2) |

CSV: [`2026-07-22-chathls-native-vs-c2hls-gemm-flatten-lat.csv`](2026-07-22-chathls-native-vs-c2hls-gemm-flatten-lat.csv)

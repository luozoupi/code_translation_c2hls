# ChatHLS native vs c2hls gf98 (gemm_flatten_v1 98 skills)

Generated: 2026-07-22 05:46 UTC

- **c2hls:** `batch_parallel_chathls_fd_ds_rag2_lat_20260722_032844_chathls_gf98`
- **ChatHLS:** hybrid-u280-split-20260717-001649 (+ matmul legal fallback)
- **Skills:** gemm_flatten_v1 (98) + latency-opt

## Headline

- Legal pairs: **14**
- c2hls wins: **7** / ChatHLS wins: **7**
- **Geomean speedup c2/ch = 1.123×** (>1 = c2hls faster)

## Per-bench

| Bench | ChatHLS | c2hls | speedup | winner | stage | vs 215722 |
|-------|--------:|------:|--------:|--------|--------|----------:|
| syr2k | 394,721 | 26,307 | **15.00×** | c2hls | flash | 0.03× |
| syrk | 388,881 | 26,272 | **14.80×** | c2hls | flash | 1.09× |
| gemm_blocked | 2,097,166 | 258,127 | **8.12×** | c2hls | dataflow | 1.22× |
| transformer | 80,676 | 10,017 | **8.05×** | c2hls | dataflow | 0.48× |
| matmul | 16,398 | 3,225 | **5.08×** | c2hls | dataflow | 1.31× |
| covariance | 10,750 | 6,183 | **1.74×** | c2hls | flash | 0.77× |
| atax | 915 | 615 | **1.49×** | c2hls | dataflow | 0.54× |
| mvt | 1,234 | 1,828 | **0.68×** | chathls | dataflow | 0.37× |
| bicg | 843 | 1,885 | **0.45×** | chathls | dataflow | 1.66× |
| gesummv | 482 | 1,184 | **0.41×** | chathls | dataflow | 1.80× |
| gemm | 902 | 2,433 | **0.37×** | chathls | flash | 2.95× |
| gemm_ncubed | 4,627 | 12,960 | **0.36×** | chathls | dataflow | 1.00× |
| symm | 311,596 | 1,718,793 | **0.18×** | chathls | flash | — |
| 2mm | 7,582 | 826,569 | **0.01×** | chathls | flash | 1.00× |

## Excluded

| 3mm | CH FF 133% |
| mobilenet | no latency |

CSV: [`2026-07-22-chathls-native-vs-c2hls-gf98.csv`](2026-07-22-chathls-native-vs-c2hls-gf98.csv)

# ChatHLS native vs c2hls on same ChatHLS benches (legalized)

Generated: 2026-07-21 21:29 UTC

**Fair pair:** same ChatHLS `benchmark_optimization` benches / sizes.

- **ChatHLS primary:** `hybrid-u280-split-20260717-001649` (DeepSeek, U280)
- **c2hls:** `batch_parallel_chathls_fd_ds_rag2_20260721_123449` (DeepSeek RAG2+skills)
- **FPGA / clock:** `xcu280-fsvh2892-2L-e`, target **3.33 ns**

## Legalization gates (both sides)

1. Target clock = **3.33 ns** (±0.02)
2. Device `Utilization (%)` ≤ **100** for BRAM, DSP, FF, LUT, URAM
3. Same bench + size family; csynth geomean uses csynth only
4. ChatHLS fallback (if primary fails): same-bench U280 hybrid only, legal, and ≤50× primary latency
5. c2hls = **best legal** latency among flash/dataflow

CSV: [`2026-07-21-chathls-native-vs-c2hls-chathls-ready.csv`](2026-07-21-chathls-native-vs-c2hls-chathls-ready.csv)

## Headline (csynth, legal pairs only)

- Legal paired benches: **14**
- Excluded / incomplete: **2** (3mm, mobilenet)
- c2hls wins: **3**
- ChatHLS wins: **11**
- **Geomean speedup of c2hls over ChatHLS** = **0.408×**

Interpretation: **>1 means c2hls is faster**.

## Per-bench csynth (legalized)

| Bench | size | ChatHLS | c2hls | **speedup c2/ch** | winner | CH source | c2 stage | notes |
|-------|------|--------:|------:|------------------:|--------|-----------|----------|-------|
| matmul | chathls_imported_mini | 16,398 | 6,298 | **2.60×** | c2hls | legal_fallback | dataflow | primary_illegal(dsp=221%); fallback opt-matmul-hybrid-u280-20260716-214729 |
| covariance | chathls_imported_mini | 10,750 | 6,504 | **1.65×** | c2hls | primary_session | dataflow |  |
| gemm_blocked | machsuite_64 | 2,097,166 | 1,319,078 | **1.59×** | c2hls | primary_session | flash |  |
| syr2k | chathls_imported_hlsfactory_sized | 394,721 | 407,673 | **0.97×** | chathls | primary_session | flash |  |
| transformer | unknown | 80,676 | 94,209 | **0.86×** | chathls | primary_session | flash |  |
| atax | chathls_imported_mini | 915 | 1,098 | **0.83×** | chathls | primary_session | flash |  |
| bicg | chathls_imported_mini | 843 | 1,893 | **0.45×** | chathls | primary_session | flash |  |
| syrk | chathls_imported_hlsfactory_sized | 388,881 | 979,273 | **0.40×** | chathls | primary_session | flash |  |
| mvt | chathls_imported_mini | 1,234 | 5,007 | **0.25×** | chathls | primary_session | flash |  |
| gesummv | chathls_imported_mini | 482 | 2,314 | **0.21×** | chathls | primary_session | flash |  |
| 2mm | chathls_imported_hlsfactory_sized | 7,582 | 37,645 | **0.20×** | chathls | primary_session | flash |  |
| gemm_ncubed | machsuite_64 | 4,627 | 34,135 | **0.14×** | chathls | primary_session | flash |  |
| symm | chathls_imported_hlsfactory_sized | 311,596 | 2,846,553 | **0.11×** | chathls | primary_session | flash |  |
| gemm | chathls_imported_mini | 902 | 32,930 | **0.03×** | chathls | primary_session | flash |  |

## Excluded / incomplete

| Bench | CH lat | CH note | C2 lat | C2 note |
|-------|-------:|---------|-------:|---------|
| 3mm | 7018 | illegal_or_missing(ff=133%) | 93850 | ok |
| mobilenet | — | illegal_or_missing(no_latency) | — | illegal(no_legal_c2) |

## Clock / resource snapshot

| Bench | CH clk | CH DSP%/FF% | C2 clk | C2 DSP%/FF% | pair legal |
|-------|-------:|-------------|-------:|-------------|------------|
| 2mm | 3.33 | 12/79 | 3.33 | 8/5 | 37645 |
| 3mm | 3.33 | 31/133 | 3.33 | 9/6 | False |
| atax | 3.33 | 0/0 | 3.33 | 1/1 | 1098 |
| bicg | 3.33 | 0/0 | 3.33 | 3/1 | 1893 |
| covariance | 3.33 | 0/0 | 3.33 | 1/1 | 6504 |
| gemm | 3.33 | 3/4 | 3.33 | 0/0 | 32930 |
| gemm_blocked | 3.33 | 0/0 | 3.33 | 0/0 | 1319078 |
| gemm_ncubed | 3.33 | 15/47 | 3.33 | 1/1 | 34135 |
| gesummv | 3.33 | 0/0 | 3.33 | 0/0 | 2314 |
| matmul | 3.33 | 0/0 | 3.33 | 0/5 | 6298 |
| mobilenet | 3.33 | 7/2 | — | None/None | False |
| mvt | 3.33 | 0/0 | 3.33 | 0/0 | 5007 |
| symm | 3.33 | 0/5 | 3.33 | 0/0 | 2846553 |
| syr2k | 3.33 | 0/0 | 3.33 | 0/0 | 407673 |
| syrk | 3.33 | 0/1 | 3.33 | 0/0 | 979273 |
| transformer | 3.33 | 2/4 | 3.33 | 1/2 | 94209 |

## Cosim (legal pairs with both measured)

- n=4; geomean speedup c2 over ch = **119.372×**

| Bench | ChatHLS cosim | c2hls cosim | speedup c2/ch |
|-------|--------------:|------------:|--------------:|
| gemm_blocked | 1,187,910 | 3,955 | 300.36× |
| gesummv | 4,339 | 20 | 216.95× |
| covariance | 74,630 | 1,081 | 69.04× |
| mvt | 2,979 | 66 | 45.14× |

## Notes

- Prior table used illegal ChatHLS matmul **660** (DSP 221%) and 3mm **7018** (FF 133%).
- Matmul @ 10 ns (1043) excluded for clock mismatch; legal CH matmul fallback **16398** @ 3.33 ns.
- 3mm excluded: no legal competitive ChatHLS (fallback would be ~2.3M-cycle gold-like).

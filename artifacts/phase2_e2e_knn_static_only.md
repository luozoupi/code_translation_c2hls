# Phase 2 End-to-End: knn

_generated 2026-05-06T23:31:35 from existing on-disk artifacts._

- Vitis HLS 2025.2 / `xc7a100t-csg324-1` / 4 ns clock
- agent: claude-haiku-4-5-20251001
- philip's reference: Vitis 2023.2 / xilinx_u280 — different (toolchain, FPGA), compare ratios not absolutes


## Static-order run (current production behavior)

| step | success | effect | lat_cyc | latency_ns | ii | BRAM | DSP | FF | LUT | Fmax | routing |
|------|:-------:|--------|--------:|-----------:|---:|-----:|----:|---:|---:|-----:|---------|
| baseline | True | baseline | 72351885 | 3206000000.0 | 110100622 | 29 | 5 | 7071 | 4961 | 22.57 | - |
| tiling | True | - | 5320925 | 240000000.0 | 5320926 | 32 | 5 | 5559 | 5587 | 22.17 | - |
| pipeline | False | reverted | 4327645 | 23138000.0 | 4327646 | 32 | 6 | 7453 | 6579 | 187.02 | - |
| unroll | False | reverted | 37247197 | 1651000000.0 | 37247198 | 33 | 5 | 5756 | 6078 | 22.57 | - |
| doublebuffer | False | reverted | 3890922 | 176000000.0 | 5615339 | 35 | 10 | 11576 | 9768 | 22.17 | - |
| coalescing | False | - | None | None | None | None | None | None | None | None | - |

### Phase 2 verdict on static run trajectory

| step | events |
|------|--------|
| baseline | throughput_regression: interval (110100622) significantly exceeds latency (72351885; ratio 1.52x) — thr |
| tiling | clean |
| pipeline | clean |
| unroll | throughput_regression: interval grew 7.00x vs parent (5320926 → 37247198); limit 1.05x |
| doublebuffer | throughput_regression: interval (5615339) significantly exceeds latency (3890922; ratio 1.44x) — throug |
| coalescing | clean |

## Dynamic-routing run (Phase 2 behavior)

_no dynamic results on disk yet — run still in progress or not started._

## Reference (philip's knn jsonl) — for context

| step | latency_avg | latency_worst | ii_max | BRAM | DSP | FF | LUT |
|------|------------:|--------------:|-------:|-----:|----:|---:|---:|
| baseline | 1048818 | 1048818 | 1048819 | 30 | 14 | 8012 | 5802 |
| tiling | 4276372 | 4276372 | 4276373 | 33 | 14 | 7900 | 6118 |
| pipeline | 4276372 | 4276372 | 4276373 | 33 | 14 | 7900 | 6118 |
| unroll | 4044880 | 4044880 | 4044881 | 31 | 28 | 43839 | 13203 |
| doublebuffer | 1740530 | 3464580 | 3464581 | 32 | 28 | 76651 | 27758 |
| coalescing | 262480 | 510530 | 510531 | 30 | 224 | 101850 | 23346 |

### Retroactive Phase 2 verdict on reference

Applies Pillar 9's no-op trap and throughput-regression checks to philip's reference trajectory. Surfaces ground-truth-side issues the Phase 2 hooks were designed to catch:

| step | events |
|------|--------|
| baseline | clean |
| tiling | throughput_regression: interval grew 4.08x vs parent (1048819 → 4276373); limit 1.05x |
| pipeline | no_op: all of ('latency_cycles', 'interval', 'bram', 'dsp', 'ff', 'lut') unchanged from previous step (lat=4276372, ii=4276373, bram=33, dsp=14, ff=7900, lut=6118) |
| unroll | clean |
| doublebuffer | throughput_regression: interval (3464581) significantly exceeds latency (1740530; ratio 1.99x) — throug |
| coalescing | throughput_regression: interval (510531) significantly exceeds latency (262480; ratio 1.95x) — throughp |


### hw_emu reference (for reference cycle counts)

| variant | status | runtime_cycles | runtime_us |
|---------|:------:|---------------:|-----------:|
| baseline | pass | 1049027 | 3496.756 |
| tiling | pass | 4264936 | 14216.453 |
| pipeline | pass | 4264936 | 14216.453 |
| unroll | pass | 4033397 | 13444.658 |
| doublebuffer | pass | 3453991 | 11513.302 |
| coalescing | pass | 502008 | 1673.361 |

## Static vs Dynamic head-to-head

_one or both runs not finished yet._

# Phase 2 End-to-End: knn

_generated 2026-05-07T02:35:37 from existing on-disk artifacts._

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

| step | success | effect | lat_cyc | latency_ns | ii | BRAM | DSP | FF | LUT | Fmax | routing |
|------|:-------:|--------|--------:|-----------:|---:|-----:|----:|---:|---:|-----:|---------|
| baseline | True | baseline | 1048841 | 4195000.0 | 1048842 | 33 | 6 | 46649 | 6826 | 256.74 | - |
| coalescing | True | absorbed | 1048841 | 4195000.0 | 1048842 | 159 | 6 | 46723 | 6836 | 256.74 | `prompt-coalescing` ← `matched bottleneck 'interval_exceeds_latency' → skill 'pr…` |
| doublebuffer | False | synth_failed | 1048841 | 4195000.0 | 1048842 | 159 | 6 | 46723 | 6836 | 256.74 | `prompt-doublebuffer` ← `matched bottleneck 'interval_exceeds_latency' → skill 'pr…` |
| tiling | False | synth_failed | 4202765 | 22536000.0 | 4202766 | 207 | 6 | 8813 | 7518 | 186.5 | `prompt-tiling` ← `matched bottleneck 'interval_exceeds_latency' → skill 'pr…` |
| pipeline | False | synth_failed | 1048841 | 4195000.0 | 1048842 | 159 | 6 | 46723 | 6836 | 256.74 | `no actionable bottleneck found; advancing static order` |
| unroll | True | improved | 524753 | 2099000.0 | 524754 | 54 | 12 | 13316 | 10592 | 284.66 | `no actionable bottleneck found; advancing static order` |

### Phase 2 verdict on dynamic run trajectory

| step | events |
|------|--------|
| baseline | clean |
| coalescing | clean |
| doublebuffer | no_op: all of ('latency_cycles', 'interval', 'bram', 'dsp', 'ff', 'lut') unchanged from previous step (lat=1048841, ii=1048842, bram=159, dsp=6, ff=46723, lut=6836) |
| tiling | throughput_regression: interval grew 4.01x vs parent (1048842 → 4202766); limit 1.05x |
| pipeline | no_op: all of ('latency_cycles', 'interval', 'bram', 'dsp', 'ff', 'lut') unchanged from previous step (lat=1048841, ii=1048842, bram=159, dsp=6, ff=46723, lut=6836) |
| unroll | clean |

## combo_full run (Phase 3 single-shot all-in-one)

| step | success | effect | lat_cyc | latency_ns | ii | BRAM | DSP | FF | LUT | Fmax | routing |
|------|:-------:|--------|--------:|-----------:|---:|-----:|----:|---:|---:|-----:|---------|
| baseline | True | baseline | 1048841 | 4195000.0 | 1048842 | 33 | 6 | 46649 | 6826 | 256.74 | - |
| combo_full | False | reverted | 3278989 | 17519000.0 | 3278990 | 37 | 6 | 8891 | 7645 | 187.16 | - |

### Phase 2 verdict on combo_full run trajectory

| step | events |
|------|--------|
| baseline | clean |
| combo_full | throughput_regression: interval grew 3.13x vs parent (1048842 → 3278990); limit 1.05x |

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

## Strategy head-to-head: best PPA achieved per run

| run | strategy | best lat_cyc | best lat_ns | wall observed |
|-----|----------|-------------:|------------:|---------------|
| static | tiling→pipeline→… | 5320925 | 240000000.0 | 64.8 min |
| dynamic | bottleneck-routed (Phase 2) | 524753 | 2099000.0 | 66.4 min |
| combo_full | single-shot all-in-one (Phase 3) | 1048841 | 4195000.0 | _see run log_ |

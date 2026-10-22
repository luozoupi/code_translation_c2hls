# Manual true DATAFLOW + ping-pong of ctl_cc_refnf_r2

Source design: `runs_controlled/skills_ctl_cc_refnf_r2_*/autosa_mm/autosa_mm_selected.cpp` (Opus 5.5, HLSmith-SingleStep-
Refined-Expertise + Resource-Requirement in words). Its original report: 1,248,397 cycles (worst = best), 5,120 DSP
(56.7%), LUT 312,231 (23.9%), BRAM 346 (8.6%), URAM 256 (26.7%), FF 572,436 (22.0%), csim passed on the old bench; DATAFLOW
/ ping-pong / streams: No / No / No. That design reads C (C += A·Bᵀ), so it fails the fixed bench.

All variants below use the fixed bench (C = A·Bᵀ, C poisoned), xcu280-fsvh2892-2L-e, 3.33 ns, Vitis HLS 2023.2.

| Variant | Change | csim | Latency best / worst | DSP | LUT | FF | BRAM | URAM | est. clk | dfdiag |
|---|---|---|---|---|---|---|---|---|---|---|
| v0 | Original structure; C load removed, accumulators start at 0 | Passed | 1,246,225 / 1,246,225 | 5,120 | 311,780 | 571,725 | 346 | 256 | 2.494 | NO_DATAFLOW: B preload 65.5K + 16 × (A load 4.1K + compute 65.6K + store 4.1K), all serial |
| v1 | 16 row tiles in a DATAFLOW loop: load_A(t+1), compute(t), store_C(t−1) overlap through PIPO A_t/C_t. B streamed into URAM one beat per cycle during tile 0 (next 16-row block while computing the current one), so there is no serial preload. Compute order (jb, kb, ii), accumulator distance 16. | Passed | 1,062,813 / 1,071,597 | 5,120 | 319,735 | 577,689 | 602 | 256 | 3.562 | PINGPONG, FULL: per tile L = 70,884, II = 66,714 (compute); 16 tiles → 1,071,596 ≈ pipelined prediction 1,071,594 (serial would be 1,134,144) |
| v2 | v1 with the 64 column-block loops flattened into one II=1 pipeline (removes ~18 cycles of refill × 64 blocks per tile) | pending | pending | | | | | | | |

Compute floor at 5,120 DSP is 1,048,576 cycles. v1 worst case is 1.022× the floor. AutoSA st0_c6 reaches 1,048,845 /
1,054,811 cycles.

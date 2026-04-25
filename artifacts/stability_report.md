# Full-corpus stability sweep — variants[-1] on U50 @ 3.33 ns

**Command:**
```bash
python verify_corpus_stability.py --last-only --validated-only --n-runs 3
```

**Target:** `xcu50-fsvh2104-2-e` @ 3.33 ns, Vitis HLS 2025.2, threshold cv ≤ 0.05 on `latency_ns`.
Ran on 17 benchmarks; one record per `variants[-1]` (mirrors what the c2hls pipeline picks as ground truth).

## Headline

- **13 / 17 stable.** Every stable variant produces **bit-identical** numbers across all 3 runs (`cv = 0.0000` for `latency_ns`, `fmax_mhz`, `lut`, `ff`, `bram`, `dsp`).
- **0 cv-bound failures** — no benchmark synthesised but drifted between runs.
- **4 failures** all on the synth-probe step (none reached the N-run pass).

## Per-benchmark roll-up

Sorted by stable → fail; metrics are the across-run mean.

| Benchmark      | Variant                          | OK   | Stable | latency_ns | fmax (MHz) | LUT   | sec |
|----------------|----------------------------------|------|--------|-----------:|-----------:|------:|----:|
| StreamCluster  | streamcluster_4_coalescing       | OK   | Y      |     71,675 |     411.35 |  4,041 | 174 |
| aes            | aes_0_baseline                   | OK   | Y      |      3,247 |     411.35 |  8,342 |  51 |
| gemm_ncubed    | gemm_ncubed_0_baseline           | OK   | Y      |  1,747,000 |     374.11 |  4,540 |  44 |
| hotspot        | hotspot_6_multiddr               | OK   | Y      | 13,559,000 |     411.35 | 16,320 | 169 |
| kmeans         | kmeans_6_multiddr                | OK   | Y      |  2,962,000 |     411.35 | 18,260 |  80 |
| knn            | knn_5_coalescing                 | OK   | Y      |    471,000 |     426.99 | 12,640 |  53 |
| lavaMD         | lavaMD_5_coalescing              | OK   | Y      |  6,053,000 |     300.30 | 17,820 | 148 |
| md_knn         | md_knn_0_baseline                | OK   | Y      |     17,969 |     411.35 |  3,400 |  43 |
| nw             | nw_5_coalescing                  | OK   | Y      |    110,000 |     411.35 |  9,860 | 139 |
| pathfinder     | pathfinder_5_coalescing          | OK   | Y      |     59,783 |     411.35 |  6,420 | 117 |
| spmv_crs       | spmv_crs_0_baseline              | OK   | Y      |      5,558 |     238.27 |  3,270 |  37 |
| srad           | srad_5_coalescing                | OK   | Y      |    251,000 |     411.35 | 11,950 | 111 |
| stencil2D      | stencil2D_0_baseline             | OK   | Y      |     80,799 |     300.30 |  3,890 |  37 |
| fft            | fft_0_baseline                   | FAIL | N      |          — |          — |     — |   — |
| lud            | lud_3_unrolling                  | FAIL | N      |          — |          — |     — |   — |
| sort_merge     | sort_merge_0_baseline            | FAIL | N      |          — |          — |     — |   — |
| viterbi        | viterbi_0_baseline               | FAIL | N      |          — |          — |     — |   — |

(Per-variant LUT figures come from each benchmark's `artifacts/stability/<bench>.json`; hand-checked a few for accuracy.)

## Failure classification

| Benchmark | Class | Root cause |
|---|---|---|
| `lud` | **synth timeout** | `lud_3_unrolling` exceeds 1200 s synth budget on U50 @ 3.33 ns — same finding as the earlier top-3 rerun. Real performance issue with this rodinia variant on data-center parts. Not a corpus bug. |
| `fft` | **header-resolution** | Vitis: `'support.h' file not found (./fft.h:9:10)`. |
| `sort_merge` | **header-resolution** | same — `./sort.h:4:10` |
| `viterbi` | **header-resolution** | same — `./viterbi.h:10:10` |

The three header-resolution failures share a root cause: `c2hls._ground_truth_candidates` (introduced in commit `c3c5bf7`, "Stabilize HLS validation pipeline") loads the per-variant header from the **upstream** `source_path` directory when one exists. For machsuite-derived benchmarks, that upstream header still contains `#include "support.h"`. The local `benchmarks/<bench>/<bench>.h` was cleaned to drop that include during corpus prep, but the cleaned local header is bypassed when the upstream copy is on disk.

The pipeline's main `c2hls.py --bench fft` flow still passes — it falls back to the local baseline header through a different path. The verifier hits the bug because it goes through `_ground_truth_candidates` directly.

## Determinism finding

For the 13 successful benchmarks, **Vitis HLS 2025.2 on this U50 box is bit-deterministic at variants[-1]**. No metric drifts between runs. This is the cleanest signal possible for any downstream RL reward function — the GT report can be cached after a single run with no statistical loss.

This does **not** generalize across machines or Vitis versions. The numbers above are valid only for this `(part, clock_ns, vitis_version, hardware)` tuple. To detect cross-machine drift later, the same script + threshold can be re-run on a different host and the summary diffed.

## Follow-up work (not in this commit)

1. **Header-resolution fix.** In `_ground_truth_candidates`, prefer the local `benchmarks/<bench>/<header_file>` over the upstream header when both exist, OR always materialize `support.h` alongside if it appears in the upstream header's includes. Easiest path: prefer local; fall back to upstream only when local is missing. Affects `fft`, `sort_merge`, `viterbi`. Should restore the 16/17 stability count.
2. **`lud_3_unrolling` synth timeout.** Distinct from above — needs either a longer timeout (`C2HLS_SYNTH_TIMEOUT=2400`) or pipeline-level walk-back to `lud_2_coalescing` for stability measurement. Walking back was explicitly out of scope for this sweep per the user's "final step optimized version" framing.
3. **Cross-host repeat.** Re-run this sweep on a second machine with the same Vitis 2025.2 install; diff the per-benchmark `latency_ns_mean` values. Differences > 0 there would establish the variance budget the rubric/RL reward needs to tolerate.

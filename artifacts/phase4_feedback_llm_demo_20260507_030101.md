# FeedbackAgent: deterministic vs LLM-aided composition

_generated 2026-05-07T03:01:05_


Real regression case from the static-knn run: the LLM's `unroll` step produced a kernel that synthesized 6.88x slower latency than the parent step (240M → 1651M ns) with LUT 1.80x, FF 2.65x, DSP 2.40x. Phase 1's regression-revert kicked in.

This script asks the FeedbackAgent for a retry-guidance prompt twice:
- **Deterministic template** (default, no LLM call): `fa.render('regression', ...)`
- **LLM-aided composition** (Phase 4 stretch path): `fa.compose_with_llm('regression', kernel_diff=..., prior_template=..., bottleneck_record=...)` with `C2HLS_FEEDBACK_LLM=1`. Routes through the FeedbackAgent's own model (Haiku 4.5 in this run).

## Deterministic template

```
Your previous attempt at the `unroll` step was REJECTED because it regressed against the previous step's metrics:
  - latency_ns regressed 6.88x (240000000 -> 1651000000)

Produce a more conservative version that PRESERVES or IMPROVES on the previous step's latency and does not inflate resource usage. If the requested optimization cannot help here, return the previous code with only minor tweaks.
```

_399 chars; emitted with zero LLM cost._

## LLM-aided composition

```
The 6.88x latency regression indicates aggressive unrolling is causing critical path delays and memory contention rather than improving throughput. Remove the `#pragma HLS UNROLL factor=2` directives from the inner feature loop (the innermost `for(int j = 0; j < NUM_FEATURE; ++j)` block) and the STORE phase loop—these are creating excessive parallel memory ports that violate timing constraints. Keep only the outer point loop (`for(int i = 0; i < NUM_PT_IN_BUFFER; i += UNROLL_FACTOR)`) with `#pragma HLS UNROLL factor=2`, and reduce the cyclic array partitioning factors on `search_tile` and `distance_tile` from factor=2 to factor=4 (coarser grain) to match only the single unrolled loop level. This conservative approach preserves the tiling/buffering strategy while eliminating the cascading unroll that bloated resource usage and destroyed latency.
```

_856 chars._

## Compare

| dimension | deterministic | LLM-aided |
|-----------|---------------|-----------|
| length | 399 chars | 856 chars |
| reads kernel diff? | no | yes (4407 chars in) |
| names specific construct (loop label / pragma) | no (just numbers) | inspect output above |
| cost | 0 LLM calls | 1 small-model LLM call |
| latency | µs | ~1-3s |

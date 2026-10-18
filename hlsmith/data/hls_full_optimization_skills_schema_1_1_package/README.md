# HLS Skills JSON: Coalescing-Extended Version

This package contains an extended `skills.json` for an agentic HLS optimization flow. It keeps the original top-level structure as much as possible:

```json
{
  "saved_at": "...",
  "schema": "1.0",
  "skills": [...]
}
```

The file is intended to remain benchmark-independent. It should not reveal whether hidden tests come from any specific benchmark suite, repository, or optimization sequence.

## What changed

The original skill entries are preserved, but coalescing-related skills were strengthened. In particular:

- `axi-burst-coalescing-narrow-safe`
- `axi-burst-widening-512`
- `prompt-coalescing`

were updated to emphasize that coalescing is not complete after adding AXI pragmas.

New generic coalescing skills were added:

- `hls-coalescing-512-compound-transform`
- `hls-coalescing-compute-lane-parallelism`
- `hls-coalescing-contiguous-access-rewrite`
- `hls-coalescing-lane-parallel-reduction`
- `hls-coalescing-partition-lane-buffers`
- `hls-avoid-coalescing-interface-only`
- `hls-avoid-benchmark-provenance-assumption`

## Main coalescing principle

The agent should understand coalescing as a compound transformation:

1. **Interface/data-movement widening**
   - Use `m_axi`.
   - Prefer `max_widen_bitwidth=512`.
   - Use burst length and outstanding transaction pragmas.

2. **Burst-friendly memory access**
   - Rewrite global memory loops to be contiguous, monotonic, and unit-stride where legal.
   - Separate load, compute, and store when mixed logic blocks burst inference.
   - Stage data into local buffers when useful.

3. **Compute-side exploitation**
   - Widening the memory interface improves memory bandwidth utilization, but it does not directly parallelize computation.
   - To benefit fully, compute should be rewritten around:
     ```cpp
     LANES = 512 / element_bitwidth
     ```
   - The lane loop should be unrolled when iterations are independent.
   - Local lane buffers should be partitioned to feed the unrolled compute lanes.

## Important warning

Adding this pragma is not enough by itself:

```cpp
#pragma HLS INTERFACE m_axi port=A max_widen_bitwidth=512
```

The agent should not mark coalescing as complete unless it also checks:

- whether the access pattern can actually burst,
- whether local staging is needed,
- whether compute remains scalar,
- whether lane-level parallelism can be legally exploited,
- whether tails and boundaries are handled,
- whether synthesis reports confirm widening/burst success.

## Schema compatibility

The top-level structure remains the same:

```json
{
  "saved_at": "...",
  "schema": "1.0",
  "skills": [...]
}
```

The following fields existed in the original skill objects and are still present:

- `applicable_fpgas`
- `applicable_versions`
- `bottleneck_kinds`
- `confidence`
- `id`
- `last_used_at`
- `mean_advantage`
- `occurrences`
- `origin`
- `pattern`
- `sec_pass`
- `strategy`
- `tags`
- `template`

Some new skills also include optional fields:

- `kind`
- `guards`
- `required_steps`

These optional fields are intended to improve agent behavior, but they are not required for backward compatibility.

## How to handle optional fields in the agentic flow

If your current agent loader accepts arbitrary JSON fields, no change is needed.

If your loader is strict, use one of these approaches.

### Option 1: Ignore unknown fields

Keep your existing schema and ignore fields that are not recognized:

```python
KNOWN_FIELDS = {
    "applicable_fpgas",
    "applicable_versions",
    "bottleneck_kinds",
    "confidence",
    "id",
    "last_used_at",
    "mean_advantage",
    "occurrences",
    "origin",
    "pattern",
    "sec_pass",
    "strategy",
    "tags",
    "template",
}

public_skill = {k: v for k, v in skill.items() if k in KNOWN_FIELDS}
```

This preserves backward compatibility.

### Option 2: Use optional fields in retrieval/prompting

For better behavior, use:

- `pattern` for retrieval matching,
- `bottleneck_kinds` for report-based filtering,
- `strategy` for the high-level action,
- `guards` for safety constraints,
- `required_steps` as a checklist,
- `template` only as an example, not as code to copy blindly.

Recommended agent prompt construction:

```text
Skill: {id}
When applicable: {pattern}
Strategy: {strategy}
Required steps: {required_steps}
Guards: {guards}
Template/example: {template}
```

### Option 3: Flatten optional fields into `strategy`

If your agent only consumes `strategy`, merge optional fields before prompting:

```python
def flatten_skill(skill):
    strategy = skill.get("strategy", "")
    if "required_steps" in skill:
        strategy += "\nRequired steps:\n" + "\n".join(f"- {x}" for x in skill["required_steps"])
    if "guards" in skill:
        strategy += "\nGuards:\n" + "\n".join(f"- {x}" for x in skill["guards"])
    skill = dict(skill)
    skill["strategy"] = strategy
    return skill
```

## Recommended retrieval policy for coalescing

When the report indicates memory bandwidth, failed AXI burst inference, narrow global accesses, or underutilized wide memory:

1. Retrieve `hls-coalescing-512-compound-transform`.
2. Retrieve `hls-coalescing-contiguous-access-rewrite`.
3. Retrieve `hls-coalescing-compute-lane-parallelism`.
4. If local port conflicts appear after unrolling, retrieve `hls-coalescing-partition-lane-buffers`.
5. If the computation is a reduction, retrieve `hls-coalescing-lane-parallel-reduction`.
6. Always include `hls-avoid-coalescing-interface-only` as a guardrail.

## Hidden-test / benchmark-leakage policy

The agent-visible skill file should stay neutral. Do not expose:

- benchmark suite names,
- repository names,
- kernel names,
- target variant names,
- step-by-step target sequences,
- one-shot target assumptions.

The agent should optimize based only on:

- current source code,
- synthesis reports,
- compiler errors,
- csim/cosim feedback,
- latency/resource metrics,
- explicit interface constraints.

This makes the agent robust whether the evaluator uses:

- one-shot baseline-to-final tests,
- stepwise optimization tests,
- repair-after-failed-optimization tests,
- or a future test type.

## Notes on 512-bit coalescing

For common scalar types:

- `float`: 512 / 32 = 16 lanes
- `int`: 512 / 32 = 16 lanes
- `double`: 512 / 64 = 8 lanes

The agent should compute lane count from the actual element type and avoid hard-coding `16` unless the element is known to be 32-bit.

## Validation checklist

After applying coalescing, the agent/controller should check:

- Did csim pass?
- Did cosim pass?
- Did synthesis infer wider/better AXI access?
- Did burst length improve?
- Did end-to-end latency improve?
- Did II improve or remain acceptable?
- Did resource usage remain within budget?
- Did tail handling preserve correctness?
- Did the compute loop actually exploit the widened memory movement?


# Tiling Extension

This version extends the coalescing-focused skills with a stronger tiling skill family. The goal is to avoid superficial tiling and teach the agent that tiling is a compound HLS transformation.

## Main tiling principle

Tiling should not mean only:

```text
large array -> local buffer
```

A useful tiling transformation should mean:

```text
identify reusable working set -> reshape loops into tiles -> stage tile locally -> rewrite compute to use local tile coordinates -> exploit reuse and local parallelism
```

Tiling should enable at least one of these: data reuse from local memory, fewer repeated DRAM/global-memory accesses, improved locality, improved pipeline II by replacing global-memory access with local-memory access, inner-loop parallelism over local buffers, local-buffer partitioning/banking, or dataflow/double buffering across tiles.

## New tiling skills

- `hls-tile-1d-reuse-and-compute-restructure`
- `hls-tile-2d-locality-and-halo`
- `hls-tile-compute-inner-parallelism`
- `hls-tile-doublebuffer-load-compute`
- `hls-tile-partition-local-buffers`
- `hls-avoid-superficial-tiling`
- `hls-avoid-tiling-without-reuse`

The existing `prompt-tiling` skill was strengthened so it no longer means only "large array -> tile it." It now emphasizes reuse, local compute restructuring, boundaries, tails, and local parallelism.

The existing `local-axi-staging-for-ii` skill was also strengthened to connect local staging with tiling-aware compute restructuring.

## 1D tiling

Use `hls-tile-1d-reuse-and-compute-restructure` when a 1D loop or flattened array traversal repeatedly accesses a bounded contiguous global-memory region that can be staged locally.

The agent should choose a compile-time tile size, rewrite the loop around tile blocks, load the tile with pipelined unit-stride accesses, rewrite compute indexing from global indices to local tile indices, pipeline the tile compute loop, unroll independent local/lane loops if useful, partition local buffers to support parallel local access, and handle tails safely.

## 2D / halo tiling

Use `hls-tile-2d-locality-and-halo` when the computation operates on a 2D grid, matrix, image, dynamic-programming table, or neighbor/stencil pattern.

The agent should identify tile height and width, identify halo radius, allocate local tile storage including halo, load interior and halo safely, rewrite neighbor accesses to local coordinates, compute only valid output points, and preserve boundary behavior.

## Tiling and compute restructuring

The skill `hls-tile-compute-inner-parallelism` is important because tiling should affect the computation, not only memory movement.

After data is staged locally, the agent should check whether the local tile enables `PIPELINE II=1`, partial `UNROLL`, local-buffer `ARRAY_PARTITION`, reduced global-memory dependence, or more parallel reads/writes from local memory.

If the compute remains scalar and unchanged, the tiling may be superficial.

## Tiling and double buffering

Use `hls-tile-doublebuffer-load-compute` when tiled load and compute phases serialize across tile iterations.

This is state-based, not sequence-based. The agent should not assume a hidden test expects double buffering. It should apply this only when reports or code structure show serialized load/compute phases and potential overlap.

## Avoid rules

### `hls-avoid-superficial-tiling`

Prevents the agent from treating local-buffer copy insertion as completed tiling. Tiling should reduce memory traffic, improve locality, enable parallelism, or improve scheduling.

### `hls-avoid-tiling-without-reuse`

Prevents tiling of pure streaming loops with no reuse or scheduling benefit. For pure streams, coalescing, pipelining, or dataflow may be more appropriate.

## Recommended tiling retrieval policy

When reports or code structure indicate poor locality, repeated global accesses, repeated neighbor access, or memory-bound loops with local reuse:

1. Retrieve `hls-tile-1d-reuse-and-compute-restructure` for 1D/flattened access.
2. Retrieve `hls-tile-2d-locality-and-halo` for grids, images, matrices, DP tables, or stencils.
3. Retrieve `hls-tile-compute-inner-parallelism` after a tile is staged but compute remains serial.
4. Retrieve `hls-tile-partition-local-buffers` if local buffer ports limit II after unrolling.
5. Retrieve `hls-tile-doublebuffer-load-compute` if tiled load/compute phases serialize.
6. Always include `hls-avoid-superficial-tiling` and `hls-avoid-tiling-without-reuse` as guardrails when tiling is proposed.

## Hidden-test / benchmark-leakage reminder

These skills are deliberately neutral. They do not encode benchmark suite names, kernel names, variant names, or assumptions about whether tests are stepwise or one-shot.

The agent should optimize only from current code, synthesis reports, csim/cosim feedback, compiler errors, latency/resource metrics, and explicit interface constraints.

# Pipeline Extension

This version extends the coalescing and tiling package with a stronger pipeline skill family. These skills are benchmark-independent and do not encode hidden test provenance.

## Main pipeline principle

Pipelining is not only `#pragma HLS PIPELINE`. A useful pipeline transformation is:

```text
choose a schedulable hot loop -> request a realistic II -> restructure memory/dependencies/local buffers/body latency -> verify achieved II
```

The agent should distinguish requested II from achieved II and should address the scheduling bottleneck reported by HLS.

## New pipeline skills

- `hls-pipeline-hot-loop-achieve-ii`
- `hls-pipeline-local-compute-after-tiling`
- `hls-pipeline-bank-local-buffers`
- `hls-pipeline-unroll-small-inner-loops`
- `hls-pipeline-stage-global-memory`
- `hls-pipeline-resolve-false-dependence`
- `hls-pipeline-handle-true-recurrence`
- `hls-pipeline-recurrence-with-shift-register`
- `hls-pipeline-realistic-ii-selection`
- `hls-avoid-pipeline-pragma-only`

The existing skills `prompt-pipeline`, `dependence-inter-false-on-accum`, `partition-cyclic-on-port-conflict`, and `local-axi-staging-for-ii` were strengthened to align with this model.

## Pipeline after tiling

Use `hls-pipeline-local-compute-after-tiling` when data is already staged locally but the compute loop over the local tile is still serial or underuses local memory bandwidth.

```text
tiling/local staging creates the working set
pipeline schedules compute over the working set
partitioning/unrolling make the pipeline feedable
```

## Pipeline and local buffer banking

Use `hls-pipeline-bank-local-buffers` when local memory ports block II. Partition the dimension actually accessed in parallel, and avoid complete partitioning of large arrays.

## Pipeline and small inner loops

Use `hls-pipeline-unroll-small-inner-loops` when a pipelined outer loop contains small fixed-bound inner loops that serialize the body. Unroll only when iterations are independent or safely reducible, and partition local arrays as needed.

## Pipeline and global memory

Use `hls-pipeline-stage-global-memory` when direct `m_axi` access prevents low II. Separate load/compute/store when legal, stage bounded data locally, and pipeline the local compute loop.

## False dependence vs true recurrence

Use `hls-pipeline-resolve-false-dependence` only when a dependence is proven false. Use `hls-pipeline-handle-true-recurrence` when the dependence is real. Do not suppress true recurrences with `DEPENDENCE inter false`.

For structured recurrences, use `hls-pipeline-recurrence-with-shift-register`, which describes rolling buffers, previous/current row buffers, or shift-register-like state.

## Realistic II

Use `hls-pipeline-realistic-ii-selection` when II=1 is unrealistic because of true dependency, resource limits, memory-port limits, or arithmetic latency. Optimize for achieved II and end-to-end latency, not just the appearance of `II=1`.

## Avoid rule

`hls-avoid-pipeline-pragma-only` prevents the agent from treating pragma insertion as a completed pipeline transformation. Success requires achieved-II improvement, latency improvement, or actual resolution of the scheduling bottleneck.

## Recommended pipeline retrieval policy

When reports show high interval, non-pipelined hot loops, or II target miss:

1. Retrieve `hls-pipeline-hot-loop-achieve-ii`.
2. If data is already tiled/staged locally, retrieve `hls-pipeline-local-compute-after-tiling`.
3. If local memory ports block II, retrieve `hls-pipeline-bank-local-buffers`.
4. If small inner loops serialize the pipeline body, retrieve `hls-pipeline-unroll-small-inner-loops`.
5. If global memory blocks II, retrieve `hls-pipeline-stage-global-memory`.
6. If dependence blocks II, distinguish false dependence, true recurrence, and structured recurrence.
7. If II=1 is unrealistic, retrieve `hls-pipeline-realistic-ii-selection`.
8. Always include `hls-avoid-pipeline-pragma-only` as a guardrail.

## Hidden-test / benchmark-leakage reminder

The pipeline skills are neutral. They do not encode benchmark names, kernel names, step numbers, or assumptions about whether tests are stepwise or one-shot.


# Remaining Optimization Extensions

This version extends the coalescing, tiling, and pipeline skills with the remaining major HLS optimization families:

- unrolling / parallelization,
- double buffering / DATAFLOW,
- multi-DDR / memory-bank mapping,
- general local array partitioning.

All skills are benchmark-neutral and should not expose hidden test provenance.

## Unrolling / parallelization

Unrolling should not mean only:

```text
add #pragma HLS UNROLL
```

A useful unrolling transformation should mean:

```text
prove independence -> choose unroll factor -> feed lanes with partitioned/banked memory -> validate resources and latency
```

New unrolling skills:

- `hls-unroll-independent-loop`
- `hls-unroll-with-array-partition`
- `hls-unroll-reduction-partial-sums`
- `hls-unroll-independent-tasks-processing-elements`
- `hls-avoid-unroll-memory-bound-loop`
- `hls-avoid-unroll-resource-explosion`

The existing `prompt-unroll` and `avoid-over-unroll-axi-dep` skills were also strengthened.

### Unroll retrieval policy

When the code/report suggests compute-bound latency or arithmetic throughput limits:

1. Retrieve `hls-unroll-independent-loop`.
2. If local memory ports block the unrolled lanes, retrieve `hls-unroll-with-array-partition`.
3. If the loop is a reduction, retrieve `hls-unroll-reduction-partial-sums`.
4. If the workload has independent jobs or work items, retrieve `hls-unroll-independent-tasks-processing-elements`.
5. Always include `hls-avoid-unroll-memory-bound-loop` and `hls-avoid-unroll-resource-explosion` as guardrails.

## Double buffering / DATAFLOW

Double buffering should not mean only:

```text
create two buffers
```

A useful double-buffering transformation should mean:

```text
split load/compute/store -> ping-pong buffers -> overlap stages -> handle prologue/epilogue -> verify actual overlap
```

New double-buffering skills:

- `hls-doublebuffer-load-compute-store`
- `hls-doublebuffer-pingpong-local-buffers`
- `hls-doublebuffer-dataflow-stage-split`
- `hls-doublebuffer-first-last-guards`
- `hls-avoid-doublebuffer-without-overlap`
- `hls-avoid-doublebuffer-memory-overuse`

The existing `prompt-doublebuffer` skill was strengthened.

### Double-buffer retrieval policy

When load, compute, and store phases serialize across tiles or chunks:

1. Retrieve `hls-doublebuffer-load-compute-store`.
2. Retrieve `hls-doublebuffer-pingpong-local-buffers`.
3. If code is not stage-separated, retrieve `hls-doublebuffer-dataflow-stage-split`.
4. Retrieve `hls-doublebuffer-first-last-guards` for startup/drain correctness.
5. Always include `hls-avoid-doublebuffer-without-overlap` and `hls-avoid-doublebuffer-memory-overuse`.

## Multi-DDR / memory-bank mapping

Multi-bank optimization should not mean only:

```text
rename bundles
```

A useful multi-bank transformation should mean:

```text
identify independent high-bandwidth streams -> assign to distinct bundles/banks -> update host/linker mapping if required -> balance traffic
```

New multi-bank skills:

- `hls-multibank-separate-independent-arrays`
- `hls-multibank-balance-memory-traffic`
- `hls-avoid-single-gmem-for-independent-streams`
- `hls-avoid-multibank-without-host-platform-support`

### Multi-bank retrieval policy

When multiple independent high-bandwidth arrays share one memory bundle or memory bank:

1. Retrieve `hls-multibank-separate-independent-arrays`.
2. Retrieve `hls-multibank-balance-memory-traffic`.
3. Include `hls-avoid-single-gmem-for-independent-streams`.
4. Include `hls-avoid-multibank-without-host-platform-support`.

## General partitioning

The new `hls-partition-select-complete-cyclic-block` skill helps the agent choose between complete, cyclic, and block partitioning.

Use it whenever unrolling, pipelining, tiling, or coalescing introduces multiple local-buffer accesses per cycle.

## Final agentic-flow reminder

The agent should optimize only from:

- current source code,
- synthesis reports,
- csim/cosim feedback,
- compiler errors,
- latency/resource metrics,
- explicit interface and platform constraints.

It should not infer benchmark names, kernel names, hidden target variants, or whether evaluation is stepwise or one-shot.


# Schema 1.1 Update: Explicit New Fields

This package updates the skill schema usage so every skill object explicitly includes:

- `kind`: classifies the skill, such as `compound_transformation`, `supporting_transformation`, `transformation_operator`, `diagnostic_or_transform`, or `avoid_rule`.
- `required_steps`: a checklist the agent should follow when applying the skill.
- `guards`: safety constraints and cases where the transformation should not be applied.

The original top-level structure is still preserved:

```json
{
  "saved_at": "...",
  "schema": "1.1",
  "skills": [...]
}
```

Most original fields are unchanged, including:

- `id`
- `pattern`
- `strategy`
- `bottleneck_kinds`
- `confidence`
- `tags`
- `template`
- `origin`
- `mean_advantage`
- `occurrences`
- `last_used_at`
- `sec_pass`
- `applicable_fpgas`
- `applicable_versions`

## Integration guidance

For best results, the agent should retrieve skills based on `bottleneck_kinds`, `pattern`, and `tags`, then include the following fields in the prompt:

```text
Skill: {id}
Kind: {kind}
Pattern: {pattern}
Strategy: {strategy}
Required steps: {required_steps}
Guards: {guards}
Template/example: {template}
```

If the current loader is strict, it can either:

1. ignore the new fields, or
2. flatten `required_steps` and `guards` into the `strategy` text before prompting.

Avoid-rules should be retrieved together with positive transformation skills to prevent shallow or unsafe edits.

"""
Prompts for C-to-HLS translation pipeline.
"""

# System instruction
Instruction_c2hls = """You are an expert in FPGA High-Level Synthesis (HLS) using Xilinx Vitis HLS. Your task is to add HLS pragmas and optimizations to plain C/C++ code to make it synthesizable and efficient on FPGAs.

Key HLS optimization techniques you know:
- Interface pragmas: #pragma HLS INTERFACE (m_axi, s_axilite, ap_ctrl)
- Loop pipelining: #pragma HLS PIPELINE II=N
- Loop unrolling: #pragma HLS UNROLL [factor=N]
- Array partitioning: #pragma HLS ARRAY_PARTITION variable=X [complete|cyclic|block] [factor=N] [dim=D]
- Dataflow: #pragma HLS DATAFLOW
- Inline control: #pragma HLS INLINE [off]

You must preserve the original algorithm's correctness while adding HLS directives.
Always provide complete code in a ```cpp code fence."""

# Phase A: Validate the input C code compiles
q_validate_c_code = """The following C/C++ code is an algorithm kernel. Verify it is complete and correct.
If it has any issues, fix them. The code should compile with g++ -c (no main needed, just the kernel function).

Provide the validated code in a ```cpp code fence.

Here is the code:
```cpp
{c_code}
```

Here is the benchmark-specific guidance:
{benchmark_context}

Here is the header file content:
```cpp
{header_code}
```"""

# Phase B: Translate C to HLS
q_translate_c_to_hls = """Convert the following plain C/C++ kernel code into Xilinx Vitis HLS-optimized code.

Requirements:
1. Use a top-level `workload()` function wrapped in `extern "C" {{ }}`.
   If the input already contains a `workload()` wrapper, preserve and upgrade that wrapper instead of creating a second wrapper.
   If there is no wrapper yet, add one that calls the kernel.
2. Add HLS INTERFACE pragmas to the workload function:
   - `#pragma HLS INTERFACE m_axi port=<ptr> offset=slave bundle=gmem` for pointer arguments
   - `#pragma HLS INTERFACE s_axilite port=<arg> bundle=control` for all arguments
   - `#pragma HLS INTERFACE s_axilite port=return bundle=control`
   IMPORTANT: every s_axilite port MUST share the SAME bundle name (use `bundle=control` for all, including `port=return`).
   Vitis kernel mode rejects `bundle=control_r` or any auto-generated split bundle — it produces
   `[HLS 214-219] s_axilite ports must be bundled into one bundle`. Do not omit the `bundle=control`
   on any s_axilite line; do not introduce a second bundle name.
3. Add performance pragmas to the kernel:
   - `#pragma HLS PIPELINE` on innermost loops where appropriate
   - `#pragma HLS UNROLL` where beneficial for parallelism
   - `#pragma HLS ARRAY_PARTITION` for arrays that need parallel access
4. Keep the original algorithm logic UNCHANGED
5. Include the original header file
6. Do NOT copy or re-declare structs, typedefs, constants, or function prototypes that already exist in the header; include the header once and reuse its declarations
7. The code must be synthesizable with Vitis HLS in vitis kernel flow (Alveo U280 / Virtex UltraScale+ HBM)

Benchmark-specific guidance:
{benchmark_context}

Checklist before returning:
- Include the header exactly once.
- Reuse existing function names and signatures from the plain input when possible.
- Match the exact `workload()` argument order and linkage expected by the testbench when benchmark guidance provides it.
- Preserve the plain-input helper and wrapper structure unless a change is required for valid Vitis HLS pragmas.
- Prefer minimal edits to the plain input over creative rewrites.
- Do not redeclare header-owned structs/types like `bench_args_t`.
- Do not invent undeclared helper arrays or buffers like `l_*`; if a local buffer is needed, declare it and fill it explicitly.
- Keep every `#pragma HLS` inside a function body or loop body, never at global scope.

Here is the header:
```cpp
{header_code}
```

Here is the plain C kernel:
```cpp
{c_code}
```

Provide the complete HLS-optimized code in a ```cpp code fence."""

# Phase B: conservative functional translation for multistep mode.
# This is intentionally narrower than q_translate_c_to_hls. In multistep
# experiments the optimization trajectory should own PIPELINE/UNROLL/
# DATAFLOW/coalescing changes; Phase B should only make the kernel legal,
# callable, and structurally simple.
q_translate_c_to_hls_functional = """Convert the following plain C/C++ kernel code into a FUNCTIONAL Xilinx Vitis HLS kernel.

Goal:
- Produce a synthesizable and testbench-compatible Vitis kernel baseline.
- Preserve the original algorithm and data layout.
- Keep the baseline conservative so later optimization steps can add tiling, pipeline, unroll, double buffering, and coalescing intentionally.

Requirements:
1. Use a top-level `workload()` function wrapped in `extern "C" {{ }}`.
   If the input already contains a `workload()` wrapper, preserve and upgrade that wrapper instead of creating a second wrapper.
   If there is no wrapper yet, add one that calls the kernel.
2. Add only required HLS INTERFACE pragmas to the workload function:
   - `#pragma HLS INTERFACE m_axi port=<ptr> offset=slave bundle=gmem` for pointer arguments
   - `#pragma HLS INTERFACE s_axilite port=<arg> bundle=control` for all arguments
   - `#pragma HLS INTERFACE s_axilite port=return bundle=control`
   IMPORTANT: every s_axilite port MUST share the SAME bundle name (`bundle=control`).
3. Do NOT add performance optimizations in this Phase B baseline:
   - no `#pragma HLS PIPELINE`
   - no `#pragma HLS UNROLL`
   - no `#pragma HLS DATAFLOW`
   - no `#pragma HLS ARRAY_PARTITION`
   - no `ap_uint<512>` wide-bus rewrites or `memcpy_wide_bus_*` helpers
4. Keep helper functions simple and synthesizable. Add `static` / `inline` only when it preserves behavior and helps Vitis specialize small helpers.
5. Include the original header file exactly once.
6. Do NOT copy or re-declare structs, typedefs, constants, or function prototypes that already exist in the header; include the header once and reuse its declarations.
7. The code must be synthesizable with Vitis HLS in vitis kernel flow (Alveo U280 / Virtex UltraScale+ HBM).

Benchmark-specific guidance:
{benchmark_context}

Checklist before returning:
- Include the header exactly once.
- Reuse existing function names and signatures from the plain input when possible.
- Match the exact `workload()` argument order and linkage expected by the testbench when benchmark guidance provides it.
- Preserve the plain-input helper and wrapper structure unless a change is required for valid Vitis HLS pragmas.
- Prefer minimal edits to the plain input over creative rewrites.
- Do not redeclare header-owned structs/types like `bench_args_t`.
- Do not invent undeclared helper arrays or buffers like `l_*`; if a local buffer is needed for correctness, declare it and fill it explicitly.
- Keep every `#pragma HLS` inside a function body, never at global scope.

Here is the header:
```cpp
{header_code}
```

Here is the plain C kernel:
```cpp
{c_code}
```

Provide the complete functional HLS baseline code in a ```cpp code fence."""

# Fix HLS synthesis errors
hls_synthesis_fix = """The HLS code failed synthesis with the following error:

{synth_error}

Here is the current code:
```cpp
{hls_code}
```

{attempt_history}Benchmark-specific guidance:
{benchmark_context}

Repair-specific guidance:
{repair_guidance}

Here is the header:
```cpp
{header_code}
```

Fix the code so it synthesizes successfully with the active configured Vitis HLS target:
{target_context}.
Common issues:
- Variable-length arrays are not supported; use fixed sizes from #defines
- Dynamic memory allocation (malloc/new) is not supported
- Recursive functions are not supported
- All loops should have bounded trip counts
- INTERFACE pragmas must be inside the top-level function

Before returning, verify:
- no header structs/prototypes/macros are duplicated in the source
- every identifier you reference is declared
- every `#pragma HLS` appears inside a function body
- the wrapper remains `workload()` unless the input already defines it differently
- the `workload()` signature and `extern "C"` linkage still match the expected testbench-visible declaration

Before writing the corrected code, in ONE sentence at the top of your reply name (a) the
category of mistake your last attempt made and (b) the smallest specific change you'll make
to fix it. Then provide the corrected code in a ```cpp code fence."""

# Fix synthesis timeout — tells LLM to simplify
hls_synthesis_timeout_fix = """The HLS code TIMED OUT during synthesis (exceeded {timeout}s).
This means the code is too complex for the synthesis tool to handle.

Here is the current code:
```cpp
{hls_code}
```

{attempt_history}Benchmark-specific guidance:
{benchmark_context}

Repair-specific guidance:
{repair_guidance}

Header:
```cpp
{header_code}
```

You MUST simplify the code to reduce synthesis complexity:
1. Remove deeply nested loop structures — flatten where possible
2. Reduce unroll factors (use smaller factors like 2 or 4, not full unrolling)
3. Remove excessive array partitioning (especially complete partitioning of large arrays)
4. Avoid complex dataflow regions with many parallel stages
5. Use simple PIPELINE pragmas on innermost loops only
6. Keep array sizes reasonable — do NOT partition arrays larger than ~256 elements completely
7. Prefer BLOCK or CYCLIC partitioning with small factors over COMPLETE partitioning
8. Prefer preserving the existing helper/kernel structure over inventing new wrapper-side buffering schemes
9. When the plain input already has a valid wrapper with local buffers and copy loops, keep that wrapper shape and only add the minimum pragmas needed

The goal is synthesizable code that completes within a few minutes, NOT maximum performance.

Provide the simplified code in a ```cpp code fence."""

# Pillar 9 / Phase 9: correctness gate — synth passes but csim or cosim fails
hls_correctness_repair_fix = """The HLS code SYNTHESIZES cleanly but FAILS the correctness gate:

  - {gate_name} verdict: FAILED
  - Error / log excerpt:
{gate_error}

The csynth report is fine, so the issue is NOT a synthesis bug — the current
kernel computes the wrong values. The source C behavior or previously accepted
step was correct; your `{step_name}` code broke the algorithm. Common
defects after this kind of optimization:
  1. Loop bounds shifted (off-by-one after tiling/unroll)
  2. Buffer indices stale (doublebuffer/coalescing reordering)
  3. Reduction order changed in a non-associative way
  4. AXI burst widening dropped tail elements (e.g. when total size is
     not a multiple of the burst width)
  5. Pipelined inner loop reads stale values from a parent loop's
     iteration variable

Here is the current (synth-OK but functionally-broken) code:
```cpp
{hls_code}
```

Header:
```cpp
{header_code}
```

{attempt_history}Benchmark-specific guidance:
{benchmark_context}

Fix the correctness defect introduced by your `{step_name}` edit. Keep the
optimization intent (do NOT revert the whole step) but restore byte-equivalent
output to the testbench. If the optimization is fundamentally incompatible
with the testbench's data shape, fall back to a minimal-pragma version of
this step rather than producing wrong values.

Before writing the corrected code, in ONE sentence at the top of your reply
name (a) the specific defect category (loop bounds / indexing / ordering /
buffering / burst-tail) and (b) the smallest specific change you'll make to
fix it. Then provide the corrected code in a ```cpp code fence."""

# Quality-aware post-synthesis repair. Reference/gold reports are deliberately
# not exposed here; the controller keeps them offline for scoring.
hls_quality_repair = """The current HLS code already synthesizes, but the controller flagged a latency, timing, or device-budget issue.

Current HLS code:
```cpp
{hls_code}
```

Current synthesis report:
{current_report}

Controller diagnostics:
{quality_context}

Benchmark-specific guidance:
{benchmark_context}

Quality-repair guidance:
{quality_guidance}

Improve the code while preserving functional behavior, wrapper/signatures, the current benchmark structure, and any existing passing CSim/Cosim behavior.
Priorities:
1. Fix timing/slack/Fmax when requested.
2. Reduce BRAM/FF/LUT/DSP overuse called out above.
3. Do not make latency dramatically worse just to save minor area.
4. Prefer minimal changes such as reducing partition/unroll factors, removing unnecessary complete partitioning, keeping large arrays in memories, and avoiding duplicated logic.
5. Preserve the plain-input helper and workload wrapper structure unless a smaller safe change is enough.
6. Do not change the `workload()` argument order or drop `extern "C"` linkage.

Provide the improved code in a ```cpp code fence."""

# Fix C compilation errors
c_compilation_fix = """The C++ code has compilation errors:

{compile_error}

{attempt_history}Benchmark-specific guidance:
{benchmark_context}

Repair-specific guidance:
{repair_guidance}

Here is the current code:
```cpp
{hls_code}
```

Before writing the corrected code, in ONE sentence at the top of your reply name (a) the
category of mistake your last attempt made and (b) the smallest specific change you'll make
to fix it. Then provide corrected code in a ```cpp code fence.
Do NOT duplicate declarations from the header file; include the header and remove redundant structs/prototypes/macros from the source.
Do NOT invent new undeclared buffers or helper arrays; either declare and initialize them properly or use the existing arrays/signatures from the input.
Preserve the exact `workload()` signature and `extern "C"` linkage expected by the benchmark/testbench."""

# Synthesis report comparison prompt
synthesis_comparison = """Compare the synthesis reports of the generated HLS code vs the ground truth.

Generated code report:
- Latency: {gen_latency} cycles
- BRAM: {gen_bram}
- DSP: {gen_dsp}
- FF: {gen_ff}
- LUT: {gen_lut}
- Fmax: {gen_fmax} MHz

Ground truth report:
- Latency: {gt_latency} cycles
- BRAM: {gt_bram}
- DSP: {gt_dsp}
- FF: {gt_ff}
- LUT: {gt_lut}
- Fmax: {gt_fmax} MHz

Analyze the differences. Is the generated code reasonably optimized compared to the ground truth?
Answer YES if the generated code is within 2x of the ground truth latency and resource usage, NO otherwise.
Start with YES or NO on the first line."""

# End prompt
end_prompt_hls = """The HLS translation and synthesis were successful. Provide the final, clean version of the HLS code in a ```cpp code fence."""

# Prompt to extract just HLS code from a response
extract_hls_prompt = """Extract only the C++ HLS code from the response. Return it in a ```cpp code fence."""

# ============================================================================
# Multi-step optimization prompts
# ============================================================================

# System instruction for incremental optimization (more detailed than base)
Instruction_c2hls_multistep = """You are an expert in FPGA High-Level Synthesis (HLS) using Xilinx Vitis HLS.
You apply HLS optimizations incrementally, one technique at a time, to systematically improve performance.

Key optimization techniques (in typical order):
1. **Tiling**: Buffer data into local arrays to improve memory locality. Separate load/compute/store phases.
2. **Pipeline**: Add `#pragma HLS PIPELINE II=1` to inner loops. Add `#pragma HLS DEPENDENCE` for false dependencies.
3. **Unroll**: Add `#pragma HLS UNROLL factor=N` to parallelize loop iterations.
4. **Double buffering**: Use two sets of buffers and alternate between them to overlap load and compute.
5. **Coalescing**: Use wide memory bus (ap_uint<512>) with burst transfers for higher memory throughput.

Rules:
- Preserve the algorithm's correctness at each step.
- Keep the `extern "C" workload()` wrapper with proper INTERFACE pragmas.
- Each step should build on the previous code, adding ONE optimization technique.
- Always provide complete code in a ```cpp code fence.
- Do NOT add optimizations beyond the one requested."""

# Step-specific optimization prompts
# Each takes {current_code}, {header_code}, and optionally {synth_report}

q_optimize_tiling = """Apply TILING optimization to the following HLS code.

Tiling means:
- Buffer input data from global memory into local arrays (use memcpy or manual loops)
- Separate the code into load(), compute(), store() phases
- Process data in tiles/chunks of a reasonable size (e.g., 256 elements)
- The compute phase should operate on local buffers instead of directly on AXI memory

Keep all existing INTERFACE pragmas. Keep the extern "C" workload() wrapper.

Current synthesis report:
{synth_report}

Header:
```cpp
{header_code}
```

Current HLS code:
```cpp
{current_code}
```

Provide the complete tiling-optimized code in a ```cpp code fence."""

q_optimize_pipeline = """Apply PIPELINE optimization to the following HLS code.

Pipeline means:
- Add `#pragma HLS PIPELINE II=1` to the innermost compute loops
- Add `#pragma HLS ARRAY_PARTITION` on local arrays that need parallel access within the pipeline
- Add `#pragma HLS DEPENDENCE variable=X inter false` where loop-carried dependencies are false
- Add `#pragma HLS LOOP_TRIPCOUNT min=N max=N` for variable-bound loops

Do NOT change the algorithmic structure. Only add pipeline/partition/dependence pragmas.

Current synthesis report:
{synth_report}

Header:
```cpp
{header_code}
```

Current HLS code:
```cpp
{current_code}
```

Provide the complete pipeline-optimized code in a ```cpp code fence."""

q_optimize_unroll = """Apply UNROLL optimization to the following HLS code.

Unroll means:
- Add `#pragma HLS UNROLL factor=N` to inner loops where parallelism is beneficial
- Increase array partitioning factors to match unroll factors
- The unroll factor should be a power of 2 (2, 4, 8) and divide the loop bound evenly
- Focus on the dimension/feature loops that can benefit from data parallelism

Do NOT change the algorithmic structure. Only add unroll pragmas and adjust array partitioning.

Current synthesis report:
{synth_report}

Header:
```cpp
{header_code}
```

Current HLS code:
```cpp
{current_code}
```

Provide the complete unroll-optimized code in a ```cpp code fence."""

q_optimize_doublebuffer = """Apply DOUBLE BUFFERING optimization to the following HLS code.

Double buffering means:
- Create TWO copies of each local buffer (e.g., buffer_A_1 and buffer_A_2)
- In the outer loop, alternate between buffer pairs: when loading into buffer_1, compute from buffer_2, and vice versa
- Use a flag (e.g., `(iteration/tile_size) % 2`) to select which buffer set to use
- This allows the load and compute phases to overlap in time

The load() and compute() functions should accept a flag parameter to select buffers.
Keep all existing pipeline/partition pragmas.

CORRECTNESS REQUIREMENTS — double buffering must NOT change observable output:
1. The total number of compute steps and their input/output mapping must be identical to
   the prior (single-buffer) step. A common mistake is writing the buffer pair swap so that
   the FIRST or LAST iteration is computed from the wrong buffer (off-by-one in the prologue
   or epilogue), which produces zeros / stale data at the boundaries.
2. The standard prologue/epilogue pattern is: outer loop runs N+2 iterations.
     iter 0    : load(0) → A   ; compute(skip)        ; store(skip)
     iter 1    : load(1) → B   ; compute(0) from A    ; store(skip)
     iter 2    : load(2) → A   ; compute(1) from B    ; store(0) ← from compute iter 1
     ...
     iter N+1  : load(skip)    ; compute(skip)        ; store(N-1)
   Use explicit flag arguments to load/compute/store and gate them with `if (flag)` so the
   skipped iterations write nothing. Don't change the algorithm, only the schedule.
3. Output values must be byte-identical to the prior step's output. If you cannot prove this
   from the schedule, prefer the simpler structure that you can prove correct over a more
   aggressive variant.

Current synthesis report:
{synth_report}

Header:
```cpp
{header_code}
```

Current HLS code:
```cpp
{current_code}
```

Provide the complete double-buffer-optimized code in a ```cpp code fence."""

q_optimize_coalescing = """Apply MEMORY COALESCING optimization to the following HLS code.

Memory coalescing means:
- Preserve the existing `workload()` function signature and pointer element types unless the prompt
  explicitly says the active benchmark variant is wide-bus ABI compatible.
- Do NOT include `../../../common/mc.h`, `MARS_WIDE_BUS_TYPE`, or `memcpy_wide_bus_*` helpers in
  the default generated kernel. The stripped-C benchmark input does not guarantee those helpers are
  available, and changing the ABI can break the host/testbench contract.
- Improve burst behavior while keeping the narrow ABI: add/adjust m_axi pragmas such as
  `max_read_burst_length=64`, `max_write_burst_length=64`, `num_read_outstanding=16`, and
  `num_write_outstanding=16` where legal.
- Stage contiguous memory ranges into local buffers, compute locally, and write back in contiguous
  order so Vitis can infer long bursts.
- Add cyclic array partitioning with appropriate factors for local buffers

CRITICAL — every `s_axilite` port (including `port=return`) MUST share the same `bundle=control`.
Vitis kernel mode rejects split bundles with `[HLS 214-219]`. Do NOT introduce `control_r` or any
auto-generated second bundle. Each `#pragma HLS INTERFACE s_axilite ...` line must end with
`bundle=control`.

CORRECTNESS REQUIREMENTS — coalescing changes the memory access schedule, NOT the algorithm:
1. Output values must be byte-identical to the previous (narrow-bus) step. A single off-by-one
   index, missing tail iteration, or reordered store breaks the testbench.
2. Keep all original scalar loop bounds unless you are only tiling/staging internally. If you split
   a range into tiles, keep an explicit scalar tail path for non-divisible bounds.
3. Do not mix narrow loads with wide stores or change array layout visible to the caller.
4. If a skill block mentions a 512-bit wide-bus ABI, treat it as reference knowledge only unless
   the prompt explicitly permits wide ABI for this benchmark variant.

Keep all existing double-buffering, pipeline, and unroll optimizations.

Current synthesis report:
{synth_report}

Header:
```cpp
{header_code}
```

Current HLS code:
```cpp
{current_code}
```

Provide the complete coalescing-optimized code in a ```cpp code fence."""

# Generic "apply optimization X" prompt (for custom step names)
q_optimize_generic = """Apply the following optimization to the HLS code: **{optimization_name}**

{optimization_description}

Keep all existing optimizations and INTERFACE pragmas intact.

Current synthesis report:
{synth_report}

Header:
```cpp
{header_code}
```

Current HLS code:
```cpp
{current_code}
```

Provide the complete optimized code in a ```cpp code fence."""

# --- Phase 3: combo/flash prompts (apply multiple techniques in one shot) ---
#
# The Rodinia-HLS reference trajectory shows that *individual* steps can
# regress PPA (e.g., tiling alone is +4x latency over baseline) — the win
# only materializes once the full combination is in place. The agent's
# revert-on-regression heuristic kills these enabling steps. Combo mode
# sidesteps the issue by asking the LLM to apply a *bundle* of techniques
# in a single rewrite, then synthesizing once at the end of the bundle.
#
# Two bundle modes are exposed via OPTIMIZATION_PROMPTS:
#   - combo_full         : everything at once
#   - combo_progressive  : two checkpoints, structural + parallelization

q_optimize_combo_full = """Apply ALL of the following HLS optimizations to the kernel
in a SINGLE rewrite. Do not produce intermediate "minimum-change" steps —
combine them so the kernel exhibits all of these techniques together:

1. **TILING**: split the iteration space; load tiles into local buffers;
   separate load() / compute() / store() phases.
2. **PIPELINE**: annotate every innermost feasible loop with
   `#pragma HLS pipeline II=1` and the supporting
   `#pragma HLS array_partition` to remove memory-port conflicts.
3. **UNROLL**: apply `#pragma HLS unroll factor=N` (4 or 8 typical) on the
   innermost data-parallel loop, keeping the unroll factor in lockstep
   with the array_partition factor on consumed buffers.
4. **DOUBLEBUFFER**: create two ping-pong copies of the load buffer and
   alternate them across iterations so global-memory load overlaps
   compute. Annotate with `#pragma HLS dataflow` at the workload level
   if applicable.
5. **COALESCING**: improve contiguous AXI burst behavior while preserving
   the public kernel ABI by default. Add legal burst/outstanding m_axi
   pragmas and local staging. Only change pointer types to ap_uint<512>
   or use memcpy_wide_bus_* helpers when the prompt explicitly says the
   active host/testbench variant supports a wide-bus ABI.

Justification: applying these techniques *together* avoids the trap
where an intermediate step (e.g., tiling alone) shows worse PPA in
isolation but is a structural prerequisite for later wins.

Keep the `extern "C"` workload() top function and existing INTERFACE
pragmas (m_axi + s_axilite). Header file is below for reference; do not
modify it. Provide the complete optimized code in a single ```cpp ...```
code fence.

Current synthesis report (for reference; the goal is the *combined*
endpoint, not a small per-technique improvement):
{synth_report}

Header:
```cpp
{header_code}
```

Current HLS code:
```cpp
{current_code}
```
"""


q_optimize_combo_structural = """Apply the STRUCTURAL HLS optimizations
to the kernel in a SINGLE rewrite — these are typically prerequisites
for the parallelization combo:

1. **TILING**: split the iteration space; load tiles into local buffers;
   separate load() / compute() / store() phases.
2. **DOUBLEBUFFER**: create two ping-pong copies of the load buffer and
   alternate them across iterations so global-memory load overlaps
   compute.
3. **DATAFLOW** at the workload level if applicable.

Note: these structural changes alone may *increase* per-call latency
relative to the baseline because the load/compute/store separation
introduces extra control. That is **expected**; the parallelization
combo (next step) will recover and exceed baseline.

Keep `extern "C" workload(...)` and INTERFACE pragmas. Provide the
complete code in one ```cpp ...``` fence.

Current synthesis report:
{synth_report}

Header:
```cpp
{header_code}
```

Current HLS code:
```cpp
{current_code}
```
"""


q_optimize_combo_parallel = """Apply the PARALLELIZATION HLS optimizations
to the (already-structural) kernel in a SINGLE rewrite:

1. **PIPELINE** the innermost feasible loops with `#pragma HLS pipeline II=1`
   plus the supporting `#pragma HLS array_partition` to clear port conflicts.
2. **UNROLL** the innermost data-parallel loop with factor 4–8, keeping
   the array_partition factor in lockstep with the unroll factor.
3. **COALESCING** contiguous AXI accesses while preserving the public ABI
   by default: add legal burst/outstanding m_axi pragmas and local
   staging. Only change pointer types to ap_uint<512> or use
   memcpy_wide_bus_* helpers when the prompt explicitly says the active
   host/testbench variant supports a wide-bus ABI.

Keep `extern "C" workload(...)` and INTERFACE pragmas. Provide the
complete code in one ```cpp ...``` fence.

Current synthesis report:
{synth_report}

Header:
```cpp
{header_code}
```

Current HLS code:
```cpp
{current_code}
```
"""

q_optimize_flash = """FLASH MODE: produce one aggressively optimized HLS design in a SINGLE rewrite.

This is not a step-by-step trajectory. Treat the current code as the
functional starting point and return the best complete endpoint you can
safely synthesize. Use any combination of the following when it naturally
fits the kernel:

1. Preserve the public kernel ABI, `extern "C"` top, and existing
   m_axi/s_axilite INTERFACE pragmas.
2. Pipeline hot inner loops with `#pragma HLS pipeline II=1` only when
   memory ports and loop-carried dependencies can support it.
3. Unroll small data-parallel loops with a bounded factor, usually 2, 4,
   or 8, and match any local-buffer array partitioning to the unroll factor.
4. Use local scalar or array staging for repeated accesses; partition only
   small hot buffers, not large global-size arrays.
5. Add DATAFLOW/load-compute-store structure only when the added buffering
   is simple and the kernel has clear streaming phases.
6. Improve contiguous AXI access with legal burst/outstanding m_axi pragmas
   and local staging while preserving the narrow host/testbench ABI by
   default. Do NOT introduce `ap_uint<512>`, `MARS_WIDE_BUS_TYPE`, or
   `memcpy_wide_bus_*` helpers unless the benchmark context explicitly says
   the active harness supports a wide-bus ABI.
7. Keep AXI pragma values Vitis-legal: burst lengths must be powers of two
   in [1, 256], and m_axi ports sharing the same bundle must use the same
   adapter parameters (`latency`, outstanding counts, burst lengths, etc.).
   If unsure, preserve the existing INTERFACE pragmas exactly.

Prefer a compact, synthesizable implementation over a heroic rewrite. For
small PolyBench/HLSFactory-style kernels, simple pipelining, modest unroll,
and a few local buffers often beat complex tiling/dataflow.

Current synthesis report for the functional starting point:
{synth_report}

Header:
```cpp
{header_code}
```

Current HLS code:
```cpp
{current_code}
```

Provide the complete optimized code in one ```cpp ...``` fence.
"""


# Map step names to prompts
OPTIMIZATION_PROMPTS = {
    "tiling": q_optimize_tiling,
    "pipeline": q_optimize_pipeline,
    "unroll": q_optimize_unroll,
    "doublebuffer": q_optimize_doublebuffer,
    "coalescing": q_optimize_coalescing,
    # Phase 3 combo-step prompts.
    "combo_full": q_optimize_combo_full,
    "combo_structural": q_optimize_combo_structural,
    "combo_parallel": q_optimize_combo_parallel,
    "flash": q_optimize_flash,
}

# Default optimization order (matches rodinia-hls convention)
DEFAULT_OPT_STEPS = ["tiling", "pipeline", "unroll", "doublebuffer", "coalescing"]

# Phase 3 combo orderings.
COMBO_FULL_STEPS = ["combo_full"]
COMBO_PROGRESSIVE_STEPS = ["combo_structural", "combo_parallel"]
FLASH_STEPS = ["flash"]

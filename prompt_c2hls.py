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
3. Add performance pragmas to the kernel:
   - `#pragma HLS PIPELINE` on innermost loops where appropriate
   - `#pragma HLS UNROLL` where beneficial for parallelism
   - `#pragma HLS ARRAY_PARTITION` for arrays that need parallel access
4. Keep the original algorithm logic UNCHANGED
5. Include the original header file
6. Do NOT copy or re-declare structs, typedefs, constants, or function prototypes that already exist in the header; include the header once and reuse its declarations
7. The code must be synthesizable with Vitis HLS targeting Artix-7 (vivado flow)

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

# Fix HLS synthesis errors
hls_synthesis_fix = """The HLS code failed synthesis with the following error:

{synth_error}

Here is the current code:
```cpp
{hls_code}
```

Benchmark-specific guidance:
{benchmark_context}

Repair-specific guidance:
{repair_guidance}

Here is the header:
```cpp
{header_code}
```

Fix the code so it synthesizes successfully with Vitis HLS (vivado flow, Artix-7).
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

Provide the corrected code in a ```cpp code fence."""

# Fix synthesis timeout — tells LLM to simplify
hls_synthesis_timeout_fix = """The HLS code TIMED OUT during synthesis (exceeded {timeout}s).
This means the code is too complex for the synthesis tool to handle.

Here is the current code:
```cpp
{hls_code}
```

Benchmark-specific guidance:
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

# Quality-aware post-synthesis repair
hls_quality_repair = """The current HLS code already synthesizes, but its implementation quality is worse than the validated gold baseline on important metrics.

Current HLS code:
```cpp
{hls_code}
```

Current synthesis report:
{current_report}

Validated gold baseline report:
{ground_truth_report}

Current comparison against the gold baseline:
```json
{comparison_summary}
```

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

Benchmark-specific guidance:
{benchmark_context}

Repair-specific guidance:
{repair_guidance}

Here is the current code:
```cpp
{hls_code}
```

Fix the compilation errors and provide corrected code in a ```cpp code fence.
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
- Change pointer arguments in the workload() function to use wide bus types: `ap_uint<512>*` (or `ap_uint<LARGE_BUS>*`)
- Use wide bus read/write helper functions (memcpy_wide_bus_read_float, memcpy_wide_bus_write_float, etc.)
- Include the wide bus header: `#include "../../../common/mc.h"` (defines LARGE_BUS=512, MARS_WIDE_BUS_TYPE, and provides helper functions)
- Update INTERFACE pragmas to use the wide bus pointer types
- Increase burst lengths where possible (max_read_burst_length=256, max_write_burst_length=256)
- Add cyclic array partitioning with appropriate factors for local buffers

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

# Map step names to prompts
OPTIMIZATION_PROMPTS = {
    "tiling": q_optimize_tiling,
    "pipeline": q_optimize_pipeline,
    "unroll": q_optimize_unroll,
    "doublebuffer": q_optimize_doublebuffer,
    "coalescing": q_optimize_coalescing,
}

# Default optimization order (matches rodinia-hls convention)
DEFAULT_OPT_STEPS = ["tiling", "pipeline", "unroll", "doublebuffer", "coalescing"]

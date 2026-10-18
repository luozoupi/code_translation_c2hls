"""Readable prompt profile for architecture experiments.

Changing this file changes the model input. Keep the named profile and its
content hash with results; the paper profile remains the reproduction default.
"""

_INTERFACE = """Preserve the public interface: function names, argument order, pointer element
types, data layout, and the testbench's expected `extern "C"` linkage. Keep one
`workload()` wrapper; reuse an existing wrapper rather than adding another.
Include the supplied header once and reuse its declarations. Declare and
initialize any new buffers, including live boundaries and tails. Place HLS
pragmas inside functions or loops. Keep all `s_axilite` ports, including return,
in `bundle=control`; do not introduce a second control bundle.
"""

_INPUTS = """
Current synthesis report:
{synth_report}

Header:
```cpp
{header_code}
```

Current implementation:
```cpp
{current_code}
```

Return the complete implementation in one ```cpp code fence."""

Instruction_c2hls = """Help an FPGA architecture researcher turn an algorithmic C/C++ kernel into a
correct, efficient Vitis HLS implementation. Relate proposed changes to memory
traffic, available memory ports, loop dependencies, and operator parallelism.
Use interface, pipeline, unroll, partition, dataflow, and inline directives
where the source and target support them. Preserve the algorithm's behavior.
Return complete compilable source in a ```cpp code fence."""

q_validate_c_code = """Check this kernel as a complete C/C++ translation unit that compiles with
`g++ -c`; it does not need a main function. Fix compilation or completeness
issues while preserving its computation and public interface. Use the supplied
header and benchmark guidance.

Kernel:
```cpp
{c_code}
```

Benchmark guidance:
{benchmark_context}

Header:
```cpp
{header_code}
```

Return the validated source in a ```cpp code fence."""

q_translate_c_to_hls = """Translate this kernel for the Vitis kernel flow on Alveo U280 / Virtex
UltraScale+ HBM. Keep the computation unchanged and prefer small, justified
edits to the existing helpers and wrapper.

""" + _INTERFACE + """
If a wrapper is missing, add `workload()` with `extern "C"` linkage and call
the kernel from it. Give pointer arguments an `m_axi` interface with
`offset=slave bundle=gmem`. Give all arguments and return an `s_axilite`
interface in `bundle=control`. Use pipelining, bounded unrolling, and local
array partitioning where useful and legal.

Benchmark guidance:
{benchmark_context}

Header:
```cpp
{header_code}
```

Algorithmic kernel:
```cpp
{c_code}
```

Return the complete HLS source in a ```cpp code fence."""

q_translate_c_to_hls_functional = """Build a conservative, synthesizable baseline for the Vitis kernel flow on
Alveo U280 / Virtex UltraScale+ HBM. Preserve the algorithm and data layout;
later steps will explore performance transformations.

""" + _INTERFACE + """
If needed, add one `workload()` wrapper with `extern "C"` linkage. Give pointer
arguments `m_axi port=<ptr> offset=slave bundle=gmem`; put each argument and
return on `s_axilite` with `bundle=control`.
Add only the interfaces required for a callable kernel. Do not add PIPELINE,
UNROLL, DATAFLOW, ARRAY_PARTITION, wide-bus pointer types, or wide-copy helpers.
Keep helpers simple; use static/inline only when behavior is preserved.

Benchmark guidance:
{benchmark_context}

Header:
```cpp
{header_code}
```

Algorithmic kernel:
```cpp
{c_code}
```

Return the complete baseline source in a ```cpp code fence."""

hls_synthesis_fix = """Repair this Vitis HLS synthesis failure for the configured target.

Compiler diagnostic:
{synth_error}

Current source:
```cpp
{hls_code}
```

{attempt_history}Benchmark guidance:
{benchmark_context}

Repair guidance:
{repair_guidance}

Header:
```cpp
{header_code}
```

Target:
{target_context}

Use fixed-size storage, bounded loops, and synthesizable helpers. Remove
dynamic allocation and recursion if present.
""" + _INTERFACE + """
Begin with one sentence identifying the defect and the smallest repair.
Then return the complete corrected source in a ```cpp code fence."""

hls_synthesis_timeout_fix = """Synthesis exceeded its {timeout}-second limit. Simplify the implementation so
the tool can finish while preserving the computation and interface.

Current source:
```cpp
{hls_code}
```

{attempt_history}Benchmark guidance:
{benchmark_context}

Repair guidance:
{repair_guidance}

Header:
```cpp
{header_code}
```

Reduce excessive unrolling and partitioning, especially complete partitioning
of arrays larger than about 256 elements. Prefer factors of two or four and
small block/cyclic banks. Simplify deep loop nests and complex dataflow
regions without changing iteration coverage. Preserve existing helpers,
wrapper buffers, copy loops, signatures, and linkage. Simple inner-loop
pipelining is a useful starting point.

Return the complete simplified source in a ```cpp code fence."""

hls_correctness_repair_fix = """The `{gate_name}` output check failed after the `{step_name}` transformation.
Synthesis completed, but the numerical result needs repair. Use the diagnostic
to distinguish loop-bound, indexing, reduction-order, buffering, and burst-tail
errors; a failed check alone does not identify which one occurred.

Diagnostic:
{gate_error}

Current source:
```cpp
{hls_code}
```

Header:
```cpp
{header_code}
```

{attempt_history}Benchmark guidance:
{benchmark_context}

Repair the `{step_name}` edit while retaining its intent where it is legal.
Check prologue/epilogue iterations, initialized boundaries, final-buffer
selection, reduction ordering, and non-divisible tails. Restore the prior
step's observable outputs. If this transformation cannot fit the input shape,
use a simpler version with minimal pragmas.
""" + _INTERFACE + """
Begin with one sentence identifying the specific defect and the smallest
repair, then return the complete corrected source in a ```cpp code fence."""

hls_quality_repair = """Improve this synthesizable design in response to the measured latency,
timing, or resource issue. Preserve the behavior and any passing C/RTL checks.

Current source:
```cpp
{hls_code}
```

Synthesis report:
{current_report}

Controller diagnostics:
{quality_context}

Benchmark guidance:
{benchmark_context}

Repair guidance:
{quality_guidance}

Address requested timing violations first, then the reported BRAM, FF, LUT,
or DSP excess. Prefer small changes: reduce unroll/bank factors, keep large
arrays in memory, and remove duplicated logic. Avoid a large latency cost
for a minor area saving.
""" + _INTERFACE + """
Return the complete revised source in a ```cpp code fence."""

c_compilation_fix = """Fix the C++ compilation error without changing the kernel's computation or
testbench interface.

Compiler diagnostic:
{compile_error}

{attempt_history}Benchmark guidance:
{benchmark_context}

Repair guidance:
{repair_guidance}

Current source:
```cpp
{hls_code}
```

Reuse header-owned types, constants, and prototypes. Remove duplicate
declarations; declare and initialize any needed buffers. Preserve the exact
`workload()` argument order and expected `extern "C"` linkage.
Begin with one sentence explaining the defect and smallest repair, then
return the complete corrected source in a ```cpp code fence."""

synthesis_comparison = """Compare these synthesis estimates with the supplied reference design.

Generated design:
- Latency: {gen_latency} cycles
- BRAM: {gen_bram}
- DSP: {gen_dsp}
- FF: {gen_ff}
- LUT: {gen_lut}
- Fmax: {gen_fmax} MHz

Reference design:
- Latency: {gt_latency} cycles
- BRAM: {gt_bram}
- DSP: {gt_dsp}
- FF: {gt_ff}
- LUT: {gt_lut}
- Fmax: {gt_fmax} MHz

Apply the existing comparison rule: answer YES when latency and resource use
are within twice the reference values, otherwise NO. Start with YES or NO on
the first line, then explain the relevant differences."""

end_prompt_hls = """The translation and synthesis checks passed. Return the final complete HLS
source in a ```cpp code fence."""

extract_hls_prompt = """Extract the complete C++ HLS source from this response. Return only that
source in a ```cpp code fence."""

Instruction_c2hls_multistep = """Work with an FPGA architecture researcher on incremental Vitis HLS
optimization. Apply the requested technique to the current accepted source,
using the compiler report to understand its bottleneck. The usual sequence
considers local tiling, pipelining, unrolling, double buffering, and contiguous
memory transfers. Each step should preserve behavior and build on prior work.
Keep the public interface, `extern "C" workload()` wrapper, and HLS interfaces.
Apply only the requested technique and its required support. A wide-bus
interface is appropriate only when the active harness explicitly permits it.
Return a complete implementation in a ```cpp code fence."""

q_optimize_tiling = """Improve locality by tiling this kernel. Stage data from global memory into
local buffers, compute on those buffers, and write results back. Choose tile
sizes suited to the kernel and available storage; 256 elements is one starting
point. Make load, compute, and store responsibilities clear. Preserve input
coverage, boundaries, and the scalar tail for partial tiles.
""" + _INTERFACE + _INPUTS

q_optimize_pipeline = """Pipeline the compute loops that can support overlapped iterations. Request
II=1 where dependencies and memory ports permit it. Add local-array partitions
needed to supply simultaneous accesses. Mark a dependency false only when the
source establishes that it is false; add trip-count information for variable
bounds when useful. Keep the algorithmic structure and limit edits to pipeline,
partition, dependency, and trip-count directives.
""" + _INTERFACE + _INPUTS

q_optimize_unroll = """Expose data parallelism with bounded unrolling. Use a factor such as two,
four, or eight that divides the loop bound, and match the local memory banks
to concurrent accesses. Focus on independent inner dimensions or feature
loops. Preserve the algorithmic structure; change only unroll directives and
the array partitioning needed to support them.
""" + _INTERFACE + _INPUTS

q_optimize_doublebuffer = """Overlap transfers and computation with two alternating local-buffer sets.
Keep the existing pipeline and partition directives. Let load, compute, and
store select their buffer explicitly, and preserve each tile's input/output
mapping and total number of computations.

For N tiles, one three-stage schedule uses N+2 iterations: iteration zero
loads tile zero; iteration one loads tile one and computes tile zero;
iteration two loads tile two, computes tile one, and stores tile zero.
Drain the final compute/store stages without reading or writing invalid tiles.
Gate skipped stages explicitly. Check initial boundaries, the first/last
tile, and the final output buffer. Preserve the previous step's output bytes;
prefer a simple correct schedule when overlap is difficult to establish.
""" + _INTERFACE + _INPUTS

q_optimize_coalescing = """Improve contiguous AXI transfers while preserving the narrow public ABI.
Stage contiguous ranges into local buffers, compute locally, and write back
in order. Where legal, consider read/write burst lengths of 64, outstanding
counts of 16, and cyclic partitioning of small local buffers. Retain existing
double buffering, pipelining, and unrolling.

Keep caller-visible array layout and scalar iteration coverage unchanged.
Handle partial tiles explicitly and preserve the previous step's output
bytes. Do not mix narrow loads with wide stores. Do not introduce
`../../../common/mc.h`, `MARS_WIDE_BUS_TYPE`, `memcpy_wide_bus_*`, or
`ap_uint<512>` unless the active benchmark context explicitly permits the
wide-bus ABI; examples in a recipe alone do not grant that permission.
""" + _INTERFACE + _INPUTS

q_optimize_generic = """Apply this requested transformation: {optimization_name}.

{optimization_description}

Retain the existing optimizations and interface pragmas. Preserve computation,
public signatures, storage initialization, and iteration coverage.
""" + _INTERFACE + _INPUTS

q_optimize_combo_full = """Explore a combined implementation using tiling, pipelining, unrolling,
double buffering, and contiguous transfers in one rewrite. Stage tiles into
local buffers; pipeline feasible compute loops; match a bounded unroll factor
(often four or eight) to memory banking; and alternate buffers where useful
to overlap load and compute. Use DATAFLOW when the stages support it.

Judge the combined endpoint rather than requiring every intermediate change
to improve latency: local buffering may enable later parallelism. Keep the
header unchanged. Preserve the public narrow ABI, using legal AXI burst and
outstanding settings; use wide pointer types or wide-copy helpers only when
the active host/testbench explicitly supports them.
""" + _INTERFACE + _INPUTS

q_optimize_combo_structural = """Prepare the storage and transfer structure for a later parallelization
step. In one rewrite, tile the iteration space, introduce local load/compute/
store stages, and use alternating buffers and DATAFLOW where applicable.
Keep every iteration's computation, initialization, and final store correct.
Extra control and buffering can increase this checkpoint's latency; record
that cost rather than assuming the structural step improves performance.
""" + _INTERFACE + _INPUTS

q_optimize_combo_parallel = """Add compute parallelism to the existing staged implementation. Pipeline
feasible inner loops and provide enough local memory ports. Unroll independent
work with a bounded factor, typically four or eight, matched to array banking.
Improve contiguous AXI transfers with legal burst/outstanding settings and
local staging. Preserve the narrow public ABI; use wide pointer types or
wide-copy helpers only if the active host/testbench explicitly permits them.
""" + _INTERFACE + _INPUTS

q_optimize_flash = """Propose one complete optimized endpoint from this functional starting point.
Choose the combination of transformations that fits the kernel's bottleneck.
Pipeline hot loops when dependencies and memory ports permit it. Unroll
independent work modestly, usually by two, four, or eight, and match local
banking to simultaneous accesses. Stage reused data in small local buffers.
Use tiled load/compute/store or DATAFLOW only when the added structure and
buffering are justified.

For contiguous AXI accesses, prefer legal burst/outstanding settings and
local staging while retaining the narrow ABI. Do not introduce wide pointer
types, MARS_WIDE_BUS_TYPE, or memcpy_wide_bus_* unless the active benchmark
context explicitly supports them. Burst lengths must be powers of two from
1 to 256. Ports sharing an m_axi bundle must agree on adapter parameters,
including latency, outstanding counts, and burst lengths; preserve existing
settings when unsure. A compact pipeline with modest parallelism is often a
useful design for small PolyBench/HLSFactory kernels.
""" + _INTERFACE + _INPUTS

TEMPLATES = {name: value for name, value in tuple(globals().items())
             if not name.startswith("_") and isinstance(value, str)}

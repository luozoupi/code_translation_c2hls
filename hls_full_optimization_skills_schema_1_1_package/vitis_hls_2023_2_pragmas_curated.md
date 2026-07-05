# Vitis HLS 2023.2 Pragmas — Curated Usage Guide

**Scope:** AMD Vitis HLS **UG1399, 2023.2 English**.  
**Source basis:** AMD/Xilinx Vitis HLS User Guide UG1399 2023.2, especially the `HLS Pragmas` index and each individual pragma page.  
**Important version note:** This file intentionally covers the pragma list present in the **2023.2** UG1399 pragma index. Pragmas that appear in later documentation but not in the 2023.2 pragma index, such as `array_stencil`, are excluded.

Main AMD source:

- <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls>
- HLS pragmas index: <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/HLS-Pragmas>

---

## 1. Mental model

Vitis HLS pragmas are source-level directives that guide how C/C++ is scheduled, bound, partitioned, interfaced, and lowered into RTL. They usually target one or more of these goals:

- **Reduce initiation interval (II)**: start a new loop/function iteration more frequently.
- **Reduce latency**: reduce total cycles from input to output.
- **Increase throughput**: run operations, loop iterations, or tasks concurrently.
- **Control resources**: select BRAM/URAM/LUTRAM/DSP usage or limit replication.
- **Expose memory bandwidth**: partition/reshape arrays or use streams/FIFOs.
- **Define RTL interfaces**: AXI, AXIS, FIFO, register, block-control protocols.
- **Improve reports**: tell HLS loop bounds when C code has variable iteration counts.

A pragma is not a guarantee. Always confirm the result in:

- `csynth.rpt` loop/function latency and II tables
- schedule viewer
- dataflow viewer
- resource utilization report
- C/RTL cosimulation
- implementation timing report when the directive materially changes timing/resource pressure

---

## 2. Fast decision map

| Design goal | First pragmas to try | Usual code region |
|---|---|---|
| One loop iteration per cycle | `pipeline`, sometimes `dependence`, `array_partition` | Innermost compute/load/store loop |
| More parallel loop work | `unroll` + `array_partition`/`array_reshape` | Small fixed-bound loops, reductions, vector lanes |
| Overlap load/compute/store | `dataflow` + `stream` | Top-level kernel body or coarse subfunction region |
| Improve variable-bound reports | `loop_tripcount` | Loops with runtime bounds |
| Reshape nested loops for better scheduling | `loop_flatten`, `loop_merge` | Perfect/semi-perfect loop nests; consecutive loops |
| Reduce memory port bottlenecks | `array_partition`, `array_reshape`, `bind_storage` | Local arrays, buffers, tiles, line buffers |
| Reduce area/resource replication | `allocation`, `bind_op`, `bind_storage`, `inline off` | Hot operators/functions/arrays |
| Control kernel/RTL ports | `interface`, `alias`, `protocol`, `top` | Top function arguments or protocol regions |
| Handle structs/classes | `aggregate`, `disaggregate`, `inline`, `function_instantiate` | Struct ports, class/member-function code |
| Mark unchanged configuration input | `stable` | Config arrays/scalars used during kernel execution |
| Control reset behavior | `reset` | Static/global/state variables |

---

## 3. Full pragma table for Vitis HLS 2023.2

| # | Pragma | Syntax skeleton | Intention | Expected improvement | Cost / risk | Expected place to use | Best verification signal | AMD source |
|---:|---|---|---|---|---|---|---|---|
| 1 | `aggregate` | `#pragma HLS aggregate variable=<var> compact=<none/bit/byte/auto>` | Group struct/class fields into one wider scalar/vector-like object. Used especially for struct arguments or arrays of structs. | Fewer ports/objects; better packed memory layout; cleaner wide transfers. | Can create very wide ports; may hurt timing or make fine-grained field access harder. | Near struct variable/argument declaration or in the relevant scope. | RTL port count/width; interface report; memory layout. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-aggregate> |
| 2 | `alias` | `#pragma HLS alias ports=<p1>,<p2> distance=<int> offset=<int>` | Inform HLS about possible aliasing/offset relation between `m_axi` pointer arguments. | Enables safer scheduling/optimization when pointers may refer to overlapping external memory. | Wrong alias information can produce incorrect hardware behavior. | Top-level pointer arguments mapped to `m_axi`. | Schedule dependencies; memory interface behavior; C/RTL cosim. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-alias> |
| 3 | `allocation` | `#pragma HLS allocation instances=<name> limit=<N> operation/function/core` | Limit number of function/operator/core instances used by HLS. | Area reduction; controlled sharing of expensive operators/functions. | Can increase latency or II because operations serialize through fewer resources. | Function body, loop body, or top region where replication is excessive. | Resource report decreases; schedule/II may increase. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-allocation> |
| 4 | `array_partition` | `#pragma HLS array_partition variable=<array> type=complete/block/cyclic factor=<N> dim=<D>` | Split an array into smaller memories or registers to increase parallel read/write ports. | Higher memory bandwidth; often enables unroll/pipeline II=1. | More BRAM/LUTRAM/registers; top-level `m_axi` arrays cannot be partitioned directly. | Local tile buffers, accumulators, small arrays, dimensions accessed in parallel. | Memory port conflicts disappear; II improves; resource report changes. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-array_partition> |
| 5 | `array_reshape` | `#pragma HLS array_reshape variable=<array> type=complete/block/cyclic factor=<N> dim=<D>` | Reshape array by packing multiple elements into wider words while reducing memory count versus full partitioning. | More data per memory access; can reduce BRAM count compared with partitioning. | Wider datapaths/ports; may not help if access pattern is random or not aligned with reshape. Top-level `m_axi` arrays are not reshaped directly. | Local arrays where parallel accesses are adjacent/regular and full partitioning is too costly. | BRAM count/width; II; memory access schedule. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-array_reshape> |
| 6 | `bind_op` | `#pragma HLS bind_op variable=<var> op=<op> impl=<impl> latency=<N>` | Bind an operation result to a specific implementation/resource, e.g. DSP or fabric implementation and latency. | Resource/timing control; can force DSP use or avoid DSP use; can pipeline expensive ops. | Wrong binding can hurt Fmax, II, or resource balance. DSP multi-op matching can complicate expectations. | On variable receiving result of arithmetic operation. | Operator binding table; resource report; timing estimate. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-bind_op> |
| 7 | `bind_storage` | `#pragma HLS bind_storage variable=<var> type=<RAM/FIFO/ROM> impl=<BRAM/LUTRAM/URAM/...> latency=<N>` | Choose memory/storage type and implementation for arrays/variables/arguments. | Resolve memory-port bottlenecks; move large arrays to URAM/BRAM; reduce LUT/register pressure. | Storage choice can increase latency, reduce ports, or hurt timing. | Local arrays, FIFOs, ROMs, large buffers. | Storage binding table; BRAM/URAM/LUT utilization; II. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-bind_storage> |
| 8 | `cache` | `#pragma HLS cache port=<m_axi_port> lines=<N> depth=<N> ...` | Add cache-like temporary storage for read accesses from an `m_axi` adapter when locality exists. | Lower external memory traffic; better throughput for non-burst or reused reads. | Only useful with locality; extra on-chip memory/control; read-oriented use case. | Top-level `m_axi` read port with repeated/local accesses. | AXI transaction count; dataflow/latency; memory performance. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-cache> |
| 9 | `dataflow` | `#pragma HLS dataflow` | Enable task-level pipelining: functions/loops in the region run concurrently with channels between them. | Overlap load/compute/store; large throughput improvement when stages are balanced. | Possible deadlocks; channel sizing required; canonical coding style matters. | Top function body or region with producer/consumer functions or loops. | Dataflow viewer; channel report; stage II/latency; cosim deadlock-free. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-dataflow> |
| 10 | `dependence` | `#pragma HLS dependence variable=<var> inter/intra true/false distance=<N> type=RAW/WAR/WAW` | Declare or remove loop dependencies that HLS cannot infer accurately. | Can lower II when false dependencies block pipelining. | Very dangerous if used incorrectly; can produce functionally wrong RTL. | Pipelined loops with arrays/pointers where dependence analysis is conservative. | II improves; schedule report removes dependency; C/RTL cosim passes. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-dependence> |
| 11 | `disaggregate` | `#pragma HLS disaggregate variable=<var>` | Split a struct/class into individual fields/elements. | More independent field access; more ports; can improve scheduling. | More ports/signals; may increase interface complexity. | Struct variables or top-level struct arguments when field-level concurrency matters. | RTL ports/fields; schedule; interface report. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-disaggregate> |
| 12 | `expression_balance` | `#pragma HLS expression_balance off` | Control expression tree balancing. HLS normally balances integer expressions to reduce latency. | Lower latency/depth for associative integer arithmetic trees. | Disabled for floating point by default because reassociation can change numerical behavior. Turning off can increase latency. | Arithmetic expressions, reductions, logic trees. | Schedule depth/latency; numerical equivalence checks. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-expression_balance> |
| 13 | `function_instantiate` | `#pragma HLS function_instantiate variable=<const_or_arg>` | Create unique RTL instances of a function based on a variable/constant argument. | Specialization can reduce area or improve performance for calls with different constant behavior. | More generated modules; possible area increase if overused. | Inside reusable functions called multiple times with different constants/template-like parameters. | Function call graph; module count; latency/resource comparison. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-function_instantiate> |
| 14 | `inline` | `#pragma HLS inline` or `#pragma HLS inline off` | Inline function hierarchy into caller, or explicitly prevent inlining. | Inlining can expose optimization and remove call overhead; `off` can preserve sharing/hierarchy. | Inlining can duplicate hardware and increase area; preventing inline can limit optimization. | Small functions in hot loops; helper functions inside dataflow regions; or use `off` for resource sharing/debug hierarchy. | Function hierarchy report; resource duplication; II/latency. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-inline> |
| 15 | `interface` | `#pragma HLS interface mode=<mode> port=<arg> bundle=<name> ...` | Define RTL interface protocol for top-level function arguments and block control. | Correct host/kernel integration; AXI bandwidth; streaming interfaces; register maps. | Wrong interface mode can block cosim/integration or throttle performance. | Top-level kernel function arguments and return/control port. | Interface summary; exported RTL ports; cosim; host integration. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-interface> |
| 16 | `latency` | `#pragma HLS latency min=<N> max=<N>` | Constrain desired min/max latency for a function, loop, or region. | Can guide scheduling toward a latency target or reveal infeasible timing/scheduling. | Cannot force impossible schedules; may increase resource use or warnings. | Function body, loop body, region needing bounded latency. | Latency table; warnings if target cannot be met. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-latency> |
| 17 | `loop_flatten` | `#pragma HLS loop_flatten` or `off` | Flatten nested loops into one loop hierarchy. | Removes loop transition overhead; may improve pipelining across nested loops. | Requires perfect/semi-perfect nests; may not help imperfect nests; can complicate debug. | Inside outer loop of nested loops with regular bounds. | Loop hierarchy; latency reduction; pipeline schedule. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-loop_flatten> |
| 18 | `loop_merge` | `#pragma HLS loop_merge force` | Merge consecutive loops into one loop. | Reduces latency from loop overhead; can improve sharing/locality. | Only legal under rules; different loop bounds or side effects can prevent/complicate merge. | Consecutive loops in same scope with compatible structure. | Loop hierarchy; latency/resource report. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-loop_merge> |
| 19 | `loop_tripcount` | `#pragma HLS loop_tripcount min=<N> max=<N> avg=<N>` | Provide loop iteration estimates for variable-bound loops. | Better latency/performance estimates in reports. | Does **not** affect synthesis, scheduling, or RTL behavior. | Variable-bound loops where reports otherwise show `?`. | Report latency estimates become meaningful. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-loop_tripcount> |
| 20 | `occurrence` | `#pragma HLS occurrence cycle=<N>` | Tell HLS that a region/function call inside a pipeline occurs less frequently than the pipeline II. | Enables resource sharing or less aggressive scheduling for conditional/infrequent work. | Incorrect occurrence rate may cause over-optimistic schedule/resource assumptions. | Conditional code inside pipelined loops. | Schedule/resource sharing; II; cosim. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-occurrence> |
| 21 | `performance` | `#pragma HLS performance target_ti=<N> target_tl=<N>` | Specify high-level throughput interval or latency targets; HLS infers lower-level optimizations. | Faster design-space exploration; target-oriented optimization. | Target is not guaranteed; variable loops need `loop_tripcount`; inferred choices may overuse resources. | Around loop/function regions with desired target throughput/latency. | Whether target II/latency is met in report; inferred directives. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-performance> |
| 22 | `pipeline` | `#pragma HLS pipeline II=<N> style=flp/frp/stp rewind` | Pipeline loop/function so new iteration can start before prior iteration finishes. | Lower II; higher throughput; often most important HLS pragma. | Can increase registers/resources; loop-carried dependencies or memory ports may prevent II target. | Usually innermost compute/load/store loops; sometimes function level. | Loop II in csynth report; dependency/memory-port warnings. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-pipeline> |
| 23 | `protocol` | `#pragma HLS protocol fixed/floating` | Define a protocol region where operations follow strict order and no clock cycles are inserted unless explicit. | Precise I/O/control behavior for protocol-sensitive regions. | Easy to overspecify; can limit scheduling freedom. | Small code regions requiring exact read/write ordering or protocol behavior. | RTL schedule/order; cosim waveform. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-protocol> |
| 24 | `reset` | `#pragma HLS reset variable=<var> off` | Add or remove reset behavior for state variables. | Better initialization semantics; can reduce reset fanout/resources when reset disabled. | Wrong reset policy can cause stale state or simulation/hardware mismatch. | Static/global/state variables. | RTL reset network; simulation after reset; resource/timing. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-reset> |
| 25 | `stable` | `#pragma HLS stable variable=<var>` | Mark input/config data as unchanged during kernel execution. | Avoid unnecessary synchronization/register fanout; in dataflow networks can dramatically improve II. | Must really be stable while kernel runs; only appropriate for configuration-like data. | Top-level scalar/array config inputs; dataflow regions using config arrays. | II improvement; synchronization/channel behavior; cosim. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-stable> |
| 26 | `stream` | `#pragma HLS stream variable=<array> type=fifo/pipo/shared/unsync depth=<N>` | Implement array/channel as streaming FIFO/PIPO/shared channel instead of RAM. | More efficient producer-consumer communication; lower area with shallow FIFO; enables dataflow throughput. | Deadlock risk if depth too small; FIFO only works for sequential access patterns. | Dataflow channels; local arrays produced/consumed sequentially. | Dataflow channel table; FIFO depth; cosim deadlock-free; area. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-stream> |
| 27 | `top` | `#pragma HLS top name=<string>` | Attach a synthesis top name to a function, often useful for C++ member functions/classes. | Build-flow clarity; supports `set_top` with renamed function. | Not a performance optimization by itself. | Inside intended top-level function. | Top function selected correctly; generated RTL top name. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-top> |
| 28 | `unroll` | `#pragma HLS unroll factor=<N> skip_exit_check` | Replicate loop body so multiple iterations run in parallel; full or partial unroll. | Higher throughput/lower latency; exposes vector-style parallelism. | Increases operators/memory ports; full unroll needs compile-time loop bounds; may need partition/reshape. | Small fixed loops, inner reduction/vector loops, per-head/lane loops. | Loop unroll factor in report; resource increase; II/latency improvement. | <https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/pragma-HLS-unroll> |

---

## 4. Practical usage patterns

### 4.1 Loop pipelining: safest default for memory copies

For a simple 2D copy, pipeline the **inner loop** by default:

```cpp
load_A_i: for (int i = 0; i < M; i++) {
    load_A_j: for (int j = 0; j < N; j++) {
#pragma HLS pipeline II=1
        local_A[i][j] = A[i][j];
    }
}
```

Why: this asks for one element per cycle. Pipelining the outer loop can implicitly require many inner-loop operations to overlap and can explode memory-port/resource pressure unless the inner loop is intentionally unrolled and memories are partitioned.

Use outer-loop/function-level pipelining only when you understand the unrolling/flattening/resource consequences.

---

### 4.2 Unroll requires memory bandwidth

Bad pattern:

```cpp
for (int j = 0; j < 16; j++) {
#pragma HLS unroll
    acc += A[j] * B[j];
}
```

This may fail to achieve the expected improvement if `A` and `B` are each one RAM with only 1-2 ports.

Better:

```cpp
#pragma HLS array_partition variable=A complete dim=1
#pragma HLS array_partition variable=B complete dim=1

for (int j = 0; j < 16; j++) {
#pragma HLS unroll
    acc += A[j] * B[j];
}
```

Rule: **unroll compute lanes only after exposing enough memory ports** through `array_partition`, `array_reshape`, wider vector types, or multiple memory banks.

---

### 4.3 Dataflow load/compute/store pattern

```cpp
void top(const data_t *A, data_t *C, int n) {
#pragma HLS interface m_axi port=A bundle=gmem0
#pragma HLS interface m_axi port=C bundle=gmem1
#pragma HLS interface s_axilite port=A
#pragma HLS interface s_axilite port=C
#pragma HLS interface s_axilite port=n
#pragma HLS interface s_axilite port=return

    hls::stream<data_t> a_s;
    hls::stream<data_t> c_s;
#pragma HLS stream variable=a_s depth=32
#pragma HLS stream variable=c_s depth=32

#pragma HLS dataflow
    load(A, a_s, n);
    compute(a_s, c_s, n);
    store(c_s, C, n);
}
```

Check after synthesis:

- Are `load`, `compute`, and `store` concurrent in the dataflow viewer?
- Are channel depths sufficient?
- Does C/RTL cosim pass without deadlock?
- Is the slowest stage determining throughput?

---

### 4.4 `array_partition` vs `array_reshape`

Use `array_partition` when:

- You need independent parallel element accesses.
- The array is small enough to become registers or many small memories.
- You are matching an `unroll factor`.

Use `array_reshape` when:

- You want fewer memories but wider words.
- Accesses are adjacent/regular.
- You want bandwidth without fully exploding the number of banks.

Simple rule:

- **Partition** = more independent banks/ports.
- **Reshape** = wider words, often fewer BRAMs than partitioning.

---

### 4.5 `dependence` is a correctness-sensitive repair

Use only when you can prove the dependence is false.

Example pattern:

```cpp
#pragma HLS dependence variable=buf inter false
for (int i = 0; i < N; i++) {
#pragma HLS pipeline II=1
    ...
}
```

This can fix an II bottleneck from conservative dependence analysis, but if the claimed false dependency is real, the RTL can be wrong while C simulation still looks fine. Always run C/RTL cosimulation and preferably randomized tests.

---

### 4.6 `loop_tripcount` is report-only

Use it for variable-bound loops:

```cpp
for (int i = 0; i < n; i++) {
#pragma HLS loop_tripcount min=1 max=1024 avg=512
#pragma HLS pipeline II=1
    ...
}
```

It improves report estimates. It does **not** optimize the RTL by itself.

---

### 4.7 `stable` for configuration arrays/scalars

Use `stable` when a scalar or array is configuration data that does not change during kernel execution:

```cpp
void kernel(const int cfg[16], data_t *A, data_t *C) {
#pragma HLS stable variable=cfg
#pragma HLS dataflow
    ...
}
```

This is especially useful when the config data is read by dataflow processes and would otherwise introduce unnecessary synchronization.

---

## 5. Recommended first-pass pragma recipe for matrix/attention-style kernels

For matrix multiplication, attention, softmax-like kernels, and tiled operators:

1. **Start with interfaces**
   - Use `interface m_axi` for large arrays.
   - Use `s_axilite` for scalar control arguments and `return`.
   - Use separate `bundle`s when real memory bandwidth exists.

2. **Tile into local buffers**
   - Use `bind_storage` to choose BRAM/URAM/LUTRAM for local tiles.
   - Use `array_partition` on the dimension consumed by unrolled lanes.
   - Use `array_reshape` if the access is vector-like and contiguous.

3. **Pipeline the innermost useful loop**
   - Add `pipeline II=1` to load/store/compute loops.
   - If II fails, inspect dependency and memory-port messages before adding more pragmas.

4. **Unroll only when ports exist**
   - Match `unroll factor` to partition/reshape factor and available multipliers/DSPs.
   - For reductions, expect adder-tree/resource growth.

5. **Use `dataflow` for coarse overlap**
   - Split into load, compute, store functions/loops.
   - Add `stream` channels with explicit depths.
   - Verify with the dataflow viewer and C/RTL cosim.

6. **For variable problem sizes**
   - Use `loop_tripcount` for meaningful reports.
   - Do not expect it to change hardware.

7. **Only use `dependence false` with proof**
   - This is a repair for conservative analysis, not a generic speed pragma.

---

## 6. Common failure patterns and fixes

| Symptom | Likely cause | Pragmas/fixes to check |
|---|---|---|
| `pipeline II=1` not achieved | Memory port conflict | `array_partition`, `array_reshape`, `bind_storage`, separate `m_axi` bundles |
| `pipeline II=1` not achieved | Loop-carried dependency | Rewrite recurrence; use `dependence` only if false dependency is proven |
| Unroll increases area but not speed | Memory not partitioned or external bandwidth bottleneck | Partition local arrays; widen memory; reduce factor |
| Dataflow deadlock in cosim | FIFO depth too small or non-canonical producer/consumer | `stream depth`, dataflow viewer, simplify region |
| BRAM usage too high after dataflow | Default ping-pong buffers too large | `stream variable=... type=fifo depth=...` |
| Timing worsens after unroll/partition | Too much parallel logic/fanout | Reduce unroll factor, pipeline reduction tree, bind ops/storage |
| Reports show `?` latency | Variable loop bounds | `loop_tripcount` |
| Struct interface is awkward/wide | Wrong aggregation/disaggregation choice | `aggregate` or `disaggregate` |
| Config input hurts dataflow II | HLS synchronizes config as data | `stable` |

---

## 7. Minimal examples by category

### Pipeline

```cpp
for (int i = 0; i < N; i++) {
#pragma HLS pipeline II=1
    C[i] = A[i] + B[i];
}
```

### Partial unroll + partition

```cpp
#pragma HLS array_partition variable=A cyclic factor=4 dim=1
#pragma HLS array_partition variable=B cyclic factor=4 dim=1

for (int i = 0; i < N; i++) {
#pragma HLS unroll factor=4
    C[i] = A[i] + B[i];
}
```

### Bind local buffer to URAM

```cpp
data_t tile[4096];
#pragma HLS bind_storage variable=tile type=ram_2p impl=uram
```

### Interface

```cpp
void kernel(data_t *A, data_t *C, int n) {
#pragma HLS interface m_axi port=A bundle=gmem0
#pragma HLS interface m_axi port=C bundle=gmem1
#pragma HLS interface s_axilite port=A
#pragma HLS interface s_axilite port=C
#pragma HLS interface s_axilite port=n
#pragma HLS interface s_axilite port=return
    ...
}
```

### Latency target

```cpp
#pragma HLS latency min=10 max=20
for (int i = 0; i < N; i++) {
    ...
}
```

### Stable config

```cpp
void kernel(const int config[16], data_t *A, data_t *C) {
#pragma HLS stable variable=config
    ...
}
```

---

## 8. Priority order when optimizing

A practical tuning order:

1. Correct C simulation.
2. Define `interface` pragmas.
3. Add `loop_tripcount` for variable loops so reports are readable.
4. Add `pipeline II=1` to inner loops.
5. Fix memory bottlenecks with `array_partition`, `array_reshape`, and `bind_storage`.
6. Add `unroll` where parallelism is intended and memory bandwidth exists.
7. Use `dataflow` to overlap coarse stages.
8. Tune `stream depth` and dataflow channels.
9. Control resources with `allocation`, `bind_op`, and `bind_storage`.
10. Apply specialized directives such as `dependence`, `stable`, `occurrence`, `protocol`, and `reset` only when the report or design semantics justify them.

---

## 9. Checklist before accepting a pragma change

For every pragma change, record:

- Did C simulation still pass?
- Did C/RTL cosimulation pass?
- What changed in latency?
- What changed in II?
- What changed in LUT/FF/BRAM/URAM/DSP?
- Did timing estimate or implementation timing get worse?
- Did the pragma create warnings about unsatisfied constraints?
- Is the performance gain worth the resource cost?


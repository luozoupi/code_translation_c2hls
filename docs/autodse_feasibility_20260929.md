# AutoDSE baseline feasibility — 2026-09-29

**Verdict: relevant baseline, feasible in principle, not ready to run in this environment.** This is a static compatibility investigation; no Merlin installation, container launch, Vitis invocation, model request, or experiment mutation was performed. A public-source checkout exists only at `/tmp/c2hls-autodse-investigation-20260929`.

## Identity and scientific scope

The intended tool is UCLA-VAST **AutoDSE: Enabling Software Programmers to Design Efficient FPGA Accelerators**, Sohrabizadeh, Yu, Gao, and Cong, TODAES 27(4), Article 32 (2022), [DOI 10.1145/3494534](https://doi.org/10.1145/3494534). It is not the unrelated `rockyco/autoDSE` project. I inspected the accessible [author preprint v2](https://arxiv.org/pdf/2009.14381v2), official [repository](https://github.com/UCLA-VAST/AutoDSE), and [tutorial](https://ucla-vast.github.io/AutoDSE/).

The comparison is important precisely because AutoDSE already uses compiler feedback to prioritize bottlenecks. Its Merlin backend can restructure source code to implement architectural choices. We cannot claim that iterative feedback or source transformation alone distinguishes HLSmith. Our testable distinction should be open-ended source proposals plus explicit step objectives and evidence-backed acceptance, compared with AutoDSE's compiler-supported parameter space. Whether this broader freedom improves validated performance per unit search cost remains an experimental question.

The preprint's evaluation uses EC2/VU9P and MachSuite/Rodinia plus vision workloads; its published speedups are not directly comparable to our U280 measurements. Its methodological limitations include dependence on informative cycle breakdowns. These are motivation for a fresh common-backend comparison, not grounds to assume our method wins.

## Pinned implementation and local compatibility

Inspected AutoDSE commit: `bfc37862d1b9c6bf69cc7763ffdcfaa1fabd6c50` (2023-10-23). Preserve this identifier and any compatibility patch diff in a future run manifest.

| Requirement | Evidence / current state | Consequence |
|---|---|---|
| Merlin compiler | No `merlincc`, `merlin_flow`, or Merlin installation found in PATH or searched user/tool roots. The [Merlin README](https://github.com/Xilinx/merlin-compiler) describes CentOS 7.4, GCC 4.9.4, LLVM/Clang 6, Boost 1.67. | Provision and validate an isolated compatible runtime before any comparison. An unsearched installation elsewhere cannot be ruled out. |
| Container/runtime | AutoDSE [Dockerfile](https://github.com/UCLA-VAST/AutoDSE/blob/bfc37862d1b9c6bf69cc7763ffdcfaa1fabd6c50/docker/Dockerfile) starts from `merlin:latest` and uses Python 3.6. Host is Ubuntu 22.04.5. Docker CLI exists, but current-user read/write access to `/var/run/docker.sock` is false. | A CLI alone does not establish runnable Docker or availability of the Merlin image. Native isolated build or an accessible runtime is required; no group/daemon changes were attempted. |
| Generator ABI | Both bundled `ds_generator` files are ELF binaries linked to `librose.so.0` and Boost 1.67; `readelf` reports build-machine RPATHs. | The `OS_compatible` copy does not establish compatibility with this host. Check runtime libraries or use the supplied generated example search space first. |
| Vendor backend | Vitis/Vitis_HLS/Vivado 2023.2 and XRT 2023.2 directories exist under `/mnt/data/luo00466`; local protocol specifies U280 `xcu280-fsvh2892-2L-e`, 3.33 ns. | The documented generic Vitis dependency is not evidence that Merlin's report flow supports 2023.2. A one-design probe must establish this. |
| Reports | [MerlinAnalyzer](https://github.com/UCLA-VAST/AutoDSE/blob/bfc37862d1b9c6bf69cc7763ffdcfaa1fabd6c50/autodse/evaluator/analyzer.py) expects `merlin.log`, `hierarchy.json`, `topo_info.json`, and `perf_est.json` below `.merlin_prj/.../report_merlin/final_report`. | Replacing the HLS command with our raw Vitis runner is insufficient. Reimplementing those reports would be a port, which must be labeled and validated separately. |
| Correctness | The inspected evaluator executes transformation, HLS, and optional bitstream stages. Resource-feasible HLS results can be marked valid without our five-case CPU/C/RTL checks. | Reuse our independent oracle for parent and selected output. Do not interpret AutoDSE's `valid` field as semantic correctness. |

A native Merlin build is possible in principle, so Docker access is not a fundamental algorithmic blocker. However, the official [build file](https://github.com/Xilinx/merlin-compiler/blob/main/trunk/CMakeLists.txt) requires separate LLVM, GCC, Boost, Clang, and ROSE roots, and the [detailed README](https://github.com/Xilinx/merlin-compiler/blob/main/trunk/README.md) assumes a matching dependency bundle and compiler headers. The repository includes a ROSE directory, but this investigation did not establish that its binaries and headers can form a working Ubuntu 22.04/Vitis 2023.2 installation. Merely cloning the public compiler is not a completed tool installation.

An actual raw-Vitis backend port would need (1) a legal design-space generator and implementation of Merlin-level tile/pipeline/parallel semantics, including buffering and partition transformations, (2) stable source-loop identifiers before/after transformation, (3) hierarchy and compute/communication cycle attribution for bottleneck prioritization, (4) consistent resource/latency extraction, and (5) independent semantic validation and common budget accounting. Swapping only `make mcc_estimate` for `vitis_hls` does not provide these. Restricting to direct Vitis pragma edits is a different search space and must be named accordingly.

Source licensing is explicit: AutoDSE has [BSD-3-Clause](https://github.com/UCLA-VAST/AutoDSE/blob/bfc37862d1b9c6bf69cc7763ffdcfaa1fabd6c50/LICENSE); Merlin's [license file](https://github.com/Xilinx/merlin-compiler/blob/main/LICENSE) identifies Apache-2.0. Preserve notices for redistributed source/patches. These do not replace the vendor-tool license terms. This investigation did not test a vendor license checkout.

## Strong benchmark overlap

The upstream `examples/mvt/mvt.c` uses the same PolyBench operations, `double`, and N=120 as our `benchmarks_external/HLSFactory/polybench_float_small/hlsfactory_mvt/plain.cpp`. The local source omits braces around nested single-statement loops; AutoDSE documents a brace requirement. Add braces and the Merlin kernel annotation only in an isolated adapter, preserving dimensions, types, initialization, and arithmetic order. This is the best initial smoke test, after rechecking the parent with the common oracle.

Upstream `examples/gemm-ncubed` and local `benchmarks/gemm_ncubed` both use MachSuite double GEMM, 64×64. Our local source additionally exposes a `workload` wrapper. Fix one common kernel boundary rather than count a wrapper's overhead for just one method. These are static matches, not demonstrated compiled equivalence.

Use MVT and GEMM for compatibility, then one TRMM or stencil case to exercise structural transformation. Do not use the known-broken FDTD harness as a pilot oracle. Do not replace the registered main-study kernels silently; treat these as a separately registered baseline pilot.

## Budget and isolation details that matter

- [Tutorial timeouts](https://ucla-vast.github.io/AutoDSE/) are **minutes**. The sample/generated setting `timeout.exploration=1200` means 20 hours; it is not a 1,200-second pilot. Generator code overwrites defaults, so inspect/freeze the generated config before exploration.
- The sample uses four partitions and two workers per partition. Set one evaluator worker for the first probe and record partition count explicitly. Account for scope-map and partition profiling syntheses as well as search evaluations; the built-in exploration timer is not an end-to-end budget.
- The public config has no explicit global HLS-call limit. A fair synthesis-budget comparison needs an audited external evaluation counter/gate or a documented small patch. Also report cumulative tool CPU time and wall time; model-call equality is not a meaningful resource match to a zero-LLM baseline.
- **Do not use the stock launcher on a shared Redis instance.** `autodse/database.py:250` calls `flushdb()`, the default port is 6379, and the launcher uses host networking. A future run needs its own Redis process/port and data directory or isolated container network. Prefer explicit `python -m autodse` options over the wrapper's stale optional-argument documentation.
- Keep source copies, caches, logs, temporary directories, and scheduler accounting separate from the four current backfill workers. No use of system time manipulation or modification of live tool settings is needed for this investigation.

## Bounded next experiment

1. **Compatibility gate (one engineering day maximum).** Obtain a pinned usable Merlin runtime; test one default MVT design with one worker. Require generated source, correctly mapped scope/latency/resource reports, and the same five-case parent/selected-output checks. Record setup failures explicitly. Do not delay the main controlled study indefinitely for an older compiler stack.
2. **Two-kernel feasibility pilot.** MVT and GEMM; compare plain common parent, Merlin without DSE, and Merlin+AutoDSE (`search.algorithm.name=bottleneck`, fast synthesis mode). The plain and Merlin-only arms separate backend transformations from exploration. Proposed initial cap: ten new HLS evaluations per kernel including scope/profiling, 60 minutes total per kernel, 20 minutes per HLS call, with all timeouts/failures retained. These are engineering limits, not claims of a fully optimized AutoDSE comparison.
3. **Final-selection validation.** Freeze the selected source before independent five-case RTL validation. Re-synthesize extracted generated kernels with the common Vitis 2023.2/U280/3.33 ns harness if interfaces can be faithfully adapted; preserve both native AutoDSE reports and common-harness reports. Distinguish HLS estimates from RTL cycles and routed timing.
4. **Comparison only after feasibility.** Register a larger common HLS-call and wall-time budget with both HLSmith modes, balanced ordering, resource limits, unchanged inputs, and failure denominators. Report anytime quality against both evaluation count and elapsed time. A tiny pilot or different compiler/device does not support a superiority claim.
5. **Fallback if gate fails.** Report AutoDSE as investigated but unavailable under the common stack, publish the exact compatibility evidence, and run the already planned zero-LLM numeric DSE. If we port its algorithm to raw Vitis, label it an AutoDSE-inspired port and compare it separately; do not silently call it the authors' tool.

This adds a useful traditional baseline without changing the existing flash/multistep artifacts or converting failed runs into passes. Reviewer-enabled step-goal checking is a separate ablation; keep its model calls and tool evaluations inside the same budget as the other agent variants.

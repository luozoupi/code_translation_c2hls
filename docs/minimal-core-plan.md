# Minimal HLSmith implementation plan

Revised October 26, 2026, following the request for a small, straightforward
implementation that uses open-source tools where they can perform the required
work. This is a plan; the new runner and simulator backends are not implemented.

## Current position

The canonical checkout remains `hlsmith-clean` at
`/home/luo00466/code_translation_c2hls`. Archive verification, historical version
organization and the first package migration are complete. The package still
contains about 41,000 Python lines, including a 10,071-line controller. Its CLI
delegates to that controller, so the current implementation is not yet minimal.

At October 23, 19:36 UTC (14:36 Chicago), there were no active experiment,
supervisor, observer or Vitis sessions. The saved-design gate finished all 648
scheduled checks, consuming 209.209 of the approved 250 worker-hours including
the initial trial. Of 324 design pairs, 239 passed both evaluators with matching
sources, cycles, resources and timing; 83 pairs had matching failures/timeouts;
two passed the frozen evaluator and timed out in the cleaned evaluator. Five
historical comparator identities remain unresolved. This is not an equivalence
sign-off or complete paper reproduction.

DeepSeek stopped on October 22 at 16:03 UTC: 40 completed cells, one execution
error and 1,597 pending, out of 1,638. Completed means processed, not necessarily
correct. The settled ledger records $0.71905 and 14.417 worker-hours against
the approved $100 and 500-hour caps. Both API calls in the failed cell are
settled. An early generation-failure return omitted usage metadata, causing
the controller to misclassify the outcome as an accounting error. The saved
history contains that metadata. The observer subsequently exited normally
because it treats stopped errors as terminal; it is not currently monitoring.

On October 26 at 15:36 UTC, a separately frozen recovery controller resumed
pending cells without retrying the failed generation. The original failure is
retained and its scientific classification is a generation nonpass. The new
observer keeps watching stopped states and polls every 60 seconds. This closes
the bookkeeping/monitoring work below; the independent minimal runner,
Verilator pilot and LightningSim evaluation remain unimplemented. See
[campaign operations](campaign-operations.md) for receipts and live status.

## Product scope

Continue development on `hlsmith-clean` in the canonical checkout. Preserve a
tagged research baseline and immutable evaluator snapshots before removing
runtime dependencies. Avoid another permanent checkout or parallel product.

The default product has three commands: `c2hls run`, `c2hls validate` and
`c2hls report`. One small controller performs:

1. Load the public C source, interface, test inputs and target settings.
2. Establish the original-C output oracle and check the public interface.
3. Translate through one OpenAI-compatible provider adapter, with bounded
   compiler repair and explicit token, request and time limits.
4. Optionally optimize in one bounded loop, retaining the best candidate that
   actually satisfies the selected validation and resource requirements.
5. Run final independent validation and write a compact result and artifacts.

The new controller must not import or wrap the existing large `pipeline.py`.
Reuse small correctness, budget, process-cleanup and prompt helpers only when
their dependencies remain small. Use a few ordinary tool functions rather than
an agent registry or plugin framework. Report missing capabilities explicitly.

Move dated campaigns, training, router/adapters/GRPO tooling, corpus exports,
ablations and trajectory analysis outside the default installation and CLI.
Preserve their sources and reproduction entrypoints in the research baseline
and archive. Inventory documented CLI/Python interfaces before pruning. Keep
thin aliases for shared core operations in the slim install; preserve complete
legacy behavior in the tagged research environment and document how to invoke
it. The slim runtime must not load the old controller to provide those aliases.
Keep benchmark bytes, licenses, hashes and attribution. Keep both prompt
profiles, with `paper-sigmetrics-2026` remaining the default until the readable
profile passes its separate evaluation.

## Tool responsibilities

| Stage | Intended tool | Evidence it provides |
| --- | --- | --- |
| Original-C and candidate checks | GCC/Clang | Outputs for the recorded inputs and comparator; not RTL correctness |
| C-to-RTL and U280 synthesis estimates | Optional Vitis HLS 2023.2 | Generated RTL, target-specific resource and timing estimates |
| Executed RTL simulation | Verilator, after a compatibility pilot | RTL outputs and cycles under a recorded interface/memory harness |
| Faster latency exploration | Optional LightningSim | Trace-based latency estimates from a specific Vitis solution |
| Physical implementation or board checks | Explicit vendor/board flow when required | Measurements from that actual implementation or board |

Verilator consumes Verilog/SystemVerilog, not C. LightningSim expects Vitis HLS
and a Vitis solution directory. Neither replaces C-to-RTL synthesis or U280
physical implementation. A fully vendor-free fresh C-to-RTL path would require
an additional HLS compiler and a separately validated backend; that is outside
this first minimal release. Already exported portable RTL can be simulated
without rerunning Vitis, provided its dependency closure is available.

Use native checks without vendor tools for supported standard C/C++ inputs.
Identify vendor headers/types and unsupported IP dependencies explicitly.
Keep Vitis out of installation/help/report paths. Retain the existing vendor
simulation backend for frozen paper reproduction and unsupported RTL cases.
The existing `hw_emu` path does not by itself establish placed-and-routed timing.

## Delivery order and acceptance

1. **Recover experiment bookkeeping independently.** Preserve the error,
   history and ledger. Verify the history usage against the settled requests
   and add a correction receipt for the generated-code nonpass. Do not request
   another model response or erase the original error. Use a new execution
   epoch that dispatches pending cells only; keep frozen numerical runtimes,
   prompts, decoding, budgets and selections unchanged. Extend the observer
   to keep watching states that need attention until actual completion or an
   explicit stop. A plain restart is insufficient: the old launcher rejects
   resuming this stopped campaign.
2. **Build the independent minimal runner.** Start with native checks and a
   small optional Vitis synthesis/simulation boundary. Demonstrate one complete
   example, deterministic saved-response replay, required interface checks and
   installation from a clean Git export. The default wheel excludes research
   campaigns and does not import the historical controller.
3. **Pilot Verilator on saved RTL before model-driven use.** Start with simple
   interfaces and an actual AXI-array design. Preserve reset/start/done,
   address maps, depths, packing, burst/backpressure and memory timing. Compare
   the same RTL, inputs, output comparator and cycle-count boundary against
   the vendor simulator. Require matching outputs and no unexplained cycle
   differences; exercise wrong outputs, hangs, missing modules and mismatched
   fixtures. Record unsupported IP instead of replacing it with stubs. Do not
   promise coverage of all 26 kernels before this pilot.
4. **Evaluate LightningSim only if useful.** Pin a tested version and check
   compatibility and estimate accuracy against the actual Vitis 2023.2 saved
   solutions. Keep it optional. Its estimates cannot replace independent RTL
   correctness checks. A changed ranking backend is a new registered study.

Each result records source/RTL/input/harness identities, tool versions, target,
limits and failure status. Keep HLS estimates, executed RTL cycles, trace
estimates and implementation measurements in separate fields. Unsupported or
unmeasured quantities remain unavailable, never zero. Preserve existing frozen
schemas; use an explicit version/adapter for new results.

Use the small deterministic fixture gate before any new paid sweep. Do not
alter the existing 1,638-cell comparison to evaluate the new backend or imply
that it validates later code. New backend measurements and slim-runner results
remain distinct from preserved paper evidence and the current frozen-versus-
clean campaign. Review and explain the two hardware timeout asymmetries while
preserving their original outcomes; register any follow-up separately. Close
the remaining acceptance gaps before default-branch promotion.

## Tool references

- [Verilator overview](https://verilator.org/guide/latest/overview.html)
- [LightningSim requirements and usage](https://github.com/sharc-lab/LightningSim)
- [Paper measurement contract](paper-contract.md)
- [Existing migration and preservation record](migration.md)

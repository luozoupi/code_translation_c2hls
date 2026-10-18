# Queued July routing validation and PolyBench success within K attempts

This is a new prospective queue requested on October 6, 2026. It does not modify
the registered 216-cell recipe study, its continuation, or the historical timeout
replay. The queue waits for both currently active campaigns to finish and their
controllers to exit. It then runs one historical backfill worker and one new
agent-attempt worker, subject to CPU, memory, storage, and unrelated-Vitis checks.

## Where to observe it

- [Canonical queue summary](../results_sweeps/sigmetrics_followup_queue_20261006/summary.json)
- [Canonical queue state](../results_sweeps/sigmetrics_followup_queue_20261006/state.json)
- [Queue registration](../results_sweeps/sigmetrics_followup_queue_20261006/manifest.json)
- [July routing inventory](../results_sweeps/july_routing_cosim_20261006/inventory.csv)
- [Pass-within-K protocol](../results_sweeps/polybench_passk_20261006_v2/manifest.json)
- [Reference wall-time audit](../artifacts/sigmetrics2027/reference_walltime_20261006/report.md)

The child campaigns' original `prepared` inventories are preserved. Their
`queue_registration.json` files point to the coordinator, which owns live task
states and outcomes. Read the canonical summary rather than interpreting an
unchanged child preparation summary as progress.

## July skill-routing backfill

The broader frozen July inventory contains 2,553 historical rows. This follow-up
queues **1,401 rows mapped to 1,139 distinct validation identities**, preserves
643 rows with existing evidence or a registered replay, and retains 330 blocked
and 179 out-of-scope rows in the inventory. Existing failures, timeouts, and
not-run outcomes are not silently retried.

Included experiments cover the corrected routing matrix, Sonnet and Gemma skill
sweeps, earlier Qwen/Gemma routing policies, skill-dose trials, and agentic SFT
comparisons. Preserve campaign/model/mode/policy/repeat columns when analyzing
effects; the pooled queue is not a randomized routing-policy comparison.

Deduplication checks saved selected-kernel hashes, all public input hashes,
toolchain, and the audited output-shape registry. Completed validations from the
earlier HPCA backfill, SIGMETRICS registry backfill, reviewer A/B, and timeout
replays are checked before a task is admitted. All compact inputs are copied into
a separate staging directory and verified. The frozen 62-file validator is
reused; historical generation artifacts and measurements remain unchanged.

Limits: native-oracle/CSim checks precede RTL, CSim is bounded at 180 seconds,
and the candidate's fresh synthesis plus co-simulation invocation is bounded at
3,600 seconds. The direct worker applies no cycle-estimate skip. A failed
independent C-output gate is `failed_csim` with RTL `not_run`, not a fabricated RTL
failure. Harness/input failures and infrastructure errors remain distinguishable.

Generation-reference audit flags remain **336 false and 1,065 unknown** among
queued historical rows. New RTL validation cannot resolve historical reference
leakage or make these development experiments held-out evidence.

## New agent experiment

| Setting | Registered choice |
| --- | --- |
| Model | DeepSeek V4.1 Flash through `deepseek-flash` |
| Benchmark scope | All 28 packaged PolyBench kernels, including Doitgen |
| Agent modes | Flash and dynamic/multistep, reported separately |
| Skill routing | Frozen `smart_exhaustive_v2` policy and skill library |
| Attempt budget | At most five full agent invocations per kernel/mode, including the first |
| Search allowance per invocation | At most five model requests across all roles and five selection syntheses |
| Internal repair-turn limit | Three, separately recorded from the outer K budget |
| Decoding | Temperature 0.7, thinking disabled, fresh context per full invocation |
| Success gate | Feasible selected design, source-attributed independent C and RTL checks, positive measured RTL cycles, complete accounting/reference audit |
| Stop rule | Stop that kernel/mode after its first verified pass |
| Overall size | 56 kernel/mode groups; at most 280 full invocations |
| API accounting | Existing shared $300 cap; conservative additional upper bound $22.16256 |

The API identifier is documented by the provider as V4.1 Flash; aliases do not
guarantee immutable weights. Preserve the requested model, response metadata,
configuration, and run dates. [Provider model documentation](https://api-docs.deepseek.com/quick_start/pricing/).

This measures **observed success within k full agent attempts**, for k=1..5.
It is not the combinatorial IID pass@k estimator over individual completions.
Timeouts consume attempts. Internal repair calls also consume the per-invocation
allowance and cannot reset it. Retries begin from the same frozen public input
with a fresh agent context. A first successful run ends its group; unsuccessful
groups rotate so every untouched kernel/mode gets an initial attempt before
later attempts are prioritized.

All 28 kernels remain in reporting denominators, including any input-blocked
ones. A provider/accounting uncertainty or inconsistent evidence stops the agent
lane rather than silently rerolling a sample or erasing its reserved API cost.
This corpus was used for development and skill mining; it is not held-out
generalization evidence. The experiment is separate from the recipe-representation
study's prompts, parent qualifications, repeats, and estimand.

Public-only staging retains the original `plain.cpp`, header, and public
testbench. It removes expert-variant paths and reduced co-simulation overrides;
`testbench_cosim.cpp` and `gold_kernel_for_cosim.cpp` are not candidate inputs.
Native preflight checks every public testbench without model calls or Vitis runs.

## Wall-time early termination

The new opt-in flag is `C2HLS_COSIM_WALL_TIMEOUT_SECONDS`. Its clock starts at a
flushed Tcl marker immediately before `cosim_design` and ends after that command.
It measures elapsed host time including simulator preparation, elaboration,
RTL execution, and output checking, **excluding preceding synthesis**. It is
not hardware cycles, simulated microseconds, or XSIM-only execution time.

The reference audit found pass/timeout/not-run records for the corpus, but the
older cache does not contain host phase durations. Eight parent configurations
retain separate completed `cosim_design` and XSIM durations; seven share public
source/header/testbench bytes with the target. Validator, wrapper, tool-build,
and logging equivalence has not been established, so they are diagnostic
evidence rather than automatically usable time limits. GramSchmidt's parent
also uses different dimensions. Doitgen has no executed reference co-simulation.

The prospective rule supports `min(10800, max(120, 2 * reference_seconds))` only
for an uncensored, successfully completed reference with matching inputs,
toolchain, and timing scope. **All 28 currently use the explicit 3,600-second
phase fallback.** No duration is derived from kernel cycles, hardware
microseconds, whole-worker times, or a timed-out reference's lower bound.
Updating reference calibration requires a new protocol version before inference.

A separate invocation watchdog adds a 1,200-second synthesis allowance, giving
4,800 seconds at the current fallback. Phase expiry is recorded as a resource
timeout; it terminates only the owned Vitis process tree and permits the next
fresh agent attempt within K. Execution cleanup uncertainty is an error and
halts new agent dispatches.

Full Vitis logs remain on disk. Complete output-dump evidence and postcheck
markers are checked even when the returned diagnostic log is shortened. Oversized
or malformed evidence fails closed. Scratch is archived and verified before
deletion; surviving owned processes prevent scratch removal.

## Preparation revisions and operation

`polybench_passk_20261006` is a preserved, unlaunched preparation revision.
Review found that a bounded log tail could omit earlier output dumps; the queued
`polybench_passk_20261006_v2` fixes full-log evidence handling. No API or Vitis
experiment ran under the earlier revision.

The queue controller persists in tmux. It polls predecessor completion rather
than requesting changes to their resource gates. Interrupted queue tasks never
automatically restart; their saved outcomes, scratch, and any unknown API usage
must be reconciled before a separate recovery attempt is registered.

To inspect without launching an experiment:

```bash
cat results_sweeps/sigmetrics_followup_queue_20261006/summary.json
```

### Registered launch and verification

The controller started at **2026-10-06 16:40:27 CDT** (21:40:27 UTC) in tmux
session `c2hls_sigmetrics_followup_queue_20261006`. The
[launch receipt](../results_sweeps/sigmetrics_followup_queue_20261006/launch_receipt.json)
records the exact frozen command and queue manifest hash:
`0d383358474c09db3c79b31e59ff5f97f29894e3359e60f4bebfcbeeb769a916`.
The queued passK V2 manifest hash is
`b88105d340fafa04b19c693db96282f6766b1c6c0053f6731d2d8c85536c0cb4`.

Startup verification confirmed a live controller,
`waiting_for_existing_campaigns`, 1,139 pending historical validations,
56 pending agent groups, and zero dispatched attempts. Both predecessor
controllers remained alive. No new API or Vitis experiment had started.
Process identity checks must run in the host process view: a restricted sandbox
can read the saved summary while being unable to see the tmux controller PID.

Validation: **119 focused tests passed**, covering scheduling, bounded retries,
historical deduplication, reference timing provenance, phase timing and process
ownership, full-log correctness evidence, API configuration/accounting checks,
and reproducibility metadata. All **28/28 native-input preflights passed**;
both frozen queue and passK source verifications succeeded. Native preflights
do not establish HLS synthesizability or RTL correctness; those remain queued.

# September 30 main-study launch record

The author requested the 216-cell study in parallel with the historical RTL
backfill. The campaign was launched at **11:39:24 CDT (16:39:24 UTC)**. This record
includes the verified launch identities and initial activity.

## Registration and isolation

The study comprises 12 kernels × two providers × three recipe representations
× three independent requests/runs. Each run has at most five model calls and
five selection syntheses. The three arms share the generic HLS prompt and
feedback policy; flat and structured guidance contain the same recipe content.
Flash mode and the disabled QoR reviewer are fixed. This is the representation
study, not the proposed multi-step prerequisite-enforcement experiment.

Each kernel starts from unchanged public C checked on five frozen domain-valid
inputs. ATAX's validated parent is reused. Remaining parents are measured before
any trial for that kernel. Per-kernel recipe selections and initial prompts are
frozen before generated outcomes. Repeats are fresh API calls, not claims of
independent model weights or supported seed control. Alias behavior is recorded.
Failed parent validation leaves all 18 affected cells visible in the denominator.

Two additional workers share parent preparation and eligible study cells. The
historical backfill retains four worker slots and its frozen validator,
3600-second limit, source identities, and saved measurements. A separately
recorded scheduler handover drains current jobs and permits only the registered
main controller's process tree to coexist. Scratch directories and Vitis user
homes are separate for every job. Unexpected external Vitis jobs remain subject
to the historical scheduler's resource policy.

## API cost check

The existing shared ledger limit remains **$300**. Its frozen conservative
per-million-token rates were checked on September 30 against primary pages:

| Model | Input | Cached input | Output | Source |
|---|---:|---:|---:|---|
| `gpt-5.6-luna` | $0.20 | $0.02 | $1.20 | [Official OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna) |
| `deepseek-flash` | $0.30 | $0.006 | $1.20 | [Official DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/) |

DeepSeek figures use peak rates; actual off-peak rates are half. Its current
alias serves V4.1-Flash. The campaign does not assume immutable provider weights.
OpenAI long-context pricing requires a separate request-size bound, enforced
below the 272k-token threshold. No cache discount is assumed in the upper bound.

Five calls with at most 16,384 output tokens each and a 500,000 aggregate input
allowance per run give a full-campaign conservative estimate of **$48.233664**.
That input allowance explicitly scales the one-call pilot's 100,000 limit by
five. Request reservations and actual usage settlement enforce the shared cap;
these are rate-based estimates, not invoices. The prior pilot plus compatibility
probes used an estimated $0.0079645.

## Snapshot-only corrections

The main campaign uses an explicit source amendment. Whole-kernel flash
instructions are made consistent across arms. Call-limit handling retains a
previously evaluated feasible design when the final attempt fails or a local
request allowance stops search. Real API/accounting failures halt new work and
remain visible. Shared production files, the pilot snapshot, and backfill
validation code are not altered by these corrections.

See [runner protocol](sigmetrics_recipe_main_20260930.md) for the final frozen
policy, tests, and reproduction command. The campaign records exact hashes of
its source amendment, input suite, per-kernel parents, selected recipes, and
first prompts. New generated results must pass accounting and RTL-source
identity checks before entering the paper.

## Observed launch

Launched at **September 30, 11:39:24 CDT**:

- Main tmux: `c2hls_sigmetrics_recipe_main_20260930`; controller PID `2634148`.
- Handover tmux: `c2hls_sigmetrics_cosim_coexist_20260930`.
- Main registration SHA-256: `643db88bffa116a976080ad134c461e6ccdf41bae06bdcc4b7ea4c7ee6432c4a`.
- Frozen main runner SHA-256: `486db1ee738688822283ccf1b4276ba67cff21ca3c5d5178f28aebf9ea70c541`.
- Handover request SHA-256: `09c6380f9e38d92cbb6319cf81d0960f407e2a7d3fdb8f05de441295c869bfc0`.

At **11:40:30 CDT**, the main study has one active cell (ATAX, DeepSeek,
structured guidance, repeat 1), one active parent measurement (GEMM), ten pending
parents, and the qualified ATAX parent. Two provider requests have completed
and settled, estimated at $0.003171; no cells have completed yet. Vitis processes
for the new study were observed under its authenticated controller, with no
live unrecognized Vitis processes in the immediate isolation check.

The backfill is at **387/405 completed: 276 pass, 24 fail, 87 timeout**, with
four active and 14 queued. Only its controller received SIGTERM after checking
PID start ticks; active validators were not signaled. New backfill dispatch is
temporarily paused while those jobs drain. The persistent handover wrapper
waits for their normal completion, verifies preserved evidence, and then resumes
four backfill workers with the registered main campaign exempted from the
external-Vitis check. The waiting wrapper itself does not count as a Vitis worker.

The main and coexistence focused suites passed (55 and 32 tests respectively),
including candidate retention, exact prompt checks, accounting failures,
PID reuse, process ancestry, and progress reporting. All six ATAX initial
prompts passed offline checks before any provider call.

[Launch receipt](../results_sweeps/sigmetrics_recipe_main_20260930/launch_receipt.json)
and [initial combined snapshot](../artifacts/sigmetrics2027/status/20260930T164030435017Z/report.md)
record the observed state. Refresh with `python3 scripts/snapshot_sigmetrics_progress.py`.
The controller performs per-cell reconciliation and halts new submissions on
execution/accounting failure. This launch is not a completed 216-cell result.


At **11:43 CDT**, the first main cell had completed all five provider requests,
with all five ledger entries settled and an estimated $0.007398684 total. The
cell and GEMM parent validation were still running; no completed main-cell
result was claimed. All 248 frozen manifest files matched their registered hashes.
[Updated combined snapshot](../artifacts/sigmetrics2027/status/20260930T164301619813Z/report.md).

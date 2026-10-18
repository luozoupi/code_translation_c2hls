# Multi-step goal verification audit — 29 September 2026

This is a read-only audit of the current SIGMETRICS worktree controller. No experiment, result, production code, or campaign configuration was modified. Current implementation evidence must not be retroactively assigned to older artifacts without matching their recorded source version and environment.

## Finding

Multi-step currently provides intermediate compilation, exercised-input correctness checks, feasibility filtering, optional QoR review, and accepted-state propagation. It does **not** establish that each step accomplished its stated architectural objective. The word `completed` in dynamic routing means a technique was visited, including a failed visit; it does not certify successful source promotion or goal completion.

The scientifically useful distinction is rewrite granularity and state progression: flash requests a complete endpoint rewrite, whereas multi-step applies separately targeted transformations to the currently accepted design. Both use the same evaluation and optional review machinery. Flash can itself contain adaptive repair/improvement calls; it is not necessarily one LLM call or an independent sample at every attempt.

## Implementation evidence

| Question | Evidence in current worktree | Consequence |
|---|---|---|
| What defines a multi-step edit? | `prompt_c2hls.py:352–367`, step prompts at `372–554`; default order at `745` | Natural-language technique request, not a typed architectural goal with a pass predicate. |
| What defines flash? | `prompt_c2hls.py:682–727`; `c2hls.py:8064–8087` | One named endpoint-rewrite stage; dynamic technique routing is bypassed. |
| Are flash prompts internally consistent? | All stages use the multi-step system instruction at `c2hls.py:7023–7025`, whose lines `prompt_c2hls.py:365–367` prohibit multiple techniques | Flash’s combination-permitting user instruction conflicts with the common single-technique system instruction. Freeze a mode-specific fix in a new cohort. |
| What is tested per candidate? | `c2hls.py:5267–5335`, `7349–7443` | CSim with supplied testbench; synthesis; co-simulation only when configured/required and not selected-only; feasibility checks in the reference-blind profile. Never equate CSim with universal semantic correctness or per-step RTL validation. |
| Is the reviewer mandatory? | `qor_profile_review.py:19–20`; invocation at `c2hls.py:7210–7219` | `C2HLS_QOR_PROFILE_REVIEW` defaults off; inspect each run’s fingerprint. |
| What does the reviewer verify? | `qor_profile_review.py:89–156` | Deterministic aggregate metric deltas, source-family evidence and bottleneck persistence, not an independent LLM or a proof of architectural intent. Records explicitly say recipe correctness is not proven and per-skill causality is not isolated. |
| Does failed intent block acceptance? | Reviewer retry condition `qor_profile_review.py:134–135`; outer retry/acceptance `c2hls.py:6127–6196` | Retry is recommended for some feasible unchanged/regressed candidates. A faster candidate can be accepted even when the intended bottleneck persists or declared skill lacks new source evidence. |
| Does pragma presence prove effect? | `skill_usage.py:90–159`, called at `c2hls.py:7079–7088` | `verified_applied_skill_ids` means a newly observed source-family marker. This separate marker detector also scans comments/strings; the QoR reviewer strips them. Neither establishes scheduled hardware effect. |
| What does no-op rejection detect? | `c2hls.py:1793–1827`, `6136–6175` | Equality of an aggregate latency/interval/resource tuple, with sufficient populated fields. Any unrelated resource change can defeat this heuristic; equal aggregates do not prove no structural change. |
| Is every accepted step faster? | `c2hls.py:1620–1643`, `6188–6196`, best-state selection `7712–7745`, `8350–8372` | Reference-blind mode disables reference-calibrated relative regression thresholds; feasible enabling transformations may be retained. End-of-run best-state selection limits endpoint degradation. |
| Are failed steps retried later? | `c2hls.py:8201–8203` | After the bounded attempt path, the step is appended to `completed` and removed from `available` even on ordinary failure. There is no separate achieved/blocked/unknown goal ledger. |
| Does failure overwrite the incumbent? | Deferred commit `c2hls.py:6533–6536`; success-only commit `6188–6196`; rollback `6313–6327` | No: route visitation and source promotion are distinct. A failed normal step retains the prior accepted design. |
| Can legacy control consult an expert trajectory? | `c2hls.py:6261–6307`, gated initialization `4467–4471` | Optional GT-aware enabling-regression logic exists. Keep disabled in the reference-blind comparison and stratify historical runs by provenance. |
| Is scope-resolved evidence available? | `hls_feedback.py:475–523`, `735–751`, `915–1000` | Existing parsing exposes loop IDs, achieved PipelineII/depth/trip count and AXI burst inference. These are useful building blocks for verifiers; missing/raw diagnostics remain unknown. |

## Deterministic sanity checks

A tool-free synthetic-report check reproduced the main limitation: parent cycles 100 and candidate cycles 80, with the same `loop_A` II=4 bottleneck and no new PIPELINE evidence, yielded `latency_effect=improved`, `retry_recommended=false`, `unsupported_declarations=[pipe]`, and `profile_effect=target_bottleneck_still_reported`. This establishes reviewer behavior, not a rate in real experiments.

A second synthetic check placed `#pragma HLS PIPELINE II=1` only inside a source comment: the separate legacy `verify_declared_skill_usage` nevertheless returned `verified_applied_skill_ids=[pipe]`. The newer QoR reviewer strips comments and would flag absent real source evidence. Rename/qualify legacy marker credit in analysis and fix its detector before any new enforcement cohort.

## Precise paper wording

“Flash requests a complete endpoint rewrite. Candidate branches begin at the same accepted parent; configured repair or improvement attempts may revise a preceding candidate using compiler feedback. Multi-step instead carries accepted transformations into subsequent, separately targeted stages. Both modes share the correctness, feasibility, and candidate-selection policy.”

Avoid calling all flash attempts independent. Separate branch roots reuse `self.hls_code` and `self.synth_report` (`c2hls.py:6426–6440`, `6558–6562`), but within-branch improvement uses the previous candidate source/report (`7502–7542`), outer retries include earlier failure evidence (`6113–6134`), and reviewer-on can inject the latest review across branches (`6576–6581`). Historical independence needs inspection of the frozen source and actual messages.

“Completed step” should become “visited stage” in descriptions of the present scheduler. Reserve “goal achieved” for a recorded predicate over a defined scope and evidence.

## Minimal step-goal contract to implement in a separate revision

Before generation, freeze a `StepGoal` containing: goal ID/schema version; parent source/report hashes; targeted loop/array/interface scope and source mapping; bottleneck observation; intended transformation; measurable predicate; safety/ABI invariants; tool/part/clock/shape identity; evidence types; and a maximum retry count within the run’s existing budget. The target should be achievable given known recurrence and memory constraints; it is not always II=1.

After evaluation, attach `StepReview`: candidate and raw evidence hashes, scope mapping status, before/after observations, `achieved | not_achieved | unknown | not_applicable | blocked_by_constraint`, functional-check status, timing/device feasibility, endpoint latency/resource changes, and the controller action. `unknown` is required when diagnostics are missing or scopes cannot be matched; absence of a warning is not proof of resolution.

Keep three decisions separate:

1. Functional and physical eligibility: no mismatch, valid required checks, target timing and device fit.
2. Architectural goal status: did the measured scoped predicate hold?
3. Search-state promotion: retain the candidate, retry the same goal, explore an enabling state, defer a dependent step, or roll back.

A correct faster design whose intended goal failed can remain a useful endpoint candidate, but must not be credited as accomplishing that goal. It must not unlock steps whose prerequisites depend on the unproven goal. Track the best feasible endpoint separately from the active exploration state. A successful structural enabler may increase aggregate cycles; do not demand monotonic endpoint speedup from every step. Handle `not_applicable`/`blocked_by_constraint` explicitly instead of repeating an impossible optimization.

| Goal | Sufficiently concrete initial predicate | Insufficient evidence |
|---|---|---|
| Pipeline a hot loop | Stable scope correspondence; achieved loop II meets a preregistered feasible target; schedule/recurrence and port diagnostics preserved | PIPELINE pragma or improved top-level cycle count alone |
| Relieve a banking bottleneck | Intended array mapping/bank count plus compatible accesses per bank; targeted port-conflict/schedule bound changes as predicted | ARRAY_PARTITION pragma; fewer warnings without scope match |
| Increase useful compute parallelism | Target loop’s effective unroll/issued work and achieved II, plus operation/bank evidence and resource cost | More DSPs alone; UNROLL pragma alone |
| Form local tiled storage | Access/source mapping demonstrates target compute uses bounded local tiles; generated memory/interface diagnostics support the intended traffic change | Variable names containing `tile`; guessed external bandwidth |
| Coalesce external accesses | Interface width and burst inference for the named array/bundle under the fixed ABI; report before/after inferred transaction properties | `ap_uint` or `max_widen_bitwidth` text alone |
| Overlap load and compute | Process/channel schedule or dataflow/RTL trace demonstrating overlap; correct ping-pong ownership, fill/drain and boundary behavior | Two buffers, DATAFLOW text or lower estimated cycles alone |

Implement cheap deterministic checks first, initially in offline/shadow mode on archived source/report pairs. Human-audit a stratified sample of positive, negative, and unknown cases. Only then enable control-flow gating. Optional LLM commentary may help interpret evidence, but must not set factual predicates when compiler evidence is absent. Preserve every rejected candidate and review; never rewrite old run outcomes.

Useful focused tests before launching: faster overall candidate with unchanged targeted II; identical pragma text on a different loop; wrong-scope/renamed loop; missing report returns unknown; already-present bank structure; a legal enabler with temporary latency increase; dependent step blocked until prerequisite is achieved; retry exhausting the total budget; rollback restoring all incumbent source/report/CSim/co-sim hashes; flash receiving the correct system instruction.

## Evaluation implications

The six-cell SIGMETRICS pilot and planned 216-request same-content study are flash-only with QoR reviewer disabled. They cannot establish a mode effect or an effect of verified step goals. Preserve that registered study as a representation comparison; create a separate registered extension.

A minimal treatment matrix is mode (flash, multi-step) × goal-review control (off, on), with a fixed recipe representation, common frozen parent, toolchain/shape/clock, total LLM-generation and synthesis caps, explicit timeouts/retry policy, independent API repeats, and selected-source RTL validation. The review-on condition should expose equivalent endpoint-goal evidence to flash; otherwise an observed benefit mixes mode with information. Holding nominal caps equal is not enough: report actual calls, failed attempts, token/context lengths, tool process time and wall time. Deterministic review adds no model calls; any LLM review calls count toward the declared budget or a separately identical allowance.

The existing September 25 campaign already crosses flash/dynamic with reviewer-off/on over 27 kernels and three repeats (`artifacts/experiment_matrix/deepseek_flash_reviewer_ab27_20260925.json`, `scripts/run_qor_review_ab.py:30–65`). It is a valuable audit source but tests a **bundle** of review feedback and hardware-tradeoff prompt guidance, uses two within-branch flash attempts versus one dynamic attempt, and explicitly disables co-simulation. It does not implement this new scoped goal contract. Separate bundled legacy review, new goal verification, and mode effects in all claims.

Primary measures: fraction of registered runs yielding an RTL-passing selected design; fraction improving the frozen parent; paired eligible speedups with cluster uncertainty; best feasible estimated cycles versus consumed synthesis/call/time budget; architectural-goal achievement/unknown rates; accepted-but-unachieved steps; resource/timing tradeoffs; and attribution of repair failures. Checkpoint curves need checkpoint RTL validation to be called RTL-validated curves. A selected-parent fallback is a valid outcome but not a generated improvement.

First perform the archived audit with no paid calls: pair matching parents and fingerprints; reconstruct every accepted/rejected edge; classify intended step, source evidence, compiler evidence, scoped outcome, eligibility and promotion; report missing evidence explicitly. Follow with 4–6 stratified kernels for verifier calibration, then a preregistered wider comparison. Do not choose only successful historical trajectories.

## Historical transcript confirmation: September 25 reviewer campaign

The prompt inconsistency is confirmed in saved actual histories for the ATAX repeat-1 flash cells in **both** reviewer arms. The adjacent system message says “adding ONE optimization technique” and prohibits optimizations beyond the request; the user message begins “FLASH MODE” and permits “any combination.” These are saved conversation messages, not an inference from today’s source:

- `hlsfactory_atax_flash_r1_review_off`: `/mnt/data2/luo00466/c2hls_rl/deepseek_flash_reviewer_ab27_20260925/results/agentic_no_streamcluster_deepseek_flash_reviewer_ab27_20260925_hlsfactory_atax_flash_r1_review_off/hlsfactory_atax_deepseek_flash_matched/hlsfactory_atax_history.json`, adjacent zero-based JSON `messages[0]` (system) and `messages[1]` (user). History SHA-256 `4860a15f9534d6d895b000e89109f9875d7eae928c45a950695b2e184ee0a468`.
- `hlsfactory_atax_flash_r1_review_on`: `/mnt/data2/luo00466/c2hls_rl/deepseek_flash_reviewer_ab27_20260925/results/agentic_no_streamcluster_deepseek_flash_reviewer_ab27_20260925_hlsfactory_atax_flash_r1_review_on/hlsfactory_atax_deepseek_flash_matched/hlsfactory_atax_history.json`, adjacent zero-based JSON `messages[0]` (system) and `messages[1]` (user). History SHA-256 `e75de18549ee40a1903447fc794c2df38459ee94bc3218455208201a30e5c3d8`.

The same files also retain flash requests in `event_history`. This establishes the inconsistency for these inspected historical cells. It does not quantify its performance impact or establish that every historical cohort shared it. Treat it as a treatment-definition caveat when using this campaign for mode comparisons; it is common to both inspected reviewer arms, so it is not by itself evidence of differential reviewer-arm contamination. Correct and freeze mode-specific system prompts for a new experiment version.

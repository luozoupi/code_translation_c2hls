# HPCA 2027 evidence readiness

This document separates evaluation infrastructure from measured paper
evidence.  An implementation marked ready below is not a result, and no
unmeasured field may be filled from a prediction or a legacy run.

## Readiness snapshot

| Evidence item | Status | Paper use |
|---|---|---|
| Independent CPU-golden comparison with hash-bound HLSFactory shapes, reference-isolation audit, immutable per-call run fingerprint, frozen-skill controls, complete candidate events, typed result statuses, matched budgets, and selected-winner cosim binding | Implemented in the isolated `hpca2027-paper-eval` worktree | Infrastructure only until a fingerprint-complete campaign is run |
| Reference-isolated Rodinia-HLS baseline/expert audit | Not yet measured | The expert-recovery claim is blocked until at least eight valid pairs pass the gate below |
| Nine-kernel reference-blind method/model matrix and fixed-seed repeats | Not yet measured | No performance, solve-rate, cost, or model-comparison claim is currently supported |
| Three-kernel reference-guided oracle ablation | Not yet measured | May be reported only as an explicitly labelled upper bound |
| Frozen-skill transfer benefit | Not yet measured | Retain the claim only if the paired bootstrap interval improves without reducing correct-solve rate |
| Compact-model enablement | Not yet measured | Retain the claim only if dynamic C2HLS beats matched-budget one-shot on the reference-isolated performance profile |
| Post-route implementation of five stratified winners | Not started | Use `paper_eval/post_route_implementation_manifest.json`; never substitute synthesis estimates for implementation evidence |
| Physical U280 runtime | Unavailable in the current environment | Report as unavailable unless device access changes and an independently fingerprinted board campaign is completed |

The implemented controls still require the focused test suite, matrix
preflight, frozen skill snapshot, immutable model revisions, and toolchain
identity to pass immediately before launch.  Generated paper tables and
figures must come from accepted artifacts, not from this status document.

## Active HLSFactory queue is exploratory

A read-only observation at `2026-07-13T11:07:50-05:00` found the tmux queue
`c2hls_matrix_goldprecheck_full28_20260710` still active.  Its Qwen 27B,
flash, skill-off arm had completed all 28 rows (27 successful and one failed
at Doitgen).  The next, skill-on arm had completed `2mm` and was synthesizing
`3mm`.  The queue script contains 12 full-28 arms in total.

These artifacts are not HPCA headline evidence.  Their HLSFactory
testbenches dump computed values but do not independently compare them with a
CPU-golden output.  The active command also uses legacy
gold-relative cosim prechecks (`C2HLS_COSIM_SKIP_SLOWER_THAN_GOLD=1`, ratio
10), trusted external reference validation, and persistent skills.  For
example, a visible row records a predicted-timeout skip from the expert-cycle
ratio rather than an executed RTL measurement.  Consequently, a successful
legacy CSim status is not a golden correctness proof, and a predicted cosim
skip is not a measured cycle result.

The active repository, result directories, caches, benchmark inputs, and
processes must remain untouched.  They may later be revalidated as
development evidence only after independent golden checking; they must not be
resumed into the reference-isolated Rodinia campaign.

## Baseline/expert go/no-go gate

For each of the nine reference-isolated Rodinia-HLS kernels, re-synthesize all eligible
upstream variants with Vitis 2023.2, `xcu280-fsvh2892-2L-e`, and a 3.33 ns
target.  A valid pair requires:

1. a canonical public workload/ABI audit for every upstream variant, with all
   incompatible variants visibly excluded before CSim or synthesis;
2. an independently golden-checked plain-C baseline with positive executed
   RTL co-simulation cycles;
3. at least one independently golden-checked, successfully synthesized,
   device-fitting, timing-meeting expert variant; and
4. a reproducible expert frontier chosen as the minimum-latency valid expert
   variant under the same settings, also with positive executed RTL
   co-simulation cycles.

The valid-pair count is currently unknown.  Proceed with the headline
expert-recovery claim only when the count is at least eight.  If the gate
fails, remove that headline claim, preserve every failure and exclusion in
the artifact set, and report only analyses supported by valid kernels.

The term *reference-isolated* is intentional.  The paper profile prevents
expert code and QoR from entering an evaluation run, but earlier controller
development inspected Rodinia behavior.  The suite is therefore not claimed
to be an unseen development holdout; this limitation must remain visible in
the manuscript.

## Post-route and hardware protocol

Select exactly five primary-campaign winners without replacement using a
stratification rule frozen before implementation starts.  Each selected item
must already have golden correctness, synthesis feasibility, target timing,
executed selected-winner cosim, and a complete run fingerprint.  Record the
selection rationale and source artifact hashes in the machine-readable
manifest before invoking implementation.

For each selected winner, retain implementation reports and checksums,
post-route timing, resource use, tool status, and elapsed time.  Keep
implementation and board execution separate: a routed design is not a board
measurement.  The current host exposed neither `/dev/xclmgmt0` nor
`/dev/dri/renderD128`, no Xilinx PCI function through `lspci -d 10ee:`, and no
`xbutil` or `xrt-smi` on `PATH`.  Recheck access before the evidence freeze;
if it remains absent, state that physical hardware execution was unavailable.

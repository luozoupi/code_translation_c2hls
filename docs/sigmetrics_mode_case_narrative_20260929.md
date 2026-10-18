# When flash and multi-step find different architectures

September 29, 2026. This is a retrospective analysis of the frozen September 25
cohort, an offline goal-checking prototype, and a plan for a new controlled
comparison. It launches no model, HLS, or RTL jobs and does not alter the live
controller. Original failures and selections remain unchanged.

## Main finding and narrative choice

Use **BICG reviewer-off repeat 1** for a multi-step advantage and **ATAX
reviewer-off repeat 2** for a flash advantage. Both modes improve the frozen
parent in these examples; the reviewer setting, model, toolchain, inputs and
parent match within each pair. Within BICG, repeat 1 is the only reviewer-off multi-step win
where flash also improves its parent. ATAX is the median paired flash advantage
in that arm. These are post-hoc explanatory selections, not an unbiased estimate
of a mode effect. All six pairs per benchmark are retained in the evidence.

| Case | Flash / multi-step synthesis cycles | Model calls F / M | DSP F / M | BRAM18K F / M | URAM F / M |
|---|---:|---:|---:|---:|---:|
| BICG off r1 | 76,314 / 23,146 | 2 / 5 | 12 / 126 | 20 / 16 | 4 / 0 |
| ATAX off r2 | 107,767 / 167,354 | 2 / 5 | 11 / 22 | 16 / 16 | 0 / 0 |

The ratios are **3.30× in favor of multi-step for BICG**, and **1.553× in favor
of flash for ATAX**. These are worst-case synthesis cycle fields, with recorded
C-output checks and synthesis timing/resource feasibility, not RTL or routed
measurements. ATAX checks 124 final `y` values; `tmp` is not independently
compared. The cases do not have equal realized search opportunities.

The architectural distinction is **which coupled storage and scheduling change
the search reaches**, not a general “matrix kernels need multi-step” rule.
Both benchmarks are matrix-vector computations. The winning policy must supply
operands as well as parallel operators; a recipe label or a requested pragma
does not show that either condition holds.

## BICG: recover from an infeasible rewrite and change the working set

BICG forms products with a matrix and its transpose, sharing matrix data but
accumulating along different axes. In the selected flash source, the full
matrix is staged and partitioned by four, but a comment about unrolling is not
accompanied by an actual unroll directive. Its flattened scalar compute loop
has II=5 and 14,384 iterations. The source changed storage without realizing the
intended wider computation.

Multi-step first adds a genuine factor-four unroll. The identical inner-loop
scope changes trip count 116→29 and II160→32; Vitis explicitly records the
partial unroll. A later full-staging candidate estimates 89,145 cycles but
fails timing at 4.296 ns. The controller keeps the earlier feasible source.
The eventual tiling prompt contains exactly that accepted source and its
II32 report, verified by code hashes; the rejected staging candidate is not
silently treated as an enabling parent.

The selected 32×32-tiled design pipelines a row loop, causing Vitis to fully
unroll its inner compute loop by **32**, despite an explicit factor-four
request. The row loop still has II5. Thus the key change is scheduled work per
iteration and working-set organization, not an II5→1 improvement. DSP use
increases 12→126 while memory changes from 20 BRAM + 4 URAM to 16 BRAM + 0 URAM.
The arithmetic-resource cost must accompany the 3.30× cycle reduction.

This is evidence of **preserving feasible partial progress and changing the
next rewrite after feedback**. It does not show that all five visited stages
composed successfully, that prerequisites were enforced, or that a feasible
latency regression enabled the winner. At two calls, the reconstructed
multi-step incumbent is 158,290 cycles and flash already has 76,314. Additional
opportunities materially contribute to this chosen multi-step win.

## ATAX: place reused state locally before replicating computation

ATAX computes a row dot product, then consumes that scalar while updating the
output vector. Flash stages reused `x` and accumulated `y` locally, pipelines
the two consumers, and writes back `y` once. It still reads matrix `A` globally.
This is vector locality, not proof of eliminating external traffic or staging
the entire working set.

The retained multi-step source actually unrolls both column loops by four,
but keeps `x` and `y` accesses global. Vitis confirms the replication, while
floating-point carried dependencies and shared AXI requests still limit the
schedule. Its dot loop has II32, trip31, depth114, latency1,073; flash has II5,
trip124, depth17, latency631. Its update has latency219 versus flash142.
**Do not divide II32 by II5 and call that speedup:** one scheduled iteration
contains different work, and pipeline fill cost differs.

Flash therefore uses half as many DSPs and fewer cycles in the median pair.
Multi-step does attempt staging later: that candidate reaches 94,845 cycles
but fails timing at 4.296 ns, followed by two output mismatches and a later
pipeline timing failure. The controller correctly retains its feasible design.
The missed opportunity is finding a feasible locality rewrite, not bypassing
correctness or timing checks.

Aggressive flash repeats also coordinate full-matrix staging and outer-loop
pipelining, at roughly 475 DSPs and 264–268 BRAMs. However, multi-step reviewer-on
repeat 2 reaches the same architecture and 5,866 cycles. The architectures are
not exclusive to either mode. In all six ATAX dynamic runs, the final incumbent
is already available within two calls; additional attempts do not improve it.

## Patterns supported by the full cohort, and their limits

Across six pairs, BICG favors multi-step in 5/6; ATAX favors flash in 4/6.
BICG's flash/multi-step geometric-mean ratios are 5.77 off and 63.97 on; four
flash runs retain the parent, inflating the pooled contrast. ATAX's reciprocal
flash advantages are 2.52 off and 5.20 on; its two multi-step wins are near ties.
Neither a best repeat nor these unequal-budget means is a causal mode effect.

The complete 27-kernel audit has 161 eligible pairs, with 76 multi-step wins,
9 ties and 76 flash wins. Fifteen kernels strictly change geometric-mean winner
between reviewer-off/on arms, whose prompt content differs. Broad source-defined
classes contain opposite outcomes: BICG/ATAX, covariance/correlation, and
SYRK/SYMM. Three API requests per stratum are insufficient to train and validate
a general mode selector on the same tasks.

The bounded SYRK check also cautions against inventing a prerequisite chain:
the median reviewer-off multi-step winner combines row buffering, partitioning,
pipelining and unrolling in a later candidate after pragma-only attempts fail.
Flash retains its parent after timing failures. This corroborates a **late
coupled rewrite**, not clean evidence that each earlier skill completed a
necessary architectural goal.

Test the following hypotheses on held-out workloads, with groups assigned from
plain source and baseline reports before looking at mode outcomes:

| Observable constraint / source structure | Proposed decision to test | Evidence required before making a claim |
|---|---|---|
| A concrete global rewrite repeatedly fails timing, while a feasible partial design exposes a scoped next constraint | Multi-step may profit from retaining that design and exploring bounded tiles or different compute width | Exact prompt-parent edges, legal state preservation, candidate-level outcomes, actual calls and resource cost |
| Small reused vectors or reduction state can be moved locally with clear initialization/writeback boundaries | Flash may reach a useful combined locality/scheduling change early | Source access paths plus compiler loop latency, recurrence/port diagnostics, boundary-output checks |
| A target schedule requires coordinated banking, accumulator structure and loop placement | Permit a coupled rewrite in either mode; do not enforce one pragma per step | Realized unroll, loop work/trip count, achieved II, storage binding and timing |
| A true recurrence remains after replication, or external requests cannot feed added lanes | Resolve the dependence/operand-supply constraint before demanding more lanes | Named carried-dependence and port witnesses; do not infer this from DSP growth alone |

These are proposed rules for selecting the next search action. The current
framework does not already implement or validate them as a mode-routing policy.
“Instinct” in the paper should be replaced with the concrete decision and its
observed compiler consequence.

## Offline predicates implemented before any enforcement

[The checker](../scripts/audit_sigmetrics_mode_case_goals.py) reads a frozen
manifest, verifies result/source/report hashes, and evaluates manually bound
loop metrics. Each probe records semantic scope, exact source line, compiler
scope, metric/operator/threshold, expected change, prerequisites and required
evidence. It reports **achieved / unmet / unknown**, separately from recorded
correctness, feasibility and endpoint cycles. It never changes historical
selection or enables production control-flow gating.

The four primary endpoints give 12 deliberately selected coverage probes:
**3 achieved, 5 unmet, 4 unknown**. This is not a verifier success rate or a
sample estimate. Eight probes test II≤1 in named loops; dynamic ATAX's two
thresholds are hypothetical targets, not historical directives. Four physical
bank/port probes remain unknown because binding evidence is absent. Candidate
scope mapping was manually audited; the checker does not solve generic loop
correspondence. Missing, ambiguous, or renamed scopes and hash mismatches remain
unknown. Requested II1 is not automatically a feasible target.

Ten focused tests pass: exact scope, wrong/renamed scope, missing II, rejecting
a module interval as loop II, duplicate scopes, missing bank evidence, invalid
source/report binding, a faster candidate with an unmet goal, a slower enabling
candidate with an achieved goal, and transitive unknown prerequisites. Actual
case bindings all match and derived output regeneration is deterministic.

Manual source/transform checks in the BICG and ATAX reports provide complementary
unroll/locality observations; they are not silently promoted to automated
bank-port or concurrency proofs. The generic checker is intentionally narrower.

## Next controlled moves

1. **Calibrate goals on four kernels:** BICG, ATAX, SYRK, and correlation. Keep
   these exploratory cases out of the final held-out selector evaluation.
   Choose achievable schedule targets from dependence/resource constraints,
   not a universal II1 rule. Preserve unknowns and measure evidence coverage.
2. **Freeze consistent mode prompts.** Remove the archived flash conflict
   between a one-technique system instruction and combination-permitting user
   instructions. One multi-step goal may require several coordinated edits.
3. **Equalize primary search opportunities:** five charged generation slots
   and at most five syntheses in both modes, identical repair spending and
   predeclared stopping rules. Distinguish whole-design from targeted rewrites;
   provide equivalent compiler and goal feedback. Keep historical two-call
   flash as a separately labeled low-cost operating point. Report prefix
   quality, tokens, elapsed time, failures and resource costs, not only caps.
4. **Compare advice versus prerequisite enforcement:** identical predicates,
   diagnostics and budgets; only the dependency-advancement decision changes.
   An achieved, feasible structural enabler may increase cycles. Keep an active
   exploration state separate from the best feasible endpoint. An unmet goal
   can still yield a useful endpoint, but must not earn prerequisite credit.
5. **Validate new endpoints in RTL** using the frozen independent comparator,
   complete outputs and stated tolerances. No post-RTL reselection. Register
   invalid and retained-parent outcomes alongside paired passing-design ratios.
   Test the policy on additional tasks and shapes after freezing hypotheses.

This extension remains separate from the registered recipe-representation
study. No new experiment is launched by this document or the checker.

## Evidence and reproduction

- [Complete pattern audit](../artifacts/sigmetrics2027/mode_case_study_20260929/pattern_report.md)
- [BICG source/report/prompt trace](../artifacts/sigmetrics2027/mode_case_study_20260929/multistep_bicg_report.md)
- [ATAX source/report trace](../artifacts/sigmetrics2027/mode_case_study_20260929/flash_report.md)
- [Four endpoint records](../artifacts/sigmetrics2027/mode_case_study_20260929/primary_endpoints.csv)
- [All twelve pairs](../artifacts/sigmetrics2027/mode_case_study_20260929/primary_all_repeats.csv)
- [Search trace plot](../artifacts/sigmetrics2027/mode_case_study_20260929/primary_trajectories.pdf)
- [Goal manifest](../artifacts/sigmetrics2027/mode_case_study_20260929/primary_goal_manifest.json)
  and [results](../artifacts/sigmetrics2027/mode_case_study_20260929/primary_goal_results.json)

Run `build_primary_case_bundle.py` in that evidence directory, then
`python3 scripts/audit_sigmetrics_mode_case_goals.py --manifest
artifacts/sigmetrics2027/mode_case_study_20260929/primary_goal_manifest.json
--output artifacts/sigmetrics2027/mode_case_study_20260929/primary_goal_results.json`.
Use `python3 -m unittest discover -s tests -p test_mode_case_goal_audit.py` for
checker tests. Plot generation uses `plot_primary_trajectories.py` with the
`py310_2` matplotlib environment. The artifacts directory is git-ignored;
preserve the evidence files explicitly when preparing an evidence release.

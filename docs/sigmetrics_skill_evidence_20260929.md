# Skill groups, observed outcomes, and the TRMM case

Updated September 29, 2026. This records a versioned catalog audit and
retrospective evidence, not a new HLS/model campaign or a causal estimate of
every recipe's contribution. Raw experiments and frozen libraries are preserved.

## What to claim and what to measure

Organize the library by the hardware constraint each entry addresses. For an
actual run, keep five facts separate: an entry was selected by the router; its
text reached the prompt; the model declared using it; the source/report shows
an intended change; and the candidate improved a defined outcome while passing
required checks. Only the last two connect a skill to architectural behavior.
None alone isolates the effect of that skill when several were supplied.

Use **observed association** for historical usage/outcome tables, **mechanism
evidence** for a bound source/report trace, and **causal contribution** only
for a controlled intervention. A candidate with three declarations and a 2×
cycle reduction is one candidate outcome, not three independent 2× skill gains.
Catalog `mean_advantage`, usage counters, and source-marker credit cannot be
substituted for these comparisons. Avoidance rules should be assessed by the
failures they prevent, not ranked solely by latency reduction.

## Versioned inventory

The frozen v3 file is `skill_v3_no_rmw_fc27133/skills.json`, SHA-256
`d37094bf5548739debd8bdaae98fc9bc13db959cd33e17ddf9edd7885a1f358c`.
Its copy in the six-cell SIGMETRICS pilot is byte-identical. It contains 123
entries: 84 non-avoid and 39 avoidance entries. A content-role classification
separates 80 transformation entries, 39 guards, 3 diagnostic controls, and one
reporting-only annotation. These are entries, not 80 distinct algorithms or
independently validated techniques. Prompt-origin entries can contain real
compound transformations; an origin tag is not a sufficient role classifier.

| Primary architecture family | v3 entries | Intended role and evidence to inspect |
|---|---:|---|
| External-memory access | 14 | Coalescing, interface and burst organization; inspect inferred transactions and the named interface. |
| Locality and working-set reuse | 28 | Stage reused data, tile the computation, preserve boundary and final-write behavior; inspect actual accesses and storage. |
| Local memory ports | 12 | Match bank layout to concurrent accesses; inspect bank binding, port demand and loop schedule. |
| Loop scheduling | 12 | Pipeline or restructure loops; inspect achieved II, trip count, depth and recurrences at the target scope. |
| Dependence and reduction | 17 | Preserve true dependences or restructure legal accumulations; inspect schedule and numerical behavior. |
| Stage overlap | 23 | Separate producers/consumers and manage channels/buffers; inspect concurrent schedules and ownership. |
| Compute parallelism | 11 | Replicate useful work; inspect realized unrolling, operand supply, timing and per-resource cost. |
| Controller and measurement rules | 6 | Diagnose or describe the tool outcome; do not treat report annotations as faster hardware. |

Every entry receives one primary family for counting and secondary tags for
coupled mechanisms. The mapping is a manually reviewed descriptive taxonomy;
it is not a minimality proof. The machine-readable taxonomy retains entry IDs,
content hashes, rationale, and uncertainty/conflict notes.

The earlier v2 library contains 56 entries: **42 non-avoid and 14 avoidance
rules**. Its SHA-256 is
`9f46a3de6073b1dd9fe657423b43b41051a5fb89604f437fddb8d94c4234af5e`.
The historical router table's 42 therefore must not be globally replaced by
123. Likewise, the current pilot selects only three v3 entries; catalog size is
not prompt exposure. The archived Sonnet TRMM run has a different recorded
store hash and no recovered exact snapshot. Shared IDs do not prove identical
historical text or provide a complete provenance chain.

## Observed skill associations in the September reviewer cohort

The [reproducible outcome audit](../artifacts/sigmetrics2027/skill_audit_20260929/effect_report.md)
retains 323 of 324 registered runs across 27 tasks and both modes. It contains
1,139 candidate events, 649 recorded feasible events and 404 strict incumbent
improvements. Feasible means correctness gate, synthesis, timing and resource
checks all passed, with positive worst-case synthesis cycles. Improvement is
relative to the frozen parent or best earlier feasible candidate, whichever
has fewer cycles. Run-integrity eligibility is not candidate correctness.
There is no RTL co-simulation in this cohort. Detailed per-attempt comparator
records remain for 569 of the 649 feasible events; the other 80 have passing
event telemetry. All 227 unique source-linked improving events have saved,
hash-linked comparator records with reason `match`. None was newly replayed
by this retrospective audit.

Only 25 IDs were rendered: 16 transformations, 8 guards, and 1 diagnostic entry.
636 events have matching retained source, and 526 have exact parent-hash and
worst-cycle links. For the association below, require actual rendering, model
declaration, a recorded new-family marker, matching source/catalog hashes,
a transformation role, and no static fallback. The marker is syntactic and
does not certify the recipe's architecture goal. In particular, the staging
entry's marker recognizes pipeline syntax, not staging itself.

| Entry | Rendered events | Source-linked declared-marker cases | Incumbent improvements in those cases |
|---|---:|---:|---:|
| `hls-baseline-load-compute-store-gate` | 446 | 224 | 151 |
| `hls-array-partition-match-unroll-factor` | 487 | 158 | 118 |
| `hls-fixed-tile-bound-not-runtime-tile-size` | 107 | 57 | 29 |
| `hls-canonical-dataflow-lcs-functions` | 474 | 7 | 2 |

227 unique improving events have at least one eligible transformation
association. Per-family counts are locality 153, local ports 118, scheduling
29, external access 6, overlap 2, and dependence 1. These counts overlap and
must not be summed or interpreted as causal skill gains. Frequency is not
an effectiveness ranking; low exposure and missing evidence constrain inference.

The [case records](../artifacts/sigmetrics2027/skill_audit_20260929/effect_cases.csv)
include a separate September 25 TRMM run: a staging-associated event changes
worst cycles 45,202,801→4,853,927, followed by a fixed-bound event to 1,999,861.
This is a different run from the older manuscript trajectory below. A covariance
case with a dataflow marker instead increases worst cycles 3,526,172→5,106,615
and BRAM 30→46. An improving GEMM fallback and an ATAX timing failure are
explicitly excluded from skill-improvement attribution. Keep negative,
infeasible and fallback cases in the evidence package.

## Attribution design and next experiments

The current cohort audit provides a map of where to investigate, not a causal
ranking. Use its family/task table to select both favorable and unfavorable
cases, then prespecify the family experiment before obtaining new outcomes.
Selection should cover mechanism classes and failures, not only large gains.

1. **Family removal.** Keep parent, generic instructions, safety oracle,
   toolchain, selected non-target content, model and budgets fixed; remove the
   selected transformation family. Remove its content from compound recipes too;
   removing only primary-family IDs leaves the same guidance elsewhere. Keep
   unrelated and common correctness guardrails constant. Report
   prompt length and actual work as well as the registered limits. No outcome-
   dependent replacement recipe or additional retries in the removed-family arm.
2. **Interaction test.** For memory-bound matrix kernels, compare common
   guidance, banking guidance, unrolling guidance, and both. A joint benefit
   motivates a coupling claim only after actual bank/lane/schedule evidence
   and outputs are checked. The extra content is part of this intervention;
   it is not an isolated test of structured versus flat representation.
3. **Mechanism intervention.** On a frozen source checkpoint, use legal
   bank/unroll-factor variants to separate the generated implementation's
   hardware interaction from the language model's response to a prompt. Use the
   same inputs, numerical tolerance, clock and resource limits. This is a
   deterministic source ablation, distinct from a prompt ablation.
4. **Order and prerequisites.** Within multi-step, compare the same content
   and work budget with prescribed versus alternative order. Record actual
   goal achievement; a visited stage or vanished warning is insufficient.
   Allow bounded enabling changes without requiring every intermediate to be
   faster, and retain the best feasible endpoint separately.
5. **Avoidance rules.** Report invalid-output, compile, timing and deadlock/
   timeout outcomes separately. Removing textual guards never disables the
   deterministic correctness oracle or changes tolerance. Distinguish an
   observed timeout stage from a proven deadlock.

Primary endpoints are RTL-pass yield and improvement over the validated parent
on all registered runs; report paired latency ratios only on the named passing
subset. Include resource costs, tokens/calls/tool time, actual prompt hashes,
and failed attempts. Family and per-entry results need benchmark-cluster
uncertainty and exposure denominators. The registered guidance-format study
remains separate from these new family and mode interventions.

## Existing skill-dose evidence to reuse carefully

The older Sonnet v2 dose study already has 729 records across 27 kernels, nested
positive counts 0/1/2/3/5/8/42, three requests per condition, and action-only
versus precondition-bearing guidance at counts 3 and 42. Its saved summary
reports 467 valid candidates, 123 wrong-output failures, 86 timing failures,
43 compile/interface failures, one synthesis timeout, and nine other failures.
It has no RTL co-simulation. The archived clustered log-cycle slope interval
includes zero, so the report does not support a simple “more skills always
helps” interpretation.

This is skill-count/guard-policy evidence, not an individual-skill or family
ablation. Before importing an effect into the paper, independently join the
raw sources, actual prompt contents, frozen parents, costs and invalid outcomes.
The report's old “expand to remaining 19” flag predates its 27-kernel contents
and is not a current execution instruction. The inventory and source hashes
are saved in `artifacts/sigmetrics2027/skill_audit_20260929/prior_dose_inventory.json`.

## TRMM: explain mechanism, route, and measurement together

Use the five archived checkpoint records and earlier manifest source checks
to connect the recorded routed entry to the documented changes. Exact source
checkpoints were not recovered; do not describe this as a new source-diff audit. Separate source-visible buffering, banking, tiling and
unrolling from achieved scheduling or transfer effects. The final banking/
unrolling action is a **static fallback**, so its cycle change cannot be
credited to selection of an unrolling recipe. The exact historical catalog and
prompts remain unavailable; do not replace them with current v3 text.

The new audit also finds two distinct archived latency fields. The familiar
469,321→181,621→159,267→97,914→26,233 sequence is the controller-recorded
`latency_cycles` metric; the preserved parser prefers average-case latency.
The raw `latency_cycles_worst` sequence is
935,521→941,281→303,267→339,194→61,305. The old normalized exporter copies the
first field into several labels, including `Worst-caseLatency`; that normalized
label must not be treated as an independent worst-case observation. The exact
historical parser snapshot is not recovered, so raw-field provenance is stated
explicitly and no missing XML report is implied.

The second sequence increases at the intended overlap and tiling steps.
All reported inner pipelined loops already have II=1 at the first checkpoint;
there is no supported II>1→1 claim. Later changes affect working-set
organization, repeated work, storage mapping and parallel lanes. The final
checkpoint changes a reported scale-loop trip count from 80 to 10 while its
accumulation loop remains II=1. These observations support a more useful case
than a monotonic sequence of speedups: the recipe objective, actual source
change, compiler metric and hardware cost must be checked separately.

Preserve both metrics and all original artifacts. In the manuscript, label the
controller metric explicitly, show the archived worst-case field alongside it,
retain resource/timing costs, and avoid baseline-to-final RTL speedup claims.
Only the selected final has the archived 50,040-cycle RTL count, and the
previous host replay does not independently replay those RTL outputs. The
case remains explanatory rather than a matched causal recipe ablation.


## Concrete next moves and paper integration

1. **Freeze evidence first.** Use the version-pinned taxonomy and association
   files linked below. Resolve the catalog's conflicting DATAFLOW retain/remove
   instructions in a new version; preserve the historical versions.
2. **Implement goal evidence before claiming enforced multi-step completion.**
   Follow the [controller audit](multistep_goal_verification_audit_20260929.md):
   bind the requested goal, candidate source hash, named loop/storage scope,
   report witness and verdict. Keep legality, goal completion and endpoint
   performance as distinct outcomes. Missing evidence yields unknown, not pass.
3. **Prioritize a bounded family study.** Staging and bank/unroll guidance,
   followed by fixed bounds, have enough historical exposure to motivate
   prospective tests. Include a negative overlap case. Freeze task selection,
   model, prompts and limits before launching; preserve both modes' actual costs.
4. **Recover or reproduce TRMM checkpoints.** Historical manifest hashes permit
   verification if source files are found. Otherwise reproduce a new, separately
   named cohort and retain all sources, prompts and raw reports. Then apply
   legal bank/unroll-factor interventions and independently check RTL outputs.
   Do not fabricate missing historical sources from the diagram.
5. **Report all paths.** Include selected recipes, fallback guidance, repair
   attempts, timing misses, both cycle fields, and DSP/BRAM/URAM costs. The
   current five-checkpoint picture required eight model calls, including the
   baseline and two repair attempts; it is not a five-attempt cost claim.

The SIGMETRICS draft now contains the eight-family inventory, qualified
per-entry associations, and a TRMM route/mechanism table. Its resource table
and plot show both archived cycle fields. The HPCA sources and original
numeric manifests remain preserved. This is documentation and manuscript
work; no model or Vitis experiment was launched by this audit.

Reproduction and provenance:

- [Catalog audit](taxonomy_skill_catalog_20260929.md) and
  [taxonomy generator](../artifacts/sigmetrics2027/skill_audit_20260929/taxonomy_build.py).
- [Outcome audit](../artifacts/sigmetrics2027/skill_audit_20260929/effect_report.md) and
  [event/skill generator](../artifacts/sigmetrics2027/skill_audit_20260929/effect_analyze.py).
- [TRMM raw-field audit](../artifacts/sigmetrics2027/skill_audit_20260929/trmm_report.md),
  [checkpoint evidence](../artifacts/sigmetrics2027/skill_audit_20260929/trmm_evidence.json),
  and [metric table](../artifacts/sigmetrics2027/skill_audit_20260929/trmm_metrics.csv).

The artifacts directory is git-ignored. Preserve these derived audits explicitly
when preparing an evidence commit or release; documentation alone does not
contain all event-level provenance.

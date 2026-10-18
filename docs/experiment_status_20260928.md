# Experiment status and SIGMETRICS next steps

Prepared September 28, 2026. This is a dated interpretation, not a live service.
The companion script creates fresh snapshots without changing campaign outputs.
It makes no API calls and launches no HLS jobs. Existing live experiments and
their frozen evaluator files have been left running and unchanged.

**The immediate priority is correctness and evidence accounting.** September's
completed studies offer useful synthesis and cost evidence, but cannot yet
replace the paper's headline results with independently validated RTL claims.
The historical queue helps recover old evidence; it does not replace the new
same-budget experiment or isolate the value of structured recipes.

## Tracking and result files

Run from the repository root:

```bash
python3 scripts/snapshot_experiment_status.py
```

Each invocation prints a new directory under `artifacts/experiment_status/`.

| File | Purpose |
|---|---|
| `report.md`, `snapshot.json` | Read window, controller state, code hashes, progress, failures and source summaries |
| `historical_rows.csv` | All 2,553 rows; original labels, blocked rows, audit status, CSYNTH estimates and separate COSIM cycles |
| `validations.csv` | One row per distinct queued validation identity (1,741 in the frozen manifest) |
| `failures.csv` | Non-pass completed validations, exact reasons and measurement paths |
| `registry_affected_rows.csv` | Rows whose recorded testbench does not match the default registry |
| `campaign_coverage.csv` | Per-campaign coverage for all 65 historical campaigns |
| `completed_search_costs.csv` | 432 September runs: calls, tokens, cache use, synthesis, search time, parent retention and RTL status |

The [22:12 UTC snapshot](../artifacts/experiment_status/20260928T221251763822Z/report.md)
includes explicit executed-RTL fields and passes row/identity/cost reconciliation:
**92/1,741 processed**, comprising 68 passes, 13 registry/oracle errors,
9 candidate numerical failures and 2 timeouts; 2 active and 1,647 pending.
This is 11 more processed identities than the first snapshot. All six frozen
code hashes still match. The detailed first-snapshot counts below are retained
as a dated observation.

The [first snapshot](../artifacts/experiment_status/20260928T220601296055Z/report.md)
was captured during 22:06:01–22:06:19 UTC (17:06 CDT). Later timestamped snapshots
contain fresher progress. Reads of active files are not a transaction: the
tracker records controller timestamps and gives a finished measurement
precedence over a stale active list. Two failed preflight records outside the
frozen manifest are excluded from campaign totals and remain preserved.

The script only writes new directories in `artifacts/experiment_status/`.
It does not call the controller's `report` command, which would compete with
the running controller for the same output files. It does not run continuously
after the command returns; invoke it again to refresh.

## Live session

Only one C2HLS experiment controller was observed in the current user's tmux
and Python process checks: `c2hls_july_cosim_20260928`, PID **520566**, with two
historical COSIM workers. Other unrelated processes were not modified.

At the first snapshot:

| Distinct validation jobs | Count |
|---|---:|
| Completed / total | 81 / 1,741 (4.7%) |
| Independent-golden CSim + RTL passes | 58 |
| Native-oracle/registry errors; RTL not run | 13 |
| Candidate CSim numerical mismatches; RTL not run | 8 |
| Combined synthesis/COSIM timeouts | 2 |
| Sampled active | 2 |
| Unresolved outside sampled active list | 1,658 |

The two sampled active jobs were Sonnet Jacobi-1D candidates. All six frozen
evaluator/registry hashes still matched. The resource gate reported no external
Vitis processes, about 848 GiB free local scratch, and load ratio 0.244 (limit
0.70). The controller was running and its results advanced during inspection.

The **row** totals differ because jobs are deduplicated: 59 new-pass rows,
16 existing-COSIM rows, 22 not-run rows, 3 timeout rows, 363 blocked rows,
2 active rows and 2,088 pending rows. Existing-COSIM records retain their
historical validation scope; they are not new independent-oracle passes.

There is no defensible queue ETA. Successful jobs observed so far have a
56.5-second median and 425.3-second maximum, but jobs are ordered by increasing
historical CSYNTH cycles; registry errors exit quickly, and two jobs already
hit the 3,600-second limit. Completed-only performance or pass-rate estimates
would be biased. The >10x reference-CSYNTH runtime protection also selects
which candidates execute; predicted skips are not measured timeouts.

Compact state: `results_sweeps/july_cosim_backfill_20260928/`.
Bulk measurements/logs/archives:
`/mnt/data2/luo00466/c2hls_rl/july_cosim_backfill_20260928/`.

## Problems requiring separate treatment

Counts in the detailed triage below refer to the 22:06 snapshot; the 22:12
refresh adds one more candidate numerical failure (nine total).

1. **Registry dispatch is wrong for the imported harness cohort.** Of eligible
   rows, 535 (405 distinct validations) disagree with the default registry:
   413 testbench-hash mismatches and 122 missing benchmark entries. All 535
   match `configs/hlsfactory_output_shapes_fc27133.json`. Thirteen affected
   identities had already failed and 392 remained unresolved at the snapshot.
   The backfill worker calls `_prepare_independent_golden` without choosing the
   cohort registry; scanner identities hash only the default registry. Fix both
   dispatch and identity/provenance in a new validator/campaign version. Do not
   change environment selection midway through this frozen campaign or loosen
   hash/shape checks. Preserve failed records and connect new validation IDs
   to the old IDs and unchanged candidate hashes.
2. **Eight Jacobi-1D candidates fail independent CSim numerically.** These are
   `float_mismatch` failures after a valid native oracle, distinct from the
   registry issue. Inspect saved source and error indices, including boundary
   handling and stencil update semantics. Keep failures in the denominator;
   do not relax tolerance or regenerate a replacement under the same historical
   sample. A root-cause explanation is still pending.
3. **Two Jacobi-1D runs timed out.** Inspect preserved logs for the last completed
   stage before classifying them as simulator hang, excessive execution, or
   tool overhead. The limit covers synthesis plus COSIM, so the label does not
   prove RTL itself consumed the whole hour. Any longer-timeout replay is a
   separately recorded sensitivity analysis, not a replacement sample.
4. **363 rows are blocked before validation:** 189 missing selected sources,
   89 selected-source/report mismatches, 51 summary/report cycle mismatches,
   and 34 without a successful selected synthesis. Recover exact archived
   artifacts where possible; do not replace them with a better candidate.
5. **Historical reference isolation remains unresolved.** The inventory copies
   896 false audit flags and 1,657 missing flags; no true flags are recorded.
   False does not by itself prove leakage, but RTL success cannot repair the
   generation information boundary. Trace prompt/features/training provenance
   before promoting such rows into a reference-blind claim.
6. **The shared evaluator still needs stage-aware RTL comparison.** The backfill
   locally handles Vitis's repeated C and RTL output dumps; the general evaluator
   still uses a single-dump comparison. Unify the fix in isolated development
   and require a real RTL-postcheck section, validating every dump.
7. **An integrity flag can easily be mistaken for correctness.** All 432
   September runs have `selected_cosim_measurement_valid=True` but zero
   `selected_winner_cosim_count`. None executed selected-winner RTL COSIM.
   Export and gate on an actual executed pass plus source identity, never that
   flag alone. The refreshed ledger includes `rtl_executed_pass` explicitly.

## Existing evidence and permissible use

| Study / source | Verified saved result | What remains before a stronger claim |
|---|---|---|
| [Reviewer A/B, Sep 25](../results_sweeps/deepseek_flash_reviewer_ab27_20260925/report.md) | 324 runs, 323 audit-valid feasible selections, 161/162 valid pairs over 27 kernels; GM estimated-cycle improvement **1.6683x**, benchmark-cluster 95% CI **[1.1048, 2.4647]**; 91 wins / 54 regressions / 16 ties | All 324 native-oracle and CSim checks pass, but no RTL; reviewer bundles feedback and hardware guidance, not an isolated guards/order ablation |
| [Flash numeric search, Sep 23](../results_sweeps/deepseek_flash_qor_profile27_20260923/summary.json) | 54/54 pairs, 27 kernels; **1.3532x [1.1186, 1.6873]**; 18 improve, 36 unchanged; 819 candidates, 699 feasible | CSim only; numeric search adds **705 synthesis calls** beyond 153 model-candidate synthesis calls; retaining the parent permits non-worsening selection by construction |
| [Pro numeric search, Sep 8](../results_sweeps/deepseek_v4_pro_qor_paired27_axi_v2_20260908/summary.json) | 54/54 pairs, 27 kernels; **1.0682x [1.0081, 1.1803]**; 32 improve, 22 unchanged; 822 candidates, 814 feasible | CSim only; **816 extra synthesis calls** beyond 169 model-candidate synthesis calls |
| [Router development](../artifacts/setup_router/topk_architecture_comparison_corrected_v2_20260728/report.md) | 19 development lineages, grouped five-fold OOF; best top-5 within-5% coverage 0.895, GM regret 1.056 | Architecture selection/development evidence, not the four-kernel final test |
| [Strengthened router historical test](../artifacts/setup_router/strengthened_router_corrected_v2/metrics.json) | 3/4 within 5%, GM regret 1.684; **6.75** configurations evaluated on average | Explicit historical-test-reuse caveat; not a fixed-five-budget result or untouched prospective test |
| [Qwen offline SFT](offline_sft_workflow.md) | Completed active-actor QLoRA adapter and documented lineage split; existing small held-out comparisons | Broader matched base/SFT evaluation with consistent RTL and full costs; GRPO remains exploratory, with different data |
| [Post-route implementation manifest](../paper_eval/post_route_implementation_manifest.json) | `not_started`; five slots unassigned; `paper_evidence_allowed=false` | No routed timing or board-runtime evidence; old placeholder prerequisites need a new frozen selection protocol |

Keep the two numeric-search effects separate from the reviewer on/off effect.
Do not multiply them or rename historical model aliases as current immutable
model versions. The historical queue does not cover every commercial-model
headline or supply a matched ChatHLS replay. The paper's maximum gains still
need an exact source/report/validation join before reuse.

For reviewer A/B, both arms have five-call/five-synthesis caps and the same
frozen Phase-B parent. Actual means across **all 162 attempts per arm**, including
the audit-excluded attempt, are:

| Per attempt | Review off | Review on |
|---|---:|---:|
| Model calls | 3.506 | 3.537 |
| Input tokens | 16,168 | 19,398 |
| Output tokens | 3,952 | 5,121 |
| Candidate synthesis calls | 2.778 | 2.660 |
| Search seconds, mean | 227.95 | 210.72 |
| Search seconds, median | 87.16 | 92.50 |
| Parent retained, count | 34 | 22 |
| No feasible generated candidate, count | 24 | 17 |

The excluded cell is `hlsfactory_trisolv_flash_r2_review_on`; preserve its audit
finding and cost. Do not call 323 final selections 323 successful new generations.
Search timing excludes construction of the shared frozen parent. Cached-token
counts are available, but actual dollar costs require run-date pricing or
receipts. Repeated requests used temperature zero and no supported provider
seed; they are repeated requests, not proven distinct seeds.

## Prioritized work to address the reviews

| Priority / concern | Concrete next work | Completion evidence |
|---|---|---|
| **P0: correctness/provenance (A, D)** | Isolate development while preserving September's dirty files. Implement cohort registry dispatch/fingerprints; share the stage-aware validator; fix the HLSFactory baseline adapter's missing public benchmark ID; define executed-RTL status | Checks for both registries, missing/wrong hashes, repeated dumps with one corrupt dump, absent RTL section, adapter metadata and integrity-only false passes; native/CSim/RTL smoke on both harness cohorts |
| **P0: comparable cost (A, D)** | Give candidates an origin (`llm`, `deterministic_search`, `frozen_parent`), hash and nullable LLM event link. Count retries, repairs, native checks, synthesis and RTL separately; report parent construction and total cost | Pilot ledger reconciles every call and token; global budget caps cover the entire run rather than each step |
| **P1: recipe contribution (C, D)** | Freeze recipe IDs/content from the initial parent; compare no expertise, flat same-content guidance and structured recipes with identical feedback and budgets | Prespecified paired matrix, equal knowledge for flat/structured arms, repeated requests, independent output tests and selected-winner RTL |
| **P1: baselines/scope (B, D)** | Adapt the existing zero-LLM numeric DSE as a budgeted baseline; integrate one runnable source-transforming baseline, with AgRefactor as the first target | Same inputs, parent, device, clock and tool budget; failures visible. `pragma_only` is LLM-driven and must not be labeled non-LLM DSE |
| **P1: external applications (C, D)** | Add MachSuite BFS and KMP plus a documented window/FFT/power pipeline; implement adapters and independent harnesses first | Multiple valid inputs and RTL checks; full-pipeline latency/throughput, buffering, resources and failures |
| **P2: router/adaptation (A, D)** | Freeze a fixed top-5 router before external outcomes; compare against fixed and exhaustive ten-setup policies. Compare existing Qwen base/active-actor adapter under matching conditions | Prospective external routing; lineage-disjoint SFT results separated from previously exposed tasks; actual call/tool savings |
| **P2: explanation/claims (B, D)** | Describe recipe curation/manual effort and actual enforcement; connect bottlenecks to transformations and resources/II; add background and group related work | Artifact-linked claims; guards described as guidance unless executable; LUT/FF/BRAM/DSP tradeoffs and limitations |

The next implementation should start with **P0 and a small validation pilot**.
Do not launch the full new matrix until the pilot produces a reconciled ledger.
Use a new worktree/snapshot rather than editing the six files frozen by the
running queue. A checkout of HEAD alone would omit uncommitted September work;
record copied changes and hashes explicitly.

## Proposed experiment scope and execution order

Retain the previously discussed moderate scope and **$300 total new API cap**.
The proposed core is 12 prespecified PolyBench kernels: GEMM, 2MM, ATAX, BiCG,
Jacobi-2D, Seidel-2D, LU, TriSolve, Durbin, Gram–Schmidt, Nussinov and
Floyd–Warshall. Three arms × two API models × three repeated requests is
**216 core cells**, before baseline/external/router/SFT supplements. Record
DeepSeek-V4.1-Flash and GPT-5.6 Luna request identifiers and returned metadata
at launch; do not treat old alias traces as immutable-version matches.

Use five model calls and five candidate-synthesis invocations per core run,
counting failures and repairs; retain the shared initial parent with separately
reported construction cost. Freeze token caps, timeouts, stopping and fallback
rules. The existing proposal is 16,384 response tokens and 100,000 aggregate
input tokens per run; pilot costs and reserve budget for failures before
committing the matrix. Label repeated requests honestly when seeds are unsupported.

Validate candidates against five independently chosen, domain-valid inputs;
the selected result must pass required RTL tests. Freeze inputs, tolerances,
shape contracts and selection before generation. Preserve failed selections;
do not retry until a passing sample appears. Report success probabilities over
all attempts and paired speedups with denominator/coverage disclosed; use
benchmark-cluster uncertainty for repeated requests on the same kernel.
Publish cost–quality curves, not only the best configuration's speedup.

Make historical validation a background recovery task. Once the submission
pilot is ready, drain the historical controller and give the new campaign the
two HLS slots. SIGTERM/SIGINT stops new launches while active jobs finish.
First recheck PID ownership/command and frozen hashes; do not signal an old PID
copied from this document or terminate worker trees. Resume only after old
workers are gone. No drain, restart, model generation or new HLS launch was
performed during this status task.

## Schedule and paper integration

The official fall deadlines are **October 2, 2026, 23:59 AoE for the abstract**
and **October 9, 2026, 23:59 AoE for the paper**. The Learning track covers ML
used to optimize systems. [SIGMETRICS 2027 CFP](https://www.sigmetrics.org/sigmetrics2027/pages/cfp.html)
(checked September 28).

| Date | Deliverable / gate |
|---|---|
| Sep 28–30 | P0 validator/provenance fixes, failure triage and exact ledger; external harness implementation |
| Oct 1 | Both-harness smoke tests, same-budget pilot, frozen matrix and cost estimate within $300; drain history when new HLS campaign is ready |
| Oct 2 | Register a substantive abstract; frame contributions around measured correctness, quality and optimization cost |
| Oct 3–6 | Core comparisons and RTL, non-LLM/source-transforming baselines, external pipeline; router/SFT supplements as capacity permits |
| Oct 7 | Evidence freeze: matched results, correctness, costs, uncertainty and exclusions. If essential evidence is missing, narrow claims/scope rather than relabel partial work as complete |
| Oct 8–9 | Update figures/macros and narrative from the frozen manifest; check anonymity, citations, content pages and artifact declaration |

This is a gated target schedule, not an assurance that the full matrix will
finish by October 7. Pilot HLS runtime and integration determine feasibility.
Keep recipe/orchestration, router and SFT contributions, stating their evidence
strength separately. Existing GRPO stays exploratory; a new RL campaign is
outside the critical path.

The separate `main-sigmetrics.tex` already builds in the Overleaf checkout;
the prior formatting audit found 17 technical-content pages and 19 total.
Results have not been revised by that conversion or this tracking task. Retain
the HPCA entry and regenerate claims after the evidence freeze. The inherited
missing `tab:architectural-expertise` reference also needs resolution.

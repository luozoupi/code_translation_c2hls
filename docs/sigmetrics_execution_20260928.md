# SIGMETRICS revision execution — September 28, 2026

> Latest status (September 29, 09:16 CDT / 14:16 UTC): historical registry
> recovery is **running with four workers**. Of 405 distinct validations,
> **214 finished: 185 pass, 16 fail, 13 timeout**; four are active and 187 queued.
> Eleven failures have a confirmed FDTD testbench dump-framing defect; the
> other five have numerical RTL postcheck mismatches. Preserve raw statuses
> pending separate correction/revalidation. [Session report](../artifacts/sigmetrics2027/status/20260929T141635190065Z/report.md).
> The six-cell recipe pilot is complete; the 216-cell main study has not launched.
> The manuscript visual restoration was pushed to Overleaf as `ff75c62`.



The correctness pilot passed. The user explicitly approved sending the prepared
prompts to DeepSeek and OpenAI. The frozen six-cell recipe pilot was launched
at **2026-09-29T02:15:02.672359+00:00**, with two
workers and the existing $300 API cap. The earlier automatic approval rejection
is resolved by the explicit authorization saved with the campaign.

This work is in `/home/luo00466/code_translation-c2hls-sigmetrics2027`, branch
`sigmetrics2027-validation`. The original HPCA tree and its frozen evaluator
remain preserved. The bootstrap manifest records the dirty overlay, required
ignored runtime module, public input files, and historical test fixture.
No changes have been committed or pushed.

## Completed evidence and implementation

| Item | Result / evidence |
|---|---|
| Historical queue drain | Stopped cleanly after 117 distinct validations: **85 RTL passes, 21 oracle/registry errors, 9 numerical CSim failures, 2 timeouts**; 1,624 distinct queued identities remain unresolved |
| Both-cohort validator pilot | **2/2 RTL passes**, one saved ATAX design per harness cohort; engineering smoke, not an efficacy sample |
| Corrected registry recovery | Frozen manifest for **535 rows / 405 distinct identities**, each linked to its previous validation ID and unchanged candidate hash; prepared, not launched |
| New public inputs | Twelve core kernels × five domain-valid inputs: **60/60 native oracle passes** |
| Fresh ATAX parent | Unchanged public C, no expert HLS and no model calls; **5 CSim + 1 selection synthesis + 5 RTL co-simulations**, all passed; each RTL case **2,652,285 cycles** |
| Provider compatibility | DeepSeek Flash and GPT-5.6 Luna accepted a tiny READY-only request; both usages settled in the global ledger |
| Jacobi root causes | Nine historical numerical failures inspected; separate CPU diagnostic repairs pass **9/9** against the original golden outputs and unchanged tolerances |
| Recipe pilot | Six exact prompts and all inputs frozen; approved pilot running |

The final historical snapshot is
[`20260928T233555584087Z`](../../code_translation-c2hls-hpca2027/artifacts/experiment_status/20260928T233555584087Z/report.md).
Historical row counts differ from deduplicated validation counts. The historical
speed-sorted queue is not an unbiased success-rate sample.

The new evaluator chooses an exact benchmark/testbench-hash registry and fails
closed on ambiguity or drift. An explicit incorrect registry never falls back.
C and RTL postcheck dumps share one comparator, including every complete dump;
a genuine RTL postcheck is required. The public baseline adapter preserves the
benchmark identity. A synthesis-only run no longer receives an executed-RTL
success flag.

Cost accounting distinguishes LLM candidates, deterministic search, and a
retained parent. Native checks, CSim, selection synthesis and the resynthesis
inside each RTL flow are recorded separately. The API ledger uses locked
reservations, disables hidden SDK retries, retains reservations with unknown
usage, enforces the aggregate input allowance, and blocks subsequent requests
if observed usage exceeds its reservation. Dollar amounts are estimates at
frozen rates, not provider invoices.

The opt-in five-case validator verifies source/header, harness, shape-registry,
and native-golden hashes before execution. It stops at the first failed case;
only five passes constitute success. It reports the **maximum measured RTL
cycles** across those cases. A selected design that fails RTL remains failed;
there is no search for a replacement selected design after that failure.

## Verification

The final focused regression run passed **206 tests and four subtests** in
4.23 seconds; results are saved in
[`validation_tests.xml`](../artifacts/sigmetrics2027/validation_tests.xml).
The exact prompt audit passed **21 checks**: matching messages across providers
within each arm, absent recipes in the control, and complete frozen recipe
renders in both guidance arms. `git diff --check` passed. All six original
historical evaluator hashes still match their frozen manifest.

The two READY-only compatibility checks cost an estimated **$0.0000134**.
The approved pilot adds metered experiment requests to the same ledger. Current
request costs and status are available from the progress snapshot command.

## Input and comparison decisions recorded before model outcomes

The core is GEMM, 2MM, ATAX, BiCG, Jacobi-2D, Seidel-2D, LU, TriSolve, Durbin,
Gram–Schmidt, Nussinov and Floyd–Warshall. LU inputs are well conditioned;
TriSolve diagonals are nonzero; Durbin uses stable autocorrelation inputs;
Nussinov symbols and Floyd–Warshall edges stay in their valid domains.

Historical Gram–Schmidt has M=60,N=80, so its columns cannot all be independent.
The new, explicitly named `hlsfactory_gramschmidt_m60_n40` variant uses M=60,N=40
and well-conditioned inputs. It needs its own parent and measurements. It must
not be silently merged with the old benchmark results.

Use fresh unchanged-public-C parents for the new controlled study, with
construction costs separated. This improves the reference boundary compared
with reusing historical parents. Each parent must pass all five cases before
optimization. Parent failures remain visible in the prespecified benchmark
coverage; do not replace them with a hand-optimized design or drop a kernel
silently.

The comparison isolates **additional recipe representation**. Every arm retains
the same generic HLS prompt and synthesis feedback. The arms are no added
recipes, flat same-content guidance, and structured recipes. Flat and structured
retain the same complete statements, guards and templates. Recipe IDs are chosen
once from the initial parent and frozen; outcomes cannot change their selection.
The pilot IDs are `hls-pipeline-hot-loop-achieve-ii`, `prompt-pipeline`, and
`hls-pipeline-stage-global-memory`.

The six-cell ATAX pilot uses two providers × three arms, **one model request and
one selection synthesis per cell**, 16,384 maximum response tokens and 100,000
aggregate input tokens. It is an engineering test, excluded from efficacy
claims. The proposed main study remains **216 cells**: twelve kernels × two
providers × three arms × three repeated requests, with five model calls and
five selection syntheses per run. Seed fields are not evidence of seeded
sampling; the client records that these requests have no supported seed control.

## Run and artifact locations

| Artifact | Path relative to this worktree |
|---|---|
| Original-tree overlay and hashes | `artifacts/sigmetrics2027/bootstrap/manifest.json` |
| Historical drain receipt | `artifacts/sigmetrics2027/historical_drain_request.json` |
| Both-cohort RTL pilot | `results_sweeps/sigmetrics_validator_pilot_20260928/summary.json` |
| Corrected historical recovery | `results_sweeps/historical_registry_recovery_20260928/manifest.json` |
| Public inputs and 60 native checks | `artifacts/sigmetrics2027/core_inputs_v1/` |
| Fresh parent, cost and RTL evidence | `artifacts/sigmetrics2027/parent_pilot_20260928/atax/measurement.json` |
| Six-cell registration and source snapshot | `results_sweeps/sigmetrics_recipe_pilot_20260928/manifest.json` |
| Exact outgoing prompt previews | `artifacts/sigmetrics2027/recipe_prompt_preview_final_20260928/index.json` |
| Shared $300 API cap | `artifacts/sigmetrics2027/api_budget.json` |
| Numerical-failure diagnoses | `artifacts/sigmetrics2027/jacobi_failure_diagnosis/diagnosis.json` |
| Refreshable progress snapshots | `artifacts/sigmetrics2027/status/` |

Refresh current state without launching jobs:

```bash
python3 scripts/snapshot_sigmetrics_progress.py
```

Explicit export authorization is recorded and the pilot is running in tmux
`c2hls_sigmetrics_recipe_pilot_20260928`. Its launch command is shown below (absolute `--output` is
required when launching outside this worktree):

```bash
/home/luo00466/.conda/envs/py310_2/bin/python -u \
  results_sweeps/sigmetrics_recipe_pilot_20260928/source/scripts/run_sigmetrics_recipe_pilot.py \
  run --output /home/luo00466/code_translation-c2hls-sigmetrics2027/results_sweeps/sigmetrics_recipe_pilot_20260928
```

The controller uses at most two workers and stops submitting cells after an
execution/accounting failure. Completed, failed or interrupted cells are not
silently rerolled. Its `state.json` and each `cells/<id>/result.json` are the
canonical progress and accounting records. The full matrix is not automatically
launched after the pilot.

## Next gates

1. Complete the six-cell engineering pilot and reconcile provider reservations,
   token counts, candidate identities, selection syntheses, and all RTL flows.
   Check actual rendered recipe IDs, reference isolation, selected-source hashes
   and the five-case evidence. A correctness failure can be a valid recorded
   result; an accounting failure cannot pass this gate.
2. Measure the remaining eleven public-C parents and finalize the 216-cell
   registration with immutable parent/recipe snapshots, timeouts, balanced
   ordering, failure denominators and benchmark-cluster uncertainty. Use the
   pilot costs to verify budget headroom before launch.
3. Run the new controlled comparison before giving the 405-job historical
   recovery queue sustained resources. Complete the zero-LLM numeric DSE and
   runnable source-transforming baseline under the same resource accounting.
4. Build the BFS, KMP and pipeline adapters/oracles; freeze the prospective
   router before their outcomes. Evaluate the existing Qwen base/SFT adapter
   under matching conditions; keep exploratory RL separate.
5. Update paper claims only from joined source/report/correctness/cost evidence.
   Historical synthesis estimates, new measured RTL cycles, and post-route
   implementation are separate outcomes. Routed timing remains unmeasured.

The separate ACM SIGMETRICS manuscript entry remains in the Overleaf checkout.
This turn changes evaluation infrastructure and adds diagnostic evidence; it
does not promote engineering smoke results into headline paper improvements.

## Approved execution update — September 29 UTC

The first completed cell (`deepseek-flash_none`) passed all five selected RTL
cases, with its one provider request, one selection synthesis and five RTL
resyntheses reconciled. Subsequent cells run through the two-worker controller.
Use `python3 scripts/report_sigmetrics_recipe_pilot.py` for a dated, independent
check of sent-prompt hashes, provider costs, parent retention and all five RTL
case records.

A pilot export-label defect was found: the legacy sanitizer rewrites the
public-only top-level `ground_truth_status` to `valid`, although the nested
reference record correctly says `not_evaluated`, has no expert report, and
records no expert synthesis. The development sanitizer and timing-scope labels
are corrected for future campaigns. The running pilot source and raw results
remain frozen; its independent report explicitly sets `expert_reference_status`
to `not_evaluated` and records the legacy-label warning. This defect does not
alter the prompt, candidate selection, native golden, or RTL measurements.

The two historical timeout archives both reached XSim's `run all` stage, after
synthesis and elaboration. One ends at the initial 0/1 progress marker; the
other contains 2,575 DSP OPMODE warnings and ends at simulated time 7,012,330 ps.
Neither has a completed RTL measurement. These observations locate the timeout
stage but do not establish a simulator deadlock or the cause of slow execution.
No timeout replay was launched. Extracted evidence and log hashes are in
`artifacts/sigmetrics2027/historical_timeout_diagnosis_20260929/triage.json`.

A lightweight completion monitor is running in
`c2hls_sigmetrics_pilot_report_20260928`. It makes no model or HLS calls. When
the pilot controller exits its running state, it creates a dated independent
pilot report and progress snapshot, then records their paths in
`results_sweeps/sigmetrics_recipe_pilot_20260928/completion_monitor.json`.
This provides a durable final report even when the interactive turn has ended.

## All prepared prompts delivered — September 29, 02:37 UTC

All **six** requests were accepted (three DeepSeek Flash, three GPT-5.6 Luna),
and every sent prompt hash matches its approved local preview. All six provider
usages are settled. The final delivery audit is
`results_sweeps/sigmetrics_recipe_pilot_20260928/prompt_delivery_audit.json`.
Four cells have completed RTL validation and accounting; the two remaining
OpenAI cells continue under the same frozen controller. The completion monitor
will write the final report after both finish. No larger matrix was launched.

## Pilot completion and architectural writing pass — September 29 UTC

The frozen controller completed all six cells at 02:43:27 UTC. The independent
completion report passed its evidence/accounting gate at 02:43:33 UTC:
[`final pilot report`](../artifacts/sigmetrics2027/pilot_reports/20260929T024333638806Z/report.md).
All six selected sources passed five RTL cases; DeepSeek-flat and OpenAI-flat
retained the public-C parent after candidate feasibility failures. Four cells
selected new generated designs. All approved prompts matched and usage was
settled. Estimated pilot API cost is $0.0079511, or $0.0079645 including provider
compatibility probes. This remains a one-kernel engineering pilot. The 216-cell
main study and remaining eleven parent measurements have not been launched.

The manuscript now has a separate architecture-focused prose revision, with
hardware background, precise measurement definitions, a constraint-to-rewrite
table, archived TRMM resource tradeoffs, and a reviewer-linked writing plan:
[`SIGMETRICS writing plan`](../../6a3058ea0542c2fa6f168fe8/SIGMETRICS_WRITING_PLAN.md).
The first pass uses `sections/sigmetrics/`; the original HPCA entry and source
sections are preserved. Existing historical data macros were not changed.
The ACM draft builds with 15 technical-content pages and 17 total pages,
resolved references, no overfull boxes, and embedded fonts. Its abstract and
primary evaluation remain provisional until controlled results are available.
No experiment code, frozen pilot inputs, or raw result records were changed by
the writing pass.

## Historical recovery launched at the user's request — September 29 UTC

The user requested the COSIM backfill under the SIGMETRICS worktree, superseding
the earlier scheduling preference to defer sustained historical recovery.
The already prepared corrected-registry campaign began at 04:22:30 UTC with two
workers in `c2hls_sigmetrics_cosim_recovery_20260929`. It covers 405 distinct
saved designs / 535 rows and makes no model requests. Original records remain
preserved; every new measurement is attached to the versioned frozen manifest.
The wider unresolved historical inventory is outside this subset.

See [the launch and monitoring instructions](historical_cosim_backfill.md#sigmetrics-registry-recovery-launched--september-29-2026-utc).
The progress snapshot now reads the recovery controller and summary dynamically
rather than reporting the old “unlaunched” status. Read `state.json` and
`summary.json` in `results_sweeps/historical_registry_recovery_20260928/` for
current counts; the runner refreshes them throughout the queue.

Initial launch verification at 04:24 UTC: two distinct validations completed
successfully (Jacobi-1D: 4,227 RTL cycles; BiCG: 17,522 RTL cycles), two were
active, and 401 remained queued. The website-schema export had zero invalid
records, and the frozen manifest hash still matched the launch receipt. These
are historical single-harness checks, distinct from the new five-case study.

## Four-worker scheduling handover

The user requested two additional parallel Vitis validations. A separate frozen
scheduling wrapper now waits for the original controller to finish draining,
then resumes the same deduplicated queue with four workers. Read
`parallel_scheduler_status.json` beside the campaign `state.json` for the actual
handover state. The replacement tmux session is
`c2hls_sigmetrics_cosim_x4_20260929`. Existing measurements and all frozen
validation source/input hashes are preserved; 24 focused checks passed.

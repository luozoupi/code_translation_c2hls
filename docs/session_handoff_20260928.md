# C2HLS Session Handoff: September 28, 2026

This is the entry point for another coding/analysis session. It documents the
current local implementation, recent experiments, data locations, and operational
limits. Historical descriptions are not permission to relaunch API experiments.
Read live state before acting; the numerical snapshot below was checked around
2026-09-28 20:34 UTC (15:34 America/Chicago).

## Start Here

1. Read this file and [historical_cosim_backfill.md](historical_cosim_backfill.md).
2. Check the running July COSIM queue's `state.json`, `summary.json`, and tmux
   session before editing experiment code or starting Vitis.
3. Use the [directory index](../artifacts/handoff/handoff_index_20260928.md) for
   browsing and [machine-readable index](../artifacts/handoff/handoff_index_20260928.json)
   for exact paths, archive links, repository identities, and source hashes.
4. Read `git status --short`. Many changes are uncommitted and several important
   implementation/test files are untracked. Do not reset, clean, discard, or
   assume a fresh checkout contains the current implementation.

The index is a bounded metadata inventory, not a recursive listing of every
Vitis intermediate. It lists the artifact/results roots, their direct entries,
and one level of child names. It never reads credentials or raw prompt/log text.

## Repositories And Environment

| Purpose | Absolute path | Identity checked at handoff |
|---|---|---|
| Main working tree | `/home/luo00466/code_translation-c2hls-hpca2027` | `hpca2027-paper-eval`, HEAD `a778848863a3ee51d3048c6989480882a546e3e0`, dirty |
| Legacy experiments and training utilities | `/home/luo00466/code_translation-c2hls` | `c2hls-enhanced`, HEAD prefix `1164311` |
| Imported skill-v3 / benchmark snapshot | `/home/luo00466/code_translation-c2hls-enhanced-fc27133` | detached `fc27133` |
| Additional source checkout | `/home/luo00466/code_translation_c2hls` | Separate underscore-named tree; inspect the generated index before using |
| Python used for current tests/runners | `/home/luo00466/.conda/envs/py310_2/bin/python` | Existing environment, not recreated by this work |
| Preferred Vitis settings | `/mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh` | Vitis 2023.2 |
| Device / clock / flow | `xcu280-fsvh2892-2L-e` / `3.33 ns` / `vitis` | U280 HLS part, not an arbitrary platform string |

Commands below assume the main working tree as the current directory. Historical
README examples are not universal campaign budgets; use each run's manifest.
In this session ordinary sandbox commands failed before execution with
`error building bubblewrap command: mountinfo path is not absolute`. Reviewed
escalated commands worked. This is an execution-environment issue, not a Vitis
or project error; a future session may not have the same restriction.

## Current Live Work

### July COSIM Backfill

- tmux: `c2hls_july_cosim_20260928`.
- Compact root: `results_sweeps/july_cosim_backfill_20260928/`.
- Bulk root: `/mnt/data2/luo00466/c2hls_rl/july_cosim_backfill_20260928/`.
- Active scratch: `/tmp/july_cosim_backfill_20260928/<validation_id>/`.
- Scan scope: July 1-31 by campaign date, including July campaigns whose last
  cells finished in August; both principal trees, one-shot evaluations, corrected
  matrix, and skill-dose trials. No new model calls or training.
- Frozen inventory: 2,553 rows across 65 campaigns. 2,174 eligible rows map to
  1,741 distinct code/harness/validator identities; 16 existing attributable
  COSIM passes are retained. 363 rows are blocked: 189 missing selected source,
  89 selected-source/report conflicts, 51 summary/report cycle conflicts, and
  34 without successful selected synthesis.
- Snapshot: two new passes from the smoke, two active validations, 1,737 pending.
  Read live state for newer counts. Source rows and distinct validations are
  different denominators; deduplicated results can update multiple setup rows.
- Smoke: ATAX 107,885 RTL cycles, BiCG 38,565 RTL cycles; both pass candidate
  CSim and independent native-oracle checks of RTL postcheck outputs.
- Two isolated workers maximum; no unrelated Vitis process was present at
  launch. Resource gate waits on unrelated Vitis/XSIM, load, and free space.
- Candidate RTL must be rebuilt, so a COSIM invokes candidate synthesis.
  Reference CSYNTH/COSIM is not rerun. Native plain-C oracle execution is not a
  reference HLS synthesis run.

Primary files in the compact root:

| File | Meaning |
|---|---|
| `manifest.json` | Frozen rows, input identities, toolchain, evaluator hashes |
| `state.json` | Controller status, PID, pending/active jobs, resource gate |
| `summary.json` | Row counts, distinct completed measurements, schema validation |
| `inventory_and_results.csv` / `.jsonl` | Full source provenance and separate CSYNTH/COSIM fields |
| `campaign_summary.csv`, `by_campaign/*.csv` | Per-campaign coverage and measurements |
| `schema_records.jsonl` | Executed RTL pass/fail/timeout records only |
| `report.md`, `launcher.log` | Human status and queue launch/completion messages |
| `preflight_manifest_v1.json` | Preserved unlaunched inventory from before the validator correction |

The bulk root contains `inputs/`, `recovered/`, `measurements/`, `logs/`, and
`vitis_archives/`. Archive receipts record checksums and full `tar --compare`
verification. Scratch is deleted only after successful archive verification.

Companion tables are refreshed without overwriting original historical tables:

- `results_sweeps/sonnet46_skillv2_latency_matrix_20260726/sonnet46_skillv2_latency_records_long.cosim_backfill.csv`
- `results_sweeps/skill_v3_qwen_gemma_matrix_20260731/all_rows.cosim_backfill.csv`
- `results_sweeps/qwen_gemma_oneshot_agentic_sft_20260729/comparison_records.cosim_backfill.csv`

The first preflight found that Vitis emits both C-test and RTL-postcheck dumps.
The generic single-dump parser falsely rejected repeated output names. The
backfill-only validator now requires the RTL-postcheck section and checks every
complete dump independently. Mismatches and malformed/missing sections fail;
it does not merely accept the last good dump or relax numerical tolerances.
Original failed preflight evidence remains in
`artifacts/analysis/july_cosim_smoke_20260928/`; corrected evidence is in
`artifacts/analysis/july_cosim_smoke_v2_20260928/`. The kernels were unchanged.
Validator changes alter validation IDs, so preflight failures are not silently
overwritten or counted as current measurements.

### Completed Reviewer A/B

Compact root: `results_sweeps/deepseek_flash_reviewer_ab27_20260925/`.
Bulk root: `/mnt/data2/luo00466/c2hls_rl/deepseek_flash_reviewer_ab27_20260925/`.

Completed September 26 at 05:27 UTC with 323 audit-valid/feasible final selections
and one excluded run out of 324. The 161 valid matched pairs give 91 wins,
54 regressions, 16 ties for reviewer-on; geomean cycle speedup is 1.6683x, with
benchmark-clustered bootstrap interval [1.1048, 2.4647]. Feasible final selections
may include frozen-baseline fallback; this is not 323 valid generated candidates.

The excluded cell is `hlsfactory_trisolv_flash_r2_review_on`. Its assistant
response triggered an `absolute_reference_metric` audit finding. The flag is
not by itself proof of actual leakage. Keep it excluded unless a documented
forensic review justifies a separately versioned analysis. Do not reroll the
sample or weaken the audit to improve the result.

Registration is `artifacts/experiment_matrix/deepseek_flash_reviewer_ab27_20260925.json`.
Frozen Phase-B code/reports are under
`artifacts/experiment_matrix/deepseek_flash_reviewer_ab27_20260925/`, including
`phase_b_manifest.json` and `code/`. Results include `arm_measurements.csv/jsonl`,
`paired_measurements.csv/jsonl`, `summary.json`, `report.md`, `schema_records.jsonl`,
`campaign_state.json`, and `manifest.json`.

This is 27 kernels x flash/multistep x review-off/on x three independent samples
per benchmark/mode/arm combination. Same frozen Phase-B parent, catalog, toolchain, and five-call/five-synth
caps; no numeric parameter search and no COSIM. Flash allows two attempts in
both arms so review feedback can reach a later prompt. DeepSeek does not receive
the unsupported seed parameter; repeated calls are not guaranteed different
outputs or reproducible seeded replicas. The exact model alias is `deepseek-flash`,
not `deepseek-v4.1-flash` and not an immutable weight revision.

Auditor-only smoke amendments are retained under the compact root's
`auditor_amendments/` and in manifest amendment history. They fixed rejection of
a disabled parameter-search placeholder and reporting status, not generation
inputs. The original smoke samples were retained. Separate review-only versus
tradeoff-prompt-only component ablations have not been established by this A/B.

## Code Update Map

The following files are present locally; not all are committed. The generated
index records current changed-file SHA-256 values and git status.

| Area | Main files | Update / limitation |
|---|---|---|
| Controlled QoR parameter search | `qor_design_space.py`, `hls_feedback.py`, `scripts/run_qor_ofat_campaign.py`, `scripts/run_saved_qor_step_smoke.py` | OFAT candidates, achieved-versus-requested II, realization evidence, feasibility gates, bounded controls and experiments |
| Coupled hardware controls | `qor_linked_controls.py` | GEMM lanes/banks/striped reductions and step-specific parent transformations; transformed parents require fresh CSim/CSynth |
| QoR analyses | `scripts/analyze_qor_ofat_campaign.py`, `scripts/analyze_gemm_linked_lane_campaigns.py`, `scripts/analyze_gemm_parameter_rollout.py`, `scripts/compare_qor_methodology_benchmarks.py` | Candidate tables, within-parent parameter trends, GEMM/2MM comparisons, plots |
| Model-paired parameter campaigns | `scripts/run_qor_paired_model_campaign.py`, `scripts/analyze_qor_paired_model_campaign.py` | Frozen inputs, resource-gated workers, before/after numeric-search measurements and exports |
| Skill/profile reviewer | `qor_profile_review.py`, `c2hls.py` | `C2HLS_QOR_PROFILE_REVIEW=1`; deterministic evidence review, per-reply declarations, profile/resource tradeoff guidance and bounded feedback retries |
| Reproducibility | `evaluation_repro.py` | Reviewer flag/source hashes, frozen Phase-B seed manifest, and sample identity in fingerprints |
| Reviewer A/B | `scripts/freeze_qor_ab_seeds.py`, `scripts/run_qor_review_ab.py`, `scripts/analyze_qor_review_ab.py`, `scripts/amend_review_ab_auditor.py` | Exact measured seed recovery, paired queue, audit gates, clustered analysis, tightly scoped audit amendment tooling |
| July COSIM backfill | `scripts/scan_historical_cosim.py`, `scripts/run_historical_cosim.py` | Archive/source recovery, immutable staging, resume/dedup, independent output checks, bounded workers, schema and companion tables |
| Handoff index | `scripts/build_session_handoff_index.py` | Bounded metadata inventory and selected live-state snapshots; no launches or raw secret/log ingestion |

Relevant tests: `tests/test_qor_design_space.py`, `test_qor_axi_controls.py`,
`test_qor_ofat_campaign.py`, `test_gemm_parameter_rollout.py`,
`test_qor_methodology_comparison.py`, `test_qor_paired_campaign.py`,
`test_qor_profile_review.py`, `test_qor_review_ab.py`,
`test_evaluation_repro.py`, and `test_historical_cosim.py`.
The last full suite after backfill implementation passed 382 tests, skipped 29,
and passed 57 subtests. Two actual Vitis integration smokes passed separately.

### Workflow And Scientific Meaning

`c2hls.py` has four named agent classes: `TranslatorAgent`, `SynthesisAgent`,
`QualityRepairAgent`, and `FeedbackAgent`, coordinated by `C2HLSOrchestrator`.
The orchestrator also issues optimization model calls recorded under the
`orchestrator` role. Classes, recorded model-call roles, and active SFT actors
are not interchangeable counts; some paths are deterministic or disabled.

The new reviewer is deterministic, not another review-LLM call or formal proof
that a skill is correctly implemented. It compares source-family changes,
declared/exposed skills, profile bottlenecks, cycles, II, and resource deltas.
Comments/strings do not count as implementation evidence. It encourages useful
resource spending while retaining correctness, timing, and actual device-fit
gates. Selection remains latency-first among feasible candidates, with Pareto
relations recorded; it is not a general configurable multiobjective optimizer.

More prompted skills are not guaranteed to reduce cycles. Parent-retaining
tournaments can make the best feasible measured result non-worsening as valid
candidates are added; this is different from assuming every extra skill improves
one generated kernel. Pipeline target II is also not an inverse-linear predictor
of whole-kernel latency. Achieved II, dependencies, memory bandwidth, recurrence,
loop trip count, and design realization must be inspected.

## Results And Artifact Map

All project-relative paths below are under the main tree unless marked legacy.
The generated index lists every direct campaign directory at these roots.

| Topic | Compact location / entry point | Bulk location |
|---|---|---|
| September reviewer-enabled system plus numeric search | `results_sweeps/deepseek_flash_qor_profile27_20260923/` | `/mnt/data2/luo00466/c2hls_rl/deepseek_flash_qor_profile27_20260923/` |
| DeepSeek-Pro paired 12-kernel search | `results_sweeps/deepseek_v4_pro_qor_paired12_20260906/` | `/mnt/data2/luo00466/c2hls_rl/deepseek_v4_pro_qor_paired12_20260906/` |
| DeepSeek-Pro full paired search / AXI correction | `results_sweeps/deepseek_v4_pro_qor_paired27_axi_v2_20260908/` | `/mnt/data2/luo00466/c2hls_rl/deepseek_v4_pro_qor_paired27_axi_v2_20260908/` |
| GEMM / 2MM parameter families and visuals | `results_sweeps/qor_gemm_*`, `qor_2mm_all_step_parameter_rollout_20260901/`, `qor_methodology_gemm_vs_2mm_20260901/` | Named `qor_gemm_*` directories under `/mnt/data2/luo00466/c2hls_rl/`; consult each manifest |
| Multi-model combined tables including external Devstral data | `results_sweeps/full_model_comparison_20260816/README.md`, `all_available_model_setups_long.csv`, `all_available_model_setups_cycles_wide.csv` | Per-row source paths; do not treat imported CSV as locally validated raw kernels |
| July imported comparison data | `results_sweeps/results-20260730/` | Per-source CSV provenance |
| Sonnet skill-v2 matrix, long/wide tables, charts | `results_sweeps/sonnet46_skillv2_latency_matrix_20260726/` | Raw `results_sweeps/agentic_no_streamcluster_hpca_skillv2_*` and moved scratch on NAS1 |
| Qwen/Gemma skill-v3 matrix | `results_sweeps/skill_v3_qwen_gemma_matrix_20260731/` | `/mnt/data2/luo00466/c2hls_rl/skill_v3_fc27133/{model}/` |
| Qwen/Gemma base, one-shot, SFT comparisons | `results_sweeps/qwen_gemma_oneshot_agentic_sft_20260729/` | Per-row source paths; some historical one-shot summaries are compressed on NAS1 |
| Corrected ten-setup matrix/tournament | `artifacts/corrected_setup_matrix_v2/`, `results_sweeps/corrected_v2_setup_tournament_20260728/` | `/mnt/data2/luo00466/c2hls_rl/corrected_setup_matrix_v2/` |
| Skill-dose pilot and smoke | `artifacts/skill_dose_pilot_v1/`, `artifacts/skill_dose_smoke_v1/`, `results_sweeps/skill_dose_and_corrected_v2_20260726/` | `/mnt/data2/luo00466/c2hls_rl/skill_dose_pilot_v1/` and `skill_dose_smoke_v1/` |
| Router corpus/models/architecture/top-k comparisons | `artifacts/setup_router/`, `results_sweeps/setup_router_*`, `results_sweeps/strengthened_setup_router_20260726/` | `/mnt/data2/luo00466/c2hls_rl/setup_router/`, `router_lora/` |
| External MachSuite router work | `results_sweeps/machsuite_router_corrected_v2_20260727/`, `artifacts/setup_router/machsuite_external_comparison_v1/` | `/mnt/data2/luo00466/c2hls_rl/machsuite_router_corrected_v2_20260727/` |
| Earlier Qwen COSIM supplement | `artifacts/qwen_cosim_enrichment/strong_gains_20260725_{staged,cosim}/` | `/mnt/data2/luo00466/c2hls_rl/vitis_work/qwen_strong_gains_cosim_20260725/` |
| Campaign registrations / launch scripts | `artifacts/experiment_matrix/` | Script-specific NAS roots; full index enumerates registrations |
| Reusable measured Phase-B seeds | `artifacts/phase_b_seeds/`, reviewer A/B seed directory above | Per-manifest provenance |
| Reference caches | `artifacts/reference_validation_cache/`, legacy counterpart | Reuse only matching code/input/toolchain fingerprints |
| Reference website records | `/home/luo00466/schema_records.jsonl` | Reporting/evaluation only, not model/router input |

The September 23 campaign measures the updated reviewer-enabled system plus
numeric search; its pre/post search effect is **not** the September 25 reviewer
on/off effect. Similarly, external COSIM CSVs, CSYNTH estimates, and executed
COSIM are not interchangeable performance measures.

Canonical inputs live in
`benchmarks_external/HLSFactory/polybench_float_small/hlsfactory_<name>/`,
with `plain.cpp`, header, public testbench, and metadata. Skill-v3 runs also use
the imported `code_translation-c2hls-enhanced-fc27133/benchmarks/` tree.
Use each artifact's recorded benchmark path/hashes instead of guessing that
two directories contain identical shapes. Libraries are `skills/`, `skill_v2/`,
and `skill_v3_no_rmw_fc27133/`; preserve catalog hashes and behavior versions.

## SFT And Learned Router Context

Read [offline_sft_workflow.md](offline_sft_workflow.md) before claiming any SFT
result. Training utilities primarily live in the legacy tree. Large data,
adapters, evaluation outputs, and serving metadata live under:

```text
/mnt/data2/luo00466/c2hls_sft/{corpus,runs,eval,serving,source,cache}
/home/luo00466/code_translation-c2hls/artifacts/rl_corpus
```

Documented completed Qwen training includes role-selective and active-actor
QLoRA r16 runs. The active-actor adapter is
`/mnt/data2/luo00466/c2hls_sft/runs/qwen3.6-27b_active_llm_actors_qlora_r16_1epoch_3072_20260724/adapter`.
Its recipe uses completion-only loss, NF4/double quantization, BF16 compute,
rank/alpha/dropout 16/32/0.05, one epoch, 3,072 tokens, and no sequence packing.
This is offline supervised behavioral cloning, not an online RL/GRPO run.
The active training corpus has translator/synthesis/orchestrator examples;
it does not establish training of dormant feedback/quality-repair LLM paths.
Gemma corpus exports and failed load gates exist; the documented workflow does
not establish a completed Gemma adapter. Directory names alone are not proof
of successful training.

The role-attributed SFT split uses `gramschmidt` validation and
`durbin,floyd_warshall,gemm,trmm` test; the 22 other valid kernels train.
The learned **setup router** instead holds out four validation kernels:
`atax,gramschmidt,jacobi_2d,nussinov`; test is the same four above; the remaining
19 valid kernels train; `doitgen` is excluded. Check the precise manifests:

- `artifacts/setup_router/corpus_v1/split_manifest.json`
- `artifacts/setup_router/corpus_corrected_v2/split_manifest.json`
- the SFT corpus manifest linked in `offline_sft_workflow.md`

Router decisions use source/pre-generation and Phase-B information, not gold
cycles or post-candidate winner features. Lineage-level splits keep duplicated
PolyBench tasks together. Do not describe grouped-development/top-k cross-
validation numbers as four-kernel final-test results.

The corrected router label space has ten setups: flash/multistep crossed with
`skillless`, `matched_positive`, `smart_best_fit_v2`, `smart_exhaustive_v2`, and
`all_positive_preconditions`. `legacy_v1` results are a different behavior
version. Routing/selection entry points are `setup_router.py`,
`smart_skill_router.py`, `scripts/run_setup_tournament.py`, and the router
training/evaluation scripts listed in the generated directory index.

Architecture reports include ExtraTrees, MLP/RankNet, embedding/hybrid, and
small-LM/LoRA comparisons. Start with
`artifacts/setup_router/topk_architecture_comparison_corrected_v2_20260728/report.md`
and its `topk_metrics.csv`, `per_benchmark_topk.csv`, and `metrics.json`.
The older `router_architecture_comparison_v1/report.md` recommends keeping its
committee advisory and not promoting its small-LM LoRA. Do not infer production
promotion from a good development score. Model-serving endpoints changed
repeatedly; neither Qwen/Gemma endpoint availability nor GPU idleness was
revalidated for this documentation task.

## Storage And Archives

Bulk active outputs use NAS2 `/mnt/data2/luo00466/c2hls_rl`; SFT uses `c2hls_sft`.
Keep active Vitis scratch on local `/tmp` for throughput, then archive verified
outputs. Avoid an unbounded recursive `du`, `find`, or `rg` over NAS Vitis trees.

Archive roots and manifests:

- `/mnt/data1/luo00466/c2hls_archive_20260811/manifest.jsonl`
- `artifacts/archive_manifests/c2hls_archive_20260811.md`
- `/mnt/data1/luo00466/research_archive_20260917/` for separate research archives;
  do not change CoLM/proposal outputs as part of C2HLS experiment work.
- `/mnt/data2/luo00466/code_translation-c2hls_archive_20260706_1245/`
- `/mnt/data2/luo00466/home_cleanup_20260706_approved/`

Several archive symlinks point at directories containing `payload.tar.zst`,
`metadata.json`, and `RESTORE.txt`, not transparently restored historical trees.
For example, legacy `results_sweeps` and both trees' `artifacts/vllm_vitis_smoke`
are affected. Verify the bundle checksum and recover selected members; do not
assume a nonexistent path means the artifact is lost or replace archive links.
The backfill has already recovered compact files under its NAS2 `recovered/`.

Moved intact scratch directories include:

```text
/mnt/data1/luo00466/tmp_c2hls_hpca_skill_v2_20260724
/mnt/data1/luo00466/tmp_c2hls_qwen_sft_all_agents_valtest5_nocosim_20260724
/mnt/data1/luo00466/tmp_c2hls_qwen_active_actor_sft_valtest5_nocosim_20260724
```

## Operations And Next Work

Read-only checks:

```bash
cd /home/luo00466/code_translation-c2hls-hpca2027
git status --short
tmux list-sessions
cat results_sweeps/july_cosim_backfill_20260928/state.json
cat results_sweeps/july_cosim_backfill_20260928/summary.json
tail -n 30 results_sweeps/july_cosim_backfill_20260928/launcher.log
```

Refresh this metadata inventory without launching experiments:

```bash
/home/luo00466/.conda/envs/py310_2/bin/python scripts/build_session_handoff_index.py
```

Only after confirming the controller is stopped, refresh the backfill tables:

```bash
/home/luo00466/.conda/envs/py310_2/bin/python scripts/run_historical_cosim.py report
```

The live controller already refreshes tables. Avoid concurrent manual report
writers because they use the same output paths. To resume after an interruption,
first inspect tmux/processes and ensure no old workers remain, then run:

```bash
/home/luo00466/.conda/envs/py310_2/bin/python -u scripts/run_historical_cosim.py run --workers 2
```

The queue lock rejects duplicate controllers. Finished pass/fail/timeout results
are reused rather than retried. SIGTERM/SIGINT on the controller drains active
jobs without launching more. Check the recorded PID against its current command
before signaling; never use broad `pkill` commands. Do not terminate unrelated
Vitis sessions or GPU servers. State PIDs from completed campaigns are stale.

Do not edit any file listed in the live campaign's `manifest.json/code_hashes`
or staged input hashes while it runs. Workers enforce these hashes. Another
session needing implementation changes should use a separate worktree and a
new versioned campaign, while preserving this tree's uncommitted changes.
`scan_historical_cosim.py` refuses to overwrite an existing frozen manifest.

Remaining work, not silently completed by this handoff:

1. Let the July queue finish, inspect numerical failures/timeouts and counts,
   then use the updated tables. Missing or blocked rows remain N/A, not inferred
   successes. No ETA is promised for a 1,741-validation queue.
2. Investigate the 363 blocked historical rows by exact source/report/harness
   identity. Do not synthesize a replacement kernel or substitute gold code
   while retaining the original model/setup label.
3. Investigate the excluded reviewer A/B audit finding without changing the
   frozen sample or treating an audit flag as confirmed leakage.
4. Keep CSYNTH estimates distinct from COSIM cycles and keep historical leakage
   flags visible in downstream plots, corpora, and paper exports.
5. Inspect current manifests and consent before launching new paid model sweeps,
   using external endpoints, changing training splits, or deleting archives.
   Do not read or print API key files in a handoff/index.

Additional documentation: [QoR controls and protocol](qor_design_sweep.md),
[reference-blind evaluation](hpca2027_reference_blind_evaluation.md),
[architecture/source review](hpca2027_architecture_source_review.md),
[experiment matrix](hpca2027_experiment_matrix.md),
[frozen skills](hpca2027_frozen_skills.md),
[paper artifact boundary](hpca2027_paper_artifacts.md),
[freeze normalizer](hpca2027_freeze_normalizer.md), and
[evidence readiness](hpca2027_evidence_readiness.md).

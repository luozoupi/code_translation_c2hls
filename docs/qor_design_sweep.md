# QoR Design Sweep

## Purpose

`QualityRepairAgent` has an opt-in, reference-blind design-space stage for
testing explicit HLS design choices instead of asking an LLM to make another
uncontrolled rewrite. The stage answers two questions:

1. Which tested parameter value gives the lowest feasible CSynth latency?
2. How do latency, achieved II, timing, and resource use move as that parameter
   changes while the rest of the source remains fixed?

It does not assume that a larger optimization factor must improve latency.
Unroll, partition, and tile changes often cross memory, timing, and resource
limits. Monotonicity is measured as evidence, not imposed as a selection rule.

Requested pipeline II is a scheduler target, not a promised speedup. Lowering
II only helps when the changed loop contributes materially to total latency and
Vitis can realize the target without recurrence, memory-port, timing, or
resource limits. Increasing a requested II normally preserves or reduces
throughput; it is useful as an ablation, not as an acceleration direction.

## Execution Contract

1. Freeze the current best HLS source and its CSim/CSynth evidence.
2. Discover explicit controls: pipeline II, unroll factor, array
   partition/reshape factor, stream depth, allocation limit, selected AXI
   interface options, named tile/block/lane constants, binding latency, and safe
   disable-only toggles for existing HLS directives.
3. Record which optimization step produced the frozen best state and spend the
   bounded knob budget on that step's control family first. Then generate
   bounded
   one-factor-at-a-time candidates from the same frozen parent.
4. Compile-check each candidate, run CSim first, and run Vitis CSynth only when
   CSim passes. COSIM is not run during this local search.
5. Optionally combine pairs of values that improved independently, still from
   the frozen parent.
6. Reject candidates lacking passing correctness, timing, resource-fit, or
   positive-latency evidence. Select minimum worst-case CSynth cycles, using a
   tiny resource sum only to break exact latency ties.
7. Promote the winner only when it improves the parent. The framework's normal
   selected-winner COSIM policy may validate it later, outside this stage.

Every candidate records its changed and fixed knob values, source/report
hashes, CSim result, CSynth metrics, feasibility reasons, Pareto membership,
and deterministic winner explanation. Per-knob evidence includes Spearman
correlations against cycles, achieved II, and each resource plus directional
monotonicity violations.

Compiler realization is distinct from source mutation. The synthesis feedback
records Vitis pipeline target/final II decisions and cases where an enclosing
pipeline implies complete unrolling of an inner loop. It also records partial
unroll factors, so requested and effective parallelism remain separate. A
candidate whose source
token changed but whose requested knob was overridden is labeled accordingly;
identical schedule, timing, latency, and resource signatures are reported as no
observable synthesis response rather than evidence of a weak parameter effect.

## Controls

```bash
export C2HLS_QOR_DESIGN_SWEEP=1
export C2HLS_QOR_SWEEP_MAX_KNOBS=4
export C2HLS_QOR_SWEEP_MAX_CANDIDATES=8
export C2HLS_QOR_SWEEP_VALUES=1,2,4,8,16
export C2HLS_QOR_SWEEP_II_VALUES=1,2,4,8
export C2HLS_QOR_SWEEP_TILE_VALUES=4,8,16,32,64
export C2HLS_QOR_SWEEP_WIDEN_VALUES=64,128,256,512,1024
# Optional exact-name selector for an audited linked parent.
export C2HLS_QOR_SWEEP_KNOB_NAME=GEMM_COMPUTE_II
export C2HLS_QOR_SWEEP_INTERACTIONS=1
export C2HLS_QOR_SWEEP_MAX_INTERACTIONS=2
```

Direct CLI example:

```bash
python c2hls.py \
  --bench-dir benchmarks_external/HLSFactory/polybench_float_small/hlsfactory_2mm \
  --multistep \
  --qor-design-sweep \
  --qor-sweep-max-knobs 4 \
  --qor-sweep-max-candidates 8 \
  --qor-sweep-interactions
```

Sweep-driver aliases use the `C2HLS_SWEEP_QOR_*` prefix. The driver adds the
QoR candidate cap to the existing synthesis budget after applying the paper
profile. Set `C2HLS_SWEEP_SYNTHESIS_EVAL_BUDGET` to an explicit total when a
different combined cap is required. For direct `c2hls.py` runs with a bounded
`C2HLS_SYNTHESIS_EVAL_BUDGET`, reserve capacity for both the normal LLM
candidates and the QoR candidates.

## Result Fields

- `quality_repair.design_sweep`: complete parent, candidate, trend, Pareto,
  winner, and configuration evidence.
- `qor_design_sweep`: multistep root alias for the same evidence.
- `synthesis_evaluations.qor_design_events`: deterministic tool-call events.
- `qor_synthesis_evaluation_count`: QoR CSynth calls, separated from LLM
  candidate syntheses but included in total tool cost.
- `run.qor_design_sweep`: effective controls and policies.
- Website JSONL: compact evidence under
  `implementation.origin_meta.qor_design_sweep`; generated code bodies are
  omitted.

Metric terminology:

- `latency_cycles_worst` is maximum top-level invocation latency and is the
  primary objective.
- `interval` is Vitis `SummaryOfOverallLatency/Interval-max`, the cycle spacing
  between starts of top-level invocations. It is not a loop pipeline II and is
  commonly `latency + 1` for a sequential top function.
- `achieved_pipeline_ii` contains per-loop realized IIs. The convenience field
  `achieved_pipeline_ii_max` is only a cross-loop diagnostic; it may refer to a
  different loop than the knob being changed.
- `compiler_realization` records the requested value, matched loop, scheduler
  target/final II or effective complete-unroll factor, and override status.

The stage runs once after the trajectory has selected its best snapshot. It
does not multiply synthesis cost by running a full experiment after every LLM
step. The recorded `parent_origin` and `preferred_knob_kinds` explain which
step-aware controls received priority.

Supported step-aware families are:

| Winning step | Prioritized controls |
|---|---|
| Pipeline | Pipeline II |
| Unroll | Linked compute lanes, then standalone unroll factor |
| Tiling | Tile/block size and partition/reshape factor |
| Double buffer | Existing dataflow, stream/FIFO depth, linked ping-pong tile depth, tile size, and partition/reshape controls |
| Coalescing | AXI widening, outstanding-transaction, and burst-length controls |
| Resource repair | Allocation, binding/resource latency, and existing binding/partition toggles |

Bare existing `m_axi` interfaces can receive an explicit
`max_widen_bitwidth`. Existing `DATAFLOW`, complete partition/reshape,
`BIND_OP`, `BIND_STORAGE`, and legacy `RESOURCE` directives can be disabled for
an ablation. Every such edit remains subject to CSim, CSynth, timing, and device
resource-fit gates.

The stage only mutates source-local controls or disables directives already in
the source. It does not invent a new buffer topology or switch categorical
implementations such as DSP versus fabric or BRAM versus URAM. In particular,
hard-coded ping-pong arrays and `% 2` indexing need a coordinated rewrite, not
an unsafe numeric buffer-count substitution. If no safe knob exists, the stage
records `no safely parameterized QoR knobs discovered` and retains the parent.
An audited source transformer may construct such a coordinated parent before
the sweep; that transformed parent must then pass fresh CSim and CSynth.

## Linked Architecture Controls

A named constant containing `LANE`, `PARALLEL`, or `PE_COUNT` is treated as a
`compute_lanes` control when HLS unroll and partition/reshape directives refer
to it. One candidate changes only the constant token, but its record lists all
dependent directive lines and control types. This permits an architecture
parameter to coordinate compute parallelism and the memory ports needed to
feed it, instead of testing an unroll pragma in isolation.

`qor_linked_controls.build_gemm_linked_lanes_parent` provides two audited GEMM
families used by the controlled campaign:

- `direct_reduction` removes the enclosing output-loop pipeline that previously
  forced an 80-way complete unroll, then links reduction unroll and both
  K-dimension memory banks. This diagnoses the scalar accumulation recurrence.
- `striped_accumulators` gives each lane an independent partial sum and links
  the same value to array extent, loop bounds/stride, three unroll directives,
  and both K-dimension banks. Tested values must divide `NK`.

The transformed parent is always re-run through CSim and CSynth before the
first candidate. A saved report from the pre-transformation kernel is not valid
parent evidence.

`qor_linked_controls.build_gemm_step_control_parent` prepares five additional
GEMM control families on top of the independent-accumulator architecture:

| Step | Named control | Coupled behavior |
|---|---|---|
| Pipeline | `GEMM_COMPUTE_II` | Compute reduction pipeline target; Vitis target and achieved II are recorded separately |
| Tiling | `GEMM_K_TILE` | Compile-time K strip count, inner bound, and reduction index; values divide `NK` and are multiples of the fixed lane count |
| Double buffer | `GEMM_BUFFER_DEPTH` | Both ping-pong B-buffer extents, K-tile load/compute bounds, and tile stride |
| Coalescing | `GEMM_AXI_BITS` | The same `max_widen_bitwidth` on all three `m_axi` ports |
| Resource repair | `GEMM_MUL_IMPL` | Categorical `BIND_OP` choice: 1=`fabric`, 2=`meddsp`, 3=`fulldsp`, 4=`maxdsp`, 5=`primitivedsp` |

These are separate OFAT parents. Fixed lane count, topology, source hash, and
toolchain fingerprint remain constant inside each family. The families are not
combined into one completion, and their absolute cycles should not be treated
as a factorial interaction study.

The expanded GEMM audit uses the following primary domains:

| Step | Values |
|---|---|
| Parallel lanes | `1,2,4,5,8,10,16,20,40,80` |
| Pipeline II | `1,2,3,4,5,6,8,10,12,16` |
| K strip size | `4,8,16,20,40,80` |
| Ping-pong tile depth | `2,4,8,10,16,20,40,80` |
| AXI width | `64,128,256,512,1024` |
| Multiplier implementation | `fabric,meddsp,fulldsp,maxdsp,primitivedsp` |

Consolidate completed lane and step campaigns without re-running Vitis:

```bash
python scripts/analyze_gemm_parameter_rollout.py \
  --lane-measurements results_sweeps/LANE_RUN/gemm_linked_lane_measurements.json \
  --step-measurements results_sweeps/STEP_RUN/measurements.json \
  --step-summary results_sweeps/STEP_RUN/summary.json \
  --output-dir results_sweeps/GEMM_ROLLOUT
```

The combined export includes all source families, but marks only the striped
independent-accumulator lane family and the five named step controls as primary
OFAT comparisons. Parent-normalized metrics are always computed within a
single family. A per-step winner is not automatically a valid combined winner;
composing controls requires fresh CSim and CSynth evidence.

## 2MM Methodology Replication

`qor_linked_controls.build_2mm_linked_lanes_parent` and
`qor_linked_controls.build_2mm_step_control_parent` apply the same six control
families to both reductions in HLSFactory 2MM. Before transformation,
`localize_2mm_dimensions` resolves generated `NI`/`NJ`/`NK`/`NL` symbols to the
literal dimensions in the pinned public header. The localized source hash and
replacement counts are retained in the campaign record.

| Step | Named control | Coupled behavior |
|---|---|---|
| Parallel lanes | `MM2_LANES` | Both reduction lane loops, six unroll directives, four reduction-dimension banks, partial-sum extents, bounds, and strides; boundary guards permit non-divisor values |
| Pipeline | `MM2_COMPUTE_II` | Pipeline targets on both reduction loops |
| Tiling | `MM2_K_TILE` | Phase-1 reduction strip bound and index |
| Double buffer | `MM2_BUFFER_DEPTH` | Alternating B-tile extents plus coordinated load, compute, and stride bounds |
| Coalescing | `MM2_AXI_BITS` | Shared `max_widen_bitwidth` for all five `m_axi` ports |
| Resource repair | `MM2_MUL_IMPL` | Shared double-multiply `BIND_OP` implementation for both reductions |

The replicated campaign domains are:

| Step | Values |
|---|---|
| Parallel lanes | `1,2,4,5,7,10,14,25,35,50` |
| Pipeline II | `1,2,3,4,5,6,8,10,12,16` |
| Phase-1 K strip size | `1,2,5,7,10,14,35,70` |
| Alternating B tile depth | `1,2,5,7,10,14,35,70` |
| AXI width | `64,128,256,512,1024` |
| Multiplier implementation | `fabric,meddsp,fulldsp,maxdsp,primitivedsp` |

The six families retain separate freshly validated parents. This is a
methodology replication, not a claim that GEMM's winning parameter values
transfer to 2MM. Use `scripts/compare_qor_methodology_benchmarks.py` to align
their parent-normalized responses; the comparison explicitly merges GEMM's
consolidated full-domain lane measurements with its five-control summary.

## Fresh-Model Paired Campaigns

`scripts/run_qor_paired_model_campaign.py` runs fresh model trajectories and
compares each selected pre-QoR kernel against the additive search winner.
Both arms therefore share all model calls and the exact source selected by the
agentic trajectory. `*_pre_qor.cpp` and `*_pre_qor.json` retain the control
source, report, CSim, model usage, and source hash. The normal final kernel and
schema export retain the selected QoR winner.

The DeepSeek-v4-pro 12-kernel configuration is registered in
`artifacts/experiment_matrix/deepseek_v4_pro_qor_paired12_20260906.json`.
It crosses flash and dynamic/multistep with fixed matched skills, five model
calls, and up to 18 additional parameter candidates (16 OFAT plus two
interactions). The two smoke cells gate the remaining 22 trajectories.
Model seed 42 is recorded for the paper fingerprint, while unsupported seed
is omitted from the DeepSeek API request. There is one fresh trajectory per
cell; repeated-sampling uncertainty is not estimated.

The analyzer exports all candidate measurements, paired CSV/JSON/JSONL,
schema JSONL, parent/winner charts, and cumulative candidate-budget curves.
Speedup aggregation uses benchmark clusters, with workflow modes grouped
inside each benchmark. Added synthesis cost is explicit; this experiment does
not claim an advantage over an equal-budget alternative optimizer.

```bash
python scripts/run_qor_paired_model_campaign.py prepare --config CAMPAIGN.json
python scripts/run_qor_paired_model_campaign.py run --config CAMPAIGN.json
python scripts/run_qor_paired_model_campaign.py report --config CAMPAIGN.json
```

`run` resumes completed pairs only after checking their saved artifacts and
frozen inputs. Failed cells are rerun. Changes to registered inputs require a
new campaign. `refresh-empty-preflight` is limited to smoke failures that
produced no model/evaluation JSON artifacts and archives the failed manifest.

## Work Storage

### AXI Adapter Correction And Full PolyBench Batch

`axi_bundle_linked_v2` corrects two legality defects from the 12-kernel pilot:
83 candidates changed one port's adapter options independently of sibling ports,
and one candidate requested a non-power-of-two burst length. Read/write
outstanding-transaction and burst-length controls now change every port in the
same lexical scope and AXI bundle together, as one physical factor. Missing
sibling options are made explicit, symbolic options are conservatively skipped,
and burst lengths are filtered to powers of two in `[1, 256]`. Port-local width
caps remain separate. Linked source spans and affected ports are recorded in
candidate telemetry. Width insertion can compose with adapter controls without
overlapping whole-line rewrites.

The saved-parent regression checks in
`results_sweeps/qor_axi_bundle_regression_20260908` both passed CSim/CSynth:
FDTD-2D changed shared read-outstanding count 16 to 1; Cholesky used burst length
4 after rejecting requested value 3 from the domain. Their latencies stayed
unchanged. These demonstrate legal edits, not measured speedups.

The expanded registration is
`artifacts/experiment_matrix/deepseek_v4_pro_qor_paired27_axi_v2_20260908.json`:
27 kernels (the established PolyBench set excluding doitgen), flash and
dynamic/multistep, 54 fresh paired trajectories. DeepSeek-v4-pro, matched skills,
five model calls, 16 OFAT plus up to two interaction candidates, Vitis 2023.2,
U280, 3.33 ns and no COSIM are unchanged. FDTD-2D multistep and GESUMMV flash
gate the remaining 52 cells; the saved-parent regression outputs are also
required and hashed at preflight. At most two isolated synthesis workers run.

`parameter_effects.md`, `candidate_effects.csv`, `parameter_trends.csv`, and
`candidate_outcomes.png/pdf` supplement the paired table and latency/budget
charts. They distinguish faster, unchanged, slower-but-feasible and invalid
candidates; report resource, interval and achieved-II deltas; and show rejection
causes and control-response measurements. Invalid candidates have no imputed
cycles. Winner explanations separate measured associations from unproven
critical-path explanations. Final retention is non-regressing by construction,
not evidence that more aggressive parameters monotonically improve hardware.
Fresh corrected-versus-historical trajectories are not an isolated ablation of
the legality fix; the exact saved-parent regressions supply that narrow check.

### Placement

Run active Vitis CSim/CSynth work directories on local storage through
`C2HLS_TMP_ROOT` and `C2HLS_VITIS_USER_HOME`. Archive completed work directories
to `/mnt/data2/luo00466/c2hls_rl` afterward. Running Vitis directly against the
NAS can make compilation and timeout cleanup block on network-filesystem I/O;
the subprocess wrapper bounds cleanup, but local scratch remains the supported
execution path.

## Profile-Aware Skill Review (2026-09-23)

Enable `C2HLS_QOR_PROFILE_REVIEW=1` to add the reference-blind
`profile_skill_tradeoff_v1` reviewer to `QualityRepairAgent`. It runs after
each optimization candidate's CSim/CSynth evaluation, including invalid
candidates, without an additional model call. The default remains disabled
for legacy configurations; the flag and implementation hashes enter the run
fingerprint. Repair responses have their own skill declarations, rather than
inheriting claims from the initial completion.

The reviewer records parent/candidate source hashes, exposed and declared
skills, best-effort changed source-family evidence, measured bottleneck
persistence, cycle/II/resource deltas, device utilization, and Pareto relation.
Comments and string literals are excluded from source evidence. A changed
pipeline II is visible even when pipelining already existed. Skillless mode
does not query the skill library. Missing measurements are unknown, not zero.
Static evidence is not proof of recipe correctness; a passing CSim covers only
exercised inputs, and aggregate cycle changes do not isolate individual skills.

Coder prompts allow spending more DSP/BRAM/LUT/FF/URAM to reduce measured
latency. They request coupled compute-lane, banking, reduction, and buffer
changes supported by the current profile, rather than minimum resource use or
an arbitrary utilization target. Correctness, device fit, and timing remain
hard gates. Feasible non-improving candidates with unresolved profile evidence
can receive one evidence-guided retry within the existing attempt/model budget.
An equal-latency candidate that saves resources is not retried solely for that
reason. Later optimization prompts receive the preceding candidate review,
explicitly distinguished from the currently selected kernel.

`qor_profile_reviews` in each full result and `skill_prompt.qor_review` in
candidate records retain this evidence even if an attempt is rejected. Campaign
exports add `skill_profile_reviews.jsonl/csv`; existing website schema records
continue through the existing exporter unchanged.

The approved registration is
`artifacts/experiment_matrix/deepseek_flash_qor_profile27_20260923.json`:
27 PolyBench kernels, flash and dynamic/multistep, matched skills, five model
calls, and the existing 16 OFAT plus up to two interaction candidates. The
API rejected `deepseek-v4.1-flash`; the user approved the exact available ID
`deepseek-flash`. Its timestamped alias does not identify immutable weights.
BICG flash and GEMM multistep gate the other 52 cells. The runner requires
review records, passing reference-isolation checks, matching frozen inputs,
valid CSim/CSynth, and intact pre/post-QoR pairs before releasing the main
batch. At most two workers use isolated local scratch; results and archived
work directories use `/mnt/data2/luo00466/c2hls_rl`.

This batch measures the updated system and isolates the additive numeric
search through exact pre/post pairs. It does not isolate the reviewer prompt
effect from historical model changes. That causal question requires a future
same-model, same-parent, equal-budget reviewer-on/off ablation.

## Frozen-Parent Reviewer A/B (2026-09-25)

Registration: `artifacts/experiment_matrix/deepseek_flash_reviewer_ab27_20260925.json`.
This primary comparison has 324 runs: 27 kernels, flash/dynamic, reviewer off/on,
and three independent API samples. Both arms use `deepseek-flash`, temperature
0, top-p 1, the same frozen skill catalog and routing policy, U280, Vitis 2023.2,
3.33 ns, five model calls and five synthesis evaluations. Parameter search and
COSIM are disabled. Samples are separate requests, not claimed seeded replicas
or guaranteed distinct outputs. Seed 42 is registered for the paper-profile
fingerprint, but the DeepSeek client omits this unsupported API parameter.

`freeze_qor_ab_seeds.py` uses the prior campaign's dynamic Phase-B result for
each benchmark, without selecting by final performance. It matches the entire
baseline report hash to a passing, feasible synthesis event, then recovers code
whose SHA-256 matches that event after the existing interface normalization.
It does not assume the first assistant reply passed. The seed manifest freezes
the source/header hashes, exact code, executed CSim, CSynth report and toolchain.
Every arm and repetition shares that same parent; no Phase-B model call occurs.
Run fingerprints now include the seed manifest and independent sample ID.

Arm A (`review_off`) retains generic feedback and the previous prompt wording.
Arm B (`review_on`) enables the bundled skill/profile reviewer and hardware
tradeoff guidance. Other implementation fixes, including accurate repaired-reply
skill declarations, remain shared. Later dynamic skill routes can diverge as
candidate code changes; that is part of the treatment, not a changed catalog.

Flash now permits two candidate attempts in both arms. The prior one-attempt
flash sweep could record reviews but had no follow-up coding turn. In this
protocol A receives the generic candidate-continuation prompt; B can receive
the measured review. Multistep retains one attempt per optimization step plus
the existing bounded retry policy. Five-call/five-synthesis caps apply equally;
actual utilization is reported rather than assumed equal. Frozen baseline
fallback is explicit, and is not counted as a successful generated candidate.

ATAX flash A/B and GEMM multistep B/A gate the remaining 320 cells. Both B smoke
histories must show reviewer feedback reaching a subsequent prompt. The runner
keeps A/B blocks adjacent with alternating order and uses two isolated workers.
Completed, audited cells are preserved on restart. Failed or interrupted cells
are retried by an explicit restart. Code/input drift stops execution.

The report provides per-arm resource/cycle/token/call measurements, matched-pair
speedups, wins/ties/losses, validity, and benchmark-clustered bootstrap intervals.
Invalid/unpaired outcomes receive no imputed cycles. Partial summaries are
provisional, and the API alias is not an immutable weights revision. This
isolates the bundled reviewer change before numeric parameter search; it does
not separately identify the review-only and tradeoff-prompt-only effects.

```bash
python scripts/run_qor_review_ab.py prepare --config artifacts/experiment_matrix/deepseek_flash_reviewer_ab27_20260925.json
python scripts/run_qor_review_ab.py run --config artifacts/experiment_matrix/deepseek_flash_reviewer_ab27_20260925.json
python scripts/run_qor_review_ab.py report --config artifacts/experiment_matrix/deepseek_flash_reviewer_ab27_20260925.json
```

Compact reports and website-schema JSONL stay in the project. Active Vitis work
uses local `/tmp`; each completed cell is archived to NAS with a checksum before
its scratch is removed. Report generation and archiving run with wall-time
limits so a slow report/archive does not indefinitely block scheduling.

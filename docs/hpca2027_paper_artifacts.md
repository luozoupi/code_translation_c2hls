# HPCA 2027 paper artifact pipeline

`scripts/generate_hpca_paper_artifacts.py` is the only supported path from a
frozen experiment set to numerical paper fragments. It does not discover runs
or choose among duplicate outputs. The experiment freeze must first name every
raw result in an explicit, hash-pinned freeze index, run
`scripts/normalize_hpca_freeze_index.py` to produce result-manifest schema 2,
and pin that file, plus audit artifacts, in an evidence manifest. See
[`hpca2027_freeze_normalizer.md`](hpca2027_freeze_normalizer.md) for the raw
runner adapters and their current fail-closed telemetry gaps.

## Publication boundary

The generator refuses the complete bundle if any of these conditions holds:

- the evidence manifest does not say `frozen: true`;
- an input or audit SHA-256 does not match;
- an explicitly expected `(kernel, seed, method)` cell is missing, duplicated,
  or supplemented with an unplanned cell;
- a run ID is reused;
- a success lacks correctness, synthesis, resource-fit, timing, reference
  isolation, or executed RTL co-simulation evidence;
- a cycle count is predicted, estimated, or paired with a failed co-simulation;
- a passing synthesis lacks BRAM/DSP/FF/LUT/URAM counts, device capacities,
  exact derived utilizations, or positive Fmax;
- a failed outcome does not carry an explicit failure class;
- an attested matched-budget run exceeds the declared candidate/synthesis cap;
- a candidate trace is truncated, non-monotonic, inconsistent with its run
  totals, or contains predicted/estimated rather than Vitis-reported latency;
  or
- the token/synthesis checkpoints do not cover the complete frozen traces.

Failures are records, never absent rows. They receive an infinite performance
ratio, remain in solve-rate denominators, and receive zero bounded
performance-profile AUC score.

## Result manifest

The result manifest is JSON with `schema_version: 2` and these top-level keys.
Version 2 makes complete candidate-event traces mandatory for paper figures:

- `methods`: ordered `{id, display_name}` objects;
- `expected_cells`: the exact ordered `(kernel, seed, method)` coverage;
- `normalization_provenance`: the supported freeze-normalizer schema, fixed
  XCU280 target, capacity source, and exact five-class device-capacity table;
- `baseline_expert`: one ordered object per expected kernel, each containing a
  `kernel`, `baseline` record, and `expert` record; and
- `evaluation_units`: one ordered object per represented `(kernel, seed)`, with
  a `results` object containing exactly the methods listed for that unit by
  `expected_cells`, in global method order. Sparse replication is intentional,
  not missing evidence.

Every outcome record contains:

```text
run_id
terminal_status                 success | failure
correctness_status              passed | failed | not_run | tool_failure | timeout
synthesis_status                passed | failed | not_run | tool_failure | timeout
resource_fit                    true | false | null
timing_met                      true | false | null
cosim_status                    passed | failed | not_run | tool_failure | timeout
cycle_source                    executed_rtl_cosim | predicted | estimated | none
executed_cosim_cycles           positive integer | null
failure_class                   null for success; explicit class for failure
synthesis_metrics               object for passed synthesis; null otherwise
```

The synthesis-metrics object is derived from Vitis CSynth evidence:

```text
source                          vitis_csynth_report
report_sha256                   canonical source-report SHA-256
fmax_mhz                        positive finite number
resources.{bram,dsp,ff,lut,uram}.used          non-negative integer
resources.{bram,dsp,ff,lut,uram}.capacity      positive integer
resources.{bram,dsp,ff,lut,uram}.utilization   exactly used / capacity
```

`run_id` must be an opaque 1--128 character identifier using only letters,
digits, `.`, `_`, `:`, or `-`; filesystem paths are rejected so the checked-in
provenance cannot leak an author home directory.

Generated-method records additionally require
`reference_isolation_status: passed`, the independently authenticated Boolean
`provider_failure`, `tokens`, `llm_calls`,
`synthesis_calls`, `selection_synthesis_evaluations`, `wall_time_seconds`, and
`candidates_evaluated`. These fields must include failed attempts and be
non-negative; call/candidate counts are integers. `synthesis_calls` is total
tool attribution (including a selected-winner co-simulation if it reruns
synthesis), while `selection_synthesis_evaluations` is the matched-budget
quantity capped by the preregistration.

Every generated record also carries `candidate_events`, with exactly one
ordered event for every evaluated candidate. An event contains:

```text
event_id                         opaque unique ID within the run
candidate_index                  contiguous 1-based index
code_sha256                      exact candidate-source SHA-256
report_sha256                    source-report SHA-256 after passed synthesis; null otherwise
cumulative_tokens                actual non-negative integer
cumulative_llm_calls             actual non-negative integer
cumulative_synthesis_evaluations actual non-negative integer
cumulative_elapsed_seconds       actual non-negative time
correctness_status               passed | failed | not_run | tool_failure | timeout
synthesis_status                 passed | failed | not_run | tool_failure | timeout
resource_fit                     true | false | null
timing_met                       true | false | null
synthesized_latency_cycles       positive integer | null
latency_source                   vitis_csynth_report | none
failure_class                    null only for a feasible candidate
selected_for_executed_cosim      boolean
```

Counters must be monotonic and the final event must agree exactly with the run
totals. Each event consumes zero or one new selection synthesis. A candidate
that passes CSim must be synthesis-evaluated. A synthesized latency is accepted
only with `latency_source: vitis_csynth_report`; predicted, LLM-estimated, or
gold-relative latency is rejected. Exactly one feasible candidate is marked as
the executed-cosim winner for a successful run, and it must be the
minimum-latency feasible synthesized state (ties are allowed). Fields carrying
predicted or gold-relative latency are forbidden even when a measured value is
also present. The selected event's code digest must match the record's selected
code and cosim-target digests, and its report digest must match
`synthesis_metrics.report_sha256`; bundle generation fails closed on a mixed or
stale winner.

Accepted failure classes are defined in the generator and cover malformed or
wrong output, compile/interface failure, synthesis timeout, tool failure,
resource/timing infeasibility, co-simulation failure/timeout, reference
isolation failure, missing executed co-simulation, invalid reference, budget
exhaustion, and `other`. `other` requires `failure_detail`.

## Evidence manifest

The evidence manifest is JSON with this shape (placeholders are descriptive,
not generator-ready values):

```json
{
  "schema_version": 2,
  "frozen": true,
  "evidence_freeze_timestamp": "<RFC-3339 timestamp with timezone>",
  "run_set": {"path": "<results.json>", "sha256": "<64 hex>"},
  "expected_kernels": ["<kernel IDs in paper order>"],
  "expected_methods": ["<method IDs in result order>"],
  "expected_cells": [
    {"kernel": "<kernel ID>", "seed": "<fixed seed>", "method": "<model-qualified method ID>"}
  ],
  "headline_units": [{"kernel": "<one table unit per kernel>", "seed": "<fixed seed>"}],
  "profile_units": [{"kernel": "<reference-isolated profile unit>", "seed": "<fixed seed>"}],
  "bootstrap_units": [{"kernel": "<paired replicate unit>", "seed": "<fixed seed>"}],
  "claim_methods": {
    "primary": "<method ID>",
    "one_shot": "<method ID>",
    "dynamic_no_skill": "<method ID>",
    "dynamic_frozen_skill": "<method ID>"
  },
  "profile_taus": [1.0, 1.25, 2.0, 4.0, 10.0],
  "profile_tau_max": 10.0,
  "budget_synthesis_checkpoints": [1, 2, 3, 4, 5],
  "budget_token_checkpoints": ["<sorted positive checkpoints covering all traces>"],
  "bootstrap": {
    "confidence": 0.95,
    "replicates": 10000,
    "seed": 2027
  },
  "policy": {"minimum_valid_baseline_expert_pairs": 8},
  "artifacts": [
    {"id": "<stable ID>", "path": "<audit file>", "sha256": "<64 hex>"}
  ],
  "gate_evidence": {
    "transcript_leakage_audit": {
      "status": "passed",
      "artifact_id": "<attested artifact ID>"
    },
    "matched_budget": {
      "status": "passed",
      "candidate_limit": 5,
      "synthesis_limit": 5
    },
    "candidate_validation_audit": {
      "status": "passed",
      "artifact_id": "<CSim/cosim audit artifact ID>"
    },
    "complete_candidate_event_stream": {
      "status": "passed",
      "artifact_id": "<complete event stream artifact ID>"
    },
    "fingerprint_consistency_audit": {
      "status": "passed",
      "artifact_id": "<resume/fingerprint audit artifact ID>"
    },
    "frozen_skill_snapshot": {
      "status": "passed",
      "artifact_id": "<attested artifact ID>",
      "frozen_before_evaluation": true,
      "no_evaluation_persistence": true
    },
    "post_route_validation": {
      "status": "passed",
      "artifact_id": "<attested artifact ID>",
      "stratified_winner_count": 5
    }
  }
}
```

All relative paths resolve from the evidence manifest. Referenced artifacts are
opened and hashed before analysis. Missing optional gate evidence blocks only
the associated claim; malformed core run evidence blocks generation.
`headline_units` must select exactly one preregistered unit for every kernel;
`profile_units` and `bootstrap_units` are explicit non-empty subsets. Every
profile unit must contain all profile methods, whereas a bootstrap unit need
only contain the three paired claim methods. This permits the registered
three-method extra-seed campaign without manufacturing pragma/flash cells. This
prevents extra replication seeds from silently reweighting the primary
reference-isolated performance profile.
The evidence-freeze timestamp controls PDF creation/modification metadata and
the TeX fallback's `SOURCE_DATE_EPOCH`, so regenerating the same frozen bundle
does not inject wall-clock time into figure bytes.

## Statistics and claim rules

- Expert recovery is exactly `log(B/G) / log(B/E)`. It is emitted only when
  baseline, generated, and expert are successful executed co-simulations and
  `B > E > 0`. Values above one are retained.
- A performance ratio is generated cycles divided by the best successful
  method for the same preregistered unit. Failure is positive infinity.
- The CSV and grayscale SVG emit the exact empirical step profile at every
  observed finite breakpoint (plus 1 and the declared maximum); configured
  `profile_taus` provide stable figure ticks rather than interpolated data.
- Performance-profile AUC is the mean per-unit area over the explicitly
  declared interval `[1, profile_tau_max]`, normalized to `[0, 1]`. A failure
  scores zero. This makes the frozen-skill paired comparison failure-aware.
- At each synthesis/token checkpoint, the budget curve uses the best feasible
  Vitis-CSynth latency observed up to that checkpoint. It is divided by the
  best final feasible candidate across methods for the same unit, converted to
  the same bounded profile-AUC score, and averaged over all primary profile
  units. An unsolved unit scores zero. `budget_curves.csv` records solve rate
  and the exact failure count at every plotted point.
- The frozen-skill confidence interval is a deterministic paired percentile
  bootstrap of per-unit AUC-score differences. The claim passes only when its
  lower bound is strictly positive and correct-solve rate does not decrease.
- The bootstrap CSV also reports paired AUC and correct-solve-rate intervals
  for dynamic versus one-shot and frozen-skill versus no-skill comparisons;
  the manifest fixes confidence, replicate count, seed, and paired unit sets.
- Dynamic workflow/compact-model enablement requires exact empirical
  performance-profile dominance over matched one-shot: no lower CDF value at
  any observed finite breakpoint and a strict improvement at one or more.
- The headline pair threshold defaults to eight and is read from policy.

Claim decisions and every constituent Boolean gate are written to
`claim_decisions.json`; a failed gate never silently changes prose into a
positive claim.

## Running and publishing

```sh
python scripts/normalize_hpca_freeze_index.py \
  --freeze-index paper_eval/freeze/freeze-index.json \
  --output paper_eval/freeze/results-v2.json

python scripts/generate_hpca_paper_artifacts.py \
  --evidence paper_eval/freeze/evidence.json \
  --output-root paper_eval/paper_artifacts
```

The evidence manifest's `run_set` must pin the exact `results-v2.json` bytes.
Do not point the generator directly at a runner output or construct schema-2
rows by hand.

The destination is
`paper_eval/paper_artifacts/<result-manifest-sha256>/`. Generation occurs in a
temporary sibling and the complete directory is renamed into place. Existing
bundles are immutable and never overwritten.

The bundle contains generated LaTeX fragments, per-kernel recovery CSV,
performance-profile CSV/SVG, grayscale recovery SVG, paired-bootstrap CSV,
generated-method and baseline/expert failure accounting (including invalid
frontier relations), a row-complete `resource_utilization_fmax.csv`, a derived
resource/Fmax table, cost attribution with per-method mean Fmax/utilization,
claim decisions, cell-to-run
provenance, and an output checksum manifest. It also contains `recovery.pdf`
and `budget.pdf`: the first plots normalized executed RTL co-simulation and the
second combines failure-aware QoR versus synthesis/token budget with the final
component ablation. Resource CSV rows include synthesized failures as measured
and non-synthesized failures as explicit empty measurements; neither is
silently dropped. At evidence freeze, publish the four `.tex` fragments and
these vector PDFs from one bundle together, then record
the same 64-character run-set hash in `claims.yaml`. Do not copy individual
files from different bundles.

`verify_bundle(path)` re-hashes every output named by
`artifact_manifest.json`, rejects missing/extra files, and verifies the
hash-named directory identity. A tampered resource CSV therefore fails before
reuse; generation never repairs an existing bundle.

Matplotlib is the preferred renderer and is pinned in
`requirements-paper.txt`. It sets `pdf.fonttype=42`, producing embedded,
subsetted TrueType fonts. If matplotlib is unavailable, the generator uses a
documented `pdflatex` + pgfplots fallback with embedded TeX vector fonts. If
neither renderer exists, generation fails before publishing a bundle; it never
substitutes a placeholder or raster image. `render_provenance.json` and the
artifact manifest record the backend and font policy.

```sh
python -m pip install -r requirements-paper.txt
```

Run the focused tests with:

```sh
python -m unittest tests.test_hpca_paper_artifacts -v
```

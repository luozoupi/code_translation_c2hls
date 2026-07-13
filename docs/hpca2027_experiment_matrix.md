# HPCA 2027 experiment-matrix contract

The source of truth is
[`configs/hpca2027_experiment_matrix.json`](../configs/hpca2027_experiment_matrix.json).
It defines the reference-isolated Rodinia-HLS matrix without launching it.  The checked-in
validator is deliberately a dry-run tool: it has no subprocess path that calls
C2HLS, an LLM endpoint, or Vitis.

## Frozen study shape

The primary campaign crosses nine Rodinia-HLS kernels, both models, all five
methods, and seed 0: 90 rows.  The representative-kernel campaign adds seeds 1
and 2 for the three methods named in the paper plan: 36 rows.  The separately
labelled oracle upper bound runs dynamic C2HLS with frozen skills on the same
three kernels, both models, and seed 0: 6 rows.  The full declarative matrix is
therefore 132 rows.

The representative strata are fixed as follows:

| Stratum | Kernel |
|---|---|
| Memory-bound | `pathfinder` |
| Compute-bound | `lavaMD` |
| Irregular | `StreamCluster` |

The analysis contract is also fixed before launch: failure-aware performance
profiles integrate over $[1,10]$; paired comparisons use a percentile
bootstrap of the mean per-unit profile-AUC difference with 95% confidence,
10,000 resamples, and seed 2027. Frozen-skill transfer requires a lower bound
strictly above zero and correct-solve rate no lower than dynamic skill-off.

Every row carries the same scientific budget contract: at most five generated
optimization candidates and five synthesis evaluations, CSim plus independent
golden checking before synthesis, feasibility-plus-latency winner selection,
and executed cosim only for the selected winner.  These are acceptance limits,
not just suggested runner arguments.  A downstream result ingester must reject
an artifact whose recorded counters exceed either cap or whose selected winner
lacks the required executed-cosim status.

## Runner support is explicit

The matrix distinguishes a method required by the paper from a method currently
implemented by a faithful runner.

| Method | Mapping | Meaning |
|---|---|---|
| Matched-budget best-of-five one-shot | Supported | Maps to `run_paper_baseline.py`; five independent full translations use candidate seeds `base_seed + index` where supported. |
| Pragma-only search | Supported | Maps to `run_paper_baseline.py`; one full translation is followed by four independent revisions guarded by exact non-pragma token equivalence. |
| Flash C2HLS | Supported | Maps to `run_agentic_sweep.py` with `strategy=flash`, skills off. |
| Dynamic C2HLS without skills | Supported | Maps to the agentic sweep with `strategy=dynamic`, skills off. |
| Dynamic C2HLS with frozen skills | Supported | Maps to the agentic sweep with a required immutable `skills.json` path and matching snapshot hash. |

All 132 rows now have a supported runner mapping. “Supported” means that a
runner mapping exists; it does not waive paper
preflight checks.  Model revisions, credentials/endpoints, the frozen skill
hash, transcript isolation, global counters, golden correctness, target timing,
device fit, and selected-winner cosim must still validate. The validator also
pins each baseline to its dedicated runner, preventing a convenient but
scientifically different implementation from being substituted silently.

Skill-on rows set `C2HLS_SKILL_LIBRARY_PATH` to the freezer-produced
`skills.json`. The controller loads that file exactly: it does not merge the
mutable default store, built-ins, or packaged catalog. The fingerprint hashes
only this explicit snapshot as skill evidence and requires its aggregate hash
to equal `C2HLS_SKILL_SNAPSHOT_SHA256` before any model or Vitis call.

The baseline runner never loads expert source or expert metrics in its search
path. It loads only `plain.cpp`, the public header and testbench, and public
support files. Every request/response is hashed and retained in a transcript,
which is audited after search against expert-only identifiers, paths, code
signatures, and absolute metrics. Resume accepts only a complete, byte-exact
fingerprint covering the implementation, public inputs, prompts, model
revision, decoding, seed policy, target, and budgets.

For `pragma_only`, whitespace and comments may change, and complete `#pragma`
logical directives may be inserted, removed, moved, or replaced. The runner
removes backslash-continued pragma directives as units, lexes the rest of both
sources, and requires identical token sequences. Changes to constants, types,
expressions, macros, includes, declarations, loops, or helper functions are
therefore rejected before CSim or synthesis.

The Anthropic API does not currently expose deterministic seed control.  The
Claude rows retain the same seed labels to define matched repeated trials and
fingerprints, but the result provenance must retain the controller's
`seed_supported=false` evidence.  They must not be described as bitwise
reproducible seeded samples. The primary local endpoint identity is
`qwen3.6-27b`; its rows require an immutable weights revision through
`C2HLS_QWEN27B_REVISION`. Claude rows require a provider-reported model version
through `C2HLS_SONNET46_REVISION`.

## Validation and expansion

Validate the study contract and the checked-in benchmark/testbench files:

```bash
python scripts/validate_hpca2027_matrix.py --check-repo
```

Write row-level dry-run JSONL for review:

```bash
python scripts/validate_hpca2027_matrix.py \
  --check-repo \
  --out /tmp/hpca2027_matrix.expanded.jsonl \
  --format jsonl
```

The expansion names required environment variables but never serializes API
credentials. `--resolve-env` may resolve only model revision and frozen-skill
identity values. Use `--fail-on-blocked` in a launch-preparation gate; it
returns nonzero while any required environment variable is unresolved. With
the two model revisions, the local endpoint, the Anthropic credential, the
frozen `skills.json` path, and its hash supplied, all 132 rows are executable.

The paper profile sets `C2HLS_FORCE_SELECTED_COSIM=1`, so every primary row
attempts RTL cosimulation for its selected winner even when legacy benchmark
metadata marked cosim unsupported.  The executed status remains mandatory and
visible: a missing depth/interface contract, timeout, or tool failure rejects
the row rather than silently converting it into a measured result.

Reference-frontier construction first excludes any upstream variant whose
public workload macros or testbench-visible ABI differ from the canonical
plain-source/header/testbench contract.  The current static contract partition
admits 38 of 51 Rodinia variants and explicitly excludes 13 workload-changing
or wide-ABI variants before they consume tool calls.  It then synthesizes and
golden-checks every admitted variant and selects the fastest correct,
device-fitting, timing-feasible expert.  The designated baseline and selected
expert each execute RTL cosim once outside every method search budget.  Both
measurements are cached under the full reference-input, contract, and cosim
policy fingerprint; a missing positive cycle count for either invalidates the
pair rather than falling back to a synthesis estimate.

Reported method wall time is `search_elapsed_seconds`.  Common CPU-golden and
reference-frontier work is recorded separately as
`preflight_elapsed_seconds`; optional post-route work is also separate.  The
fingerprint probes the invoked Vitis executable and rejects a configured
version that does not match the probe.

The test suite checks exact campaign coverage, budget equality, reference-blind
environment invariants, oracle separation, baseline runner mappings,
self-checking golden-testbench evidence, and secret redaction:

```bash
python -m unittest tests/test_hpca2027_experiment_matrix.py
python -m unittest tests/test_paper_baselines.py
```

Do not use this matrix to resume the active exploratory HLSFactory queue.  It is
for the isolated HPCA 2027 evaluation worktree and its immutable fingerprints.

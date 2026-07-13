# HPCA 2027 reference-blind evaluation guardrails

`run_agentic_sweep.py` now defaults to the
`hpca2027_reference_blind` profile. The profile force-disables every known
reference-dependent controller path: GT-aware rollback, baseline alignment,
GT prepopulation, gold-relative cosim skipping, reference metrics/code in
prompts, persistent skills, and online skill statistics. A reference-guided
run must explicitly set `C2HLS_SWEEP_PROFILE=legacy` and must be reported as
an oracle ablation.

Start each model/method/seed invocation from a filled copy of
`configs/hpca2027_reference_blind.env.example`. In particular, replace
`C2HLS_MODEL_REVISION` with an immutable hosted-model version, repository
commit, or weights digest. Runs with a model alias, provider-default sampling,
or an unbounded synthesis budget remain executable for debugging but carry a
machine-readable `run.reproducibility.complete=false` marker and are not
paper-valid.

For `skill_on`, first build a reviewed content-addressed snapshot with
`scripts/freeze_hpca_skill_snapshot.py`, then set
`C2HLS_SKILL_LIBRARY_PATH` to its `skills.json` and
`C2HLS_SKILL_SNAPSHOT_SHA256` to the `sha256` returned by
`evaluation_repro.skill_snapshot_manifest(repo)`. The controller loads that
file exactly and never merges built-ins, the mutable default store, or the
packaged catalog. A missing path, malformed snapshot, or configured hash
mismatch aborts before any LLM or synthesis call; an omitted hash marks the
run incomplete. `skill_off` does not load any library or require an expected
skill hash. The wrapper also re-hashes the snapshot after every run and
rejects the run if it changed; it does not silently restore or absorb it.

## Immutable resume identity

Before any LLM or synthesis call, the sweep computes a content fingerprint
over:

- controller and feedback source files plus Git HEAD;
- prompt-template files and the exact post-run prompt hashes;
- all benchmark source, testbench, metadata, support, and data files;
- model ID, revision, per-agent models, temperature, top-p, seed, output token
  limit, and the actual identity/decoding record for every provider call;
- skill snapshot hashes, skill mode, frozen state, persistence, and online
  update state;
- configured/probed Vitis version, executable and settings digests, FPGA
  part/platform, clock, and flow target; and
- strategy, action list, turns, candidate/synthesis limits, and tool timeouts.

The benchmark manifest also hashes offline expert files for artifact identity;
only their digests are used by this provenance layer. They are not returned to
the controller. Resume accepts a result only if the schema, canonical payload,
and SHA-256 digest all match. Legacy results and partially populated
fingerprints are rejected and recomputed.

## Reference-isolation audit

After a run, `reference_isolation.py` checks controller-visible system, user,
and tool messages for expert paths, variant identifiers, tokenized partial-code
signatures, and labelled or distinctive exact expert metrics from every
frontier variant. It checks
assistant messages for explicit paths and expert metrics, but does not treat
independently recovered code as proof of leakage. Findings contain only rule
names, offsets, lengths, and one-way hashes; they never reproduce expert code.
The paper profile fails the run if the transcript is missing or the audit has
any finding. The audit is embedded in the result and also written as
`<benchmark>_reference_isolation_audit.json` beside it.

## Result fields

Every newly written result contains:

- `run_fingerprint` and `run.run_fingerprint`;
- `run.evaluation_profile`, `run.reference_blind`, model revision, separately
  recorded configured/effective decoding, exact prompt/response hashes,
  elapsed time, LLM usage, and the controller-reported synthesis count;
- `run.reproducibility` with explicit completeness issues;
- `reference_isolation_audit`; and
- `evaluation_status`, which keeps functional correctness, synthesis,
  executed cosim, predicted cosim skips, timeouts, and tool failures separate.
  The same primary classifications are surfaced as top-level
  `correctness_status`, `executed_cosim_status`, `predicted_cosim_skip`,
  `timeout_status`, and `tool_failure_status` fields.

The sweep driver does not infer predicted timeouts as measured RTL results:
`cosim_predicted_skip=true` is paired with
`cosim_execution_status=not_run`.

The sweep wrapper does not guess effective sampler settings or a synthesis
count. Until the controller reports them, the fields remain null and
`run.reproducibility` includes `effective_decoding_unreported` and/or
`synthesis_evaluation_count_missing`.

Paper runs forbid role-specific model overrides: translator, synthesis,
quality-repair, and feedback calls must use the single revisioned model in the
matrix row. Each provider event is checked against the fingerprint for model,
revision, prompt hash, temperature, top-p, seed support, and token cap.
Anthropic seed non-support is recorded explicitly rather than treated as a
provider-enforced seed.

HLSFactory development correctness uses the 28-kernel manifest and
`configs/hlsfactory_output_shapes.json`. The registry is audited from public
testbench print loops, records dense or triangular emission layouts, and binds
each contract to the testbench SHA-256. It replaces the old flat
`[value_count]` fallback for every HLSFactory kernel.

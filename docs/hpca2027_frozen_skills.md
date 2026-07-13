# HPCA 2027 frozen-skill protocol

The primary Rodinia evaluation must use a skill library frozen before any
primary evaluation kernel is run. `scripts/freeze_hpca_skill_snapshot.py` creates that
library from a deliberate review manifest. It does not launch Vitis or an LLM,
does not infer skills from arbitrary result directories, and never writes to
`skills/skills.json`.

## Evidence admitted by the freezer

Every trajectory must be labelled `source_suite: HLSFactory` and
`benchmark_role: development` in the review manifest. The referenced result
artifact must independently agree: its benchmark must have the
`hlsfactory_...` prefix or carry an HLSFactory source declaration. The freezer
hard-rejects all nine reference-isolated Rodinia kernel names (`StreamCluster`, `hotspot`,
`kmeans`, `knn`, `lavaMD`, `lud`, `nw`, `pathfinder`, and `srad`) and any
additional evaluation names listed by the review manifest.

A trajectory is admitted only when the saved result contains all of the
following executed evidence:

- a generated CPU oracle from the pragma-stripped C and public testbench, with
  nonempty typed outputs and hashes for both output and comparator spec;
- a CSim result whose independent comparator passed and whose oracle hash
  matches that CPU oracle;
- at least one synthesis evaluation marked both executed and successful;
- a feasible selected candidate and a positive synthesized latency.

Old HLSFactory artifacts whose testbenches merely printed values do not meet
this contract. Predicted correctness, a tool exit status without the golden
comparison, or a synthesis report without the executed-event record is also
insufficient.

## Review-manifest format

Paths are relative to the review manifest. Each file is pinned by its exact
SHA-256. Absolute paths are rejected so a frozen record cannot embed an author
home directory. A review manifest can live at the repository root and refer to
`skills/skills.json` and result artifacts below it.

```json
{
  "schema_version": "hpca2027.validated-hlsfactory-skills.v1",
  "source_suite": "HLSFactory",
  "benchmark_role": "development",
  "skill_source": {
    "path": "skills/skills.json",
    "sha256": "<exact 64-hex file hash>"
  },
  "evaluation_kernels": [],
  "trajectories": [
    {
      "path": "results_sweeps/<run>/hlsfactory_adi/<result>.json",
      "sha256": "<exact 64-hex file hash>",
      "kernel": "adi",
      "source_suite": "HLSFactory",
      "benchmark_role": "development",
      "validated_skills": [
        {
          "id": "pipeline-inner-loop-ii1",
          "relative_advantage": 0.18
        }
      ]
    }
  ]
}
```

`validated_skills` is an explicit review decision, not a list automatically
scraped from prompts. It is also not sufficient by itself. For every declared
ID, the pinned result must contain exactly one unambiguous successful step in
`steps` where all of the following hold on that same step:

- `routing_decision.skill_id` selects the skill;
- `skill_prompt.injected` is true and `injected_skill_ids` contains the skill;
- the step's own executed CSim and independent-golden comparison pass and use
  the trajectory's pinned golden-output hash;
- the step's own feasibility record is eligible; and
- the step carries a positive synthesis-report latency.

A final/top-level pass cannot mask an unused skill, a routed-but-not-injected
skill, a failed injected attempt, a failed step CSim, or an infeasible step.
The baseline and every earlier accepted step must have their own passing
evidence because they define the comparison parent.

`relative_advantage` is optional and is never trusted as the output value. The
freezer derives `(previous accepted latency - step latency) / previous accepted
latency` directly from the pinned reports, using the controller's scalar
latency order (`latency_cycles_worst`, `latency_cycles`,
`latency_ns_worst`, `latency_ns`). If a reviewer supplies
`relative_advantage`, it is only a cross-check and any mismatch rejects the
manifest. The source definition must exist in the schema-1.1 skill library.
Avoid-tier rules are not accepted as successful transformations.

The output skill statistics are rebuilt only from admitted observations:
source-library occurrence counts, pass counts, mean advantages, timestamps,
and confidence promotions are not copied. This prevents earlier exploratory or
primary evaluation runs from entering routing through mutable statistics.

## Freeze, verify, and launch

Freeze after HLSFactory validation and before Rodinia evaluation:

```bash
python scripts/freeze_hpca_skill_snapshot.py freeze \
  --manifest hpca2027_validated_skill_inputs.json \
  --output-root paper_eval/frozen_skills
```

The output directory is `sha256-<content-id>/` and contains only:

- `skills.json`, the filtered schema-1.1 library;
- `snapshot_manifest.json`, including exact trajectory hashes and the mapping
  from every skill to its validating development trajectories;
- `SHA256SUMS`, covering both files.

The content ID commits to the input manifest, source library, filtered skill
bytes, isolation policy, every step-level proof, derived advantage, and all
accepted evidence. If the source `skills.json` has a sibling
`snapshot_manifest.json` or `SHA256SUMS`, the freezer treats it as a snapshot
bundle: it verifies the full content address and checksums and fingerprints
both sibling files in the new descriptor. A partial or tampered sibling bundle
is rejected instead of silently consuming only `skills.json`. The freezer
creates the new bundle atomically. Repeating the command only verifies an
identical existing bundle; it never overwrites one. Any altered file,
unexpected file, hash mismatch, or address collision is an error.

Verify the bundle independently:

```bash
python scripts/freeze_hpca_skill_snapshot.py verify \
  --snapshot paper_eval/frozen_skills/sha256-<content-id>
```

For a skill-on paper run, point `C2HLS_SKILL_LIBRARY_PATH` at the frozen
`skills.json`, set `C2HLS_SKILL_SNAPSHOT_SHA256` to
`evaluation_repro.skill_snapshot_manifest(REPO)["sha256"]` after exporting
that path, and keep
`C2HLS_SKILL_LIBRARY_FROZEN=1`, `C2HLS_SKILL_LIBRARY_PERSIST=0`, and
`C2HLS_SKILL_UPDATE_STATS=0`. Do not regenerate or extend the snapshot after
the first Rodinia evaluation begins. Skill-off runs must not load this file at
all.

The exact-snapshot loader returns a `FrozenSkillLibrary`, not the ordinary
mutable store. Its load/save/add/remove/statistics/tier mutators raise, and
queries return defensive copies so a caller cannot mutate a skill or list
field in place. Environment guards remain as redundant policy checks, but
immutability no longer depends on every present and future controller call
site remembering them.

No snapshot has been checked in as a result of adding this utility. A real
snapshot should be created only after the new independent-golden HLSFactory
trajectories have completed and passed review.

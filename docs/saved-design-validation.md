The saved-design gate rechecks the fixed implementations behind the paper with
both the frozen v6 evaluator and the cleaned evaluator. It performs no model
inference, repair, search, or reselection. This is separate from the prospective
DeepSeek comparison and from regenerating the paper's original model studies.

The commercial registration supplies 144 distinct implementations covering all
156 selected/comparator roles. The open-model figure's identity join supplies
216 fixed roles: 211 link to 180 distinct archived implementations and five
comparators have unresolved identities. Those five roles remain blocked in each
evaluator cohort. The 324 available designs produce 648 validation cells.

Each registration copies the exact selected source, original public plain C,
header, testbench, support files, and matching authoritative output registry.
All copies and both evaluator trees are hashed. Metadata retains its original
bytes for provenance; the worker explicitly uses the public testbench and never
selects a reduced co-simulation testbench or applies size overrides. Original
reported synthesis cycles and historical generation reference-isolation flags
are retained. New validation cannot resolve historical generation exposure.

Registration, preflight, and report do not run synthesis, simulation, or a model.
Preflight with settings performs only bounded tool version queries. Both Vitis
entrypoints must report 2023.2; both cohorts then use the same verified
`vitis-run --tcl --input_file` binary. The target is U280
`xcu280-fsvh2892-2L-e` at 3.33 ns.

From the canonical checkout, register a fresh run with:

```bash
python -m hlsmith.saved_designs register \
  --commercial-manifest /home/luo00466/code_translation-c2hls-sigmetrics2027/results_sweeps/commercial_selected_cosim_20261008/manifest.json \
  --recovery-manifest /home/luo00466/code_translation-c2hls-sigmetrics2027/results_sweeps/historical_registry_recovery_20260928/manifest.json \
  --open-identity reproducibility/paper/generated/sigmetrics/skill_v3_rtl_evidence/figure8_identity_join.json \
  --registry-root /home/luo00466/code_translation_c2hls \
  --frozen-source /mnt/data1/luo00466/c2hls_archive_20261018/runtime/v6 \
  --clean-source /home/luo00466/code_translation_c2hls \
  --output runs/saved-design-validation-20261018

python -m hlsmith.saved_designs preflight \
  --output runs/saved-design-validation-20261018 \
  --vitis-settings /mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh

python -m hlsmith.saved_designs report \
  --output runs/saved-design-validation-20261018
```

The frozen tree is the v6 documentation snapshot `95b9a966b6efd3818e34407e8fa885bcff88e5bf`;
its runtime matches sanitized source snapshot `42de088a7716a5d00ec69696d07b1b8b1b75840f`.
Code changes invalidate an existing registration. Preserve that run and register
a new output directory after the final source freeze. Old measurement files are
never imported as new results.

After an aggregate worker-hour allowance is explicitly approved, the concrete
launch command is:

```bash
python -m hlsmith.saved_designs launch \
  --output runs/saved-design-validation-20261018 \
  --vitis-settings /mnt/data/luo00466/Xilinx/Vitis/2023.2/settings64.sh \
  --max-compute-hours "${C2HLS_SAVED_DESIGN_HOURS:?Set the approved worker-hour cap}" \
  --scratch-root /mnt/data1/luo00466/c2hls_runs_20261018/scratch/saved-designs
```

The launch runs one cell at a time and stops before starting a cell whose full
4,800-second reservation would exceed the remaining allowance. The worker has a
4,790-second process limit, with ten seconds reserved for cleanup; measured
elapsed time, including cleanup, is charged without truncation. An unexpected
overrun stops the campaign and is retained in accounting. Native compilation and
execution each have a 180-second limit, CSim has 180 seconds, and RTL has 3,600
seconds. The complete timeout envelope is 864 worker-hours, a worst-case bound
from 648 cells rather than a prediction. Historical one-hour commercial and
open validations used approximately 25.51 and 69.71 summed job-hours; two fresh
evaluator cohorts may differ substantially. There are no API charges from this
gate.

A launcher crash retains its full reserved allowance and active-cell identity.
It requires evidence review before resuming; the executor does not silently
refund or repeat uncertain work. Partial completed runs can resume with a new
explicit cap, retaining all terminal measurements. Tool workspaces and logs are
kept under the requested scratch directory for inspection. An interrupted or
failed tool cell is retained; no candidate is changed or replaced.

A passing cell requires a full independent CPU oracle with the frozen output
identity (where recorded), a matching native candidate, CSim and RTL output
comparators bound to that same oracle, positive integral measured RTL cycles,
and complete resource/timing evidence from the synthesis inside that RTL run.
A selected source is decoded without changing its newlines and its bytes must
round-trip to the registered hash. Vitis' existing interface-depth adaptation
can add depth annotations from the frozen public metadata. The worker records
the selected input hash and the actual materialized source hash separately,
checks the latter against that evaluator's exact depth adaptation, and checks
the materialized testbench/header/support hashes. Reports show any difference
between evaluator cohorts' materialized sources. Source and public-input
associations, the launch's verified runner, and fresh synthesis report files
remain checked when a report is produced.
The requested clock comes from the actual synthesis report's
`UserAssignments/TargetClockPeriod` or an existing parsed report field and the
literal `create_clock` command in that cell's Tcl. Available sources must agree;
each source path and file hash is retained. Missing, ambiguous, dynamic, or
conflicting clock evidence cannot qualify a pass. The campaign's configured
period is never substituted for missing tool evidence.
A tool's bare success flag cannot qualify a pass. Reports keep all 648 cells,
the 324 fixed design pairs, five blocked open slots per cohort, failures,
timeouts, pending work, and paired status/cycle differences. They do not claim
complete paper reproduction or hide failures behind jointly passing pairs.

Offline tests exercise the fixed denominator and selected hashes, public input
and evaluator tampering, preflight/report CLI dispatch, mandatory numeric caps,
stopping before a cell, crash reservation retention, rejection of bare tool
passes, native failures, mismatched comparator oracles, invalid cycles, and
missing fresh synthesis evidence. They execute no model or HLS jobs. Actual
native/CSim/RTL results remain pending execution. The user approved 250 aggregate
worker-hours for this gate; it will run under that explicit cap.

`reproducibility/studies.json` distinguishes these saved-design checks from
other studies. Adapter comparison scripts and router fusion scripts summarize
existing outputs; they do not supply a fresh inference/evaluation launcher.
The original 540-cell open-model generation command, per-fold router bindings,
and matched adapter serving/evaluation registrations remain incomplete. Both
Qwen adapters and the orchestrator-only adapter declare the same Qwen snapshot;
their configuration/training-manifest metadata is recorded, while payload
hashes are not yet authenticated against historical serving revisions. The
two distinct historical Gemma serving revisions remain unauthenticated. GRPO
retains only an external preliminary summary and lacks the original package,
verified task split, raw candidates, and checkpoint hashes. No retraining was
requested or performed.

The initial stopped hardware trial exposed a validator parser omission: its
XML recorded a 3.33 ns target under `UserAssignments`, but the legacy parser
only looked under `SummaryOfTimingAnalysis`. Native, CSim, and RTL checks had
passed; qualification incorrectly lacked the requested clock. The validator
now binds that existing XML field to the recorded Tcl and records their
hashes. A read-only check of the preserved report confirms target 3.33 ns,
estimate 2.692 ns, and derived slack 0.638 ns. The original trial measurement
remains unchanged, including its original failure classification. Its 114.65
charged seconds remain in the aggregate accounting when registering a new
validation run. The numerical XML excerpt and exact Tcl are retained in
`tests/fixtures/vitis2023_2_nussinov`; regression cases cover a different 5 ns
target, conflicting XML/Tcl periods, missing evidence, and clock-file changes.

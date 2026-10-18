# Commercial-model archive and RTL backfill, October 8, 2026

The owner supplied `c2hls_newskills_commercial_sweep.tar.gz` from the HPCA
checkout. Its SHA-256 is
`1235bf8226e5bd84b87dc9458323b293856e8c3d987148e1538954d155dad80d`.
All 4,578 payload checksums match. Extraction rejected unsafe paths and links;
the imported payload is retained separately from the existing experiments.

## Recovered results

The archive contains 546 run records: three commercial models (Sonnet, Opus,
GPT-5.5), seven configurations, and 26 benchmarks. Of these, 540 report success
and six report failure. All 546 generated-candidate co-simulation fields are
`not_run`; reported success does not establish functional or RTL correctness.
Across the entire grid, 539 records have linked source, synthesis report, and
CSim completion; four source/report associations are unresolved and three
records lack saved generated source.

All **156 Figure 7 selected/comparator slots**, representing **144 distinct
model/benchmark/configuration selections**, now have exact source/report/CSim
associations. The imported schemas contain the same parsed records as the
paper's three teacher inputs. Sonnet differs in serialization, not record
content. This resolves the missing-source block recorded on October 7.

Evidence, under `artifacts/sigmetrics2027/commercial_sweep_20261008/`:

- `import/import_receipt.json`: archive and per-file checksums.
- `import/source_audit_summary.json`: grid coverage and paper joins.
- `import/full_grid_inventory.csv`: all 546 records.
- `import/selected_inventory.csv` and `import/selected_slot_join.json`:
  exact saved implementations and their paper slots.
- `input_audit/report.json`: input identity and independent golden checks.
- `input_audit/selected_native_preflight.csv`: native candidate outcomes.
- `input_audit/report.md`: discrepancy diagnosis and harness limitations.

## Validation contract and discrepancies

The archive does not contain the original testbench or metadata. All 546
first-translator prompts recover plain kernels and headers matching the local
public enhanced-fc27133 benchmark inputs after outer-whitespace normalization.
The new validation uses those public, full-size testbenches, with explicit
input, registry, and metadata hashes. It does **not** claim byte-identical
historical testbenches. No reduced dimensions, replacement generated kernels,
expert implementation, or new model calls are used.

All 26 independent native goldens pass shape and finite-output checks. All 144
selected generated kernels compile and run without changing their source.
Independent native output comparison passes for **137** and fails for **7**
at absolute tolerance 1e-6 and relative tolerance 1e-5. These correspond to
147 passing and nine nonpassing paper slots because some selections are shared.
Native checks are not Vitis CSim or RTL verification.

Three discrepancies reproduce apparent code errors: Opus GEMM `ms_noskills`
overwrites a parity buffer before storing its previous result; Sonnet
correlation `ms_noskills` uses the wrong normalization factor; Sonnet Jacobi2D
`ms_noskills` omits initialization of local boundary values. Four Gram-Schmidt
cases require numerical analysis of the original M=60, N=80 fixture; do not
classify these automatically as structural code bugs. Preserve every failed
comparison and confirm with Vitis CSim under the same contract. RTL dispatch
requires a passing independent C-output comparison. Failed candidates remain
in the original cohort denominator; do not select another configuration after
observing validation results.

## Skill-policy interpretation

The archive contains two extra libraries with 99 and 33 entries, covering 129
distinct skill IDs. Their historical launchers name both files. All 86 skill
IDs found in rendered recipe blocks and all ten recorded routing IDs occur in
these archived libraries. The external base-55 library and historical loader
are absent, so the exact merged store and tag precedence remain unproven.

In **77 of 78 `ms_noskills` runs**, histories still record skill-ID-based dynamic
step routing and associated controller messages, although no rendered
`[skill ...]` recipe block is present. This arm disables recipe prompt blocks;
it is not an entirely skill-library-free controller. `ms_curated` also has no
such rendered blocks in this export. Do not infer the absence of all
skill-derived text solely from missing markers. Figure/table interpretation
should retain this qualification when attributing improvements to recipes.

## Scheduling policy

Prioritize the 144 paper selections before expanding to other archived arms.
Register each validation using source, full input contract, toolchain,
evaluator, and timeout scope, and check existing outcomes and reserved July
identities before dispatch. A saved failure or timeout is evidence, not an
implicit request to retry or replace a design.

The resource audit found 128 logical CPUs, approximately 650 GiB available RAM,
768 GiB free storage, and system load around 30. Two additional Vitis workers
fit the current headroom. The existing recovery controller rejects unrelated
Vitis peers, so the new scheduler coordinates ownership rather than launching
an unregistered competing controller. Existing active jobs drain normally;
only never-started recovery jobs transfer to the successor. The original
recovery manifest, evaluator, 10,800-second limits, and terminal outcomes are
preserved. New commercial validations use a separate fixed 3,600-second Vitis
invocation limit; this includes tool stages and is not a simulated-cycle limit.

The combined scheduler reserves two recovery workers and two commercial
workers, checks storage/RAM/load before dispatch, and records source and state
snapshots. It maintains the old recovery operational summary after taking its
locks so existing reports and follow-up dependencies remain meaningful.
The old recovery resource gate will list the commercial processes as
`unrelated_vitis` while draining. That is expected: its recorded stop request
already prevents new dispatch. The combined controller explicitly admits the
authenticated predecessor and assumes its pending queue after both active
jobs finish. No Vitis subprocess receives a handoff signal.

## Launch verified at 02:09 CDT

The frozen combined controller started in tmux session
`c2hls_commercial_parallel_cosim_20261008`. At 02:09 CDT, two commercial jobs
started alongside the original FDTD-2D and correlation recovery jobs:
GPT-5.5 2mm `flash_allpos` and `ms_noskills`. Both reached synthesis/RTL build
after the independent Vitis CSim gate. **Two commercial tasks are running and
142 are pending; this snapshot contains no completed commercial RTL result.**
The original two `xsimk` process IDs and start times were unchanged after
commercial dispatch. The resource gate observed approximately 691 GiB
available RAM, 768 GiB free storage, and load 27 on 128 logical CPUs.

Live state is in
`results_sweeps/commercial_parallel_cosim_20261008/summary.json` and
`state.json`. The prepared child is
`results_sweeps/commercial_selected_cosim_20261008`; its original queued state
is deliberately retained, with `parallel_registration.json` pointing to the
combined controller's canonical live state. Raw measurements and archived
tool workspaces will appear under
`artifacts/sigmetrics2027/commercial_selected_cosim_20261008/raw/`.

The combined manifest SHA-256 is
`9ddeea58ade6d0e0dd94dc1df94591118ed5a40bf81184c279581c795559aa7c`.
Its output directory contains `launch_receipt.json`,
`graceful_drain_request.json`, and `parallel_process_verification.json`.
All 144 identities were checked against 1,758 existing/reserved semantic
identities with no duplicate and no unresolved coverage audit issue. The
adapter, input/import audits, controller, and frozen CSim gate passed **86
focused tests**. No model API calls were made.

The remaining archive configurations are inventoried but are outside this
first, paper-priority queue. Expand them only through a separately recorded
preflight/registration; do not mix their outcomes into fixed Figure 7 source
selections.

No Overleaf figure has yet been relabeled as independently RTL-verified on the
basis of this import or native preflight.

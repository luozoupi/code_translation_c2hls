# Historical COSIM Backfill

> Latest report (October 6, 14:24 CDT): the original backfill is complete,
> **405 distinct validations: 279 pass, 24 fail, 102 timeout**. Its separate
> three-hour timeout replay has **33 pass, 28 timeout, 2 running, 39 pending**;
> the separate six-hour diagnostic is still running.
> [Report, paper figures, CSVs, and failure index](../artifacts/sigmetrics2027/backfill_reports/20261006T192447053682Z/report.md).
> Eleven failed FDTD cases have a previously diagnosed early end-of-dump marker
> in the archived harness; they require separate correction/revalidation before
> judging functional status. [Original failure triage](../artifacts/sigmetrics2027/status/20260929T141635190065Z/report.md).
> The earlier HPCA campaign is stopped. Use the explicit SIGMETRICS output
> directory and frozen runner; generic commands below describe the older campaign.



The September 28 campaign covers July 1 through July 31, 2026, by campaign
date. A July campaign may contain cells that finished in August. The scan
includes both principal working trees, archived one-shot evaluations, the
corrected setup matrix, and the skill-dose trials. It does not generate new
kernels, call model APIs, train models, or rerun reference synthesis/COSIM.

## Provenance And Validation

- Archive bundles are SHA-256 checked against their original archive metadata.
  Only compact JSON, C++, and header files are recovered, without replacing
  existing archive links or extracting bulk Vitis work trees.
- Selected code is recovered from embedded final code, the saved final C++
  file, or the selected synthesis report's kernel. Available code hashes and
  synthesis-source agreement are enforced. Conflicts remain blocked.
- Historical summary cycles must agree with the selected controller report.
  Recorded benchmark-input hashes are checked when available. The current
  snapshot and limitations are recorded for older traces without hashes.
- Deduplication requires identical code, benchmark inputs, toolchain, output
  registry, and backfill-validator fingerprint. It does not merge different
  model/setup rows or replace them with their fastest result.
- New validations first build a native oracle from `plain.cpp` and the public
  testbench. They then execute candidate CSim and RTL COSIM. No oracle HLS
  design is synthesized. Historical golden-output hashes are checked when
  present.
- Vitis emits separate C-test and RTL-postcheck dumps. The backfill validator
  requires an RTL-postcheck section and independently compares every complete
  dump with the oracle. A good dump cannot conceal a mismatching dump. This
  handling is local to the backfill and does not relax the framework's parser.
- Existing COSIM is reused only when it can be attributed to the selected
  kernel. Historical leakage-audit flags remain visible; numerical validation
  does not resolve generation leakage.

The first two-kernel preflight exposed the repeated-dump parser incompatibility.
Its failed validation records, logs, archives, and manifest remain preserved.
The corrected validator has a different fingerprint and repeats the same saved
kernels, not new model samples.

## Execution

The target is Vitis 2023.2, `xcu280-fsvh2892-2L-e`, 3.33 ns, Vitis flow.
At most two workers run. New launches wait when unrelated Vitis/XSIM processes
are detected, CPU load exceeds 70% of logical CPU count, or local free space is
below 30 GiB. Each validation has isolated scratch and Vitis user directories.
CSim is bounded at 180 seconds; the combined synthesis/COSIM invocation is
bounded at 3,600 seconds. COSIM requires resynthesizing the **candidate** to
rebuild its RTL, even when historical CSYNTH metrics are available.

The existing long-runtime protection is explicit: skip a new COSIM when the
candidate's selected CSYNTH estimate exceeds ten times the same run's reference
CSYNTH estimate. This is a prediction, not a measured timeout. Unknown reference
cycles do not cause a fabricated prediction or cycle value.

```bash
python scripts/scan_historical_cosim.py
python scripts/run_historical_cosim.py run --workers 2
python scripts/run_historical_cosim.py report
```

The scanner refuses to overwrite a frozen manifest. The queue holds an
exclusive lock, verifies frozen evaluator hashes, and resumes from atomic
per-validation result files. Completed successes and failures are not silently
rerolled. SIGTERM/SIGINT stops new launches and drains active workers. After a
machine interruption, inspect for surviving workers before restarting the queue.

Compact reports live in
`results_sweeps/july_cosim_backfill_20260928`. Inputs, recovered files, logs,
measurements, and verified work archives live under
`/mnt/data2/luo00466/c2hls_rl/july_cosim_backfill_20260928`. Completed scratch is
removed only after a full `tar --compare` check and an archive checksum receipt.

## Tables And Schema

`inventory_and_results.csv` and `.jsonl` retain every historical row, including
blocked and skipped ones, alongside separate estimated CSYNTH and measured
COSIM cycles. `campaign_summary.csv` and `by_campaign/*.csv` expose per-campaign
coverage. Failure cycles are null; no estimates are substituted for RTL cycles.

Companion `*.cosim_backfill.csv` tables are refreshed in these original result
folders without overwriting their historical tables:

- `sonnet46_skillv2_latency_matrix_20260726`
- `skill_v3_qwen_gemma_matrix_20260731`
- `qwen_gemma_oneshot_agentic_sft_20260729`

The website contract permits executed `pass`, `fail`, or `timeout` RTL records,
not queued/not-run records. Consequently `schema_records.jsonl` contains only
executed measurements; queue/skip reasons remain in the accompanying tables.
Every refresh runs the repository's schema validator.

## SIGMETRICS registry recovery launched — September 29, 2026 UTC

The user requested the backfill under the SIGMETRICS worktree. The prepared
registry recovery started at **04:22:30 UTC** in tmux
`c2hls_sigmetrics_cosim_recovery_20260929`, with two workers and 405 distinct
validation identities covering 535 historical rows. The first designs are
Jacobi-1D and BiCG. This is the corrected-registry subset of the broader
historical inventory, not the full unresolved historical queue or the new
216-cell recipe study.

- Worktree: `/home/luo00466/code_translation-c2hls-sigmetrics2027`.
- Compact output: `results_sweeps/historical_registry_recovery_20260928/`.
- Frozen evaluator: `artifacts/sigmetrics2027/registry_recovery_20260928/source/`.
- Raw measurements and archives: `artifacts/sigmetrics2027/registry_recovery_20260928/raw/`.
- Manifest SHA256: `160c58961e34de45d554af5d67af148fe809d80f05a224a9ed3d7d4959e993bb`.
- Launch details and authorization: `launch_receipt.json`; process exit status:
  `launcher_status.json`; controller state: `state.json`; live results:
  `report.md`, `summary.json`, and `inventory_and_results.csv` in the compact output.

All 62 evaluator-file hashes and all 405 distinct staged source/input identities
were verified before launch. The registry and independent-golden preparation
match the passed validator pilot. The later evaluator adds optional validation
and telemetry wrappers; the campaign uses its frozen historical testbenches.
No old pilot measurement was imported because its validation fingerprint differs.
The original kernels, golden tolerances, historical failures, and old campaign
records remain unchanged. No model requests or reference HLS reruns are made.

CSim has a 180-second limit; candidate synthesis plus RTL co-simulation has a
3,600-second limit. Launches pause on unrelated HLS jobs, high CPU load, or less
than 30 GiB of local free space. Failed validations remain recorded and are not
silently retried. The queue updates compact reports while running and continues
in tmux after the interactive session ends. It is ordered by synthesis estimate,
so interim success rates are not an unbiased sample of all queued designs.

Observe without launching anything:

```bash
cat results_sweeps/historical_registry_recovery_20260928/state.json
cat results_sweeps/historical_registry_recovery_20260928/summary.json
python3 scripts/snapshot_sigmetrics_progress.py
```

To drain, first read and verify the current controller PID in `state.json`, then
send SIGTERM to that controller only. Active validations finish before exit.
Do not kill the tmux session or reuse a PID copied from this document. Resume
only after confirming the old controller and workers have exited, using:

```bash
/home/luo00466/.conda/envs/py310_2/bin/python -u \
  artifacts/sigmetrics2027/registry_recovery_20260928/source/scripts/run_historical_cosim.py \
  run --output /home/luo00466/code_translation-c2hls-sigmetrics2027/results_sweeps/historical_registry_recovery_20260928 \
  --workers 2 --timeout 3600
```

## Parallelism increase requested — September 29, 2026 UTC

The user requested one or two additional Vitis sessions. The queue is being
increased from two to **four concurrent validations**, with one scheduling
controller. At the resource check, load was approximately 4.3 on 128 logical
CPUs, about 709 GiB of memory was available, and no external Vitis job was
visible.

The old controller was sent SIGTERM to finish its active work without launching
more. The replacement session `c2hls_sigmetrics_cosim_x4_20260929` waits for its
verified PID/start-time identity to exit and for the queue to record a clean
drain. It then resumes the same manifest and raw results under the same
exclusive queue lock. `parallel_scheduler_status.json` identifies whether the
handover is waiting, running, stopped, complete, or failed.

The scheduling wrapper is `scripts/resume_historical_cosim_parallel.py`; the
launched copy and receipt are frozen under the compact output's
`scheduling_four_workers_20260929/`. It calls the existing frozen scheduler's
worker-count parameter directly; the original CLI only exposed one/two workers.
The frozen validator, 405 validation identities, test inputs, and 3,600-second
timeout are unchanged. The wrapper corrects the report's worker-count label and
records the scheduling fingerprint separately from the evaluator fingerprint.
`parallelism_change_20260929.json` records the request and hashes of all completed
measurements at the transition. Both passes and failures remain resume records.

All 24 focused provenance and historical-validation tests passed. The automatic
resource gate still delays new launches when it sees unrelated Vitis processes,
load exceeds the existing threshold, or local disk headroom falls below 30 GiB.
It does not preempt active workers or create an exclusive machine reservation.
Scheduling concurrency can affect validation wall time; the transition is
recorded so runtime-cost analysis can distinguish the two-worker and four-worker
periods.

For future restarts after a clean four-worker stop, prepare a new transition
receipt for the then-current controller state; do not rerun the first handover
launcher against its archived old PID. The existing frozen worker remains the
validation entry point. Live state, summary, and reports stay in the same
compact output directory throughout this transition.

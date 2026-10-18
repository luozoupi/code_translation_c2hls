# Experiment recovery and co-simulation timeout policy — October 6, 2026

This records the user-approved continuation after the October 4 server reboot and
the October 1 main-study infrastructure failure. Read the continuation summaries
below for live progress. Original campaign state files remain preserved evidence;
in particular, the old replay's `running` label no longer describes a live process.

## Replay recovery

The original three-hour replay saved 59 of 102 historical validations: 33 pass,
26 timeout. Four parent attempts also finished (LU passed all five cases; Jacobi-2D,
2mm, and Seidel-2D timed out). Two historical workers were interrupted by the
reboot; 41 had not started.

The recovery scheduler prepares new, linked validation identities for those 43
unfinished jobs, using the unchanged frozen worker, source, public inputs,
golden comparison, toolchain, clock, tracing settings, and 10,800-second limit.
Two recovery workers operate concurrently. Completed results are never rerun or
overwritten. Parent replay results are not substituted into the frozen main study.

Live artifacts:

- [Recovery summary](../results_sweeps/sigmetrics_timeout_recovery_20261006/summary.json)
- [Recovery state](../results_sweeps/sigmetrics_timeout_recovery_20261006/state.json)
- [Registered inputs and policy](../results_sweeps/sigmetrics_timeout_recovery_20261006/manifest.json)
- [Per-validation observations](../results_sweeps/sigmetrics_timeout_recovery_20261006/telemetry/)

## Timing observations and diagnostic policy

The [offline audit](../artifacts/sigmetrics2027/cosim_timeout_policy_20261006.json)
finds insufficient evidence to fit reliable per-benchmark completion-time limits.
The successful replay sample is selected by earlier timeouts and completion under
the current cap. Timed-out observations provide a lower bound, not an exact
completion time. Kernel latency cycles divided by whole-worker elapsed time is
not simulator throughput: those quantities cover different scopes.

The 10,800-second limit bounds the entire per-case Vitis invocation, including
fresh synthesis, co-simulation setup, RTL elaboration, XSIM, and output checking.
Historical worker elapsed time also includes native golden generation and C
simulation. The parent suite runs five cases sequentially and stops on first
nonpass; its total wall time can exceed a single case's limit.

A sidecar samples private scratch logs and owned process identities every ten
seconds. It records available phase markers and tool-reported durations, simulated
time, completed transactions, warnings, CPU deltas, and coverage gaps. Reads are
bounded and incremental. Sampled phase transitions are observation windows;
buffering or skipped bytes can make exact boundaries unknown. Reported nested
phase durations must not be summed as independent costs. Clock activity, CPU use,
and warning output do not establish useful transaction progress or rule out a
hardware deadlock. The observer never changes correctness decisions, simulator
options, or worker termination criteria.

One separate diagnostic worker retries the already-timed-out correlation design
`e9ab1d771854d65fe97f48d1425cbff09c6f05b2429ef96b7e129348ca002344`
with a fixed 21,600-second per-case cap. Selection is prospective and deterministic:
choose the smallest recorded synthesis cycle estimate among completed three-hour
timeouts, breaking ties by validation ID. This design's archived trace advances
simulation time but records zero of one completed transactions and many DSP
warnings. That evidence motivates diagnosis; it does not predict eventual success.

Six hours is a declared twofold escalation of the previous censored limit, not a
calibrated prediction. This diagnostic has its own identity and denominator and
cannot change any original pass/fail/timeout result. The same six-hour rule applies
to all cases admitted to this diagnostic policy; there is no model-specific cap.
The recovery controller runs at most two ordinary recoveries plus one diagnostic.

Future predictive budgets should separate setup from simulation and use
`setup + margin * total_simulated_clocks / observed_simulator_clock_rate`, with
matching input sizes, testbench calls, tool version, generated-design class,
tracing settings and host concurrency. The total clocks must include interface,
reset and testbench activity. A kernel's synthesis estimate is not that total.
A revised prediction rule must be fixed before comparing models/modes and retain
all timeout outcomes and measured validation costs.

## Main-study scheduling amendment

The frozen 216-cell study stopped with 71 complete, 90 parent-blocked, one
execution/accounting failure, and 54 pending. All 54 pending identities have no
prior cell artifact, log or API reservation; their seven parents remain qualified.
A separate two-worker scheduler can execute these identities through the frozen
`cell` implementation, retaining prompts, models, call/token/synthesis limits,
validation suite, tool timeouts, original selection order and shared $300 ledger.
The original controller state and all existing outcomes remain unchanged; the
continuation summary merges original statuses with the 54 new statuses over the
same denominator of 216. Both controllers share the original campaign lock where
appropriate, and recovery recognizes only an authenticated main-continuation
process tree when checking competing Vitis work.

The Nussinov/GPT/none/repeat-2 failed cell remains failed. Its second request has
no locally saved response or usage, so its $0.0240036 upper-bound reservation
remains unsettled. The parent-only RTL result in that failed run is not promoted
to a completed optimization outcome. Continuation neither rerolls that cell nor
invents its token usage. It stops new dispatches on any new execution/accounting
failure while draining active cells. Provider aliases may differ across calendar
intervals; this continuation interval must remain visible in analysis.

Live artifacts:

- [Main continuation summary](../results_sweeps/sigmetrics_main_continuation_20261006/summary.json)
- [Main continuation state](../results_sweeps/sigmetrics_main_continuation_20261006/state.json)
- [Scheduling amendment](../results_sweeps/sigmetrics_main_continuation_20261006/manifest.json)

A successful drain would leave 125 completed cells, 90 parent-blocked cells and
one quarantined failed cell, not 216 successful runs. Original uncertainties and
cost reservations remain part of the reported evidence.

## Checks and launch receipts

The new code has targeted offline tests for incomplete-job selection, preservation
of terminal measurements, cohort separation, phase/progress parsing, bounded
log reads, corrupt evidence, accounting integrity, duplicate controllers, and
owned-process cleanup. Launch identities and frozen scheduler hashes are saved
under each continuation output. Current launch observations are recorded after
startup below.

### Verified startup — October 6, 2026

All **59 targeted tests passed** under the original `py310_2` interpreter. Both
prepared campaigns also passed verification of their frozen inputs before launch.

| Detached tmux session | Launch time (CDT) | Worker allocation |
| --- | --- | --- |
| `c2hls_sigmetrics_main_continuation_20261006` | 10:17:11 | Two workers for 54 untouched main-study cells |
| `c2hls_sigmetrics_timeout_recovery_20261006` | 10:17:35 | Two workers for 43 unfinished three-hour validations, plus one separate six-hour diagnostic |

Launch receipts are saved in each output directory as
`continuation_launch_receipt.json` and `recovery_launch_receipt.json`, respectively.
Controller identities include PID start time, boot identity, and frozen-code hashes.

At 10:18 CDT, the main continuation had two running cells and 52 pending. GEMM
(DeepSeek, no recipes, repeat 2) and ATAX (DeepSeek, structured recipes, repeat 2)
had produced candidate synthesis reports; their new provider requests had settled
usage records. These are intermediate observations, not completed cell outcomes.
The 365 pre-continuation ledger entries were unchanged, including the quarantined
$0.0240036 reservation. The original aggregate remained 71 completed cells,
90 parent-blocked cells, and one failed cell.

The replay had two running recoveries (correlation and SYMM), 41 pending
recoveries, and one running correlation diagnostic. All three telemetry streams
were writing valid samples; SYMM reached RTL elaboration, replay correlation
reached co-simulation setup, and the diagnostic correlation was synthesizing.
No controller or startup errors appeared. Resource gates passed with about
802 GiB free scratch and 700 GiB available memory. Total active worker allocation
is five; worker allocation does not imply five simultaneous simulator processes.

The startup observations above are a snapshot. Read the linked live summaries
for subsequent completions, failures, timeouts, and resource-gate decisions.

## Backfill report and figures — 14:24 CDT

The [captured report](../artifacts/sigmetrics2027/backfill_reports/20261006T192447053682Z/report.md)
contains five paper-style figures (PDF/SVG/PNG), measured RTL latency tables,
paired reviewer comparisons, and an index of every failure, timeout, and
unfinished validation. Its input JSON files are preserved with SHA-256 hashes.
All plotted passing measurements passed the source, toolchain, independent
golden-output, CSim, and RTL-postcheck consistency checks.

| Cohort | Pass | Recorded failure | Timeout | Running | Pending |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original one-hour backfill (405 distinct validations) | 279 | 24 | 102 | 0 | 0 |
| Three-hour timeout replay, including recovery (102) | 33 | 0 | 28 | 2 | 39 |
| Separate six-hour diagnostic (1) | 0 | 0 | 0 | 1 | 0 |
| Reviewer A/B, repeat 1 (102 distinct validations) | 70 | 9 | 23 | 0 | 0 |

Today's recovery has added two three-hour timeouts (correlation and SYMM).
Covariance and Cholesky are active. The longer correlation diagnostic is still
running; its advancing simulated clock has not established a completed
transaction. Both controllers remain alive. The main study is separate and has
79 completed cells, 90 parent-blocked, one quarantined failure, two running, and
44 pending at the report capture.

The original 405 validations map to 535 historical rows; reviewer A/B's 102
validations map to 108 cells. Cohort plots count distinct validations, while
latency panels display each corresponding reviewer cell. A timeout/failure has
a status marker and no fabricated latency. Historical reference-audit flags and
known harness failures remain visible. In particular, both multistep TRMM cells
timed out, so these data do not establish an RTL-verified multistep TRMM advantage.

Refresh without affecting experiments:

```bash
/home/luo00466/.conda/envs/py310_2/bin/python scripts/report_cosim_backfill.py
```

Each refresh creates a new timestamped snapshot rather than replacing an earlier
report or changing an experiment result.

## Follow-up queue — 16:40 CDT

The [registered follow-up protocol](followup_experiments_20261006.md) queues
1,139 distinct July routing co-sim validations and 56 new PolyBench agent groups
(28 kernels, Flash and multistep, up to five full attempts each). Its tmux
controller is alive and waiting for the main continuation and timeout recovery
to complete; zero new API/Vitis attempts were dispatched at startup. The
[canonical live summary](../results_sweeps/sigmetrics_followup_queue_20261006/summary.json)
owns task progress for both new campaigns.

The new agent protocol measures `cosim_design` host wall time, excludes synthesis
from that phase timer, and counts expired attempts against K. Existing reference
timings are preserved in an audit; none has sufficient setup equivalence for
automatic relative calibration, so all 28 kernels currently use an explicit
3,600-second phase fallback. This is a resource limit, not proof of deadlock or
an estimate derived from hardware cycles.

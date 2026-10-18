# Experiment status — October 7, 2026, 1:04 p.m. CDT

The timeout recovery is running. The main study stopped on an infrastructure
error on October 6 at 5:17 p.m. CDT; this also blocks admission of the follow-up
queue. Host process checks confirmed the recovery and follow-up controllers are
alive and the main controller has exited. No experiment was restarted or its
protocol changed during this status audit.

| Experiment | Status and counts |
| --- | --- |
| Main recipe study, 216 cells | 93 complete; 90 parent-blocked; 2 execution/accounting failures; 31 pending |
| Original historical backfill, 405 distinct validations | Complete: 279 RTL passes, 24 validation failures, 102 timeouts at one hour |
| Three-hour replay of those 102 timeouts | Running: 37 passes, 40 timeouts, 2 running, 23 pending |
| Separate six-hour correlation diagnostic | Finished: timeout; advancing clock did not establish transaction completion or deadlock |
| Reviewer A/B RTL, repeat 1, 102 distinct validations | Complete: 70 passes, 9 validation failures, 23 timeouts |
| July routing follow-up | Waiting: 1,139 distinct validations covering 1,401 historical result rows |
| DeepSeek V4.1 Flash success within K attempts | Waiting: 28 kernels × two modes = 56 groups; zero attempts started |

The 93 completed main-study cells include 91 passing selected-design RTL
measurements, one timeout (GEMM / DeepSeek / none / repeat 1), and one failed
co-simulation (Nussinov / DeepSeek / flat / repeat 2). Completion counts are not
counts of optimization gains. Ninety cells remain blocked because five of the
twelve registered parent benchmarks failed qualification; later parent replays
do not retroactively change that frozen study.

Relative to the October 6 afternoon snapshot, the main study completed five
more cells. Three-hour replay added three passes (one SYR2K and two SYRK
implementations) and eleven timeouts. SYRK and LU are currently active, started
at 11:55 a.m. and 12:05 p.m. CDT; both simulator logs were advancing at the audit.
Across the original backfill and its replay, **316/405 distinct designs now have
passing independent RTL evidence**. This combines different time limits and is
validation coverage, not a common-budget success rate. Cohorts overlap and
their denominators must not be summed.

## Main-study blocker

The new failed cell is
`hlsfactory_gramschmidt_m60_n40__gpt-5.6-luna__structured__r3`.
Its controller recorded `ProcessLookupError: [Errno 3] No such process:
'/proc/1964036/stat'` at October 6 5:17:29 p.m. CDT. This is consistent with a
process-enumeration race; an exact stack trace was not retained. It is not an
observed kernel validation failure or simulation timeout. The resource gate
was open at the time.

The cell's single API request was settled (3,595 input tokens, 1,010 output
tokens; estimated $0.001931). The only unresolved reservation remains the
original October 1 Nussinov request ($0.0240036). The shared ledger contains
475 settled requests and one reserved request, estimated total charged or
reserved $0.833909384 against the $300 cap. These are token-cost accounting
estimates, not provider invoices.

All 31 pending cells have no output folder, run log, or API reservation. The
next recovery should fix process enumeration and register a new continuation
for these untouched cells while preserving the original worker, prompts,
validation policy, and the two failed records. The prior continuation's prepare
routine assumes the older counts and cannot simply be rerun. The follow-up
queue must then be explicitly connected to the new recovery's completion;
its current `stopped_on_failure` predecessor will never satisfy the existing
completion gate, even after timeout recovery finishes.

## Evidence and refreshed figures

- [Main continuation summary](../results_sweeps/sigmetrics_main_continuation_20261006/summary.json)
- [Main failure state](../results_sweeps/sigmetrics_main_continuation_20261006/state.json)
- [Timeout recovery summary](../results_sweeps/sigmetrics_timeout_recovery_20261006/summary.json)
- [Follow-up queue summary](../results_sweeps/sigmetrics_followup_queue_20261006/summary.json)
- [Fresh backfill report](../artifacts/sigmetrics2027/backfill_reports/20261007T180409969217Z/report.md)
- [All five updated figures, PDF](../artifacts/sigmetrics2027/backfill_reports/20261007T180409969217Z/cosim_backfill_figures.pdf)
- [Failures, timeouts, and unfinished validations](../artifacts/sigmetrics2027/backfill_reports/20261007T180409969217Z/failure_and_timeout_index.csv)

The report has zero evidence-consistency issues. It was generated with host
process visibility; a preceding sandbox-only capture could not see the two
active Vitis process identities and is explicitly superseded. These checks
validate report consistency, not all-input equivalence or historical reference
isolation.

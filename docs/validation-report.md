# Cleanup validation report

This report separates preserved paper evidence, checks of the cleaned runtime,
and new experiments. The offline checks below generated no designs and ran no
hardware jobs. New campaign state is tracked separately under
`/mnt/data1/luo00466/c2hls_runs_20261018`. Experimental equivalence remains
unestablished.

## Completed checks

The preservation archive is
`/mnt/data1/luo00466/c2hls_archive_20261018`. Its compressed payload has SHA-256
`db45bbb46836a113e1dfde275853fca3ce60bfb9ef14733beffa79f282bc529c`.
Every archived member was checked, representative files were restored, and
the three independent restored Git object stores passed `git fsck --full`.
Live queue metadata is preserved in timestamped overlays. The August archive
was retained.

The canonical repository has its own Git object store and does not depend on
the historical common Git directory. Byte-preserved benchmark, skill and
configuration files are packaged with the implementation. Compatibility
entrypoints retain the original Python module identity and CLI arguments.

The full offline suite passed **866 tests and 127 subtests**, with 29 hardware
or legacy tests skipped. The isolated wheel install passed from `/tmp` with an
empty `PYTHONPATH`, built from a clean Git archive and checked against all 406
required release files, including four CLI help commands, 58 compatibility aliases,
packaged resources and source provenance. Machine-readable evidence is in
[validation-receipt.json](validation-receipt.json).

The offline suite blocks real network connections and FPGA tool jobs. Saved
model responses and tool reports exercise translation, repair, correctness
gates, resource eligibility, failure classification and winner restoration.
The independently loaded frozen and clean runtimes produce canonical result
SHA-256 `1da9bda6aa0a7dc809b39cb1d94b7369827f1d64ae906e3638e5755bdf028ee6`.
All 23 paper prompt templates retain their original bytes. The direct replay
covers five portable saved agent paths, seven GEMM configurations and 18 extracted
source-contract helpers. An expanded private fixture additionally covers all
144 fixed commercial designs and their 156 paper roles, with exact saved
response/source associations and separate saved-source/tool-report replay.
Both runtimes agree on this expanded boundary replay. It does not reconstruct
the original complete optimization searches or establish full-corpus execution
equivalence.

The expanded fixture covers all 144 fixed commercial designs with no exclusions.
Its canonical result SHA-256 is
`cb075464e62ef8a63a1859d238951695f96193c3bffda083adfdf7a19d2daafb`.

The portable paper replay verifies 106 frozen files and regenerates numerical
exports against paper commit `ecd3c750168dea52aa80dd609f463b9b668689b0`.
It preserves the original manifest paths and hashes, including discrepancies
between early commercial validation and later independent replay. A numerical
replay does not constitute a new hardware measurement.

Vitis version queries identify both installed tools as 2023.2. Fresh comparison
cohorts use the same verified `vitis-run` binary and arguments. The target is
U280 (`xcu280-fsvh2892-2L-e`) at 3.33 ns. Version queries did not synthesize or
simulate a design. Tool hashes are in [validation-tool-versions.json](validation-tool-versions.json).

## Remaining acceptance gates

| Gate | Status | Required evidence |
| --- | --- | --- |
| Archive integrity and representative restores | Passed | Private archive manifests and restore receipts |
| Offline compatibility and paper numerical replay | Passed within the stated coverage | Test receipt and checksum-bound replay fixtures |
| Whole-corpus deterministic runtime equivalence | Not established | Broader captured request, selection and tool-boundary replay |
| Historical selected/comparator hardware replay | Running under the 250-hour cap | Fresh fixed-design native, CSim and RTL records; no repair or reselection |
| Frozen versus clean DeepSeek paper-profile comparison | Queued after the hardware campaign | 26 kernels × 7 configurations × 3 repetitions for each implementation |
| Readable prompt evaluation | Queued as a separate DeepSeek cohort | Separate clean-profile cohort with the same matrix |
| Available frozen-checkpoint studies | Evidence and assets retained; live configurations require binding | See [study inventory](../reproducibility/studies.json) |
| Default branch promotion | Pending validation | Correctness, resource eligibility and QoR evidence with uncertainty |

The complete prospective DeepSeek comparison contains 1,638 cells. Historical
token usage projects approximately $55.01 at verified peak rates, without cache
credits, and 140.14 generation worker-hours. Independent final validation adds
runtime not estimated by those historical generation records. The combined
timeout ceiling is 8,186.1 worker-hours. These figures are workload estimates,
not spending authorization or runtime guarantees.

The API identifier `deepseek-flash` is a mutable alias for the provider's declared
DeepSeek-V4.1-Flash-1017 release. Thinking is explicitly disabled and supported
seed control is not claimed. Seven inherited configuration names do not prove
seven active treatments: the frozen single-shot entrypoint does not inject
the curated/all-positive recipes. The registration records that limitation.

The user approved $100 API spend, 500 DeepSeek worker-hours, 250 historical
saved-design worker-hours and 1,000,000 aggregate input tokens per DeepSeek cell.
GPU jobs remain stopped. The saved-design gate fixes 324 available designs
(144 commercial and 180 open), with frozen and clean evaluator cohorts giving
648 cells and an 864-hour timeout ceiling. Five unresolved open comparator
identities remain blocked in each cohort. See [its workflow](saved-design-validation.md).

The DeepSeek credential is supplied through a private local key file.
The hardware gate needs no API credential. These are separate capped
campaigns; caps may stop a run before completing its matrix.
Failures, timeouts, missing slots and incomplete studies remain
in the report. Jointly passing latency ratios alone do not establish correctness
coverage or equivalence. An inconclusive comparison cannot justify promotion.

The paper prompt profile remains the default. The readable profile and the
clean development branch are candidates until their respective gates pass.


The durable supervisor has started the historical hardware campaign. It uses
immutable NAS runtimes rather than the mutable development checkout. The
DeepSeek comparison follows in sequence with the supplied private credential
and an immediate official release/pricing check. See [campaign operations](campaign-operations.md)
for live state and report commands. These launch records are not passing
hardware/QoR results; default promotion remains pending.


The first live trial exposed two issues beyond the initial offline coverage:
an ignored required module in the Git export, and a Vitis requested-clock value
stored under `UserAssignments/TargetClockPeriod` rather than the parser's old
location. The corrected release now verifies every required source/asset in
Git, its clean export and the built wheel. Fresh clock evidence is bound to
actual XML/Tcl values and hashes, with conflicting targets rejected. The
original trial's statuses and measurements are preserved. Its 114.65 seconds
are charged against the approved total. Corrected `r2` registrations use source
commit `f9e94ebd945e7db726625da3ba9bed2c554d9f62`; they do not change selected designs.


The first corrected hardware pair passed in both evaluators: the fixed Nussinov
source produced 482,203 measured RTL cycles, passed native/CSim/RTL output
checks and met resource/timing eligibility at the recorded 3.33 ns target.
The queue then advanced to the next design. This is one completed pair, not
whole-campaign acceptance or evidence of fresh-model equivalence.

At the October 20 04:17 UTC audit, 195 hardware checks had completed, covering
97 complete design pairs. Eighty-four pairs passed both evaluators with identical
RTL cycles and materialized source hashes. Six pairs failed RTL checks, four
failed native checks and two timed out in both evaluators. One LU design passed
the frozen run but timed out after 3,600 seconds in the clean run; its source,
15 generated RTL files and synthesis metrics were identical. The frozen run
took about 3,514 seconds. This remains a censored wall-time discrepancy, not
proof of equivalence or a demonstrated cycle regression. A further frozen
Cholesky cell had timed out while its clean counterpart was still active.

The user subsequently requested parallel Vitis execution. The queue transitions
to at most four workers after the active serial cell finishes, retaining its
manifest, selected sources, evaluator runtimes, per-cell timeouts and cumulative
250 worker-hour cap. A separate scheduling snapshot and timestamped handoff
receipts record the change. Serial and parallel execution conditions remain
visible when assessing timeout coverage; earlier measurements are not rewritten
or rerun. DeepSeek generation remains a later stage under its original limits.
The scheduler change passed 109 focused offline tests; four private handoff
checks covered normal progression, hardware-cap reporting, active-cell rejection
and carry-forward accounting. These checks launched no Vitis or model jobs.

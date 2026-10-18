# Co-simulation evidence for the current SIGMETRICS visuals

Audited October 7, 2026, with Figure 8 evidence captured at 3:23 p.m. CDT.
Overleaf was fetched and fast-forwarded to `a1c7562`; the online change only
affected a background sentence. No manuscript results or graphics were edited.
Figure/table numbers below are the current SIGMETRICS numbering, not HPCA's.

**Figure 8 and the validation-coverage table can be updated now. Figures 7 and
9 require evidence from their own cohorts.** Aggregate backfill counts cannot
be substituted for source-matched measurements.

| Item | Current content | Readiness and proposed change |
| --- | --- | --- |
| Figure 7 / Table 7 | Sonnet, Opus, GPT-5.5 synthesis sweeps; supplied DeepSeek/Grok RTL summaries | Preserve the existing evidence distinctions. The 405-validation backfill covers the open-model skill-v3 cohort, not these teacher/commercial runs. New teacher validations need exact selected-source and input joins. |
| Figure 8 | Gemma/Qwen, two modes, five policies, 27 kernels | Ready for an RTL validation overlay or measured recipe-disabled/selected cycle ratios. There are 108 synthesis-selected plotted designs and 108 comparator slots; preserve that selection rather than selecting new winners after RTL. |
| Figure 9 | Fourteen ChatHLS/HLSmith attempts | No new independently replayed ChatHLS outputs are supplied by these backfills. Keep its historical qualification until both systems' source, inputs, and output checks can be matched. |
| Table 4 | Evaluation matrix and evidence coverage | Update the open-model row to describe independent RTL validation, exact coverage, tested inputs, and timeout policy. |
| Table 8 | Sonnet setup-router evaluation | Keep its synthesis-based interpretation; this is a different cohort whose queued July validation has not started. |
| Table 9 / Figure 10 | Translator SFT diagnostic / five-kernel adaptation pilot | Separate cohorts; current skill-v3 passes do not upgrade their validation status. |

## Exact Figure 8 coverage

Every selected-design path joins the historical validation manifest. Of 108
recipe-disabled comparator slots, 103 join and five are blocked by a mismatch
between the historical summary's selected cycles and the saved implementation.
For all 211 linked arm slots, source-result bytes, saved kernel SHA-256,
benchmark-input hashes, and toolchain identity match. Passing pairs also have
matching inputs and toolchain across both arms. These slots share some source
and validation identities; they are not independent repeated samples.

| Model / mode | Selected RTL passes, 1 h | Both-arm passing pairs, 1 h | Selected passes including successful 3 h replay | Both-arm passing pairs including replay |
| --- | ---: | ---: | ---: | ---: |
| Gemma / Flash | 23/27 | 17/27 | 24/27 | 20/27 |
| Gemma / multistep | 19/27 | 14/27 | 21/27 | 17/27 |
| Qwen / Flash | 20/27 | 9/27 | 25/27 | 13/27 |
| Qwen / multistep | 22/27 | 12/27 | 24/27 | 15/27 |
| Total | **84/108** | **52/108** | **94/108** | **65/108** |

The latest successful SYRK replay adds a Gemma Flash comparator and raises
paired coverage from 64 to 65; selected-design passing coverage remains 94.
The three-hour replay is still incomplete. An original one-hour timeout with
no successful replay must not be presented as a completed three-hour failure.

Use the completed, uniform **one-hour cohort for the primary update** and
label longer-budget recovery as a separate timeout-sensitivity result. The
one-hour selected-design outcomes are 84 passes, five validation failures, and
19 timeouts. Comparator outcomes are 57 passes, nine validation failures,
37 timeouts, and five identity-blocked slots. Failures may include harness or
output-parser problems; they are not all demonstrated algorithmic errors.

The five blocked comparator slots are Gemma multistep Nussinov; Qwen Flash
3mm and Floyd–Warshall; and Qwen multistep ATAX and Nussinov. Mark them as
unresolved identity, rather than timeout or missing zero-height bars.

## Recommended Figure 8 design and table wording

1. Retain all 27 kernel positions in each model/mode comparison. Draw measured
   `L_off / L_selected` only for pairs with independent RTL passes. Mark timeout,
   validation failure, and unresolved identity separately where a ratio is
   unavailable. Give each model/mode panel its paired denominator.
2. Preserve the synthesis-selected implementation and its original policy
   label. Do not replace a failed selected implementation with a slower RTL
   survivor without defining and reporting a new selection rule.
3. Alternatively, retain the existing synthesis plot with explicit RTL-status
   markers and add measured absolute cycles in a compact companion panel.
   The present upstream-reference values are synthesis estimates (25 website
   schema references and two cache references). They cannot be divided by
   measured candidate RTL cycles and relabeled as measured RTL speedups.
4. Add pass/failure/timeout/identity-blocked counts and tested-input scope to
   Table 4 or an adjacent compact validation table. Keep the one-hour primary
   and three-hour sensitivity denominators separate.

Suggested coverage wording: “For the 108 implementations selected by the
archived synthesis-based policy, independent RTL checks pass for 84 under a
one-hour synthesis-plus-co-simulation limit. Both selected and recipe-disabled
implementations pass in 52 comparisons. Successful three-hour retries increase
these counts to 94 and 65, respectively; the retry cohort remains incomplete.”

These are retrospective hardware-validation results. The best-of-five
selection, advisory reference-isolation failures, differing Gemma serving
revisions, and lack of repeated seeds remain qualifications. Co-simulation
does not establish a controlled causal effect of recipes or held-out model
generalization. Validation covers the saved public testbench, not every input.

## Controlled-study figure

The separate recipe-representation study has 93 completed cells: 91 selected
RTL passes, one timeout, and one RTL failure. However, no benchmark/model block
yet has all three arms and all three repeats completed. A balanced first-repeat
panel exists for seven qualified kernels × two models × three arms (42 cells:
41 passes and one timeout). Such a panel must be labeled preliminary and retain
the timeout. Finishing the 31 pending cells and resolving the two infrastructure
error dispositions is preferable before presenting the final arm-effect plot.
The 90 parent-blocked cells remain in registered-study coverage reporting.

The existing compiled manuscript ends technical content on page 21. Prefer
replacement panels and compact coverage columns; adding standalone figures
would worsen the 20-page constraint.

## Captured audit

- [Figure 8 identity join, CSV](../artifacts/sigmetrics2027/paper_readiness/20261007T202357Z/figure8_identity_join.csv)
- [Per-mode counts and source/hash audit, JSON](../artifacts/sigmetrics2027/paper_readiness/20261007T202357Z/figure8_identity_join.json)
- [Capture manifest](../artifacts/sigmetrics2027/paper_readiness/20261007T202357Z/capture_manifest.json)

The captured source report was generated in a restricted process view. Its
live/interrupted process labels are not authoritative; the figure join uses
completed one-hour measurements and successful source-matched replay results.
Use the live controller summaries for operational status.

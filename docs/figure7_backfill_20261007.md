# Figure 7 / Table 7 source recovery

**October 8 update:** the owner supplied a commercial sweep archive. All 156
teacher slots (144 distinct selections) now have linked saved implementations,
reports, and historical CSim completion. Public full-size input reconstruction
and independent native checks are recorded in
[the October 8 audit](commercial_archive_backfill_20261008.md). The original
testbench remains unavailable; new validation explicitly uses the matching
public harness. The October 7 findings and frozen blocked inventory below are
retained as a historical snapshot, not the current admission status.

Audited October 7, 2026. The plotting pipeline is
`scripts/generate_aligned_model_expertise.py` in the Overleaf checkout. Its
source selections are intact: 26 displayed kernels for each of Sonnet 4.6,
Opus 4.8, and GPT-5.5. These are distinct from the July Sonnet skill-v2 queue
and the Gemma/Qwen cohort in Figure 8.

**The compact source data were recovered, but no exact teacher implementation
and original-input package was recovered. No Figure 7 Vitis work was launched.**
The preparation package records every required selection and blocks execution
until its identity can be established.

The experiment owner subsequently indicated that the kernels may be on another
computer and will search there. Further source search and backfill admission
are deferred pending those artifacts. The current newest skill catalog is the
owner's working reference for the skills used; it is not a recovered historical
catalog hash or proof of byte-identical prompts. Backfill of the original saved
implementations will not regenerate designs with the current catalog.

| Teacher source | CSim records | Synthesis records | RTL records | Selected/off slots |
| --- | ---: | ---: | ---: | ---: |
| Sonnet | 206 | 206 | 26 baseline only | 52 |
| Opus | 208 | 208 | 26 baseline only | 52 |
| GPT-5.5 | 204 | 204 | 26 baseline only | 52 |

The 156 plotted arm slots refer to 144 distinct model/kernel/setup selections:
12 synthesis-selected implementations also serve as their recipe-disabled
comparators. Table 7's common-cohort teacher summaries use 148 of those slots
(25 Sonnet, 26 Opus, and 23 GPT kernels, each with two arm slots).

The teacher `sw_run` records contain `status: pass`, without independent output
comparison. Candidate synthesis reports include U280/Vitis 2023.2/3.33 ns,
resources, and latency. None of the candidate schemas contains an RTL record,
saved C++ path/hash, or original testbench/header hash. The 26 RTL records in
each schema belong to benchmark reference implementations, not generated
teacher-model candidates. Thus the teachers have recorded CSim completion and
synthesis estimates, not independent functional or generated-RTL validation.

## Recovered data and exact selection inventory

The checksum-pinned raw inputs are:

- `/home/luo00466/schema_records_newskills_sonnet.jsonl` and the companion
  `/home/luo00466/newskills_sonnet_csynth_table.csv`.
- `/home/luo00466/code_translation-c2hls-hpca2027/results_sweeps/results-20260730/schema_records_newskills_opus.jsonl`.
- The same July-30 directory's `schema_records_newskills_gpt55.jsonl`,
  `newskills_opus_csynth_table.csv`, and `newskills_gpt55_csynth_table.csv`.

[The retrieval CSV](../artifacts/sigmetrics2027/figure7_backfill_20261007/teacher_selected_slots.csv)
lists all 156 arm slots with exact model/kernel/setup, published latency,
schema file hash, record line numbers, record hash, and whether the slot
contributes to Table 7. Its
[JSON counterpart](../artifacts/sigmetrics2027/figure7_backfill_20261007/teacher_selected_slots.json)
adds the input/report requirements. The full selected schema records are
preserved in `selected_schema_evidence.json` for source matching.

The corresponding source origin identifiers are
`enh_newskills__{sonnet,opus,gpt55}__{flash,multistep}__{no_skills,curated,all_positive}`.
They must be matched before staging implementations; a matching benchmark
name or cycle count is insufficient. The sole numerical lead in the existing
July Sonnet inventory is FDTD-2D at 263,393 synthesis cycles. It has no proof of
the same source and inputs and therefore supplies no reusable validation.

## Source locations checked and remaining retrieval lead

The search covered the HPCA, SIGMETRICS, enhanced-fc27133, original C2HLS, and
underscore-named local C2HLS checkouts; ignored result files and the locally
available git history; the July-30 import bundle; `/mnt/data2/luo00466` C2HLS
archive manifests and scratch-directory names; and SFT/RL source/corpus JSON
records. No attributable `enh_newskills` teacher source package was found.
This is a statement about the searched local material, not proof that the
original artifacts have been lost.

Historical launchers in commit `da5c145` refer to the external WSL project
`/mnt/e/courses/UMN/c2hls/code_translation_c2hls`. That path is not mounted on
this host. The launcher uses, for example,
`results_matrix_u280_ENH_curated_multistep_skills_SONNET`,
`results_matrix_u280_ENH_allpositive_multistep_skills_SONNET`, and analogous
`_GPT55` directories. **These are retrieval leads from earlier ENH tooling,
not verified paths to the exact July `enh_newskills` candidates.** The raw
schemas themselves do not record original result paths. Find the original
exporter/run directory using the origin identifiers above before choosing a
source file.

For each of the 144 distinct selections, restore:

1. The exact saved generated/selected C++ implementation and the result/history
   that connects it to the recorded synthesis report and selected step.
2. Original `metadata.json`, `plain.cpp`, testbench, all headers/support files,
   and any separate co-simulation harness or compile-time size settings.
3. HLS Tcl/settings, synthesis report, CSim log, and any independent output
   comparison or existing RTL records. Preserve hashes for every file.

Once imported, audit source/report association and original input dimensions,
run independent native-output preflight, then deduplicate against existing
measurements and July reservations using source, all input hashes, toolchain,
and output-check contract. Only eligible new identities may enter the prepared
single-worker queue. Do not select a different implementation after seeing its
validation result. Restoring even one fully attributable selected/off pair is
enough to start useful backfill; the entire grid need not arrive first.

## DeepSeek and Grok rows

These rows are already **reported RTL**, not CSim-only. Their July-30 companion
synthesis CSVs include exact relative paths to `*_selected_report.json`
under `batch_parallel_hlsfactory_*` campaigns. All 112 arm paths are exported in
[reported_rtl_source_paths.json](../artifacts/sigmetrics2027/figure7_backfill_20261007/reported_rtl_source_paths.json),
with reported pass flags/cycles and compact-file hashes. The corresponding raw
sources/reports were not recovered locally. Preserve the paper's qualification
that these are supplied RTL summaries rather than independently replayed
results; other DeepSeek/Grok campaigns are not replacements for these records.

## Prepared status and reproduction

[summary.json](../artifacts/sigmetrics2027/figure7_backfill_20261007/summary.json)
reports **0 eligible validations and 156 blocked teacher slots**.
`eligible_rows.json` is intentionally empty. No identities can be deduplicated
as new or already validated without saved source and original-input hashes.
No existing queue, study, API budget, or source selection was changed.

Regenerate the audited inventory from the current paper selections with:

```bash
python3 scripts/audit_figure7_sources.py
```

The audit verifies all three raw schema checksums, each selected CSim/CSynth
status, exact published cycles, absence of teacher RTL records, and the join
to the aligned figure. It does not launch Vitis, infer correctness, or generate
replacement implementations.

# Skill catalog taxonomy and evidence requirements

This audit classifies the **content of frozen recipes**, not their measured causal contributions. A selected recipe, injected prompt, accepted source edit, achieved design goal, and end-to-end improvement are different observations. The historical result join is maintained separately in `artifacts/sigmetrics2027/skill_audit_20260929/`.

## Version boundaries

| Catalog | SHA-256 | Entries | Source non-avoid / avoid | Reviewed content roles |
|---|---|---:|---:|---|
| `skill_v3_no_rmw_fc27133/skills.json` | `d37094bf5548739debd8bdaae98fc9bc13db959cd33e17ddf9edd7885a1f358c` | 123 | 84 / 39 | 80 transformations, 39 avoidance guards, 3 diagnostic controls, 1 reporting annotation |
| `skill_v2/skills.json` | `9f46a3de6073b1dd9fe657423b43b41051a5fb89604f437fddb8d94c4234af5e` | 56 | 42 / 14 | 40 transformations, 14 avoidance guards, 1 diagnostic control, 1 reporting annotation |

The paper's historical **42 positive entries** refers to v2, not the 123-entry v3 catalog. These are counts of entries, not distinct optimization techniques: compound and supporting recipes overlap. “Positive” here only means non-avoid in the catalog, not demonstrated benefit. Both frozen catalogs reset `occurrences`, `sec_pass`, `mean_advantage`, and `last_used_at`; those fields cannot support success-rate or effect claims.

The September 28 recipe pilot copies the exact v3 bytes, but its manifest selects only `hls-pipeline-hot-loop-achieve-ii`, `prompt-pipeline`, and `hls-pipeline-stage-global-memory`. Thus the catalog size is not the pilot's injected recipe count. The pilot remains an engineering/accounting exercise, not the 216-cell efficacy study. The mutable `skills/skills.json` and base package have different hashes; they are inventoried but not silently treated as these frozen catalogs.

The v3 manifest records an ID-keyed base/overlay merge at source ref `fc27133f434451f273e84c97de700f101a03eb1f`; 33 existing IDs were overlaid and statistics reset. This is useful byte provenance. Strings such as `origin: manual`, `curated`, `prompt`, or `flash_no_RMW_m_axi` are self-reported metadata, not independent evidence of who authored or validated a recipe.

## Architecture families

Each entry has one primary family to make counts additive, plus secondary families for compound mechanisms. Assignment follows manual reading of `pattern`, `strategy`, `required_steps`, and `guards`; it is not inferred from the ID or keyword occurrence. Exact entry hashes pin each assignment. Generated display names simply expand IDs because there is no separate catalog name field.

| Primary family | v3 | v2 | Intended mechanism | Evidence needed to establish that mechanism |
|---|---:|---:|---|---|
| External-memory access | 14 | 10 | Improve burst formation, interface width, outstanding requests, or independent memory streams | Transfer width/length, bytes moved, arbitration and stalls; platform mapping for physical banks |
| Locality and working-set reuse | 28 | 7 | Retain reused operands and intermediate results in bounded on-chip storage | Global accesses/bytes, buffer capacity, reuse and load/store cycle contribution |
| Local memory ports | 12 | 6 | Match banking, partitioning, or replication to simultaneous accesses | Per-bank demand, compatible ports, reported conflicts, achieved II and storage cost |
| Loop scheduling | 12 | 5 | Expose legal pipeline boundaries and predictable iteration structure | Stable loop identity, trip count, achieved II, pipeline depth and total cycles |
| Dependence and reduction | 17 | 7 | Preserve or legally restructure recurrences, reductions and shared state | Dependence distance/type, recurrence latency, achieved II, numerical and boundary checks |
| Stage overlap | 23 | 8 | Enable legal producer/consumer concurrency through ownership and buffering | Concurrent task schedule, FIFO stalls/token counts, startup/drain and deadlock checks |
| Compute parallelism | 11 | 10 | Increase arithmetic issue capacity or improve operator structure | Active lanes/operators, supply bandwidth, DSP/LUT/FF cost and estimated period |
| Controller and measurement rules | 6 | 3 | Guide applicability checks, prompt behavior and report interpretation | Policy evidence and state transitions; distinguish annotations from actual hardware change |
| **Total** | **123** | **56** | | |

Role is a separate axis. Avoidance guards often concern the same architecture families as transformations. The five `prompt-*` entries request compound hardware transformations; `origin: prompt` does not make them mere formatting prompts. Conversely, `hls-loop-structure-audit-flash` mainly requests an audit table and rejection policy, although its source `kind` calls it a compound transformation. Both the source kind and reviewed role remain in the derived data.

## What can and cannot be attributed

For each candidate, preserve the exact catalog hash and injected recipe IDs, the accepted parent source hash, resulting source hash, targeted loop/array, metric evidence, correctness/feasibility decisions, and the controller's acceptance state. Record goal status as achieved, unmet, or unknown. A marker such as `ARRAY_PARTITION` or `DATAFLOW` establishes requested structure, not banking benefit or actual overlap. Absence of reports means unknown, not failure or success.

Use three distinct levels of evidence:

1. **Exposure:** a recipe was selected/injected. This establishes usage only.
2. **Observed transition:** a validated candidate changed a relevant intermediate metric and endpoint performance relative to its accepted parent. This supports a mechanism-consistent association, while co-edits and search history remain confounders.
3. **Controlled contribution:** matched-budget family or module ablation, with the same parent, model, prompts outside the treatment, validation rules and seeds. Report all attempts/failures and the fraction of improved valid outputs; report paired speedups only with their eligible denominator.

Do not average skill occurrence counters or multiply stage gains to assign credit to individual recipes. The current taxonomy is an analysis overlay; it changes no frozen library, prompts, policy or running experiment.

For the historical TRMM case, map visible source transformations to architecture families and explain the measured cycle/resource/timing transition. Do **not** attach current v3 IDs merely because step names resemble a recipe. An exact catalog or preserved injected content is required for a historical recipe-level join. The final banking/unroll change combines mechanisms, and a static fallback is not evidence that the LLM executed that recipe. Only the available endpoint RTL result should be called RTL-validated.

## Priority controls before family ablation

Several recipes contain overbroad prescriptions or conflicting controls. These are recorded per entry in `classification_note`; flags identify content/attribution review needs, not all classification uncertainty.

- `avoid-dataflow-without-stream-or-overlap-proof` instructs the model to retain DATAFLOW, while `hls-dataflow-ping-pong-gate-flash` permits removing it when overlap cannot be established. Freeze a resolved future catalog before comparing enforcement policies.
- `loop-tripcount-when-bound-runtime` is **reporting-only**. A changed latency estimate after annotation is not a hardware speedup.
- Distinct `gmem` bundle names alone do not establish independent physical-memory bandwidth; require host/linker/platform evidence.
- Fixed bounds or a PIPELINE pragma do not by themselves establish II=1. Store requested and achieved II separately.
- `hls-fp-reduction-partial-accumulators` lacks an explicit reassociation-tolerance guard in its own entry. Preserve numerical-contract checks outside the recipe; lane privatization can change floating-point results.
- Old/new state separation and tile-overlap double buffering belong to different mechanisms. The former must preserve iteration semantics and must not silently change an in-place update to Jacobi semantics.

A family ablation should remove **all relevant content modules**, not just recipes whose primary label matches: many compound recipes repeat banking, unrolling, staging and pipeline guidance. Otherwise a supposed “no banking” arm can still receive banking instructions through tiling or coalescing. Retain common correctness guards across arms; evaluate performance guidance and enforcement policies as separate factors. Version the new factored catalog rather than rewriting historical bytes.

Start with targeted experiments: (a) TRMM local-storage supply versus compute replication, (b) a reduction kernel's dependence transformation, and (c) a legal stage-overlap case with a schedule report. Include a case where additional storage or parallelism does not improve feasible performance. Use these to select an adequately powered family study; do not launch an eight-family Cartesian sweep before confirming instrumented goals and content isolation.

## Reproduction and outputs

From the SIGMETRICS worktree:

```bash
python artifacts/sigmetrics2027/skill_audit_20260929/taxonomy_build.py
```

The standard-library script validates both entire-catalog hashes and every canonical entry hash. It requires full one-to-one coverage and fails on unseen or changed entries. It writes only derived `taxonomy_*` artifacts:

- `taxonomy_annotations.json`: explicit reviewed assignments, definitions and version-specific notes.
- `taxonomy_entries.csv` / `.json`: v3 entry-level primary/secondary families, roles, source fields and evidence requirements.
- `taxonomy_v2_entries.csv` / `.json`: independently pinned v2 assignments; no cross-version pooling.
- `taxonomy_summary.json`, `taxonomy_v2_summary.json`: counts by family, role, source kind and origin.
- `taxonomy_provenance.json`: source hashes, frozen merge manifest, pilot selection, and interpretation limits.

The v2 catalog contains 52 entries identical to v3, three entries with differing templates reviewed separately, and one v2-only in-place-stencil recipe. Same IDs alone never establish that historical prompts were identical.

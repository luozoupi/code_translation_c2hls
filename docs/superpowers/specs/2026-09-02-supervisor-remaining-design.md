# Design: close remaining Aug 25 supervisor asks

**Date:** 2026-09-02  
**Status:** Approved (user: do all unaddressed items; plan and use subagents)  
**Do not commit** unless the user asks.

## Goal

Turn the Aug 25 meeting gaps into measured tables and slide copy. Reuse completed campaigns. Only launch jobs whose numbers do not already exist.

## Locked

- Metric: **latency min, latency max, DSP**. Do not compare interval on slides.
- Hardware: DeepSeek-v4-flash, U280 `xcu280-fsvh2892-2L-e`, 3.33 ns, cosim off.
- Do not overwrite: locked `autosa_mm` / `autosa_mm_32x8` recipes, `20260830_mmflow`, family A/B/C, Aug 18 slide file (4285 vs 4228). Add a **new** 2026-09-02 brief.
- Do not relaunch with-skills mm-flow. Do not relaunch enforcement (numbers exist). Do not relaunch 32×8 (numbers exist).
- Slides: compute rewrite / hide load-store. Never “DSE”.

## What already exists (do not rerun)

| Ask | Artifact | Latency min–max | DSP |
|-----|----------|-----------------|-----|
| Zero-shot | `20260902_mmzs` | 40454–42758 | 320 |
| Generic HLS skills (flash) | frozen `20260830_mmflow` flash | 139484–139484 | 10 |
| Skills + enforcement | `batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043` | 12893–12893 | 320 |
| With-skills systolic | frozen `20260830_mmflow` stream | 4292–4292 | 320 |
| No-skills systolic | `20260902_mmns` stream | 4216–4216 | 320 |
| Ping-pong on multi-PE (Aug 23 seed) | overlap `20260826_102425` | 8678 (gate fail) | 320 |
| 32×8 overlay | `20260830_mm32x8` stream | 4583–4583 | 1280 |
| Wave1 within 2% | `20260825_wave1_aav_n_gf` | 4/6 pass (see below) | recipe DSP |

Enforcement reason: `already ping-pong DATAFLOW`. Interval 4553 vs latency 12893 — overlap gate fail. That **is** column 3.

Aug 23 overlap 8678/320 is ping-pong vs stream, but the seed is not frozen mmflow. Re-run overlap **only** on `20260830_mmflow` `*_dse.cpp` (13160 / 352) so the A/B is the same kernel as stream 4292.

## Four design-point columns (mm)

1. **No expertise** — zero-shot. 40454–42758 / 320.
2. **Generic HLS skills** — flash 90-skill pack, no enforcement, no PE-recipe systolic rewrite. 139484 / 10. This step **slows down** vs zero-shot.
3. **Skills + enforcement agent** — flash skills + ping-pong/DATAFLOW judge (II, partition, resources, double-buffer). 12893 / 320. Structure passes; one-run finish time is still load+compute+store class.
4. **Application-specific systolic** — compute rewrite then hide load/store. 4292 with-skills / 4216 no-skills JSON / AutoSA 4228 / 320.

## Pack-level skill effectiveness (not per-skill)

Do **not** ablate skill 3 of 5. Report major packs:

| Pack | Seed | Result | Takeaway |
|------|------|--------|----------|
| Flash 90-skill | plain.cpp | 139484 / 10 | Legal kernel; no PE array. Worse than zero-shot. |
| Multi-PE (5 skills) | flash | 13160 / 352 | Workers exist; sequential I/O. |
| Overlap ping-pong (3 skills) | multi-PE | 8678 / 320 (Aug 23); new mmflow run TBD | Coarse tile overlap ≠ systolic I/O. |
| Stream hide-load-store (6 skills) | multi-PE | 4292 / 320 | Closes rank-1. |
| Same flow, skills JSON empty | PE-recipe kept | 4216 / 320 | JSON templates not required once recipe + architecture text exist. |

## Ping-pong vs stream (new job)

- Seed: frozen `artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow` `autosa_mm_dse.cpp`.
- Tool: existing `post_flash_overlap.py` / `start_autosa_pe_overlap.sh`.
- **mm only.** Do not wait on wave1 overlap (already 0/6).
- Must not emit `mm_pe` FIFO pack. Must not overwrite `*_selected.cpp`.
- Compare latency min/max + DSP to stream 4292 / 320 and AutoSA 4228 / 320.
- Prior Aug 23 8678 stays in the table as the first measurement.

## Rest of benches (2% of AutoSA)

Wave1 targets (`autosa_rank1_u280_targets.json`), agent selected:

| Bench | Rank-1 | Agent selected | vs 2% gate (×1.02) |
|-------|--------|----------------|--------------------|
| autosa_mm (locked) | 4228 / 320 | 4292 / 320 (with-skills); 4216 noskills | pass |
| autosa_mm_hcl | 4230 / 320 | 4294 / 320 | pass (1.51%) |
| autosa_mm_hcl_intel | 4226 / 320 | 4292 / 320 | pass (1.56%) |
| autosa_mm_int16 | 4219 / 64 | 4280 / 64 | pass (1.45%) |
| autosa_mm_catapult | 8286 / 96 | 8351 / 96 | pass (0.78%) |
| autosa_mm_intel | 4178 / 640 | 4527 / 640 | **fail 8.35%** |
| autosa_mm_getting_started | 2194 / 640 | selected 4307 / 640; stream architecture miss on 4384 / 320 | **fail ~2×** (tile wrap) |

**This round:** retry `autosa_mm_intel` and `autosa_mm_getting_started` via `start_autosa_wave1_dse_stream_retry.sh --benches ...` on the existing `20260825_wave1_aav_n_gf` matrix (`C2HLS_DSE_FORCE=1`, `C2HLS_STREAM_FORCE=1`). Do not start a new wave1 flash campaign.

**Not this round:** cnn / lu / large_* / HBM. No PE recipes (or no rank-1 package). Document as wave 2/3. P&R and extra HBM channels stay later work.

## Why 320 DSP (no new 16×4 job)

U280 has thousands of DSPs. Rank-1 320 is **iso-compute**: 16 PE × SIMD 4 × ~5 DSP/float MAC. AutoSA chose that array for 64³, not device fill.

Evidence already in-tree:

- 32×8 overlay (`C2HLS_PE_RECIPE=autosa_mm_32x8`): compute 9782 / 1344, stream **4583 / 1280**. More DSP, **worse** latency than 16×4 stream 4292 because I/PE=2 (`pe_kj=512` × two tiles).
- Family C IO-mesh vs AutoSA cand 9 **1846–2161 / ~1296 DSP** (different search, not the locked slide): iso-DSP 32PE point is **2245 / 1280** (misses ≤1846). Selected `io4_16x8` is **1571 / 5120** — faster only by oversubscribing DSP, not a 16×4 slide replacement.
- Paper device was often U250; this lock is U280 3.33 ns csynth.

Write a one-page memo. Do not relaunch 32×8.

## Docs / slides

Create, do not overwrite Aug 18:

- `docs/pc2/2026-09-02-supervisor-remaining-slide-brief.md` — four columns, pack waterfall, ping-pong vs hide-load-store, 320 DSP, wave1 2% table.
- Update `docs/pc2/2026-08-30-autosa-agentic-flow-one-pager.md` with 4216 / 4292 / 40454 / 12893 / 8678.
- Extend `artifacts/pc2/autosa_mm_ablation_20260901/comparison.md` with enforcement + overlap + 32×8 + wave1.
- `docs/pc2/2026-09-02-why-320-dsp.md`
- Canvas: four-column + packs + wave1 (durable figure for the meeting).

## Jobs to submit

1. Overlap mm-only on `20260830_mmflow` → login5:18092.
2. Wave1 retry intel + getting_started → same endpoint.

Flash token floor 65536 / continuations 8 on any new LLM job.

## Out of scope

- Rerun enforcement or with-skills mm-flow or 32×8.
- Per-individual-skill ablation.
- Cosim, bitstream, extra HBM, P&R.
- Git commit.

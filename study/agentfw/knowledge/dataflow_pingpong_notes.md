# What ping-pong DATAFLOW is (and how the judge checks it)

**Date:** 2026-09-09  
**Applies to:** new `C2HLS_ENFORCEMENT` runs. Does **not** rewrite frozen `20260830_mmflow`, the 12893 artifact, family C, or Aug 18–Sep 2 briefs.

Ping-pong DATAFLOW is **explicit double-buffering of tiles**, not
`#pragma HLS DATAFLOW` plus arrays declared inside the loop.

It is **double-buffering of tiles** so that, inside one GEMM call:

- `load(t+1)` fills the next tile,
- `compute(t)` uses the current tile,
- `store(t−1)` drains the previous tile,

and those three rooms run together. The C must have a **tile loop around
DATAFLOW**, explicit `buf[2]` / `[2][...]` (or a ping/pong pair) **and**
index `t & 1` / `t%2`, INLINE-off load / compute / store, and full **B
loaded once outside** the tile loop. **Arrays inside the DATAFLOW loop
without `buf[2]` are not ping-pong** (101836 wrap: single `A_loc`/`B_loc`/`C_loc`,
`load_B` every tile).

Csynth evidence is the **one-call latency**, not the initiation interval:

- **Overlap:** latency min/max ≈ `max(load, compute, store)` plus a small fill/depth. For the 16×4 nest that is compute-bound near **~4816**, not **~12893**.
- **Serial LCST:** kernel latency within ~10% of the sum of the rooms. Enforcement `20260829_123043` is **12893 ≈ 4170 + 4552 + 4169**. HLS may still print pipeline type `dataflow` and interval 4553 (the *next kernel launch*). That interval is not in-GEMM overlap.

The judge therefore requires all of:

1. Tile/iteration loop around DATAFLOW (reject one-shot full-matrix `load(); compute(); store();`).
2. **Explicit** `buf[2]` + `t & 1` (reject DATAFLOW + arrays inside with no double buffer; reject `load_B` inside the tile loop unless `C2HLS_PP_LOAD_B_IN_DF=1`).
3. Csynth process latencies: parent/kernel tracks `N * max(rooms)`, not `N * sum`, and is **not** within 10% of the serial LCST sum.
4. **Keep the flash kernel.** If flash already has LANES=16 (512-bit) loads ~256 and iso-compute ~320 DSP, the candidate must keep those bodies. Dropping `k0 += LANES` so `load_B` becomes ~4096 is a fail. Kernel latency **> flash × 1.10** is a fail even if rooms overlap (canonical: flash **4808** → enforcement **12642** on `20260909_085740`). Wrap flash load/compute with a tile DATAFLOW; do not paste a scalar II=1 skeleton.

`interval < latency` alone is a fail. Syntax-only DATAFLOW is a fail. Frozen campaigns are not re-scored.

## Opt-in: `load_B` inside the tile DATAFLOW (`C2HLS_PP_LOAD_B_IN_DF`)

Default keep-flash still prefixes `load_B` (~331) **before** the tile loop. That
prefix cannot overlap compute. A second recipe puts LANES=16 `load_B` **inside**
the tile DATAFLOW as its own INLINE-off task, with `B_local` declared in the
region (HLS channel; avoids 200-976). A/C still need `buf[2]` + `t & 1`.
101836 (no `buf[2]`) still fails.

```bash
unset C2HLS_AUTOSA_FLOW
C2HLS_PP_LOAD_B_IN_DF=1 C2HLS_SYNTH_TIMEOUT=14400 \
  ./scripts/pc2/start_autosa_mm_enforcement.sh \
  --load-b-in-df \
  --seed-flash artifacts/pc2/seeds/mmflow_flash_repro2 \
  --stamp repro2_bdf \
  --endpoint-url http://login5:18092/v1
```

Handwritten iso of repro2_enf selected.cpp (5329/354) with `load_B` in DATAFLOW:

`artifacts/pc2/manual_mm_loadb_in_df/` — `sbatch artifacts/pc2/manual_mm_loadb_in_df/job.sbatch.sh`

Keep-flash cap is still flash × 1.10. This overlay is **not** mixed into the
default keep-flash JSON.

Working bulk ping-pong of this nest still sits **above AutoSA 4228**, because bulk LCST rooms are not systolic stream I/O (4216–4292). Stream is still required to reach rank-1.

## How to synth the 2-tile kernel (no flash endpoint)

New artifact (does not overwrite frozen mmflow or 12893):

`artifacts/pc2/manual_mm_lcst_tile_pp2/`

From the repo root, on a **compute** node (Vitis 2023.2, U280, 3.33 ns):

```bash
sbatch artifacts/pc2/manual_mm_lcst_tile_pp2/job.sbatch.sh
```

or `bash artifacts/pc2/manual_mm_lcst_tile_pp2/run_csim_csynth.sh`. This does **not** call `login5:18092`. Summary: `artifacts/pc2/manual_mm_lcst_tile_pp2/csim_csynth_summary.json`.

**Measured (Slurm 2911688, 2026-09-09):** csim pass. Csynth **4662 / 352 DSP**. `load_B` 299 + parent 4359. Inner DATAFLOW `dataflow_in_loop_tile_pp_1`: latency 2263, II 2093, rooms load_A 170 / compute 2092 / store 168. Parent **4359 ≈ 2×max(2092)** (not 2×sum). Judge `pingpong_dataflow_ok`: **true**. Still **above AutoSA 4228**. Same class as handwritten `pe_pp_plus` **4688 / 352** (4 i-tiles). Neither is the old LLM enforcement **12893**. Stream I/O (4216–4292) is still what reaches rank-1.


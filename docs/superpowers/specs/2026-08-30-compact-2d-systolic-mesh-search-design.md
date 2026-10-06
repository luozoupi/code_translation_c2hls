# Design: Compact 2-D systolic mesh HLS search

**Date:** 2026-08-30  
**Status:** Approved (2026-08-30)  
**Scope:** c2hls compact PE search only. Does not call AutoSA. Does not emit AutoSA `kernel0`. Does not overwrite locked `autosa_mm` 16×4, `20260830_mmflow`, or the Aug 18 slide (4285 vs 4228).

## Goal

Clone AutoSA’s **plan** (enumerate → codegen → csim+csynth → rank) for a **2-D systolic PE mesh**, without calling AutoSA or copying `kernel0`.

The 12-point 1-D chain search ranked `pe16_simd4` at **4292**. AutoSA candidate_9 (8×4×SIMD8) and candidate_5 (16×8×SIMD8) are 2-D meshes; they were not in that grid. Add a compact 2-D generator so those geometries (and a larger legal mesh grid) get the same HLS score as the 1-D points, in one ranking.

## Decisions (locked)

| Decision | Choice |
|----------|--------|
| AutoSA binary / `kernel0` | **Do not call, do not copy** |
| ABI | Keep `autosa_mm(A[I][K], B[J][K], C[I][J])` and the existing testbench |
| 1-D grid | Keep the existing 12 chain points unchanged |
| 2-D grid | `PE_I, PE_J ∈ {4,8,16,32}`, `SIMD ∈ {2,4,8}` |
| DSP cap | Drop if `PE_I×PE_J×SIMD×5 > 0.85×9024` |
| PE-count cap | Drop if `PE_I×PE_J > 128` (keeps candidate_5; drops 512-PE DATAFLOW) |
| Ranking | One `ranking.jsonl`: 1-D ∪ 2-D |
| Cosim | Off |
| LLM | Not in the search loop |
| Job prefix | `mmsearch`; new stamp; not `mmflow` |

## Non-goals

- Invoking AutoSA or checking in `kernel_kernel.cpp`
- Host-serialize / `A_t16*` ABI
- Vivado impl / xclbin
- Other six GEMMs
- Updating the Aug 18 slide unless rank-1 beats 4285 **and** the user says so
- Changing `architecture_ok(bench=...)` agent-flow behavior

## Problem this solves

AutoSA’s advantage on mm is a **2-D PE mesh** (candidate_9: 32 PEs, 1846 cycles; candidate_5: 128 PEs, 2033 cycles), not a longer 1-D chain. Flattening mesh size into `PE_NUM = PE_I×PE_J` is still a 1-D chain (`pe64` already lost to `pe16_simd4`). The search must emit a real 2-D interconnect.

## Architecture

```
enumerate_mm_recipes()        → 12 chain PeRecipe
enumerate_mm_mesh_recipes()   → mesh PeRecipe (ids mesh{I}x{J}_simd{S})
        ↓
instantiate_mm(rec)           → chain or mesh C++ (dispatch on layout)
        ↓
validate_candidate             → compile + csim + csynth → result.json
        ↓
rank_candidates               → csim ∧ csynth ∧ architecture_ok, sort latency
```

No AutoSA. Same `_run_synth_csim_cosim` as today. Failures kept.

## Components

### `PeRecipe` (`post_flash_pe_recipe.py`)

Add fields with defaults so existing frozen recipes stay valid:

- `layout: str = "chain"` — `"chain"` or `"mesh"`
- `pe_i: int = 0` — I-side PEs; `0` means “use `pe`” (chain)
- `pe_j: int = 1` — J-side PEs; chain is `1`

Invariants:

- Chain: `pe_i` defaults to `pe`, `pe_j == 1`, `pe` is the 1-D `PE_NUM` (unchanged).
- Mesh: `pe_i` and `pe_j` set, `pe == pe_i * pe_j` (total PEs, used for DSP).
- `pack_bits = simd * 32`
- `expected_dsp = pe_i * pe_j * simd * 5` (chain: `pe * simd * 5`)
- `min_dsp = max(1, int(expected_dsp * 0.6))`
- Mesh `pe_kj = (K // simd) * (J // pe_j)`
- Mesh `i_tiles = I // pe_i`, `tile_loop = "inside_tasks"` (never wrap DATAFLOW around the array)

Do not change locked `_RECIPES["autosa_mm"]` or `autosa_mm_32x8`.

### Search (`compact_pe_search.py`)

Keep `enumerate_mm_recipes()` and `candidate_id()` for chain (`pe{pe}_simd{simd}`).

Add:

```text
candidate_id_mesh(rec) -> mesh{pe_i}x{pe_j}_simd{simd}
enumerate_mm_mesh_recipes(*, dsp_cap=0.85, max_pe=128) -> list[PeRecipe]
```

Legal mesh point:

- `I % pe_i == 0` and `J % pe_j == 0` and `K % simd == 0`
- `pe_i * pe_j <= 128`
- `expected_dsp <= dsp_cap * 9024`

Must include `mesh8x4_simd8` and `mesh16x8_simd8`.

Driver enumerates `enumerate_mm_recipes() + enumerate_mm_mesh_recipes()`.

### Instantiate (`compact_pe_instantiate.py`)

`instantiate_mm(rec)`:

- `layout == "chain"`: existing 1-D emitter (no behavior change)
- `layout == "mesh"`: 2-D emitter below

Mesh C++:

1. `#define PE_I {pe_i}` `#define PE_J {pe_j}` `#define SIMD {simd}`
2. `typedef ap_uint<{pack_bits}> vecN_bits;` plus `packN` / `unpackN`
3. `extern "C" void autosa_mm(...)` with the same INTERFACE lines as the 1-D 32×8 kernel
4. Exactly `pe_i * pe_j` explicit PE calls (plus one definition). Suggested name: `mesh_pe` so 1-D `mm_pe(` counts stay valid in 1-D tests. `architecture_ok` for mesh counts `mesh_pe(`
5. FIFOs: `fifo_A[PE_I][PE_J]`, `fifo_B[PE_I+1][PE_J]`, `fifo_C[PE_I][PE_J+1]`
6. Wiring (fixed; AutoSA-shaped, compact):
   - **A is per-PE, not a shift:** `load_A` writes `fifo_A[i][j]` for every `(i,j)` (row `i0+i`, K packed)
   - **B shifts along I:** `load_B` writes `fifo_B[0][j]`; PE `(i,j)` reads `fifo_B[i][j]`, writes `fifo_B[i+1][j]`
   - **C drains along J:** PE `(i,j)` reads `fifo_C[i][j]`, writes `fifo_C[i][j+1]`; `store_C` reads `fifo_C[i][PE_J]`
   - `drain_B` consumes `fifo_B[PE_I][j]`
7. **One `#pragma HLS DATAFLOW` at function scope.** No `for (i0 … i0 += PE_I)` around DATAFLOW.
8. I-tiles (`I/PE_I`) and J coverage live **inside** `load_A`, `load_B`, `mesh_pe`, `drain_B`, `store_C` using `for (int tile = 0; tile < I / PE_I; ++tile)` (not the 1-D forbidden string `for (int i0 = 0; i0 < I; i0 += PE_NUM)`).
9. Crow: `data_t Crow[J / PE_J]`. `BIND_STORAGE ram_2p bram`. No `ARRAY_PARTITION complete` on Crow. No `LOOP_FLATTEN off`.
10. `mesh_pe` inner loop: `#pragma HLS PIPELINE II=1` on flattened `t < (K/SIMD)*(J/PE_J)` per I-tile.

1-D tests must still pass without modification of their assertions.

### Validate, rank, launcher

- `validate_candidate` already calls `instantiate_mm` + `architecture_ok_for_recipe`. Pass the mesh recipe through; do not use `C2HLS_PE_RECIPE`.
- `result.json` gains `"layout"` and `"pe_i"` / `"pe_j"` (chain may omit or write `pe_i=pe`, `pe_j=1`). Rank eligibility unchanged.
- `compact_pe_search_main.py` concatenates both enumerations. Rank-1 `selected.cpp` may be chain or mesh.
- `start_autosa_mm_pe_search.sh --dry-run` prints **both** chain and mesh ids. Same `mmsearch` prefix. New stamp only. Walltime: 4h is tight for ~42 points; set **12:00:00** (or keep 4h only if the operator passes a smaller subset later). Default walltime **12:00:00**.

### `architecture_ok_for_recipe`

If `rec.layout == "mesh"`:

- streams + DATAFLOW
- packed `ap_uint<{pack_bits}>`
- `code.count("mesh_pe(") == 1 + pe_i * pe_j`
- function-scope DATAFLOW (no i0 wrap around DATAFLOW)
- Crow not complete-partitioned; no `LOOP_FLATTEN off`
- `dsp >= rec.min_dsp`
- compute pipeline II=1 (`pe_kj` or `mesh_kj`)
- overlap ≤ 1.15

Chain checks stay as they are today. `architecture_ok(bench=...)` is **not** rewritten.

## Testing (no Vitis)

| Test | Assert |
|------|--------|
| Mesh grid includes analog ids | `mesh8x4_simd8` in grid, `pe_i=8`, `pe_j=4`, `simd=8`, `pack_bits=256`, `expected_dsp=1280` |
| Candidate_5 analog | `mesh16x8_simd8`: `pe=128`, `expected_dsp=5120` |
| Caps | no point with `pe_i*pe_j > 128`; no `32x16`; all `expected_dsp <= 0.85*9024` |
| Instantiate 8×4 | `PE_I 8`, `PE_J 4`, `ap_uint<256>`, `mesh_pe(` count 33, `#pragma HLS DATAFLOW`, no `i0 += PE_NUM` wrap around DATAFLOW |
| `architecture_ok_for_recipe` | fake report dsp=1280, latency 1200, interval 1100, pe_kj II=1 → True for `mesh8x4_simd8` |
| Chain regression | existing `test_compact_pe_search.py` and `test_compact_pe_instantiate.py` still 6/6 |
| Driver | mocked validate: ranking first line is lowest latency among chain+mesh fakes |

## Operator run (after tests green)

```bash
./scripts/pc2/start_autosa_mm_pe_search.sh --stamp 20260831_mesh
```

Quote `ranking.jsonl`. Do not update the Aug 18 slide unless rank-1 beats 4285 and the user says so. Compact mesh latency will not be AutoSA’s 1846/2033 (`kernel0` + host-serialize).

## Out of this spec

Odyssey, polyhedral, `kernel0`, impl/xclbin, LLM-per-point, other GEMMs.

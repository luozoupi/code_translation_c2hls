# Compact IO-mesh HLS search (family C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit tiled, double-buffered Design-4 (`io4`) and Design-5 (`io5`) compact GEMM kernels, HLS-search them in one campaign, and beat AutoSA cand 9 on latency min/max at ~32 PE × SIMD 8 (~1280 DSP) without 128-PE oversubscription.

**Architecture:** Reuse packed AXI ABI (`autosa_mm_pack` / `A_t16`). One emitter (`compact_pe_io_instantiate.py`) with `io_pe(`. Tile loops (`t_j`, `t_k`) live inside every DATAFLOW task. `load_B` uses `B_ping`/`B_pong` of size `J_PART*K_PART/16` and on-chip replay of **one tile**, with `PE_J`/`PE_K` scatter unrolled so the stream trip matches PE compute (~1024), not `LAT_I` × full B. io4: C local `Crow`. io5: C systolic along K (`fifo_C_in`/`fifo_C_out`), `pe_j` = PEs along K.

**Tech Stack:** Python 3, existing `PeRecipe` / `architecture_ok_for_recipe` / `validate_candidate` / packed header+tb, pytest, Vitis HLS 2023.2, U280 `xcu280-fsvh2892-2L-e` 3.33 ns.

**Locked:** Aug 18 slide 4285 vs 4228. Do not overwrite family A/B artifacts, `autosa_mm` / `autosa_mm_32x8`, `20260830_mmflow`. Cosim off. No AutoSA binary. Do not emit `kernel0`. Do not commit unless the user asks. Rank with existing `rank_candidates` (latency min, then DSP). Comparison quote vs cand 9 is **1846–2161 / 1296 DSP** (min/max only).

**Spec:** `docs/superpowers/specs/2026-08-31-compact-io-mesh-search-design.md`

**Spec amendment (approved with this plan):** io5 identity `pe_j * simd == k_part` makes `io5_16x8_s8_k32_j64` illegal. The 128-PE Design 5 point is `io5_16x8_s8_k64_j64`.

---

## File map

| Path | Role |
|------|------|
| `post_flash_pe_recipe.py` | Add `k_part`, `j_part`, `lat_i`, `lat_j` (default 0); layout comment includes `io4`/`io5` |
| `compact_pe_search.py` | `candidate_id_io`, `search_candidate_id` dispatch, `enumerate_mm_io_recipes` |
| `compact_pe_io_instantiate.py` | `instantiate_io` — io4 and io5 C++ |
| `compact_pe_instantiate.py` | Dispatch `io4`/`io5` **before** the `pack_bits != simd*32` check |
| `post_flash_stream.py` | `architecture_ok_for_recipe` branch for io4/io5 only |
| `compact_pe_validate.py` | Packed header/tb + top `autosa_mm_pack` for io4/io5 |
| `compact_pe_io_search_main.py` | Enumerate io4 ∪ io5, rank, `selected.cpp` |
| `scripts/pc2/start_autosa_mm_io_search.sh` | Dry-run ids; job prefix `mmio` |
| `scripts/pc2/compact_pe_io_search.sbatch.sh` | 12:00:00, 16 CPU, 64G |
| `tests/test_compact_pe_io.py` | Grid, PE counts, no kernel0, architecture_ok, dry-run |

Reuse: `emit_pack_header` / `emit_pack_testbench`, `rank_candidates`, `_run_synth_csim_cosim`. Do not rewrite `architecture_ok(bench=...)`. Do not mix rows into pack/mesh `ranking.jsonl`.

Skip git commits unless the user asks.

---

### Task 1: Failing tests

**Files:**
- Create: `tests/test_compact_pe_io.py`

- [ ] **Step 1: Write the failing tests** (full file; see implementation below in this session)

Must cover:
- Grid length 6; must-include `io5_8x4_s8_k32_j32`, `io4_16x8_s8_k32_j64`, `io4_8x4_s8_k32_j32`, `io4_8x4_s8_k32_j64`, `io5_8x4_s8_k32_j64`, `io5_16x8_s8_k64_j64`
- Cand 9 analog: layout io5, pe_i=8, pe_j=4, simd=8, k_part=32, j_part=32, lat_i=8, lat_j=4, pe=32, expected_dsp=1280, pack_bits=512
- Cand 5 analog: io4 16×8, k32, j64, lat_i=4, lat_j=8, pe=128, DSP 5120
- io5 16×8 uses k_part=64 (identity), not 32
- `search_candidate_id` dispatch; chain/pack/mesh ids unchanged
- instantiate io4 8×4: `io_pe(` count 33, `PE_I 8`, `K_PART`, `B_ping`, `B_pong`, `autosa_mm_pack`, no `kernel0`, no `pack_pe`/`mesh_pe`/`mm_pe`/`PE_wrapper`, `Crow`, no `fifo_C_in`, tile loops `t_j`/`t_k`, no I-tile wrap around DATAFLOW
- instantiate io5 8×4: 33 `io_pe(`, `fifo_C_in` and `fifo_C_out`, no treating pe_j as J-columns (`LAT_J` is 4 not `J_PART/pe_j` wait: j_part=32, lat_j=4; `J_PART / LAT_J` scan exists)
- `architecture_ok_for_recipe` true with fake DSP/II report; false if `io_pe` renamed or `kernel0` injected
- validate uses packed top/header for io5
- search main ranks only io ids
- launcher `--dry-run` prints io ids, `mmio`, `compact_pe_io_search_`; does not print `pack8x4_simd8` / `mesh8x4_simd8` / `pe16_simd4`

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_compact_pe_io.py -q
```

Expected: FAIL (import / function missing)

---

### Task 2: PeRecipe fields + enumerate

**Files:**
- Modify: `post_flash_pe_recipe.py`
- Modify: `compact_pe_search.py`

- [ ] **Step 3: Add fields**

```python
layout: str = "chain"  # chain | mesh | pack | io4 | io5
pe_i: int = 0
pe_j: int = 1
k_part: int = 0
j_part: int = 0
lat_i: int = 0
lat_j: int = 0
```

- [ ] **Step 4: Enumerate + ids**

```python
def candidate_id_io(rec: PeRecipe) -> str:
    return f"{rec.layout}_{rec.pe_i}x{rec.pe_j}_s{rec.simd}_k{rec.k_part}_j{rec.j_part}"


def search_candidate_id(rec: PeRecipe) -> str:
    if rec.layout in ("io4", "io5"):
        return candidate_id_io(rec)
    if rec.layout == "pack":
        return candidate_id_pack(rec)
    if rec.layout == "mesh":
        return candidate_id_mesh(rec)
    return candidate_id(rec)
```

`enumerate_mm_io_recipes()` returns exactly the six points. Identities:

- io4: `lat_i = I // pe_i`, `lat_j = j_part // pe_j`, require `pe_j * lat_j == j_part`, `I % pe_i == 0`, `j_part % pe_j == 0`, `K % k_part == 0`, `J % j_part == 0`
- io5: `lat_i = I // pe_i`, `lat_j = 4` if `j_part % 4 == 0` else 8, require `pe_j * simd == k_part`
- `expected_dsp = pe_i * pe_j * simd * 5`, skip if `pe > 128` or DSP > `0.85 * 9024`
- `pack_bits = 512`, `tile_loop = "inside_tasks"`, `i_tiles = 1`
- `pe_kj = (k_part // simd) * lat_j` (io4 inner) or `lat_j` (io5 still names the pipeline `pe_kj`)

---

### Task 3: Emitter

**Files:**
- Create: `compact_pe_io_instantiate.py`
- Modify: `compact_pe_instantiate.py` (`instantiate_mm` dispatch io4/io5 before pack_bits check)

Shared:

- `#define K_PART`, `J_PART`, `LAT_I`, `LAT_J`, `N_J (J/J_PART)`, `N_K (K/K_PART)`
- Top: `autosa_mm_pack`, 512-bit AXI (copy pack INTERFACE)
- DATAFLOW, stream depth ≥ 256 (use 512)
- `io_pe(` count `pe_i * pe_j` calls + 1 definition
- Tile loops `t_j`, `t_k` in load_A, load_B, io_pe, drain_B, store_C
- `B_ping` / `B_pong` sized `J_PART * K_PART / 16`
- Scatter inner `pj`/`pk` **UNROLL** so load_B stream trip ≈ PE compute, not × PE_J
- No full-matrix `Bmem[J*K/16]` then `LAT_I` replay of all K,J
- `local_A` present (L1)
- No `void kernel0(`

**io4 PE:** `Crow[LAT_J]` ram_2p, not complete-partition. Flattened `pe_kj` over `(K_PART/SIMD)*LAT_J` per `ii`, accumulate over `t_k`, drain packed C after last `t_k` of each `t_j`. B shift along I. A broadcast (or L1) per row.

**io5 PE:** `pe_k` constant at each call site. `fifo_C_in` / `fifo_C_out`. First K PE: cin=0. Last K PE: accumulate `Cmem[LAT_I][J_PART]` over `t_k`, packed drain to store. Mid: `cout = cin + dot`. A reused across J (`j0==0 && jj==0`). Loop `t_j, t_k, j0 < J_PART/LAT_J, jj, ii` with `pe_kj` PIPELINE II=1 on the flattened inner (or label `pe_kj` on the II=1 loop).

- [ ] **Step 5: Dispatch**

```python
layout = getattr(rec, "layout", "chain")
if layout == "pack":
    ...
if layout in ("io4", "io5"):
    from compact_pe_io_instantiate import instantiate_io
    return instantiate_io(rec)
```

---

### Task 4: Gates, validate, driver, launcher

**Files:**
- Modify: `post_flash_stream.py` (`architecture_ok_for_recipe` only)
- Modify: `compact_pe_validate.py` (`if rec.layout in ("pack", "io4", "io5")`)
- Create: `compact_pe_io_search_main.py` (clone pack main; `enumerate_mm_io_recipes`)
- Create: `scripts/pc2/start_autosa_mm_io_search.sh` (prefix `mmio`, out `compact_pe_io_search_${STAMP}`)
- Create: `scripts/pc2/compact_pe_io_search.sbatch.sh`

io4/io5 architecture branch:

- `code.count("io_pe(") == 1 + pe_i * pe_j`
- `autosa_mm_pack(` present, `void kernel0(` absent
- `ap_uint<512>` present
- `"ping"` and `"pong"` in code (case-sensitive `B_ping`/`B_pong` satisfies)
- `"K_PART"` or `"k_part"` in code
- io5: `"fifo_C_in"` and `"fifo_C_out"`; io4: `"Crow"` and `"fifo_C_in"` absent
- no I-tile wrap around DATAFLOW (`_i_tile_loop_around_dataflow`)
- then shared stream / Crow-not-complete / pe_kj II / DSP / overlap gates

- [ ] **Step 6: Run tests to pass**

```bash
.venv/bin/python -m pytest tests/test_compact_pe_io.py tests/test_compact_pe_pack.py tests/test_compact_pe_mesh.py tests/test_compact_pe_search.py tests/test_compact_pe_instantiate.py tests/test_compact_pe_validate.py tests/test_compact_pe_rank.py tests/test_compact_pe_search_main.py -q
```

Expected: PASS

- [ ] **Step 7: Submit HLS (after tests green)**

```bash
env -u C2HLS_TMP_RUN -u C2HLS_PE_RECIPE \
  C2HLS_SYNTH_TIMEOUT=3600 C2HLS_CSIM_TIMEOUT=1800 \
  ./scripts/pc2/start_autosa_mm_io_search.sh --stamp 20260831_io
```

Quote IO `ranking.jsonl` vs cand 9 **1846–2161 / 1296 DSP**. Do not mix into pack/mesh rankings.

---

## Risks

- DATAFLOW + ping-pong deadlock: stream depth ≥ one tile (512).
- `Crow` ram_2p, never complete-partition.
- io5 C-flow trip counts must match pairwise (first PE no cin; last PE store drain after `t_k`).
- Do not treat io5 `pe_j` as J-columns.

## Self-review

- Spec grid, ids, ABI, gates, pipeline, out-of-scope: each has a task.
- No `architecture_ok(bench=...)` rewrite.
- Family A/B tests stay green.
- Commits skipped unless asked.

# Design: Compact IO-mesh HLS search (family C: io4 + io5)

**Date:** 2026-08-31  
**Status:** Approved (2026-08-31; approach 3: `io4` ∪ `io5` in one campaign)  
**Paper:** Wang, Guo, Cong, “AutoSA: A Polyhedral Compiler for High-Performance Systolic Arrays on FPGA,” FPGA’21 (`AutoSA/3431920.3439292.pdf`)  
**Scope:** Third ranking family for autosa_mm. Does not call AutoSA. Does not emit `kernel0` / `kernel_kernel.cpp`. Does not overwrite family A, family B (`compact_pe_pack_search_*`), locked 16×4, `20260830_mmflow`, or the Aug 18 slide (4285 vs 4228).

## Goal

Add AutoSA **communication management** (paper §§5–6) as a compact generator: K/J tiles, L1/L2 buffers, double buffering, latency tiles that are not PE count. One emitter family, **two space mappings** (`io4` = paper Design 4, `io5` = paper Design 5). Rank on latency **min/max** at DSP comparable to cand 9 (~1296), not by 128-PE oversubscription.

Packed family B already has 512-bit AXI + a PE mesh. Its 32-PE analog is **4434** vs cand 9 **1846–2161** because `load_B` replays full B `LAT_I` times. Rank-1 packed **1362** used **5120 DSP**. This branch exists so a 32-PE design can compete on latency without that DSP bill.

## Decisions (locked)

| Decision | Choice |
|----------|--------|
| Approach | **3:** `layout=io4` and `layout=io5` in **one** campaign, one emitter family |
| Family A / B | Frozen; do not mix rows into their `ranking.jsonl` |
| ABI / top | `autosa_mm_pack(A_t16*, B_t16*, C_t16*)` — not `kernel0` |
| Rank metric | Latency min, then latency max, then DSP (no interval) |
| Primary win | 32 PE × SIMD 8, DSP ~1280–1296, latency min **≤ 1846** |
| Secondary | 128 PE × SIMD 8 vs cand 5 **2033–2656** (scale check only) |
| Cosim | Off |
| Part / clock | U280 `xcu280-fsvh2892-2L-e`, 3.33 ns |
| Job prefix | `mmio`; artifacts `compact_pe_io_search_${STAMP}` |
| Slide | Unchanged unless the user says so |

## Space mappings (paper Fig. 11)

Both are 2-D (`space_time` analog). `pe` = total PEs = `pe_i * pe_j` for DSP.

### `io4` — Design 4 (`i,j` space)

- **PE_I along I, PE_J along J.**
- A reused along J (shift or L1 along J). B reused along I (shift along I).
- C **local** in the PE (`Crow` / `bufC`), packed drain after K tiles for that C tile.
- Cand **5** analog: `array_part[64,64,32]`, `latency[4,8]`, SIMD 8 → `PE_I=64/4=16`, `PE_J=64/8=8` = **128 PEs**.

Must-include id: `io4_16x8_s8_k32_j64`.

32-PE io4 example: `PE_I=8`, `PE_J=4` → `lat_i=I/8=8`, `lat_j=J_tile/4`. If `j_part=64`, `lat_j=16`. If `j_part=32`, `lat_j=8` and two J tiles.

Must-include id: `io4_8x4_s8_k32_j32` (and `io4_8x4_s8_k32_j64` if it stays DSP-legal).

### `io5` — Design 5 (`i,k` space) — cand 9

- **PE_I along I, PE_J means PEs along K** (second space dim). Not J-columns.
- B reused along I. A fed per `(i,k)` PE (L1 A). **C systolic along K** (`fifo_C` in/out) plus local reduce into the downstream PE / L2 C.
- Cand **9** analog: `array_part[64,32,32]`, `latency[8,4]`, SIMD 8 → `PE_I=64/8=8`, `PE_K=32/8=4` = **32 PEs**. J is time: `j_part=32`, `lat_j=4` (with an inner scan covering the J-tile). K tiled by 32; SIMD 8 sits on K.

Must-include id: `io5_8x4_s8_k32_j32`.

Do not treat cand 9 as `io4_8x4`. Design 4 with `latency[8,4]` and `array_part[64,32,32]` would be **8×8=64 PEs**, not 32.

## Tiling (derived, not a cand-9 hardcoded netlist)

For 64³ (`I=J=K=64`):

| Symbol | Meaning |
|--------|---------|
| `k_part` | K tile (paper array_part K). Default 32 if `K%32==0` else `K` |
| `j_part` | J tile (paper array_part J). 32 or 64 |
| `lat_i`, `lat_j` | Intra-PE C tile (paper **latency hiding**, §5.3). Independent of “how many PEs” except via the identities below |
| `simd` | Innermost K unroll (§5.4) |

Identities:

- io4: `pe_i * lat_i == i_tile` (i_tile = I = 64 for now), `pe_j * lat_j == j_part`
- io5: `pe_i * lat_i == I`, `pe_j * simd == k_part` (PEs along K cover one K-tile), `lat_j` is the intra-PE J tile (cand 9: 4)

Tile loops live **inside every DATAFLOW task** (same as AutoSA modules), never as one `load_B` over the whole matrix then `LAT_I` replay.

## IO stack (paper §6, compact names)

Not AutoSA module names. Semantics only:

| Level | Role |
|-------|------|
| L3 | Packed 512-bit `m_axi` bursts (`A_t16` / `B_t16` / `C_t16`) |
| L2 | Tile buffer + **ping/pong** for exterior I/O (B on io4; B and/or C on io5) |
| L1 | Small per-PE or per-row buffer (`local_A[lat_i][k_part/simd]` class) |

**Double buffering** only on those L1/L2 tile buffers (paper §6.3). Fill tile `t+1` while PEs consume tile `t`.

**Out of this emitter:** polyhedral I/O clustering / daisy-chain of dozens of `*_wrapper_*` instances. Compact: one L3 task, one L2 family per array side, PE grid, one store. Local interconnects (neighbor FIFOs) still required.

## Shared microarchitecture

- Top DATAFLOW; no I-tile wrap around DATAFLOW.
- PE MAC: SIMD float, `#pragma HLS PIPELINE II=1` on the flattened compute loop.
- `expected_dsp = pe_i * pe_j * simd * 5`, `min_dsp = max(1, int(0.6 * expected_dsp))`.
- `pack_bits = 512`.
- `io_pe(` (not `pack_pe` / `mesh_pe` / `mm_pe` / `PE_wrapper`). Count `1 + pe_i * pe_j`.
- No `void kernel0(`.

## Grid (small; DSP-efficiency first)

`simd = 8` only on the first campaign (float MAC). Drop if `pe_i*pe_j > 128` or DSP > `0.85*9024`.

| layout | pe_i × pe_j | k_part | j_part | Why |
|--------|-------------|--------|--------|-----|
| io4 | 8×4 | 32 | 32 | 32 PE Design 4, J-tiled |
| io4 | 8×4 | 32 | 64 | 32 PE Design 4, full J in PE tile |
| io4 | 16×8 | 32 | 64 | cand 5 analog |
| io5 | 8×4 | 32 | 32 | **cand 9 analog** |
| io5 | 8×4 | 32 | 64 | io5 without J-split |
| io5 | 16×8 | 64 | 64 | 128 PE Design 5 scale check (`k_part = pe_j * simd = 64`, not 32) |

Ids: `{layout}_{pe_i}x{pe_j}_s{simd}_k{k_part}_j{j_part}`  
Example: `io5_8x4_s8_k32_j32`, `io4_16x8_s8_k32_j64`, `io5_16x8_s8_k64_j64`.

io5 identity `pe_j * simd == k_part` forbids `io5_16x8` with `k_part=32` (would be 64). The 128-PE Design 5 scale point is therefore `k_part=64` (`N_K = 1`).

Optional later (not required to launch): simd ∈ {4,8}, `k_part=64`.

## Architecture gates (`architecture_ok_for_recipe`)

`elif rec.layout in ("io4", "io5")`:

- `io_pe(` count `== 1 + pe_i*pe_j`
- `autosa_mm_pack(` present, `void kernel0(` absent
- `ap_uint<512>` present
- ping and pong (or `*_ping` / `*_pong`) present
- `k_part` (or `K_PART`) appears in the compute/IO loops
- io5 only: PE has C fifo in and out (C-flow); io4 only: `Crow` / local C, no C-shift
- no I-tile wrap around DATAFLOW
- then the shared stream / Crow-not-complete / pe_kj II=1 / DSP / overlap gates

Do not change `architecture_ok(bench=...)`.

## Pipeline

```
enumerate_mm_io_recipes()     → layout io4 ∪ io5
        ↓
instantiate_mm(rec)           → compact_pe_io_instantiate
        ↓
validate (packed header/tb)   → csim+csynth, top autosa_mm_pack
        ↓
rank_candidates               → ranking.jsonl (IO family only)
```

Reuse packed header/testbench (`emit_pack_header` / `emit_pack_testbench`). New driver `compact_pe_io_search_main.py`. Launcher `scripts/pc2/start_autosa_mm_io_search.sh`.

## Out of this spec

AutoSA binary, `kernel_kernel.cpp`, mixing with family A/B rankings, Vivado/xclbin, other GEMMs, interval as a ranking key, 128-PE packed rank-1 as the success story.

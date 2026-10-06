# Design: Compact packed-mesh HLS search (family B)

**Date:** 2026-08-30  
**Status:** Approved (2026-08-30)  
**Scope:** Second ranking family for autosa_mm. Does not call AutoSA. Does not emit `kernel0` / `kernel_kernel.cpp`. Does not overwrite family A (`compact_pe_search_*`, locked 16×4, `20260830_mmflow`) or the Aug 18 slide (4285 vs 4228).

## Goal

Add AutoSA compiler *effects* (packed AXI, L3 bursts, intra-PE latency tile, 2-D PE mesh) as a compact generator, without the AutoSA compiler. Rank on the same U280 / 3.33 ns / csim+csynth path as family A, in a **separate** `ranking.jsonl`.

## Decisions (locked)

| Decision | Choice |
|----------|--------|
| Family A | Frozen (scalar `autosa_mm`, existing chain∪mesh) |
| Family B ABI | `autosa_mm_pack(A_t16*, B_t16*, C_t16*)`, `ap_uint<512>` |
| Top name | `autosa_mm_pack` — not `kernel0` |
| Grid | Same 30-point mesh grid: `PE_I,PE_J ∈ {4,8,16,32}`, SIMD ∈ {2,4,8}, DSP ≤ 0.85×9024, `PE_I×PE_J ≤ 128` |
| Ids | `pack{pe_i}x{pe_j}_simd{simd}` |
| Analogs | Must include `pack8x4_simd8`, `pack16x8_simd8` |
| `lat_i` / `lat_j` | `I/PE_I`, `J/PE_J` (derived; not AutoSA `latency[8,4]` special-case) |
| K tile | 32 when `K%32==0` else K (replay B/A from on-chip after one DRAM fill) |
| C path | Per-PE Crow, packed C word per I-row of the PE (not a `kernel0` C-shift netlist) |
| Cosim | Off |
| Job prefix | `mmpack`; artifacts `compact_pe_pack_search_${STAMP}` |
| Slide | Unchanged unless the user says so |

## Architecture

```
enumerate_mm_pack_recipes() → PeRecipe layout=pack
        ↓
instantiate_pack(rec)       → autosa_mm_pack C++
emit_pack_testbench(rec)    → serialize/deserialize host
        ↓
validate (csim+csynth)      → result.json
        ↓
rank_candidates             → ranking.jsonl (packed only)
```

## Out of this spec

AutoSA binary, `kernel_kernel.cpp`, mixing packed rows into family A ranking, Vivado/xclbin, other GEMMs.

# Slide brief: LLM agent vs AutoSA rank1 on `autosa_mm`

One kernel. Same ABI. Same FPGA and clock. Agent pipeline vs AutoSA’s published rank-1 HLS.

**Headline:** after flash → multi-PE design → stream I/O, the agent kernel is **4,285 csynth cycles** vs AutoSA rank1 **4,228**. Same DSP (320). Gap is **57 cycles (1.3%)**.

Use this file as slide copy: one slide = title + 3–5 bullets. Speaker notes are in italics.

---

## Naming (use this on slides)

Say **multi-PE design**, not DSE.

| Slide term | What it actually is | Why not “DSE” |
|---|---|---|
| **Flash** | First LLM rewrite of the C kernel (pragmas + loop order) | Fine |
| **Multi-PE design** | LLM emits a **16 PE × SIMD 4** compute nest on on-chip A/B/C | “DSE” is AutoSA’s array/SIMD/SA-size **search**. We did not re-run AutoSA. We asked the LLM for one PE×SIMD architecture. |
| **Stream I/O** | Replace bulk locals with DATAFLOW + `hls::stream` PE tasks | Fine |

*Speaker: if someone asks “is this AutoSA DSE?”, the answer is no. Same PE/SIMD width as rank1 (16×4), different generator: skills + LLM, compact kernel, original `autosa_mm` ABI.*

---

## Slide 1 — Title

**Closing the AutoSA mm gap with an LLM HLS agent**

- Kernel: `autosa_mm`, \(I=J=K=64\), `float`
- Device: Alveo U280 (`xcu280-fsvh2892-2L-e`), clock **3.33 ns**
- Agent: DeepSeek-v4-flash
- Baseline: AutoSA **rank1** HLS (`kernel0`), **4,228** cycles, **320 DSP**

*Speaker: question is not “can the LLM clone AutoSA’s 1600-line netlist”. It is: can a short agent kernel, same ports, reach rank1 latency.*

---

## Slide 2 — Fair comparison

**What is held fixed**

- Top name and ABI: `autosa_mm(A[I][K], B[J][K], C[I][J])` — **B is J×K**
- Same m_axi bundles / U280 / 3.33 ns
- Iso-compute: **16 PEs × SIMD 4** (320 DSP at II=1)
- Gold: C = A·B from zero; csim on the original testbench

**What is *not* allowed**

- Emitting AutoSA `kernel0` / `A_t16` / 1600-line PE–IO netlist
- Calling AutoSA’s array_part / SIMD DSE
- Changing the kernel name the testbench calls

*Speaker: rank1 is a systolic PE/IO machine. We keep the software-facing kernel and rebuild the hardware in three agent steps.*

---

## Slide 3 — Agent pipeline (the talk outline)

```
  C kernel
     │
     ▼
  [1] Flash              149,082 cycles   6 DSP     II=4 on k
     │
     ▼
  [2] Multi-PE design     13,160 cycles 352 DSP     compute II=1
     │                      load → compute → store, no overlap
     ▼
  [3] Stream I/O           4,285 cycles 320 DSP     DATAFLOW + streams
     │
     ▼
  AutoSA rank1             4,228 cycles 320 DSP
```

*Speaker: three different bugs, three different skills. Mixing stream skills into flash does not work. Flash cannot invent the PE array. Multi-PE cannot overlap DRAM.*

---

## Slide 4 — Results (the money slide)

Csynth latency, U280, 3.33 ns. Cosim off. csim passed on the promoted kernels.

| Stage | Cycles | DSP | vs rank1 |
|---|---:|---:|---|
| Flash | 149,082 | 6 | 35.3× slower |
| Multi-PE design | 13,160 | 352 | 3.11× slower |
| Stream (broken I/O) | 16,878–18,607 | 80 | worse than multi-PE |
| Stream (Crow BRAM) | 7,162 | 320 | 1.69× slower |
| **Stream (flatten k×j)** | **4,285** | **320** | **1.013× ( +57 cycles )** |
| AutoSA rank1 | 4,228 | 320 | 1× |

*Figure: bar chart of cycles, log y-axis, rank1 as a dashed line at 4,228.*

---

## Slide 5 — Step 1: Flash is not a systolic array

**149,082 cycles, 6 DSP**

- LLM keeps a triple loop, maybe LCST / ikj
- k-recurrence on a scalar (or muxed) FP add → **II = 4**
- Almost no MAC parallelism (6 DSP ≈ one float add/mul path)

*Speaker: flash’s job is a legal, testbench-correct HLS kernel with interfaces. It is not the PE array. Measuring flash vs AutoSA is the wrong comparison.*

---

## Slide 6 — Step 2: Multi-PE design

**13,160 cycles, 352 DSP — compute is already rank1-class**

What the LLM emits:

- `PE = 16`, `SIMD = 4`
- On-chip `local_A`, `local_B`, `local_C`
- `Crow[PE][J]`: complete-partition **PE only** (not J)
- Fused `compute_k0` × `compute_j` at **II = 1**
- Compute nest ≈ **4,816** cycles (MAC work is done)

Why the kernel is still **13,160**:

```
load A, B, C   ~4,100
compute        ~4,800
store C        ~4,100
──────────────
               ~13,160   (sequential)
```

*Speaker: 352 vs 320 DSP is a few extra FP ops in the unrolled nest, not a different array. The remaining 3.1× vs rank1 is I/O, not MAC width. Do not slap DATAFLOW on this body — shared locals are illegal.*

---

## Slide 7 — Step 3: Stream I/O (the architecture)

**Goal:** overlap load / 16 PEs / store. Keep `autosa_mm` ABI.

```
  DRAM A ──► load_A ──► fifo_A[16] ──► mm_pe × 16 ──► fifo_C[16] ──► store_C ──► DRAM C
  DRAM B ──► load_B ──► fifo_B[0] ──► PE0 → PE1 → … → PE15 → drain_B
```

- SIMD beat = `ap_uint<128>` (4 floats), not a struct of floats
- Each `mm_pe` has private `Crow[J]` as **ram_2p BRAM**
- B is forwarded (systolic); A is per-PE
- PE function named `mm_pe` (`#define PE` would eat `PE`)

*Speaker: this is I/O construction, not another PE/SIMD search. Rank1 does the same 16×4 machine with a much larger generated interconnect.*

---

## Slide 8 — Stream failures we hit (and taught)

| Attempt | Cycles | DSP | What broke |
|---|---:|---:|---|
| Struct-of-floats FIFOs | 16,878 | 80 | 4 serialized `float` reads → II=4 |
| Packed `ap_uint<128>` only | 18,607 | 80 | still II=4 |
| Crow **complete-partition** | (same class) | 80 | HLS muxes `Crow[j]` → `mux_case_0` FP recurrence, II=4 |

**Rule that fixed DSP:** `Crow[J]` = `BIND_STORAGE ram_2p bram`. Never `ARRAY_PARTITION complete` on the pipelined J index.

After that: **7,162 cycles, 320 DSP, compute_j II=1**. Promoted over 13,160. Still 1.69× rank1.

*Speaker: 80 DSP = 16 PEs × ~5 DSP at II=4. 320 DSP = 16 × 20 at II=1. DSP is a lie detector for this kernel.*

---

## Slide 9 — The last 1.7×: pipeline refill, not missing PEs

At II=1, useful MAC traffic per PE is **4,096** beats:

\[
4 \text{ i-tiles} \times 16 \text{ k-tiles} \times 64\,j = 4096
\]

Rank1 ≈ \(4 \times (1024 + \text{fill}) \approx 4228\).

Our 7,162 kernel had `#pragma HLS LOOP_FLATTEN off` on `pe_k0`:

- `compute_j` trip 64, pipeline depth ~35 → **97** cycles **per k-tile**
- 16 k-tiles × 4 i-tiles of refill ≈ **+2,200 cycles**
- Separate init/drain loops on Crow

**Fix (in the stream skills):** one flattened `pe_kj` of **1,024** beats per i-tile; fuse init (first k) and C drain (last k). Do **not** `DEPENDENCE inter false` (true dep at distance 64; ram_2p already allows II=1).

*Speaker: we had taught flatten-off as a safety rule for the Crow mux bug. Once Crow is a BRAM, flatten-off is the bug.*

---

## Slide 10 — Final stream vs rank1

| | Agent stream | AutoSA rank1 |
|---|---:|---:|
| Cycles | **4,285** | **4,228** |
| DSP | 320 | 320 |
| Compute II | 1 | 1 |
| PE loop | `pe_i0`+`pe_kj` flattened, trip **4096**, depth 28 | nested c0/c2/c4/c5/c6, inner II=1 |
| Kernel size | ~170 lines | ~1,600 lines |
| Top | `autosa_mm` | `kernel0` |

Agent PE module: **4,124** cycles. DATAFLOW interval **4,173** (load_B bound). Kernel **4,285** = interval + DATAFLOW fill.

*Speaker: 57 cycles is prologue, not missing MACs. Same DSP. We did not copy rank1’s IO network; HLS still matched the cycle count.*

---

## Slide 11 — Resources (csynth, not impl)

| | BRAM | DSP | FF | LUT |
|---|---:|---:|---:|---:|
| Flash | 98 | 6 | 23,644 | 17,013 |
| Multi-PE design | 114 | 352 | 62,121 | 44,701 |
| Stream (final) | 62 | 320 | 55,878 | 36,653 |
| Rank1 (campaign inventory) | 330 | 320 | 77,288 | 89,933 |

*Speaker: rank1 spends extra LUT/FF/BRAM on the generated PE/IO mesh. Agent stream is smaller on-chip. Not an implementation/timing-closure claim — csynth only.*

---

## Slide 12 — What the LLM had to be taught (skills, not one prompt)

Keep these as **separate** skill packs. Do not dump stream into flash.

1. **Flash** — legal kernel, interfaces, no AutoSA clone
2. **Multi-PE** — 16×4, partition PE dim of Crow, II=1 compute, B is J×K
3. **Stream**
   - `ap_uint<128>` SIMD FIFOs
   - Crow = ram_2p, never complete-partition J
   - flatten k×j to 1024 / 4096 beats
   - fuse init/drain
   - name the PE `mm_pe`

*Speaker: each failure (II=4 mux, struct FIFO, flatten-off refill) became one avoid-rule. The 4,285 number is after those rules, not from a single lucky sample.*

---

## Slide 13 — Takeaways

1. **Flash ≠ AutoSA.** 35× off, 6 DSP.
2. **Multi-PE design gets the compute.** 13,160 vs 4,228 is sequential load/store, not weak MACs (~4.8k compute).
3. **Stream I/O is a different step.** DATAFLOW + packed streams + BRAM Crow.
4. **II=1 is not enough** if you refill a 35-deep pipeline 16× per row-tile.
5. **Agent 4,285 vs rank1 4,228 (1.3%)**, **320 DSP**, original ABI, ~10× shorter code.

*Optional last line: we are not claiming we beat AutoSA’s compiler — we matched its rank1 csynth latency on this 64³ mm without emitting its netlist.*

---

## Slide 14 — Limits (stay honest)

- One kernel (`autosa_mm` 64³), one model (DeepSeek-v4-flash), **csynth + csim**, no cosim / no bitstream
- Rank1 DSP 320 is the iso-resource target; we are not claiming a better array
- Clock reports still show HLS 200-871 vs uncertainty; not Vivado WNS
- Internal pipeline still calls the middle step `dse` on disk; slides should not

---

## Appendix A — Locked numbers

| Quantity | Value |
|---|---|
| Part | `xcu280-fsvh2892-2L-e` |
| Clock | 3.33 ns |
| Problem | I=J=K=64, `data_t` = float |
| ABI | `extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J])` |
| Model | DeepSeek-v4-flash |
| Phase-B (pre-flash) | 2,445,313 cycles |
| Flash | 149,082 cycles, DSP 6 |
| Multi-PE | 13,160 cycles, DSP 352, compute ~4,816 |
| Stream final | 4,285 cycles, DSP 320, BRAM 62, interval 4,173, PE 4,124 |
| Rank1 | 4,228 cycles, DSP 320 |
| Cell | `artifacts/pc2/batch_parallel_autosa_mm_gap_ds_v4f_20260818_mm_gap_flash_ds_v4f/variants/autosa_nav_n/autosa_mm/deepseek-v4-flash__flash__autosa__nav_n/` |

## Appendix B — Suggested figures

1. **Waterfall / bar (log):** Phase-B → Flash → Multi-PE → Stream-broken → Stream-BRAM → Stream-flat → Rank1
2. **Block diagram:** load_A / load_B / 16× mm_pe / drain_B / store_C
3. **Loop cartoon:** nested k0×j with flatten-off (16× fill) vs one `pe_kj` of 1024
4. **DSP as II proxy:** 80 vs 320

## Appendix C — One-sentence claims (pick one)

- “An LLM HLS agent matches AutoSA rank1 on 64³ mm to 1.3% (4,285 vs 4,228 cycles) at 320 DSP, without cloning the AutoSA netlist.”
- “Multi-PE design gets II=1 compute; stream I/O + flattened k×j is what actually reaches rank1 latency.”
- “Calling the middle step DSE oversells it: it is a 16×4 PE design, not AutoSA’s design-space search.”

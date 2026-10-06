# How to address Fang’s points (8 Sep 2026 meeting)

Memo for Ahmad. Use for slides, paper story, and a 30-second pitch.
Metrics: **latency min, latency max, DSP only**. Never interval. Never HLS estimated clock.
Never say “DSE”. Say **compute rewrite** and **hide load-store**.
Two leaderboards — never one table: **iso-compute ~4228 / 320 DSP** vs **spend-DSP 940 / 5344**.

Kernel lock: `autosa_mm`, I=J=K=64 float, ABI `autosa_mm(A[I][K], B[J][K], C[I][J])`, B is J×K.
Device U280, Vitis HLS 2023.2, 3.33 ns. Seed is `plain.cpp`, not stripped AutoSA `kernel0`.

---

## Elevator pitch (3–5 sentences; Fang can repeat this)

We start from ordinary C for 64×64×64 float matrix multiply. A plain HLS compile of that seed is about **33k cycles and 40 DSP**; a completely unoptimized triple loop is **2.45 million cycles**. An LLM that only knows generic HLS (pipeline, unroll, partition, bursts) does not reach a human systolic design. What closes AutoSA is **domain knowledge**: build a 16-PE × 4-SIMD on-chip MAC array, then **overlap DRAM with those PEs** so one kernel call finishes in about the time of the slowest stage. At the **same 320 DSP** as AutoSA’s rank-1, the agent is **4216–4292 cycles vs AutoSA 4228**. A second, separate design spends **5344 DSP** on a fully on-chip GEMM and finishes in **940 cycles**; that is not the fair AutoSA comparison.

**30-second version:** Generic HLS skills get you a legal kernel. GEMM knowledge builds the PE array. Overlapping load with compute is what matches AutoSA at the same DSP. A faster 940-cycle kernel exists, but it uses 17× the DSPs — different question.

---

## Skill taxonomy (human language, not agent jargon)

| Bucket | What a FPGA person hears | One-liners (Rodinia-style) | What we actually ran | What it did on `autosa_mm` |
|---|---|---|---|---|
| **Generic HLS** | Things you teach in ENSC 453 / Rodinia HLS | pipeline the hot loop; unroll independent work; partition on-chip arrays; wide AXI bursts; stage DRAM into local buffers; do not slap DATAFLOW on a single sequential firing | Intended: 5–6 kernel-independent steps. **Shipped:** 90-skill dump (contaminated with 64×64 GEMM ping-pong / PE_BLK) | FLASH **139484 / 10 DSP** — legal kernel, almost no MACs. **Can lose to a skill-free rewrite.** Do not dump 90 skills as the paper method. |
| **Domain (linear algebra / GEMM)** | How you map GEMM onto FPGA, not this one size | flatten i/k/j; write C once (no RMW); PE row + SIMD-k adder tree; bank locals to match unroll | Compute-rewrite PE recipe: **16 PE × SIMD 4**; not a search | **13160 / 352 DSP**. Compute nest ~4816 at II=1. Kernel still load-then-compute-then-store. |
| **Application (this kernel / this size / this chip)** | Facts that are true of *this* 64³ float on U280 | 64³ float **fits on-chip**; 512-bit / 16-lane AXI; PE_BLK=16 output groups; **independent** `load_A` and `load_B`; do **not** DATAFLOW a one-shot full-array load | Distilled on-chip pack (8 skills). Avoids DATAFLOW/ping-pong and fused A+B. | Spend-DSP champion **940–940 / 5344**. Type **no**. Equation `max(259,258)+340+260=940`. Cosim PASS **1071**. |

**Pipeline stages → buckets (iso-compute path, comparison 1)**

1. Seed / Phase B → neither (unoptimized C).
2. FLASH → **generic HLS** (when the pack is actually generic).
3. Compute rewrite (16×4 PE array) → **domain**.
4. Hide load-store (streams, A private per PE, B forwarded, C drained) → **domain I/O + application schedule**.
5. Enforcement ping-pong DATAFLOW → a **failed** attempt to fake (4) with a pragma. Not a bucket that worked.

**Pipeline stages → buckets (spend-DSP path, comparison 2)**

1. DSP floor + 512-bit I/O + write-once C + PE_BLK=16 → **application** (on-chip fit, this size).
2. Do not present this as “the AutoSA match.”

Do not lead with HLSFactory `20260908_124722`. If asked “is this general?”: we transferred the on-chip pack to 28 PolyBench kernels as a **side test**; it is not the AutoSA-mm story.

---

## Point-by-point

### 1. Story structure he wants

**(a) What he meant.** Naive LLM, huge gap. Generic HLS (tiling / pipeline / parallelize, one sentence) closes most of it (100× → 5–10×). Remaining gap is missing **domain / application** knowledge. Then compute rewrite / overlap, another several-X, competitive with a human / AutoSA.

**(b) What is actually true.** The *shape* is right. The 90-skill dump is **not** the generic step that closed the gap.

| Rung | Latency min–max | DSP | vs AutoSA 4228 / 320 | Campaign |
|---|---:|---:|---|---|
| Unoptimized triple loop (Phase B) | **2445313–2445313** | 50 | ~578× | every `hls_synth__ref_baseline_synth` |
| Test 1 seed, no FLASH (gold HLS of `plain.cpp`) | **33357–33357** | **40** | ~7.9× | `…seed_synth_20260908_124628` (hcl, intel, hbm, getting_started, hcl_intel, block_sparse; catapult 32916/24; int16 16533/64) |
| Zero-shot LLM (one rewrite, HLS-engineer prompt) | **40454–42758** | 320 | ~10× | `20260902_mmzs` |
| FLASH, empty skills JSON | **1056894–1056894** | 11 | ~250× | `20260902_mmns` FLASH stage |
| FLASH, 90-skill “generic” pack | **139484–139484** | 10 | ~33× | frozen `20260830_mmflow` FLASH |
| Compute rewrite (16×4, sequential I/O) | **13160–13160** | 352 | **3.1×** | `20260830_mmflow` |
| DATAFLOW pragma “ping-pong” (not overlapped) | **12893–12893** | 320 | **3.0×** | `20260829_123043` |
| Hide load-store (streams) | **4292–4292** (with skills) / **4216–4216** (empty JSON) | **320** | **+1.5% / −0.3%** | mmflow / `20260902_mmns` |
| AutoSA rank-1 | **4228–4228** | **320** | 1× | AutoSA inventory |

Generic 90-skill FLASH **slowed vs zero-shot** (139k vs 40k) because it emitted a legal kernel with **10 DSP** and no PE array. The gap that matches Fang’s “several-X then competitive” is:

- **~10× → ~3×:** domain compute rewrite (PE array).
- **~3× → ~1×:** hide load-store (streaming PE I/O).

**(c) What to say / put on the slide.** Use Fang’s four-rung cartoon. Put **honest numbers** on the rungs. Caption: “Generic HLS pack as shipped did not close the gap; the PE-array rewrite and overlapping I/O did.” Optional left column: seed 33357/40 as Test 1.

**(d) What not to say.** Do not say the 90-skill dump closed 100×→10×. Do not put 940 on this waterfall. Do not call column 3 (12893) a success.

**(e) New experiment?** Yes, if he insists generic skills must *help*: a **clean 5–6 skill Rodinia pack** (no PE_BLK, no 64×64 ping-pong) vs zero-shot vs domain pack. Mark as next slide, not current claim.

---

### 2. Terminology looks too general

**(a) What he meant.** “Compute rewrite” sounds like a universal compiler pass. Ahmad said it has been GEMM-focused. He wants names a famous FPGA researcher understands in 30 seconds.

**(b) What is true.** The iso-compute middle stage is a **fixed 16×4 on-chip MAC array** (`post_flash_pe_recipe.py`), not a search. The last iso-compute stage is **streaming systolic I/O** (A per PE, B forwarded, C drained). The 940 design is **fully unrolled on-chip GEMM**, not a systolic array.

**(c) Slide glossary (use these names out loud).**

| Do not say | Say instead (30-second FPGA English) |
|---|---|
| DSE | — (never) |
| Compute rewrite | **On-chip PE array** / **parallel MAC array** (16 PEs × SIMD-4) |
| Hide load-store | **Overlap DRAM with the PE array** / **streaming systolic I/O** |
| FLASH | **First LLM rewrite** of the C kernel (pragmas + legal interfaces) |
| Enforcement | **DATAFLOW pragma on a single full-array firing** (did not overlap) |
| Spend-DSP / 940 | **Fully on-chip GEMM**, 16-wide output groups, full-K adder tree |
| 90 skills | **Generic HLS hints** (and say they are contaminated) |

**(d) What not to say.** Do not imply compute rewrite is kernel-independent. Do not say ping-pong “just works.”

**(e) New experiment?** No.

---

### 3. Compute rewrite: constraints vs teaching the rewrite; AutoSA code as ICL

**(a) What he meant.** You gave coordination constraints and snippets, not a finished netlist. If you paste AutoSA-generated `kernel0` and say “make it faster,” can the LLM learn? That is a **different study**.

**(b) What is true.** Current method: PE **recipe** (16×4) + stream **skeleton** (FIFOs, `mm_pe`, flatten `pe_kj`). We **did not** emit AutoSA `kernel0` / `A_t16` / L2 FIFOs as the result. Empty skills JSON still closed rank-1 (**4216 / 320**) — the recipe + skeleton mattered more than the 90-skill dump. **We have not tried AutoSA code as in-context example.**

**(c) What to say.** “We specify the architecture class (16×4 PEs, streaming I/O), not the 1600-line netlist. Teaching from AutoSA source is future work — a different paper.”

**(d) What not to say.** Do not promise the current paper includes “LLM learns from AutoSA code.” Do not claim reverse-engineering `kernel0` is the method.

**(e) New experiment?** **Different study.** If he wants it: ICL of AutoSA HLS, ask for a faster legal `autosa_mm` ABI. Do not mix into current tables.

---

### 4. Generic vs domain vs application

**(a) What he meant.** Clean taxonomy in human language. Not FLASH / 90 skills / skill bins.

**(b) What is true.** See the taxonomy table above. The 90-skill file is tagged kernel-independent but **contaminated** (`hls-baseline-load-compute-store-gate` encodes PE_BLK and 64×64 ping-pong).

**(c) Slide.** Three columns: Generic HLS / GEMM family / This 64³ on-chip. One example each. One measured number each. Footnote: 90-skill dump ≠ clean generic.

**(d) What not to say.** Do not present 90 skills as the generic rung. Do not mix systolic stream skills into FLASH.

**(e) New experiment?** Strip PE_BLK / 64×64 from anything labelled generic (already on the to-do list; not required to answer him tomorrow).

---

### 5. Streaming 3× vs ping-pong / DATAFLOW

**(a) What he meant.** Ping-pong **should** already overlap load/compute/store. Streaming should not be 3× better if ping-pong was real. Vitis DATAFLOW reports can lie; **manually** add module latencies like Rodinia / ENSC 453. If one ~12k number equals the sum of three ~4k modules, that is **not** overlapped DATAFLOW.

**(b) What is true.** He is right about the ~12k figure. That figure is **not** the 940 champion.

**The 12k figure (iso-compute, comparison 1)** — enforcement campaign `batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043`, report `autosa_mm_csynth.rpt`:

| Module | Latency min–max | Pipeline type |
|---|---:|---|
| load_B | 4170–4170 | no |
| load_tiles | 4170–4170 | no |
| compute_tiles | 4552–4552 | no |
| store_tiles | 4169–4169 | no |
| **Kernel `autosa_mm`** | **12893–12893** | **dataflow** |

Hand check (do this on the slide): **4170 + 4552 + 4169 = 12891 ≈ 12893**. Finish time of one call is the **sum**, not `max`. HLS labelled the parent **dataflow**. That is exactly Fang’s objection. (If asked about 4553: that is the report **interval**, not finish time — do not put it on the slide.)

Same sequential sum on compute rewrite `20260830_mmflow` (type **no**, **13160–13160 / 352**): load ~4099, compute_i0 **4816**, store ~4101. Brief equation: ~4100 + ~4800 + ~4100 ≈ 13160.

**The 3× “streaming”** is this sequential kernel vs hide load-store at the **same 320 DSP**:

- 13160 / 4292 ≈ **3.07×**
- 12893 / 4292 ≈ **3.00×**

Stream (`20260830_mmflow`) **is** overlapped. Hand check: kernel **4292–4292 / 320**, type dataflow. Modules: load_B **4172**, each `mm_pe` **4131**, store_C **4168**, load_A **1100**. **4292 ≈ max(rooms), not the sum** (sum would be >10k). Sixteen `mm_pe` tasks + load + store run together.

**The 940 champion is a different experiment** (comparison 2). Type **no**, not DATAFLOW. Sequential **by construction**: one full on-chip load, then compute, then store. Ping-pong cannot hide a single-shot dependence. See DATAFLOW paragraph below.

**(c) What to say.** Three cases, say them in this order:

1. **Pragma DATAFLOW on one firing of the full arrays** cannot overlap load-all / compute-all / store-all. Our “ping-pong” code was this. Manual sum = 12893.
2. **Tiled ping-pong** (two-slot buffers over a **tile loop**) *should* overlap. We did not get that from the agent. A later **manual** 2-tile ping-pong on the 940 array was **worse** (1074 / 5344 vs 940) because it split a 256-trip compute and paid pipeline depth 84 twice.
3. **Streaming systolic I/O** starts compute on the first beats (FIFOs into 16 PEs). That is why 4292 ≈ max(modules).

**(d) What not to say.** Do not say “Vitis ping-pong is broken.” Do not say ping-pong just works. Do not use 940 to explain the 3×. Do not quote interval 4553 as a result.

**(e) New experiment?** Optional, same-DSP: **legal tiled ping-pong** on the 16×4 sequential kernel (the 13160 design), judged by **manual module sum vs kernel latency**, not by pragma presence. Do not relaunch 940 ping-pong unless asked.

---

### 6. How we became competitive with AutoSA (mechanism)

**(a) What he meant.** Not “we closed the gap.” **How:** more DSPs? different schedule? separate loads? PE_BLK=16 K-unroll? Put the **same-DSP** table first.

**(b) What is true.**

**Comparison 1 first (same DSP, same 16×4 class).**

| Design | Latency min–max | DSP | Mechanism |
|---|---:|---:|---|
| AutoSA rank-1 | **4228–4228** | **320** | 16 PE × SIMD 4; K-tiles; B ping-pong; C drain |
| Agent stream, empty skills JSON | **4216–4216** | **320** | Same PE width; streaming A/B/C tasks; flatten `pe_kj` |
| Agent stream, 90-skill mmflow | **4292–4292** | **320** | Same |
| Aug 18 first locked stream | **4285–4285** | **320** | Same |

Mechanism at 320 DSP: **not more DSPs.** Compute rewrite builds 16×4 (compute already ~4816). Hide load-store overlaps DRAM with PEs so one call ≈ slowest room (~4200), not load+compute+store (~13k). 352 vs 320 on the sequential kernel is a few extra FP ops, not a different array. **32×8 was worse** (4583 / 1280).

**Comparison 2 second (spend DSP — do not put next to 4228 as a speedup).**

Champion `batch_parallel_autosa_mm_flash_dsp500_pe16_20260904_131622`:

| Module | Latency min–max | Type | DSP |
|---|---:|---|---:|
| load_A | 259–259 | no | 0 |
| load_B | 258–258 | no | 0 |
| compute (PE_BLK=16, full-K tree, trip 256, depth 84) | 340–340 | no | 5344 |
| store | 260–260 | no | 0 |
| **Kernel** | **940–940** | **no** | **5344** |

Equation: `max(259,258)+340+260=940`. Two **independent** load modules, started together. Nested `i` / `j += LANES` (LANES=16, 512-bit). **Not** fused A+B. Cosim PASS **1071**. DSP 5344 ≈ 16 output lanes × 64 K-unroll × ~5 DSP/float MAC (59% of U280 9024; 177% of one SLR — allowed on this comparison).

Ladder (one change per step, flash-only): 9423/1344 → 2760/636 (512-bit I/O) → 1261/2672 (write-once C, II=1) → **940/5344** (PE_BLK=16).

Fused A+B **871 csynth / 5088**, cosim **1224**, is a **failure**, even though csynth is faster.

**(c) Slide.** Left: 320 DSP table. Right: 940 module table. Big red: “do not mix.”

**(d) What not to say.** “We beat AutoSA 4.5×.” “940 is the fair match.” “Ping-pong hid the 940 I/O.” Putting 871 on the 4228 slide.

**(e) New experiment?** Not required to answer him. Efficiency vs AutoSA cand 9 (**1846–2161 / ~1296 DSP**) is **open**.

---

### 7. One design vs AutoSA family

**(a) What he meant.** Don’t sell one 64³ float as the AutoSA paper. Extend to best-of-each AutoSA bench. Philip: AutoSA more-DSP configs often fail csim/synth.

**(b) What is true.**

Iso-compute vs rank-1 is **one architecture class** (`autosa_mm` 64³ float), plus wave1:

| Bench | Agent | Rank-1 | DSP | 2% gate |
|---|---:|---:|---:|---|
| mm (locked) | 4292 / 4216 | 4228 | 320 | pass |
| hcl | 4294 | 4230 | 320 | pass 1.51% |
| hcl_intel | 4292 | 4226 | 320 | pass 1.56% |
| int16 | 4280 | 4219 | 64 | pass 1.45% |
| catapult | 8351 | 8286 | 96 | pass 0.78% |
| intel | **4525** | 4178 | 640 | **fail 8.3%** |
| getting_started | **4525** | 2194 | 640 | **fail ~2×** (rank-1 is two DATAFLOW tiles) |

Spend-DSP other `autosa_mm_*` keepers (on-chip pack, **not** mixed with 4228):

| Kernel | Csynth | DSP | Stamp |
|---|---:|---:|---|
| mm (frozen champion) | **940** | **5344** | `…pe16_20260904_131622` |
| hcl | **938** | **5344** | `…hcl…_20260908_004300` |
| hbm | **938** | **5344** | `…hbm…_20260908_010347` |
| intel | **938** | **5344** | `…intel…_20260908_010347` |
| block_sparse | **937** | **5344** | `…block_sparse…_20260908_010347` |
| getting_started | 900 | 5088 | treat as **not** 940-class until I/O shape is checked (5088 DSP matches fused-A+B risk) |
| hcl_intel | 1002 | 5344 | after 1032/5120 serial-FP failure |
| catapult | 744 | 6144 | different DSP |
| int16 | 381 | 8192 | different datatype / DSP |

CNN / LU / large_* iso-compute **not** closed. AutoSA cand-9 efficiency **not** closed.

**(c) What to say.** “Fair AutoSA match is one 64³ class at 320 DSP, plus four sibling GEMMs inside 2%. High-DSP 937–938 / 5344 on other mm variants is the **same on-chip recipe**, not a new AutoSA match. Intel and getting_started still miss at iso-compute.”

**(d) What not to say.** “We beat the AutoSA suite.” “900/5088 is a better champion.”

**(e) New experiment?** Best **valid** AutoSA point per remaining kernel (cnn, lu, large_*), still two leaderboards.

---

### 8. FLASH without skills (naive agentic baseline)

**(a) What he meant.** The agentic FLASH loop with **no skill pack**, as the naive LLM baseline — he asked for this column explicitly.

**(b) What exists (do not invent a missing column).**

| Arm | What it is | Latency min–max | DSP | Exists? |
|---|---|---:|---:|---|
| Zero-shot | HLS-engineer prompt, **one** rewrite, no skill JSON, FLASH only (`20260902_mmzs`) | **40454–42758** | **320** | **Yes** — this is the naive agentic FLASH |
| No-skills FLASH stage | Same FLASH prompt as with-skills, **empty JSON**, then compute rewrite + stream (`20260902_mmns`) | FLASH **1056894–1056894** | **11** | **Yes** as a stage; stream later **4216 / 320** |
| FLASH-only + empty JSON + default FLASH prompt, stop after FLASH | Clean column next to 139484/10 | — | — | **Not as its own campaign.** 1056894/11 is that FLASH result |

Zero-shot used K-unroll + pipeline (320 DSP from unrolling K, **not** the 16×4 PE array). Spec originally omitted the PE recipe; the locked correction is: zero-shot **had** the PE-recipe text in some writes — on disk the selected kernel is **no PE array**, 40454–42758 / 320.

**(c) Slide.** Add a column: “LLM, no skill pack (zero-shot) **40454–42758 / 320**.” Optionally a footnote FLASH-empty-JSON **1056894 / 11**.

**(d) What not to say.** That we never ran a no-skills LLM. That zero-shot is the PE array.

**(e) New experiment?** Easy next slide if he wants FLASH-only empty JSON with the **default** FLASH prompt (not zero-shot):  
`C2HLS_FLASH_ONLY=1 ./scripts/pc2/start_autosa_mm_flow.sh --flavor noskills --endpoint-url http://login5:18092/v1`  
Do **not** overwrite `20260902_mmns` or `20260830_mmflow`. New stamp. Quote latency min/max + DSP only.

---

### 9. AutoSA csim failures

**(a) What he meant.** Don’t claim LLM > AutoSA because AutoSA is buggy. Compare to **valid** AutoSA. Failures are likely AutoSA bugs. Later match best AutoSA if bugs get fixed.

**(b) What is true.** Rank-1 **4228 / 320** is a **valid** csim+csynth point. Exhaustive cand 9 **1846–2161 / ~1296 DSP** is a different, also-validated class. Philip’s point is visible on AutoSA campaign `20260908_011549_full_u280_100_simd16`: many kernels csynth with **0 csim pass** (cnn 0/91, several large_* 0 csim); others mixed (large_mm 58/60 csim, mm_hcl 45/50, mm_intel 10/10). We do not treat failed AutoSA points as the reference.

**(c) What to say.** “Reference is valid AutoSA (csim pass + csynth). We do not win by AutoSA crashing. If a faster AutoSA point becomes legal later, we compare to that.”

**(d) What not to say.** “LLM is better because AutoSA csim fails.” Ranking failed AutoSA DSP-heavy points against 940.

**(e) New experiment?** Investigate failing AutoSA candidates as AutoSA bugs (their repo). Not a c2hls slide.

---

### 10. Elevator pitch (already above)

Use the 3–5 sentence block. If he has 10 seconds: **same 320 DSP, 4216 vs 4228; a 940-cycle kernel uses 5344 DSP and is a different comparison.**

---

## Honest DATAFLOW / ping-pong vs hide load-store

Fang’s manual-sum test is the right test. Apply it to **two different reports**.

**Report A — the ~12k DATAFLOW number (comparison 1, not the champion).**  
Enforcement `20260829_123043`. Parent type **dataflow**, latency **12893**. Modules ~4170, ~4552, ~4169. **4170+4552+4169=12891**. One kernel call is load-all then compute-all then store-all. The agent put `#pragma HLS DATAFLOW` on three sequential full-array phases. It did **not** tile ping-pong (`buf[2]` over a tile loop with load(t+1)/compute(t)/store(t−1)). HLS can still print type=dataflow and an interval near max(modules). **Finish time is the sum.** That is not Vitis being mysteriously broken; it is DATAFLOW on a single firing with a true dependence chain (compute needs the whole arrays; store needs the whole C).

**Report B — hide load-store that actually overlapped (still comparison 1).**  
Stream `20260830_mmflow`: **4292 / 320**, type dataflow. Sixteen `mm_pe` (~4131) plus load_B (4172) plus store (4168). Kernel **4292 ≈ max**, not sum. Compute starts on early beats through FIFOs. That is why stream is ~3× the sequential 13k kernel **at the same DSP**.

**Report C — spend-DSP champion (comparison 2).**  
`…pe16_20260904_131622` `autosa_mm_csynth.rpt`: type **no**, **940–940 / 5344**. load_A **259**, load_B **258**, compute **340**, store **260**. **max(259,258)+340+260=940**. One full on-chip copy of A and B, then one 256-trip compute, then one store. Ping-pong cannot hide that: there is no second tile. A later legal 2-tile ping-pong on this array was **1074 / 5344** (worse): it split compute and paid depth 84 twice. LLM tile-pp was **3817 / 5120** (reloaded full A). On-chip pack **avoids** DATAFLOW/ping-pong for this reason.

If someone says “streaming shouldn’t be 3× better than ping-pong”: agree **if** ping-pong had been tiled. Ours was not. Show Report A’s sum.

---

## Slide outline (8 slides, Fang’s story)

Keep iso-compute numbers on 1–6. Spend-DSP only on 7. Do not put 940 next to 4228.

1. **Title / protocol.** `autosa_mm` 64³ float, U280, 3.33 ns. AutoSA rank-1 **4228 / 320**. Seed `plain.cpp`. Metrics: latency min/max, DSP.

2. **The story (waterfall, iso-compute).**  
   Seed **33357 / 40** (or naive **2445313 / 50**) → zero-shot **40454–42758 / 320** → generic 90-skill FLASH **139484 / 10** (hurts) → PE array sequential **13160 / 352** (~3× AutoSA) → hide load-store **4216–4292 / 320** vs AutoSA **4228 / 320**.  
   Speaker: generic pack as shipped is not the closer; domain PE array + overlap is.

3. **Taxonomy.** Generic HLS (Rodinia one-liners) vs GEMM PE array vs this-size on-chip. Map stages. Footnote: 90-skill dump contaminated.

4. **Why DATAFLOW was not 3× (manual sum).** Enforcement **12893**; 4170+4552+4169=12891. Stream **4292 ≈ max(4172, 4131, 4168)**. Three bullets: (a) one-shot DATAFLOW cannot overlap, (b) tiled ping-pong should, (c) streaming starts on first beats. Do not quote interval.

5. **Mechanism at same DSP.** 16×4, not more silicon. 32×8 **4583 / 1280** worse. Empty JSON stream **4216** beats 90-skill **4292**. Recipe + stream skeleton, not 90 skills.

6. **Family honesty.** Wave1 4/6 inside 2%. intel +8.3%, getting_started ~2×. Compare to **valid** AutoSA. AutoSA csim fails are their bugs; we do not claim victory from them. ICL of AutoSA code = future work.

7. **Separate question: spend DSP.** Champion **940–940 / 5344**, type **no**, `max(259,258)+340+260`. Cosim 1071. Other mm keepers **937–938 / 5344**. Fused A+B 871/1224 is a failure. **Not** the 4228 slide.

8. **Takeaways + next.**  
   - Same-DSP: competitive with AutoSA via PE array + overlapping I/O.  
   - Generic 90-skill dump is not the method.  
   - 940 is more DSPs, sequential on-chip GEMM.  
   - Next (if he wants): clean 5–6 generic skills; FLASH-only empty JSON as its own column; tiled ping-pong on the **13160** kernel; cand-9 efficiency.

---

## Numbers to keep in a pocket card

**Iso-compute (320-class)**  
4228 / 320 AutoSA · 4285 Aug 18 · 4292 mmflow · 4216 empty JSON · 40454–42758 zero-shot · 139484/10 90-skill FLASH · 13160/352 sequential PE · 12893/320 fake DATAFLOW · 33357/40 Test 1 seed · 2445313/50 naive

**Spend-DSP**  
940–940 / 5344 champion · Cosim 1071 · 259 ∥ 258 + 340 + 260 · type no · fused 871 csynth / 1224 cosim = fail · keepers 937–938 / 5344

**Do not mix those two blocks on one figure.**

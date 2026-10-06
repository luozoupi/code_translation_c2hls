# Slide brief: remaining supervisor asks (2026-09-02)

One kernel family. Same FPGA and clock. Finish times from synthesis.

**Do not overwrite** `docs/pc2/2026-08-18-autosa-mm-agent-vs-rank1-slide-brief.md` (4285 vs 4228). This file is the Aug 25 leftover: four columns, packs, ping-pong vs hide load/store, 320 DSP, wave1 2%.

Metric lock: **latency min, latency max, DSP**. Do not put interval on slides.
On slides say **compute rewrite** / **hide load/store**. Never “DSE”.

Quote: **4216** / **4292** / **40454** / **12893**. AutoSA rank-1: **4228 / 320**.

Use as slide copy: title + a few bullets. Speaker notes in italics.

---

## Slide 1 — Title

**Generic skills, enforcement, and systolic rewrite on AutoSA mm**

- Kernel: `autosa_mm`, 64³ float, U280, 3.33 ns, DeepSeek-v4-flash
- AutoSA rank-1: **4228 cycles, 320 DSP**
- Agent systolic: **4292** with-skills, **4216** no-skills JSON

*Speaker: Aug 18 already closed 4285 vs 4228. This meeting is the ablation: what each pack is worth, why enforcement is not a pass, why 320 DSP.*

---

## Slide 2 — Four columns (`autosa_mm`)

Csynth latency. Same ABI. Cosim off.

| | Design point | Cycles | DSP |
|---|---|---:|---:|
| 1 | No expertise (zero-shot `20260902_mmzs`) | **40454–42758** | 320 |
| 2 | Generic HLS skills (`20260830_mmflow` flash) | **139484** | 10 |
| 3 | Skills + enforcement (ping-pong judge) | **12893** | 320 |
| 4 | Application-specific systolic | **4292** with-skills / **4216** no-skills | 320 |
| | AutoSA rank-1 | **4228** | 320 |

*Speaker: column 2 is slower than column 1. Generic HLS skills without a PE recipe can hurt. Column 3 has the right DSP and a DATAFLOW shape; one-run finish is still load+compute+store. Column 4 is hide load/store on a 16×4 array.*

---

## Slide 3 — Pack waterfall (not per-skill)

Do not ablate skill 3 of 5. Report packs. With-skills mmflow unless noted.

```
  Flash (90-skill)              139484 / 10     legal, no PE array
       │
       ▼
  Compute rewrite               13160 / 352     workers exist; sequential I/O
       │
       │                        **8678 / 320**  Aug 23 seed; gate fail
       │                        mmflow re-run **2789637** LLM timed out
       ▼
  Stream hide load/store        4292 / 320      closes rank-1
```

*Speaker: flash skills can slow (139484 vs zero-shot 40454). Ping-pong is a coarse double-buffer, not systolic I/O. The 8678 cell is mm_gap 20260823, not frozen mmflow — say pending for the mmflow cell.*

---

## Slide 4 — Ping-pong ≠ hide load/store

**Enforcement (column 3): 12893 / 320**

- Judge asked for ping-pong DATAFLOW (II, partition, resources, double-buffer).
- Reason: already ping-pong DATAFLOW. Overlap gate **fails**.
- Mention 4553 only if asked: that is the reported interval, not the finish time. **12893 vs 4553 is not a pass.**

**Aug 23 overlap: 8678 / 320**

- Architecture gate fail. Not a stream pack.
- Rooms overlap at tile grain; each inner stretch still restarts.

**Stream hide load/store: 4292 / 320** (no-skills JSON **4216**)

- Load, 16 PEs, and store run together. One run ≈ the slowest room.

*Speaker: two folders on one worker is not AutoSA’s order. Systolic hide load/store is a different rewrite.*

---

## Slide 5 — Why 320 DSP

U280 is not filled. Rank-1 320 is **iso-compute: 16×4**.

| Array | Stream cycles | DSP | vs 4228 |
|-------|--------------:|----:|---------|
| 16×4 (locked) | **4292** | **320** | +1.5% |
| 32×8 overlay | **4583** | **1280** | worse |
| Family C iso-DSP 32-PE | **2245** | **1280** | misses cand 9 ≤1846; selected 1571/5120 oversubscribes |

*Speaker: 32×8 compute was 9782 / 1344. More DSP, I/PE=2, did not beat 4228. P&R and extra HBM later.*

---

## Slide 6 — Wave1 within 2% of rank-1

Gate = ×1.02. Selected vs rank-1.

| Bench | Agent | Rank-1 | Gate |
|-------|------:|-------:|------|
| hcl | 4294 | 4230 | **pass 1.51%** |
| hcl_intel | 4292 | 4226 | **pass 1.56%** |
| int16 | 4280 | 4219 | **pass 1.45%** |
| catapult | 8351 | 8286 | **pass 0.78%** |
| intel | **4525** | 4178 | **fail 8.3%** after retry |
| getting_started | **4525** / 640 | 2194 | **fail ~2×** after retry (I/PE=2) |

*Speaker: four already inside 2%. intel and getting_started stay outside after retry 2789605 (both 4525/640). cnn / lu / HBM are later waves.*

---

## Slide 7 — Takeaways

1. **Generic HLS skills can slow.** Flash 139484 / 10 is worse than zero-shot **40454**.
2. **Enforcement is not a systolic pass.** **12893** / 320; overlap fail.
3. **Hide load/store closes rank-1.** **4292** with-skills, **4216** no-skills, AutoSA **4228**, all 320 DSP.
4. **Ping-pong ≠ hide load/store.** 8678 (Aug 23) is coarse overlap; mmflow re-run timed out.
5. **320 DSP is 16×4 iso-compute.** 32×8 **4583 / 1280** is worse. Family C **2245 / 1280** misses cand 9 ≤1846; **1571 / 5120** is not this slide.

---

## Limits

- Csynth + csim. No cosim / bitstream / P&R.
- One model (DeepSeek-v4-flash). Frozen mmflow not relaunched.
- Disk still says `dse`; slides must not.

## Pointers

- Ablation table: `artifacts/pc2/autosa_mm_ablation_20260901/comparison.md`
- 320 DSP: `docs/pc2/2026-09-02-why-320-dsp.md`
- Wave1: `docs/pc2/2026-09-02-wave1-vs-rank1.md`
- One-pager: `docs/pc2/2026-08-30-autosa-agentic-flow-one-pager.md`

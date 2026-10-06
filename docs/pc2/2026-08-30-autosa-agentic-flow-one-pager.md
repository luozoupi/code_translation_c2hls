# AutoSA-ordered agentic flow (`autosa_mm`)

Supervisor note. One kernel. Finish times from synthesis, not a search story.

## What AutoSA is

Two jobs, then knobs:

1. **Many small calculators** that do not wait on one running total.
2. **Memory filling the next chapter while they work** — wide parcels; A private to each calculator; B passed along the line; C drained.

Knobs for this mm are fixed: **16 workers × 4-wide**. We do not hunt them.

## What the agent does

1. **Flash** — a correct engine (legal ports, same testbench, matching answers).
2. **Compute rewrite** — the many small calculators.
3. **I/O rewrite** — memory filling the next chapter while they work.

We **do not** start with two folders on one worker. That picture belongs after the calculators exist.

## Pass / fail

**Compute passes** when there is lots of silicon / many workers, even if the whole job is still load-then-compute-then-store. A ~3× AutoSA finish time at this step is expected. I/O is the next rewrite.

**I/O passes** when the finish time of one run ≈ the slowest room. If a second launch could start while this C is still being stored, the rooms are not overlapping.

## Measured path we already have

| Stage | Finish time | Meaning |
|-------|------------:|---------|
| Zero-shot (no skills) | **40454–42758** | One rewrite. Legal kernel, no PE array. |
| Flash (generic HLS skills) | **139484** | Legal; slower than zero-shot. 10 DSP. |
| Compute rewrite | **13160** | Workers are there; still load, then compute, then store. |
| Ping-pong overlap (Aug 23 seed) | **8678** | Rooms work together, but each inner stretch restarts. Gate fail. |
| Stream hide load/store (with-skills) | **4292** | One run ≈ the slowest room. +1.5% vs 4228. |
| Stream hide load/store (no-skills JSON) | **4216** | Same PE recipe; skills JSON empty. −0.3% vs 4228. |
| AutoSA | **4228** | Target. 320 DSP. |

## Not this flow

Enforcement-only **12893 vs 4553 is not a pass.** The last enforcement-only job put DATAFLOW and two folders on one worker. Finish time **12893** vs room time **4553**. That is not AutoSA’s order and is not a pass here.

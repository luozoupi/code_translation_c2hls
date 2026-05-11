# Nova benchmarks: direct sw_emu + hw_emu vs reference

Vitis 2023.2 / xcu280-fsvh2892-2L-e / 3.33 ns / no LLM

hw_emu measured on steps: ['baseline', 'coalescing']

## Per-variant sw_emu correctness

| bench | variant | sw_emu ours | sw_emu ref | match |
|---|---|:---:|:---:|:---:|
| lc_dilate | baseline | ✓ | pass | ✓ |
| lc_dilate | tiling | ✓ | pass | ✓ |
| lc_dilate | pipeline | ✓ | pass | ✓ |
| lc_dilate | pipeline | ✓ | pass | ✓ |
| lc_dilate | unroll | ✓ | pass | ✓ |
| lc_dilate | doublebuffer | ✓ | pass | ✓ |
| lc_dilate | coalescing | ✓ | pass | ✓ |
| lc_dilate | multiddr | ✓ | pass | ✓ |
| nw | baseline | ✓ | pass | ✓ |
| nw | tiling | ✓ | pass | ✓ |
| nw | pipeline | ✓ | pass | ✓ |
| nw | unroll | ✓ | pass | ✓ |
| nw | doublebuffer | ✓ | pass | ✓ |
| nw | coalescing | ✓ | pass | ✓ |

## hw_emu kernel runtime (subset)

| bench | variant | ours_us | ours_cycles | ref_us | ref_cycles | ref_status | ratio_us | ratio_cy |
|---|---|---:|---:|---:|---:|:---:|---:|---:|
| lc_dilate | baseline | 12072.491 | 3625372 | 12072.491 | 3621747 | pass | 1.000× | 1.001× |
| lc_dilate | coalescing | 102.034 | 30640 | 102.034 | 30610 | pass | 1.000× | 1.001× |
| nw | baseline | 97019.513 | 29134988 | 97019.513 | 29105854 | pass | 1.000× | 1.001× |
| nw | coalescing | 2066.173 | 620472 | — | — | fail | — | — |

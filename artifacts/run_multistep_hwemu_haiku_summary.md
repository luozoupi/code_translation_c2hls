# 4-bench multistep + hw_emu — claude-haiku-4-5-20251001

Vitis 2023.2 / xcu280-fsvh2892-2L-e / 3.33 ns / flow_target=vitis

Per-step AI vs GT (csynth) + final-stage hw_emu kernel runtime.

## pathfinder

  steps: 5/5 · elapsed: 10105.4s
  hw_emu: kernel_runtime_us=None cycles=None passed=False

| step | ok | gen_lat | gt_lat | ratio | csim |
|---|:---:|---:|---:|---:|:---:|
| tiling | ✓ | 7025000.0 | 10522000.0 | 0.67× | ✓ |
| pipeline | ✓ | 7025000.0 | — | — | ✓ |
| unroll | ✓ | 522000.0 | 7081000.0 | 0.07× | ✓ |
| doublebuffer | ✓ | 274000.0 | 3518000.0 | 0.08× | ✗ |
| coalescing | ✓ | — | 73560.0 | — | ✗ |

## knn

  steps: 4/5 · elapsed: 3715.9s
  hw_emu: kernel_runtime_us=192.759 cycles=57885 passed=False

| step | ok | gen_lat | gt_lat | ratio | csim |
|---|:---:|---:|---:|---:|:---:|
| tiling | ✗ | — | — | — | — |
| pipeline | ✓ | 3493000.0 | 14240000.0 | 0.25× | ✓ |
| unroll | ✓ | 3493000.0 | 13469000.0 | 0.26× | ✓ |
| doublebuffer | ✓ | 3493000.0 | 5796000.0 | 0.60× | ✓ |
| coalescing | ✓ | 3493000.0 | 874000.0 | 4.00× | ✓ |

## nw

  steps: 5/5 · elapsed: 5754.2s
  hw_emu: kernel_runtime_us=64463.457 cycles=19358395 passed=True

| step | ok | gen_lat | gt_lat | ratio | csim |
|---|:---:|---:|---:|---:|:---:|
| tiling | ✓ | — | — | — | ✓ |
| pipeline | ✓ | — | — | — | ✓ |
| unroll | ✓ | — | — | — | ✓ |
| doublebuffer | ✓ | — | — | — | ✓ |
| coalescing | ✓ | — | — | — | ✓ |

## lavaMD

  steps: 5/5 · elapsed: 3198.2s
  hw_emu: kernel_runtime_us=None cycles=None passed=False

| step | ok | gen_lat | gt_lat | ratio | csim |
|---|:---:|---:|---:|---:|:---:|
| tiling | ✓ | 35856000.0 | 57882000.0 | 0.62× | ✓ |
| pipeline | ✓ | 35856000.0 | 57380000.0 | 0.62× | ✓ |
| unroll | ✓ | 35856000.0 | 2675000.0 | 13.40× | ✓ |
| doublebuffer | ✓ | 19355000.0 | 1547000.0 | 12.51× | ✗ |
| coalescing | ✓ | 19355000.0 | 3033000.0 | 6.38× | ✗ |


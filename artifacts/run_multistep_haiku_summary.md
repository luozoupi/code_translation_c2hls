# Multistep agentic-workflow run — claude-haiku-4-5-20251001

Vitis 2023.2 / xcu280-fsvh2892-2L-e / 3.33 ns / flow_target=vitis

Per-step AI vs GT comparison at the same optimisation step.

## pathfinder

  Steps: 5/5 succeeded · elapsed: 4636.3s

| step | success | gen_lat_ns | gt_lat_ns | ratio | csim |
|---|:---:|---:|---:|---:|:---:|
| tiling | ✓ | 7022000.0 | 10522000.0 | 0.67× | ✓ |
| pipeline | ✓ | 7022000.0 | — | — | ✓ |
| unroll | ✓ | 4211000.0 | 7081000.0 | 0.59× | ✓ |
| doublebuffer | ✓ | 4211000.0 | 3518000.0 | 1.20× | ✓ |
| coalescing | ✓ | 4211000.0 | 73560.0 | 57.25× | ✓ |

## knn

  Steps: 5/5 succeeded · elapsed: 3136.9s

| step | success | gen_lat_ns | gt_lat_ns | ratio | csim |
|---|:---:|---:|---:|---:|:---:|
| tiling | ✓ | 3492000.0 | 14240000.0 | 0.25× | ✓ |
| pipeline | ✓ | 3492000.0 | 14240000.0 | 0.25× | ✓ |
| unroll | ✓ | 3492000.0 | 13469000.0 | 0.26× | ✓ |
| doublebuffer | ✓ | 3492000.0 | 5796000.0 | 0.60× | ✓ |
| coalescing | ✓ | 3492000.0 | 874000.0 | 4.00× | ✓ |

## nw

  Steps: 0/0 succeeded · elapsed: 227.2s

| step | success | gen_lat_ns | gt_lat_ns | ratio | csim |
|---|:---:|---:|---:|---:|:---:|

## lavaMD

  Steps: 3/5 succeeded · elapsed: 3642.8s

| step | success | gen_lat_ns | gt_lat_ns | ratio | csim |
|---|:---:|---:|---:|---:|:---:|
| tiling | ✓ | 36130000.0 | 57882000.0 | 0.62× | ✓ |
| pipeline | ✓ | 29337000.0 | 57380000.0 | 0.51× | ✓ |
| unroll | ✗ | — | — | — | — |
| doublebuffer | ✓ | 15985000.0 | 1547000.0 | 10.33× | ✓ |
| coalescing | ✗ | — | — | — | — |


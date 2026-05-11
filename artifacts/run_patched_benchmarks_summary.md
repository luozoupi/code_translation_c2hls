# Patched-bench rerun — claude-haiku-4-5-20251001

Vitis 2023.2 / xcu280-fsvh2892-2L-e / 3.33 ns / flow_target=vitis

Patches applied:
 - strip `#include "support.h"` from upstream-rewritten headers
 - regex-match `extern "C"` linkage detection (no-space variant)

| bench | phase | gen_lat_ns | gt_lat_ns | ratio | csim | sec |
|---|---|---:|---:|---:|:---:|---:|
| fft | complete | 125000.0 | 122000.0 | 1.02× | ✓ | 322.5 |
| sort_merge | complete | — | — | — | ✓ | 1237.9 |
| viterbi | reference | — | — | — | — | 1200.1 |
| hotspot | reference | — | 6920000.0 | — | — | 277.8 |

# Worst results — No avoids global (old `skills.json`)

**Mode:** `flash_all_skills_no_avoids_global`  
**Artifact:** `artifacts/pc2/flash_all_skills_no_avoids_global_20260620_113247`  
**Model:** `mistralai/Devstral-2-123B-Instruct-2512`  
**Overall:** 27/28 OK — champion on average, but several catastrophic per-bench regressions.

This mode injects **all positive skills** (no avoid-tier rules) from the legacy **55-skill** library. Worst cases fall into three buckets:

1. **Bench failure** — gold reference gate (`doitgen`)
2. **Massive regressions vs noskills** — skills prompt led the LLM to **drop** good optimizations
3. **Near-GT ties with poor timing** — correct structure but **II misses** and **global-memory** bottlenecks

Comparison baseline for regressions: `flash_noskills_20260620_004507`.

---

## Summary table

| Rank | Benchmark | Issue | Latency (cycles) | vs noskills | vs GT ratio | Fmax (MHz) |
|------|-----------|-------|------------------|-------------|-------------|------------|
| — | `doitgen` | **FAIL** (gold ref synth) | — | — | — | — |
| 1 | `3mm` | Naive global GEMM, no tiling | 45,441,119 | **457×** slower | 0.45 (faster than GT) | 342 |
| 2 | `heat-3d` | Global 3-D stencil, II=7 | 3,281,841 | **67×** slower | **1.00** (tie GT) | 411 |
| 3 | `correlation` | Lane unroll + in-place normalize | 8,030,247 | **4.4×** slower | 0.15 | **170** |
| 4 | `gesummv` | Moderate regression | 18,055 | 2.0× slower | — | — |
| 5 | `seidel-2d` | Moderate regression | 1,119,361 | 1.9× slower | — | — |
| 6 | `gramschmidt` | Scalar reference port | 2,270,081 | 1.8× slower | ~1.0 | — |

---

## 1. `doitgen` — bench failure (not a flash regression)

**Status:** FAIL — never reaches flash comparison.

**Error (reference gate):**

```
Gold HLS synthesis failed: ERROR: [SYNCHK 200-43] ... use or assignment of a non-static pointer
ERROR: [HLS 200-70] Synthesizability check failed.
```

**Why:** The benchmark gold HLS kernel fails Vitis synthesizability on PC2 in **every** flash mode. This is an infrastructure/benchmark issue, not specific to No avoids global.

---

## 2. `3mm` — worst regression (457× slower than noskills)

| Metric | No avoids global | Noskills |
|--------|------------------|----------|
| Latency | **45,441,119** | 99,467 |
| DSP | 14 | (lower) |
| Top bottleneck | II=75, II=161 on pipelined loops | Tiled local GEMM |

### Why

With the full skill library in context, the LLM produced a **literal PolyBench port**: three nested `i-j-k` loops reading/writing **global** `m_axi` arrays directly. Noskills (same run stamp) instead generated **local buffers**, **pipelined load/compute**, and a proper load–compute–store structure.

Synthesis reports **II target miss** (II=75, II=161) because the inner reduction loop cannot pipeline on off-chip memory.

### No avoids global (bad) — naive global GEMM

```cpp
for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
        E[i][j] = 0.0;
        for (k = 0; k < nk; ++k)
            E[i][j] += A[i][k] * B[k][j];  // global memory R/W each iteration
    }
// ... same pattern for F and G ...
```

**Source:** `hlsfactory_3mm/.../hlsfactory_3mm_final.cpp`

### Noskills (good) — local staging + pipeline

```cpp
double local_A[NI][NK];
double local_B[NK][NJ];
// ...
for (int i = 0; i < ni; i++) {
    for (int k = 0; k < nk; k++) {
#pragma HLS PIPELINE II=1
        local_A[i][k] = A[i][k];
    }
}
// Compute E = A * B on local buffers
for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
#pragma HLS PIPELINE II=1
        double sum = 0.0;
        for (int k = 0; k < nk; k++)
            sum += local_A[i][k] * local_B[k][j];
        local_E[i][j] = sum;
    }
}
```

**Lesson:** Dumping all skills can **overwhelm** the model; it sometimes **reverts to unoptimized reference code** instead of applying tiling/pipeline skills selectively.

---

## 3. `heat-3d` — 67× slower than noskills; ties ground truth

| Metric | No avoids global | Noskills |
|--------|------------------|----------|
| Latency | **3,281,841** | 48,677 |
| vs GT | **1.00** (tie) | much faster than GT |
| Bottleneck | II=7, port conflicts on 7-point stencil | Broken 2-D slice (fast but incorrect structure) |

### Why

No avoids global applies **coalescing pragmas** (`max_widen_bitwidth=512`) and pipelines the **inner `k` loop**, but the stencil still reads **seven global neighbors per iteration**:

```cpp
for (t = 1; t <= 40; t++) {          // hardcoded 40 (header default), not `tsteps`
    for (i = 1; i < n-1; i++) {
        for (j = 1; j < n-1; j++) {
        compute_b_row: for (k = 1; k < n-1; k++) {
            #pragma HLS PIPELINE II=1
            B[i][j][k] = 0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                       + ... /* 5 more global A reads */
                       + A[i][j][k];
        }}}
    // swap A/B phases similarly
}
```

**Synthesis:** II=7 (not II=1) — memory port conflicts on the 7-point stencil. Latency matches GT because the **gold kernel uses a similar full-grid global structure**; noskills got a much lower latency with a **simplified/wrong 2-D buffering** pattern that does not represent a good optimization target.

**Lesson:** Tying GT does not mean “good” — it can mean **matching a slow gold implementation**. Skills pushed coalescing without **on-chip tiling** or **time-slab buffering**.

---

## 4. `correlation` — 4.4× slower; timing collapse (Fmax 170 MHz)

| Metric | Value |
|--------|-------|
| Latency | 8,030,247 cycles |
| DSP | **685** |
| Fmax | **169.87 MHz** (worst-in-class) |
| Bottleneck | II=211, II=40, II=188 |

### Why

The LLM applied **512-bit coalescing** and **LANE unroll** (`LANES = 8`) across mean/stddev/normalize/correlation phases, including **in-place normalization** that rewrites `data[][]` in global memory:

```cpp
const int LANES = 512 / (8 * sizeof(double));
for (int j = 0; j < m; j++) {
    for (int i = 0; i < n; i += LANES) {
#pragma HLS PIPELINE II=1
        for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
            int idx = i + lane;
            if (idx < n)
                mean[j] += data[idx][j];
        }
    }
}
// ... later: in-place data[idx][j] -= mean[j]; data[idx][j] /= ...
```

Heavy **UNROLL + wide memory** increased DSP and routing pressure → **Fmax dropped to ~170 MHz**, and achieved II ≫ 1 on multiple loops.

**Lesson:** Skill recipes for coalescing/lane-parallelism were applied **without local staging**; the design is memory-bound and timing-poor.

---

## 5. Other moderate regressions

### `gramschmidt` (1.8× vs noskills)

Scalar triple-nested loops with **in-place `A` updates** on global memory — classic Gram-Schmidt port with no blocking:

```cpp
for (k = 0; k < n; k++) {
    for (i = 0; i < m; i++)
        Q[i][k] = A[i][k] / R[k][k];
    for (j = k + 1; j < n; j++) {
        for (i = 0; i < m; i++)
            A[i][j] = A[i][j] - Q[i][k] * R[k][j];  // global RMW on A
    }
}
```

### `durbin` (0.945× GT — “almost tied”, not faster)

Near-GT latency means this bench is **not a win** for this mode even though it is not a disaster.

---

## Common failure patterns (No avoids global, old skills)

| Pattern | Symptom | Example benches |
|---------|---------|-----------------|
| **Skill overload → reference code** | Drops tiling/pipeline; global nested loops | `3mm` |
| **Coalescing without staging** | II≫1, port conflicts | `heat-3d`, `correlation` |
| **Aggressive unroll + wide bus** | DSP explosion, low Fmax | `correlation` |
| **Matches slow GT** | ratio≈1.0 but still bad vs noskills | `heat-3d` |

---

## Artifact paths

| Benchmark | Final code |
|-----------|------------|
| `3mm` | `.../hlsfactory_3mm/devstral2__flash__all_skills_no_avoids_global/hlsfactory_3mm_final.cpp` |
| `heat-3d` | `.../hlsfactory_heat-3d/devstral2__flash__all_skills_no_avoids_global/hlsfactory_heat-3d_final.cpp` |
| `correlation` | `.../hlsfactory_correlation/devstral2__flash__all_skills_no_avoids_global/hlsfactory_correlation_final.cpp` |

Full root: `artifacts/pc2/flash_all_skills_no_avoids_global_20260620_113247/`

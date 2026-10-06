# Rank-1-shaped multi-PE then stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do **not** commit unless the user asks.

**Goal:** Four `autosa_mm` kernels that use AutoSA rank-1 I×K tiling (16×4), first without FIFOs then with FIFOs, each as a manual gold and an LLM-style twin; csim + csynth vs rank-1 4228/320.

**Architecture:** Official ABI and TB. On-chip this step is A 16×32, B 64×32, C 16×64. Crow lives across two K-tiles of one I-tile. No AutoSA `kernel0`. No writes into `20260830_mmflow`.

**Tech Stack:** C++ HLS, Vitis via `hls_eval`, U280 3.33 ns, official `autosa_mm` testbench.

**Spec:** `docs/superpowers/specs/2026-09-05-rank1-shaped-multipe-stream-design.md`

**Do not commit.**

---

## Files

| Path | Role |
|------|------|
| `artifacts/pc2/manual_rank1_shaped/kernel.h` | Copy of official header |
| `artifacts/pc2/manual_rank1_shaped/testbench.cpp` | Copy of official TB |
| `artifacts/pc2/manual_rank1_shaped/autosa_mm_rank1_pe.cpp` | Manual no-FIFO gold |
| `artifacts/pc2/manual_rank1_shaped/autosa_mm_rank1_pe_llm.cpp` | LLM-style no-FIFO |
| `artifacts/pc2/manual_rank1_shaped/autosa_mm_rank1_stream.cpp` | Manual FIFO gold |
| `artifacts/pc2/manual_rank1_shaped/autosa_mm_rank1_stream_llm.cpp` | LLM-style FIFO |
| `artifacts/pc2/manual_rank1_shaped/host_check_pe.sh` | g++ check for no-FIFO kernels |
| `artifacts/pc2/manual_rank1_shaped/run_csim_csynth.py` | Vitis csim+csynth one kernel |
| `artifacts/pc2/manual_rank1_shaped/comparison.md` | Numbers table after csynth |
| `c2hls_tmp/manual_rank1_shaped_*` | HLS work dirs |

---

### Task 1: Scaffold + failing host check

**Files:**
- Create: `artifacts/pc2/manual_rank1_shaped/kernel.h` (copy)
- Create: `artifacts/pc2/manual_rank1_shaped/testbench.cpp` (copy)
- Create: `artifacts/pc2/manual_rank1_shaped/autosa_mm_rank1_pe.cpp` (stub)
- Create: `artifacts/pc2/manual_rank1_shaped/host_check_pe.sh`

- [ ] **Step 1: Copy header and TB**

```bash
mkdir -p /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/manual_rank1_shaped
cp /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/related_work/benchmarks/autosa_ready/autosa_mm/kernel.h \
   /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/manual_rank1_shaped/kernel.h
cp /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/related_work/benchmarks/autosa_ready/autosa_mm/testbench.cpp \
   /scratch/hpc-prf-llmfpga/asa582/projects/c2hls/artifacts/pc2/manual_rank1_shaped/testbench.cpp
```

- [ ] **Step 2: Write stub kernel (does not compute)**

```cpp
#include "kernel.h"

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
  (void)A;
  (void)B;
  (void)C;
}
```

- [ ] **Step 3: Write host check**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:?cpp}"
g++ -O0 -I"$ROOT" -o "$ROOT/host_check.out" "$SRC" "$ROOT/testbench.cpp" -lm
"$ROOT/host_check.out"
```

chmod +x.

- [ ] **Step 4: Run host check on stub — expect FAIL**

```bash
bash artifacts/pc2/manual_rank1_shaped/host_check_pe.sh \
  artifacts/pc2/manual_rank1_shaped/autosa_mm_rank1_pe.cpp
```

Expected: `Failed with N errors!` (N > 0), non-zero or zero depending on printf-only TB (`return 0` even on fail). TB returns 0 always — check stdout contains `Failed`.

---

### Task 2: Manual no-FIFO gold (csim-correct)

**Files:**
- Modify: `artifacts/pc2/manual_rank1_shaped/autosa_mm_rank1_pe.cpp`

- [ ] **Step 1: Replace stub with rank-1-shaped ping-pong kernel**

Full file:

```cpp
#include "kernel.h"

#define PE 16
#define SIMD 4
#define KT 32

static void load_A_tile(data_t A[I][K], data_t A_buf[PE][KT], int ti, int tk) {
#pragma HLS INLINE off
  const int i0 = ti * PE;
  const int k0 = tk * KT;
load_A_p:
  for (int p = 0; p < PE; p++) {
  load_A_k:
    for (int k = 0; k < KT; k++) {
#pragma HLS PIPELINE II = 1
      A_buf[p][k] = A[i0 + p][k0 + k];
    }
  }
}

static void load_B_ktile(data_t B[J][K], data_t B_buf[J][KT], int tk) {
#pragma HLS INLINE off
  const int k0 = tk * KT;
load_B_j:
  for (int j = 0; j < J; j++) {
  load_B_k:
    for (int k = 0; k < KT; k++) {
#pragma HLS PIPELINE II = 1
      B_buf[j][k] = B[j][k0 + k];
    }
  }
}

static void compute_tile(data_t A_buf[PE][KT], data_t B_buf[J][KT],
                         data_t Crow[PE][J], int tk) {
#pragma HLS INLINE off
compute_k0:
  for (int k0 = 0; k0 < KT; k0 += SIMD) {
  compute_j:
    for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II = 1
    pe_mac:
      for (int p = 0; p < PE; p++) {
#pragma HLS UNROLL
        data_t partial = (data_t)0.0;
      simd_k:
        for (int s = 0; s < SIMD; s++) {
#pragma HLS UNROLL
          partial += A_buf[p][k0 + s] * B_buf[j][k0 + s];
        }
        if (tk == 0 && k0 == 0) {
          Crow[p][j] = partial;
        } else {
          Crow[p][j] += partial;
        }
      }
    }
  }
}

static void store_C_tile(data_t Crow[PE][J], data_t C[I][J], int ti) {
#pragma HLS INLINE off
  const int i0 = ti * PE;
store_p:
  for (int p = 0; p < PE; p++) {
  store_j:
    for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II = 1
      C[i0 + p][j] = Crow[p][j];
    }
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = C bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

tile_i:
  for (int ti = 0; ti < (I / PE); ti++) {
    data_t Crow[PE][J];
#pragma HLS ARRAY_PARTITION variable = Crow complete dim = 1
  tile_k:
    for (int tk = 0; tk < (K / KT); tk++) {
#pragma HLS DATAFLOW
      data_t A_buf[PE][KT];
      data_t B_buf[J][KT];
#pragma HLS ARRAY_PARTITION variable = A_buf complete dim = 1
#pragma HLS ARRAY_PARTITION variable = A_buf cyclic factor = SIMD dim = 2
#pragma HLS ARRAY_PARTITION variable = B_buf cyclic factor = SIMD dim = 2
      load_A_tile(A, A_buf, ti, tk);
      load_B_ktile(B, B_buf, tk);
      compute_tile(A_buf, B_buf, Crow, tk);
    }
    store_C_tile(Crow, C, ti);
  }
}
```

- [ ] **Step 2: Host check — expect `Passed!`**

```bash
bash artifacts/pc2/manual_rank1_shaped/host_check_pe.sh \
  artifacts/pc2/manual_rank1_shaped/autosa_mm_rank1_pe.cpp
```

Expected stdout: `Passed!`

---

### Task 3: LLM-style no-FIFO

**Files:**
- Create: `artifacts/pc2/manual_rank1_shaped/autosa_mm_rank1_pe_llm.cpp`

- [ ] **Step 1: Write fat nest (frozen dse style, rank-1 tiles)**

```cpp
#include "kernel.h"

#define PE 16
#define SIMD 4
#define KT 32

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t Crow[PE][J];
#pragma HLS ARRAY_PARTITION variable=Crow complete dim=1

    compute_ti:
    for (int ti = 0; ti < I; ti += PE) {
        compute_tk:
        for (int tk = 0; tk < K; tk += KT) {
            data_t A_buf[PE][KT];
            data_t B_buf[J][KT];
#pragma HLS ARRAY_PARTITION variable=A_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_buf cyclic factor=SIMD dim=2
#pragma HLS ARRAY_PARTITION variable=B_buf cyclic factor=SIMD dim=2

            load_A_p:
            for (int p = 0; p < PE; p++) {
                load_A_k:
                for (int k = 0; k < KT; k++) {
#pragma HLS PIPELINE II=1
                    A_buf[p][k] = A[ti + p][tk + k];
                }
            }

            load_B_j:
            for (int j = 0; j < J; j++) {
                load_B_k:
                for (int k = 0; k < KT; k++) {
#pragma HLS PIPELINE II=1
                    B_buf[j][k] = B[j][tk + k];
                }
            }

            compute_k0:
            for (int k0 = 0; k0 < KT; k0 += SIMD) {
                compute_j:
                for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
                    pe_mac:
                    for (int p = 0; p < PE; p++) {
#pragma HLS UNROLL
                        data_t partial = (data_t)0.0;
                        simd_k:
                        for (int s = 0; s < SIMD; s++) {
#pragma HLS UNROLL
                            partial += A_buf[p][k0 + s] * B_buf[j][k0 + s];
                        }
                        if (tk == 0 && k0 == 0) {
                            Crow[p][j] = partial;
                        } else {
                            Crow[p][j] += partial;
                        }
                    }
                }
            }
        }

        store_p:
        for (int p = 0; p < PE; p++) {
            store_j:
            for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
                C[ti + p][j] = Crow[p][j];
            }
        }
    }
}
```

- [ ] **Step 2: Host check — expect `Passed!`**

```bash
bash artifacts/pc2/manual_rank1_shaped/host_check_pe.sh \
  artifacts/pc2/manual_rank1_shaped/autosa_mm_rank1_pe_llm.cpp
```

---

### Task 4: Vitis runner + csim/csynth no-FIFO pair

**Files:**
- Create: `artifacts/pc2/manual_rank1_shaped/run_csim_csynth.py`

- [ ] **Step 1: Write runner** (clone `manual_mmflow_pe_pp/run_csim_csynth.py`; `ROOT` = this folder; argv kernel stem).

```python
#!/usr/bin/env python3
from __future__ import annotations
import json, os, shutil, sys
from pathlib import Path

REPO = Path("/scratch/hpc-prf-llmfpga/asa582/projects/c2hls")
sys.path.insert(0, str(REPO))
from c2hls_paths import apply_runtime_defaults, configure_site
configure_site("pc2")
apply_runtime_defaults()
import hls_eval

ROOT = REPO / "artifacts" / "pc2" / "manual_rank1_shaped"
stem = sys.argv[1] if len(sys.argv) > 1 else "autosa_mm_rank1_pe"
CPP = ROOT / f"{stem}.cpp"
TB = ROOT / "testbench.cpp"
HDR = ROOT / "kernel.h"
CSIM_WORK = REPO / "c2hls_tmp" / f"manual_rank1_shaped_{stem}_csim"
SYNTH_WORK = REPO / "c2hls_tmp" / f"manual_rank1_shaped_{stem}"
OUT = ROOT / f"csim_csynth_{stem}.json"

os.environ.setdefault("C2HLS_PART", "xcu280-fsvh2892-2L-e")
os.environ.setdefault("C2HLS_CLOCK_NS", "3.33")
os.environ.setdefault("C2HLS_SYNTH_TIMEOUT", "14400")

code = CPP.read_text(encoding="utf-8")
header = HDR.read_text(encoding="utf-8")
tb = TB.read_text(encoding="utf-8")

def _wipe(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

_wipe(CSIM_WORK)
csim = hls_eval.run_csim(
    code, tb, header, header_name="kernel.h", top_function="autosa_mm",
    part="xcu280-fsvh2892-2L-e", clock_ns=3.33, work_dir=str(CSIM_WORK),
)
csim_ok = bool(csim.get("success") and csim.get("passed"))
summary = {
    "stem": stem,
    "csim_success": csim_ok,
    "csim_error": (csim.get("error") or "")[:2000],
    "csim_work_dir": str(CSIM_WORK),
    "csynth_success": False,
    "latency_cycles": None,
    "latency_cycles_worst": None,
    "dsp": None, "bram": None, "ff": None, "lut": None, "interval": None,
    "csynth_error": "",
    "csynth_work_dir": str(SYNTH_WORK),
}
if not csim_ok:
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    raise SystemExit(1)
_wipe(SYNTH_WORK)
out = hls_eval.run_hls_synthesis(
    code, header, header_name="kernel.h", top_function="autosa_mm",
    part="xcu280-fsvh2892-2L-e", clock_ns=3.33, work_dir=str(SYNTH_WORK),
)
report = out.get("report") or {}
summary.update({
    "csynth_success": bool(out.get("success")),
    "csynth_error": (out.get("error") or "")[:2000],
    "latency_cycles": report.get("latency_cycles"),
    "latency_cycles_worst": report.get("latency_cycles_worst"),
    "interval": report.get("interval"),
    "dsp": report.get("dsp"),
    "bram": report.get("bram"),
    "ff": report.get("ff"),
    "lut": report.get("lut"),
    "csynth_work_dir": out.get("work_dir") or str(SYNTH_WORK),
})
OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
raise SystemExit(0 if summary["csynth_success"] else 1)
```

- [ ] **Step 2: Csim+csynth both no-FIFO kernels** (long; U280 3.33 ns)

```bash
cd /scratch/hpc-prf-llmfpga/asa582/projects/c2hls
python3 artifacts/pc2/manual_rank1_shaped/run_csim_csynth.py autosa_mm_rank1_pe
python3 artifacts/pc2/manual_rank1_shaped/run_csim_csynth.py autosa_mm_rank1_pe_llm
```

Expected: `csim_success: true`, `csynth_success: true`, DSP near 320, JSON written. If DATAFLOW + Crow fails HLS, drop inner DATAFLOW on the manual gold (keep tiling); re-csim.

---

### Task 5: Manual stream gold

**Files:**
- Create: `artifacts/pc2/manual_rank1_shaped/autosa_mm_rank1_stream.cpp`

Rank-1 PE reads A and B every beat. `load_A` repeats the 16×4 A vector across j. `load_B` replays each K-slab once per I-tile. Drain C only on last k-group of `tk == 1` (`k0 == KT - SIMD`).

- [ ] **Step 1: Write stream kernel**

```cpp
#include "kernel.h"
#include <hls_stream.h>
#include <ap_int.h>

#define PE_NUM 16
#define SIMD 4
#define KT 32

typedef ap_uint<128> vec4_bits;

static vec4_bits pack4(data_t v0, data_t v1, data_t v2, data_t v3) {
#pragma HLS INLINE
  union { unsigned int u; float f; } c0, c1, c2, c3;
  c0.f = (float)v0; c1.f = (float)v1; c2.f = (float)v2; c3.f = (float)v3;
  vec4_bits w;
  w.range(31, 0) = (ap_uint<32>)c0.u;
  w.range(63, 32) = (ap_uint<32>)c1.u;
  w.range(95, 64) = (ap_uint<32>)c2.u;
  w.range(127, 96) = (ap_uint<32>)c3.u;
  return w;
}

static void unpack4(vec4_bits w, data_t &v0, data_t &v1, data_t &v2, data_t &v3) {
#pragma HLS INLINE
  union { unsigned int u; float f; } c0, c1, c2, c3;
  c0.u = (unsigned int)w.range(31, 0);
  c1.u = (unsigned int)w.range(63, 32);
  c2.u = (unsigned int)w.range(95, 64);
  c3.u = (unsigned int)w.range(127, 96);
  v0 = (data_t)c0.f; v1 = (data_t)c1.f; v2 = (data_t)c2.f; v3 = (data_t)c3.f;
}

static void load_A(data_t A[I][K], hls::stream<vec4_bits> fifo_A[PE_NUM]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = fifo_A complete dim = 1
load_ti:
  for (int ti = 0; ti < I; ti += PE_NUM) {
  load_tk:
    for (int tk = 0; tk < K; tk += KT) {
    load_k0:
      for (int k0 = 0; k0 < KT; k0 += SIMD) {
      load_j:
        for (int j = 0; j < J; j++) {
        load_p:
          for (int p = 0; p < PE_NUM; p++) {
#pragma HLS PIPELINE II = 1
            fifo_A[p].write(pack4(
                A[ti + p][tk + k0 + 0], A[ti + p][tk + k0 + 1],
                A[ti + p][tk + k0 + 2], A[ti + p][tk + k0 + 3]));
          }
        }
      }
    }
  }
}

static void load_B(data_t B[J][K], hls::stream<vec4_bits> &fifo_B) {
#pragma HLS INLINE off
load_ti:
  for (int ti = 0; ti < I; ti += PE_NUM) {
  load_tk:
    for (int tk = 0; tk < K; tk += KT) {
    load_k0:
      for (int k0 = 0; k0 < KT; k0 += SIMD) {
      load_j:
        for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II = 1
          fifo_B.write(pack4(B[j][tk + k0 + 0], B[j][tk + k0 + 1],
                             B[j][tk + k0 + 2], B[j][tk + k0 + 3]));
        }
      }
    }
  }
}

static void mm_pe(hls::stream<vec4_bits> &fifo_A, hls::stream<vec4_bits> &fifo_B_in,
                  hls::stream<vec4_bits> &fifo_B_out, hls::stream<data_t> &fifo_C) {
#pragma HLS INLINE off
  data_t Crow[J];
#pragma HLS BIND_STORAGE variable = Crow type = ram_2p impl = bram
pe_ti:
  for (int ti = 0; ti < I; ti += PE_NUM) {
  pe_tk:
    for (int tk = 0; tk < K; tk += KT) {
    pe_k0:
      for (int k0 = 0; k0 < KT; k0 += SIMD) {
      pe_j:
        for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II = 1
          data_t a0, a1, a2, a3, b0, b1, b2, b3;
          unpack4(fifo_A.read(), a0, a1, a2, a3);
          vec4_bits bw = fifo_B_in.read();
          fifo_B_out.write(bw);
          unpack4(bw, b0, b1, b2, b3);
          data_t partial = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
          data_t acc = (tk == 0 && k0 == 0) ? partial : Crow[j] + partial;
          Crow[j] = acc;
          if (tk == (K - KT) && k0 == (KT - SIMD)) {
            fifo_C.write(acc);
          }
        }
      }
    }
  }
}

static void drain_B(hls::stream<vec4_bits> &fifo_B) {
#pragma HLS INLINE off
  const int n = (I / PE_NUM) * (K / KT) * (KT / SIMD) * J;
drain:
  for (int t = 0; t < n; t++) {
#pragma HLS PIPELINE II = 1
    (void)fifo_B.read();
  }
}

static void store_C(data_t C[I][J], hls::stream<data_t> fifo_C[PE_NUM]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = fifo_C complete dim = 1
store_ti:
  for (int ti = 0; ti < I; ti += PE_NUM) {
  store_p:
    for (int p = 0; p < PE_NUM; p++) {
    store_j:
      for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II = 1
        C[ti + p][j] = fifo_C[p].read();
      }
    }
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem2
#pragma HLS INTERFACE s_axilite port = A bundle = control
#pragma HLS INTERFACE s_axilite port = B bundle = control
#pragma HLS INTERFACE s_axilite port = C bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  hls::stream<vec4_bits> fifo_A[PE_NUM];
  hls::stream<vec4_bits> fifo_B[PE_NUM + 1];
  hls::stream<data_t> fifo_C[PE_NUM];
#pragma HLS STREAM variable = fifo_A depth = 16
#pragma HLS STREAM variable = fifo_B depth = 16
#pragma HLS STREAM variable = fifo_C depth = 64
#pragma HLS ARRAY_PARTITION variable = fifo_A complete dim = 1
#pragma HLS ARRAY_PARTITION variable = fifo_B complete dim = 1
#pragma HLS ARRAY_PARTITION variable = fifo_C complete dim = 1
#pragma HLS DATAFLOW
  load_A(A, fifo_A);
  load_B(B, fifo_B[0]);
  mm_pe(fifo_A[0], fifo_B[0], fifo_B[1], fifo_C[0]);
  mm_pe(fifo_A[1], fifo_B[1], fifo_B[2], fifo_C[1]);
  mm_pe(fifo_A[2], fifo_B[2], fifo_B[3], fifo_C[2]);
  mm_pe(fifo_A[3], fifo_B[3], fifo_B[4], fifo_C[3]);
  mm_pe(fifo_A[4], fifo_B[4], fifo_B[5], fifo_C[4]);
  mm_pe(fifo_A[5], fifo_B[5], fifo_B[6], fifo_C[5]);
  mm_pe(fifo_A[6], fifo_B[6], fifo_B[7], fifo_C[6]);
  mm_pe(fifo_A[7], fifo_B[7], fifo_B[8], fifo_C[7]);
  mm_pe(fifo_A[8], fifo_B[8], fifo_B[9], fifo_C[8]);
  mm_pe(fifo_A[9], fifo_B[9], fifo_B[10], fifo_C[9]);
  mm_pe(fifo_A[10], fifo_B[10], fifo_B[11], fifo_C[10]);
  mm_pe(fifo_A[11], fifo_B[11], fifo_B[12], fifo_C[11]);
  mm_pe(fifo_A[12], fifo_B[12], fifo_B[13], fifo_C[12]);
  mm_pe(fifo_A[13], fifo_B[13], fifo_B[14], fifo_C[13]);
  mm_pe(fifo_A[14], fifo_B[14], fifo_B[15], fifo_C[14]);
  mm_pe(fifo_A[15], fifo_B[15], fifo_B[16], fifo_C[15]);
  drain_B(fifo_B[16]);
  store_C(C, fifo_C);
}
```

- [ ] **Step 2: Vitis csim+csynth** (no g++; needs `hls_stream.h`)

```bash
python3 artifacts/pc2/manual_rank1_shaped/run_csim_csynth.py autosa_mm_rank1_stream
```

Expected: `Passed!` in csim, JSON with DSP 320 class.

---

### Task 6: LLM-style stream

**Files:**
- Create: `artifacts/pc2/manual_rank1_shaped/autosa_mm_rank1_stream_llm.cpp`

- [ ] **Step 1: Copy Task 5 kernel.** Change only comments and pack helpers inlined into `load_*` / `mm_pe` so it reads like frozen stream (one file, same loops). Keep identical trip counts and drain condition. Do not emit AutoSA `kernel0`.

- [ ] **Step 2: Csim+csynth**

```bash
python3 artifacts/pc2/manual_rank1_shaped/run_csim_csynth.py autosa_mm_rank1_stream_llm
```

Expected: same csim pass; latency/DSP within noise of Task 5 (same nest).

---

### Task 7: Comparison table

**Files:**
- Create: `artifacts/pc2/manual_rank1_shaped/comparison.md`

- [ ] **Step 1: Fill from JSON** (do not edit Aug 18 / Sep 2 slide briefs)

```markdown
# Rank-1-shaped kernels vs AutoSA rank-1

U280 3.33 ns. Latency min/max + DSP. LLM-style files are hand-written twins, not a DeepSeek run.

| Kernel | Cycles | DSP | FIFOs | Notes |
|--------|-------:|----:|-------|-------|
| AutoSA rank-1 | 4228 | 320 | yes | locked kernel0 |
| Frozen mmflow multi-PE | 13160 | 352 | no | old: all of B, no K-tile |
| Frozen mmflow stream | 4292 | 320 | yes | old: all of B once |
| Manual I-tile ping-pong | 9640 | 352 | no | all of B resident |
| rank1_pe (manual) |  |  | no | this folder |
| rank1_pe_llm |  |  | no | this folder |
| rank1_stream (manual) |  |  | yes | this folder |
| rank1_stream_llm |  |  | yes | this folder |
```

Paste real numbers from `csim_csynth_*.json`.

---

## Self-review

- Spec coverage: four kernels, official TB, no C DRAM load, I×K tiles, ping-pong DATAFLOW, stream A-every-beat, no mmflow overwrite, no commit.
- Placeholders: none.
- Names: `ti`/`tk`/`KT`/`Crow` consistent.

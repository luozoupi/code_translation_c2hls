# Why “enforcement ping-pong DATAFLOW” is 12893 / 320, not ~stream

**For:** Ahmad → Professor Fang  
**Date:** 2026-09-08  
**Question:** if it were real ping-pong DATAFLOW, one-call latency should be close to stream hide-load-store. Stream is **4216–4292 / 320 DSP**. Enforcement `20260829_123043` is **12893 / 320 DSP** (~3–4×). Why?

Metrics: **latency min/max and DSP only**. Do not quote interval on slides. Do not mix the spend-DSP champion into this comparison.

---

## Numbers (all Vitis HLS 2023.2, U280, 3.33 ns, `I=J=K=64` float)

| Design | Latency | DSP | What it actually is |
|---|---:|---:|---|
| AutoSA rank-1 | **4228** | 320 | systolic I/O (reference) |
| Frozen mmflow stream | **4292** | 320 | 16 PE FIFO chain, packed `ap_uint<128>` |
| No-skills stream `20260902_mmns` | **4216** | 320 | same architecture class |
| Compute rewrite (mmflow) | **13160** | 352 | serial load / compute / store |
| Enforcement “ping-pong DATAFLOW” `20260829_123043` | **12893** | 320 | DATAFLOW-shaped source; **one-call latency still LCST** |
| Manual `pe_pp` (buffers **inside** tile DATAFLOW loop) | **9640** | 352 | real coarse A-tile overlap; B loaded first |
| Manual `pe_pp_plus` (same + 512-bit + flattened `pe_kj`) | **4688** | 352 | working bulk ping-pong; **still above 4228** |

---

## 1. What the enforcement kernel actually is

**Selected C++**  
`artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp`

**Csynth**  
`c2hls_tmp/batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043/autosa_mm/hls_synth__step_synth/hls_proj/sol1/syn/report/autosa_mm_csynth.rpt`

**Judge** (`autosa_mm_enforcement.json`): `passed: true`, reason **“ping-pong DATAFLOW with interval < latency”**, `code_intent.dataflow=true`, `ping_pong=true`, latency **12893**, DSP **320**.

Yes, there are `load_B` / `load_tiles` / `compute_tiles` / `store_tiles` (`#pragma HLS INLINE off`), `#pragma HLS DATAFLOW`, and `A_tile[2]`, `C_tile[2]`. The ping-pong arrays are declared **in the kernel body, outside any loop**, immediately **before** DATAFLOW. There is **no tile loop around DATAFLOW**. HLS **214-397 did not fire** (that warning is for ping-pong arrays declared *outside a DATAFLOW loop*; here there is no DATAFLOW loop at all).

```67:86:artifacts/pc2/batch_parallel_autosa_mm_enf_aav_n_gf_20260829_123043/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_selected.cpp
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
    // ... m_axi gmem0/1/2 ...
    data_t B_local[J][K];
#pragma HLS ARRAY_PARTITION variable=B_local complete dim=2
    data_t A_tile[2][TI][K];
#pragma HLS ARRAY_PARTITION variable=A_tile complete dim=3
    data_t C_tile[2][TI][J];

#pragma HLS DATAFLOW
    load_B(&B[0][0], B_local);
    load_tiles(&A[0][0], A_tile);
    compute_tiles(A_tile, B_local, C_tile);
    store_tiles(&C[0][0], C_tile);
}
```

Each task has **its own** `for (t = 0; t < NT; ++t)` with `ping = t & 1` (`NT = I/TI = 4`). DATAFLOW sees **four functions invoked once** for the whole 64³. Ping-pong of **one** kernel firing is not overlap of successive tiles.

`compute_tiles` is not a 16-PE systolic array. It pipelines `(t,i,j)` and **fully unrolls `k`**, one `(i,j)` per cycle, 64-wide adder tree (trip 4096, depth 456 → 4550). DSP **320** is 64 float MACs, not 16 PE × SIMD 4.

---

## 2. Classic failure: DATAFLOW of one GEMM, not overlapping tiles

This is `load(); compute(); store();` of **one** full matrix multiply, with a `[2]` index that is never the induction variable of a DATAFLOW-region loop.

HLS implements `A_tile` as a **channel of one token** (`ap_channel_done_A_tile*` in the verbose report). `compute_tiles` waits until `load_tiles` has finished **all four tiles**. Memories are `RAM_AUTO_1R1W` (and `C_tile` is one URAM of 2048 words). Ping and pong are not concurrently accessible banks.

**Pre-repair FLASH** (same campaign, `hls_synth__flash_synth`): already **12893**, pipeline type **no**, `load_AB_flat` 4170 + `compute_C_flat` 4551 + `store_C_flat` 4167. After the “DATAFLOW + buf[2]” rewrite, **one-call latency is unchanged**.

---

## 3. Csynth: processes exist; one-call latency is still the sum

From `hls_synth__step_synth` `autosa_mm_csynth.rpt`:

```
Latency min/max: 12893 / 12893    Pipeline Type: dataflow
DSP Total: 320

Instance:
  load_B         4170
  load_tiles     4170
  compute_tiles   4552
  store_tiles     4169
```

`4170 + 4552 + 4169 = 12891` ≈ **12893**. `load_B` and `load_tiles` can start together (different AXI bundles). Then compute runs after **both full-matrix loads**. Then store. That is serial LCST of **one** call.

The judge treated **interval 4553 < 12893** as “overlap.” 4553 is the initiation interval of **another kernel invocation** (≈ `compute_tiles`). It is not hiding load/store inside **this** 64³. Do not put that interval on a slide.

`compute_tiles` itself (no inner DATAFLOW):

```
Latency 4552, Type no
Loop compute_tiles_tile_... : trip 4096, II=1, depth 456
```

---

## 4. Stream is systolic I/O, not three bulk rooms

**Frozen mmflow stream C++**  
`artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_stream.cpp`

**Csynth**  
`c2hls_tmp/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/autosa_mm/hls_synth__stream_flash_final_a0_synth/hls_proj/sol1/syn/report/autosa_mm_csynth.rpt`

**JSON:** `autosa_mm_stream_result.json` → **4292 / 320**.

Packed A load (`ap_uint<128>`) and flattened `pe_kj` (A private, B forwarded, C drained):

```38:53:artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_stream.cpp
static void load_A(data_t A[I][K], hls::stream<vec4_bits> fifo_A[PE_NUM]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=fifo_A complete dim=1
    load_i0: for (int i0 = 0; i0 < I; i0 += PE_NUM) {
        load_k0: for (int k0 = 0; k0 < K; k0 += SIMD) {
            load_p: for (int p = 0; p < PE_NUM; ++p) {
#pragma HLS PIPELINE II=1
                fifo_A[p].write(pack4(
                    A[i0 + p][k0 + 0],
                    A[i0 + p][k0 + 1],
                    A[i0 + p][k0 + 2],
                    A[i0 + p][k0 + 3]));
            }
        }
    }
}
```

```76:106:artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_stream.cpp
    pe_i0: for (int i0 = 0; i0 < I; i0 += PE_NUM) {
        data_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        pe_kj: for (int t = 0; t < (K / SIMD) * J; ++t) {
#pragma HLS PIPELINE II=1
            const int k_tile = t / J;
            const int j = t - k_tile * J;

            if (j == 0) {
                unpack4(fifo_A.read(), a0, a1, a2, a3);
            }

            vec4_bits bw = fifo_B_in.read();
            fifo_B_out.write(bw);

            data_t b0, b1, b2, b3;
            unpack4(bw, b0, b1, b2, b3);

            data_t partial = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            data_t acc;
            if (k_tile == 0) {
                acc = partial;
            } else {
                acc = Crow[j] + partial;
            }
            Crow[j] = acc;

            if (k_tile == (K / SIMD) - 1) {
                fifo_C.write(acc);
            }
        }
    }
```

Top DATAFLOW is sixteen explicit `mm_pe` tasks, not three bulk rooms:

```154:177:artifacts/pc2/batch_parallel_autosa_mm_flow_aav_n_gf_20260830_mmflow/variants/autosa_aav_n_gf/autosa_mm/deepseek-v4-flash__flash__autosa__aav_n_gf/autosa_mm_stream.cpp
#pragma HLS DATAFLOW
    load_A(A, fifo_A);
    load_B(B, fifo_B[0]);
    mm_pe(fifo_A[0],  fifo_B[0],  fifo_B[1],  fifo_C[0]);
    // ... mm_pe[1] .. mm_pe[14] ...
    mm_pe(fifo_A[15], fifo_B[15], fifo_B[16], fifo_C[15]);
    drain_B(fifo_B[16]);
    store_C(C, fifo_C);
}
```

This is **16 concurrent PE processes** plus DRAM loaders and a C drain, connected by FIFOs. A is private per PE; B is systolic-forwarded; C is streamed out. `pe_kj` is flattened. It is **not** LCST ping-pong of three on-chip rooms.

Csynth (4292 / 320 DSP, type **dataflow**):

```
load_A     1100
load_B     4172
mm_pe[0..15]  4131 each   (pe_i0_pe_kj trip 4096, II=1, depth 35)
drain_B    4098
store_C    4168
Kernel     4292
```

One-call latency **4292** is within ~120 cycles of `load_B` (4172) and of each PE (4131) and of `store_C` (4168). The processes **overlap at beat granularity**. FIFO resources are real (`FIFO` 15332 FF / 9227 LUT in the utilization table). Kernel latency ≈ the long process, not the sum.

No-skills `20260902_mmns` is the same PE/FIFO skeleton (`PE_NUM=16`, `SIMD=4`, `ap_uint<128>`, `KJ_TOTAL`, 16× `mm_pe`) at **4216 / 320**.

---

## 5. Counterfactual Fang expects — and why stream is still faster

**Compute rewrite** (`autosa_mm_dse.cpp` in frozen mmflow; say “compute rewrite”, not the internal stage name):

```
Kernel 13160, Type no, DSP 352
load_A 4099   load_B 4099   load_C 4099
compute_i0   4816   (4 i-tiles × 1204, not pipelined)
store_C 4101
```

Serial LCST of the three rooms: **~4100 + 4816 + 4100 ≈ 13160** (report total 13160; 4099+4816+4101=13016 plus a small control/AXI remainder).

If that nest had **working** ping-pong of those three rooms, one-call latency would approach

**max(4100, 4816, 4100) ≈ 4816**

That is **still above AutoSA 4228** and above stream 4216–4292.

Manual proof that coarse ping-pong is not stream:

- `artifacts/pc2/manual_mmflow_pe_pp/autosa_mm_pe_pp.cpp` — DATAFLOW **inside** `for (t = 0; t < I/PE; t++)`, `A_buf` / `C_buf` declared **inside** the loop (legal HLS ping-pong; avoids 214-397). **9640 / 352**. Csynth: `load_B` 4171 **then** `dataflow_parent_loop_proc` 5465. Per tile, load_A 1098 / compute 1066 / store 1096 overlap (`dataflow_in_loop` latency 2165). B is still a serial prefix. 4171+5465 ≈ 9640.
- `autosa_mm_pe_pp_plus.cpp` — same structure + 512-bit DRAM + flattened `pe_kj`. **4688 / 352**. `load_B` 299 + tile DATAFLOW 4385. Per tile compute **1068**, load 106, store 104. **4688 is in the 4816 ballpark**, as Fang’s counterfactual says, and **still slower than stream 4292**.

Stream is faster than even *working* bulk ping-pong because the **architecture** is different: fine-grain PE FIFOs + DRAM overlap at **beat** granularity, not because of a magic `hls::stream` keyword.

---

## 6. What the judge checked vs what Fang means

The judge scored **syntax** (`#pragma HLS DATAFLOW` + `buf[2]`) and **interval < latency**. It did not check that successive **tiles of this GEMM** overlap, that ping-pong arrays are declared **inside** a DATAFLOW loop, or that one-call latency ≈ max(modules).

So Fang is right: this is **probably not ping-pong DATAFLOW** in the expert sense. It is DATAFLOW-shaped source that did not implement overlapping tiled ping-pong. Column 12893 is a failed stand-in for hide-load-store, not a success.

---

## Oral answer (say this)

The 12893 kernel has `load` / `compute` / `store` functions, a DATAFLOW pragma, and `buf[2]`, so a syntax judge marked ping-pong DATAFLOW present. The ping-pong arrays sit outside the DATAFLOW region, and DATAFLOW wraps one call of load-compute-store for the whole 64³ — there is no tile loop around DATAFLOW, so ping-pong of one iteration cannot overlap. Csynth still has four processes, but one-call latency is 4170 + 4552 + 4169 ≈ 12893, the same serial sum as the pre-repair FLASH kernel; the tool only overlapped the two full-matrix loads, then compute, then store. Stream 4216–4292 is a different machine: a 16-PE chain with `hls::stream`, packed 128-bit beats, A private per PE, B forwarded, C drained, flattened `pe_kj`, and csynth shows those processes running together so kernel latency sits next to load_B / PE / store (~4100–4170), not their sum. Even *working* bulk ping-pong of the 13160 nest would only get you to about max(4100, 4816, 4100) ≈ 4816, and the manual tile-DATAFLOW kernels landed at 9640 and 4688 — still above 4228. Stream is ~4× because it hides DRAM behind the PEs at beat granularity; the enforcement design never did that, it only looked like DATAFLOW.

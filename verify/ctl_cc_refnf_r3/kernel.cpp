#include "kernel.h"
#include <hls_stream.h>
#include <hls_vector.h>

// ---------------- Tiling / datapath parameters ----------------
#define TI 128                    // output tile rows (rows of A)
#define TJ 128                    // output tile cols (rows of B)
#define PI 8                      // PE rows per cycle
#define PJ 8                      // PE cols per cycle
#define KL 16                     // k-lanes per cycle (dot-product width)
#define LANES 16                  // floats per 512-bit beat
#define NKC (K / KL)              // 64 k-chunks
#define NIB (TI / PI)             // 16 row sub-blocks
#define NJB (TJ / PJ)             // 16 col sub-blocks
#define NSB (NIB * NJB)           // 256 sub-blocks per k-chunk
#define NTI (I / TI)              // 8
#define NTJ (J / TJ)              // 8
#define NT (NTI * NTJ)            // 64 output tiles
#define KB (K / LANES)            // 64 beats per row
#define TILE_BEATS (TJ * KB)      // 8192 beats per B tile
#define CB (TJ / LANES)           // 8 beats per C tile row

typedef hls::vector<data_t, LANES> vec_t;

// ---------------- B streaming loader (gmem1) ----------------
static void b_loader(data_t B[J][K], hls::stream<vec_t> &bs) {
b_tile:
  for (int t = 0; t < NT; t++) {
    const int j0 = (t % NTJ) * TJ;
  b_row:
    for (int r = 0; r < TJ; r++) {
    b_beat:
      for (int kb = 0; kb < KB; kb++) {
#pragma HLS PIPELINE II=1
        vec_t v;
      b_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          v[l] = B[j0 + r][kb * LANES + l];
        }
        bs.write(v);
      }
    }
  }
}

// ---------------- 16-wide balanced dot product ----------------
static data_t dot16(const data_t a[KL], const data_t b[KL]) {
#pragma HLS INLINE
  data_t p[16];
#pragma HLS ARRAY_PARTITION variable=p complete
  data_t s8[8];
#pragma HLS ARRAY_PARTITION variable=s8 complete
  data_t s4[4];
#pragma HLS ARRAY_PARTITION variable=s4 complete
  data_t s2[2];
#pragma HLS ARRAY_PARTITION variable=s2 complete
dot_mul:
  for (int q = 0; q < 16; q++) {
#pragma HLS UNROLL
    p[q] = a[q] * b[q];
  }
dot_add8:
  for (int q = 0; q < 8; q++) {
#pragma HLS UNROLL
    s8[q] = p[2 * q] + p[2 * q + 1];
  }
dot_add4:
  for (int q = 0; q < 4; q++) {
#pragma HLS UNROLL
    s4[q] = s8[2 * q] + s8[2 * q + 1];
  }
dot_add2:
  for (int q = 0; q < 2; q++) {
#pragma HLS UNROLL
    s2[q] = s4[2 * q] + s4[2 * q + 1];
  }
  return s2[0] + s2[1];
}

// ---------------- Compute task (owns A: gmem0, C: gmem2) ----------------
static void compute_task(data_t A[I][K], data_t C[I][J], hls::stream<vec_t> &bs) {
  // row = ib*PI + p, k = kc*KL + q
  data_t Abuf[PI][KL][NIB][NKC];
#pragma HLS ARRAY_PARTITION variable=Abuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=Abuf complete dim=2
  data_t B0[PJ][KL][NJB][NKC];
#pragma HLS ARRAY_PARTITION variable=B0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=B0 complete dim=2
  data_t B1[PJ][KL][NJB][NKC];
#pragma HLS ARRAY_PARTITION variable=B1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=B1 complete dim=2
  // row = ib*PI + p, col = jb*PJ + c
  data_t acc[PI][PJ][NIB][NJB];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

  // Prefetch the first B tile into B0
prefetch_b0:
  for (int it = 0; it < TILE_BEATS; it++) {
#pragma HLS PIPELINE II=1
    vec_t v = bs.read();
    const int r = it / KB;
    const int kb = it % KB;
  prefetch_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      B0[r % PJ][l][r / PJ][kb] = v[l];
    }
  }

tile_loop:
  for (int t = 0; t < NT; t++) {
    const int ti = t / NTJ;
    const int tj = t % NTJ;
    const int i0 = ti * TI;
    const int j0 = tj * TJ;
    const bool cur = (t & 1);          // 0: compute from B0, 1: compute from B1
    const bool has_next = (t + 1 < NT);

    // Load A row block once per ti (reused across all tj)
    if (tj == 0) {
    load_a_row:
      for (int r = 0; r < TI; r++) {
      load_a_beat:
        for (int kb = 0; kb < KB; kb++) {
#pragma HLS PIPELINE II=1
        load_a_lane:
          for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            Abuf[r % PI][l][r / PI][kb] = A[i0 + r][kb * LANES + l];
          }
        }
      }
    }

    // Load C tile directly into accumulators (C += A*B^T semantics)
  load_c_row:
    for (int r = 0; r < TI; r++) {
    load_c_beat:
      for (int cb = 0; cb < CB; cb++) {
#pragma HLS PIPELINE II=1
      load_c_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          acc[r % PI][l % PJ][r / PI][cb * 2 + l / PJ] = C[i0 + r][j0 + cb * LANES + l];
        }
      }
    }

    // Main compute: k-chunk outer, sub-block inner (acc reuse distance = NSB)
  compute_kc_sb:
    for (int it = 0; it < NKC * NSB; it++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
#pragma HLS DEPENDENCE variable=B0 inter false
#pragma HLS DEPENDENCE variable=B1 inter false
      const int kc = it / NSB;
      const int sb = it % NSB;
      const int ib = sb / NJB;
      const int jb = sb % NJB;

      // Prefetch next B tile into the idle buffer (one beat per cycle)
      if (has_next && it < TILE_BEATS) {
        vec_t v = bs.read();
        const int r = it / KB;
        const int kb = it % KB;
      pf_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          if (cur)
            B0[r % PJ][l][r / PJ][kb] = v[l];
          else
            B1[r % PJ][l][r / PJ][kb] = v[l];
        }
      }

      data_t a_reg[PI][KL];
#pragma HLS ARRAY_PARTITION variable=a_reg complete dim=0
      data_t b_reg[PJ][KL];
#pragma HLS ARRAY_PARTITION variable=b_reg complete dim=0
    rd_a_p:
      for (int p = 0; p < PI; p++) {
#pragma HLS UNROLL
      rd_a_q:
        for (int q = 0; q < KL; q++) {
#pragma HLS UNROLL
          a_reg[p][q] = Abuf[p][q][ib][kc];
        }
      }
    rd_b_c:
      for (int c = 0; c < PJ; c++) {
#pragma HLS UNROLL
      rd_b_q:
        for (int q = 0; q < KL; q++) {
#pragma HLS UNROLL
          data_t b0v = B0[c][q][jb][kc];
          data_t b1v = B1[c][q][jb][kc];
          b_reg[c][q] = cur ? b1v : b0v;
        }
      }

    pe_row:
      for (int p = 0; p < PI; p++) {
#pragma HLS UNROLL
      pe_col:
        for (int c = 0; c < PJ; c++) {
#pragma HLS UNROLL
          data_t s = dot16(a_reg[p], b_reg[c]);
          acc[p][c][ib][jb] = acc[p][c][ib][jb] + s;
        }
      }
    }

    // Store C tile once
  store_c_row:
    for (int r = 0; r < TI; r++) {
    store_c_beat:
      for (int cb = 0; cb < CB; cb++) {
#pragma HLS PIPELINE II=1
      store_c_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          C[i0 + r][j0 + cb * LANES + l] = acc[r % PI][l % PJ][r / PI][cb * 2 + l / PJ];
        }
      }
    }
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16 latency=64
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16 latency=64
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16 latency=64
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW
  hls::stream<vec_t> bs("b_stream");
#pragma HLS STREAM variable=bs depth=64
  b_loader(B, bs);
  compute_task(A, C, bs);
}
#include "kernel.h"

// Manual true-DATAFLOW rewrite of ctl_cc_refnf_r2 (Opus 5.5, refined + resource requirement in words).
// Same datapath: PI x PJ x KL = 4 x 16 x 16 = 1024 fp32 MACs per cycle, TI = 64-row tiles of A / C.
// Changes:
//   1. The 16 row tiles run in a DATAFLOW loop: load_A(t+1) | compute(t) | store_C(t-1) overlap. A_t and C_t are
//      ping-pong (PIPO, depth 2) channels between the three tasks.
//   2. B is no longer preloaded serially. During tile 0 the compute task streams in the next 16-row block of B
//      (64 KB, contiguous) one 512-bit beat per cycle while it computes the current 16 output columns; B then stays
//      resident (URAM) for tiles 1..15.
//   3. C = A * B^T: accumulators start at zero; C is never read.
// Compute order is (jb, kb, ii): each accumulator is revisited every TI/PI = 16 cycles (> fp32 add latency).

#define TI     64
#define PI     4
#define PJ     16
#define KL     16
#define LANES  16
#define NT     (I / TI)          // 16 row tiles
#define NJB    (J / PJ)          // 64 column blocks
#define NKB    (K / KL)          // 64 k chunks
#define KBEATS (K / LANES)       // 64 beats per row

static void load_A(data_t A[I][K], data_t A_t[TI][K], int it) {
  load_A_beat: for (int t = 0; t < TI * KBEATS; t++) {
#pragma HLS PIPELINE II=1
    const int r = t / KBEATS, kb = t % KBEATS;
    load_A_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      A_t[r][kb * LANES + l] = A[it * TI + r][kb * LANES + l];
    }
  }
}

static void compute(data_t A_t[TI][K], data_t B[J][K], data_t C_t[TI][J], int it) {
  static data_t B_loc[J][K];
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=16 dim=2
#pragma HLS BIND_STORAGE variable=B_loc type=ram_2p impl=uram

  if (it == 0) {
    // B block 0 (rows 0..15) before the first column block can start.
    pre_B_beat: for (int t = 0; t < PJ * KBEATS; t++) {
#pragma HLS PIPELINE II=1
      const int r = t / KBEATS, kb = t % KBEATS;
      pre_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        B_loc[r][kb * LANES + l] = B[r][kb * LANES + l];
      }
    }
  }

  blk_j: for (int jb = 0; jb < NJB; jb++) {
    mac: for (int t = 0; t < NKB * (TI / PI); t++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=B_loc type=inter false
#pragma HLS DEPENDENCE variable=B_loc type=intra false
#pragma HLS DEPENDENCE variable=C_t type=inter direction=RAW dependent=true distance=16
      const int kb = t / (TI / PI), ii = t % (TI / PI);
      // tile 0 only: fetch one beat of the next B block (rows (jb+1)*16 .. +15, contiguous in memory)
      if (it == 0 && jb + 1 < NJB) {
        const int r = t / KBEATS, kbt = t % KBEATS;
        const int nj = (jb + 1) * PJ + r;
        nxt_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          B_loc[nj][kbt * LANES + l] = B[nj][kbt * LANES + l];
        }
      }
      pe_row: for (int p = 0; p < PI; p++) {
#pragma HLS UNROLL
        pe_col: for (int q = 0; q < PJ; q++) {
#pragma HLS UNROLL
          const int ri = ii * PI + p, cj = jb * PJ + q;
          data_t prod[KL];
#pragma HLS ARRAY_PARTITION variable=prod complete dim=1
          mac_lane: for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
            prod[l] = A_t[ri][kb * KL + l] * B_loc[cj][kb * KL + l];
          }
          tree_l8: for (int l = 0; l < 8; l++) {
#pragma HLS UNROLL
            prod[l] = prod[l] + prod[l + 8];
          }
          tree_l4: for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
            prod[l] = prod[l] + prod[l + 4];
          }
          tree_l2: for (int l = 0; l < 2; l++) {
#pragma HLS UNROLL
            prod[l] = prod[l] + prod[l + 2];
          }
          const data_t s = prod[0] + prod[1];
          C_t[ri][cj] = (kb == 0 ? (data_t)0 : C_t[ri][cj]) + s;
        }
      }
    }
  }
}

static void store_C(data_t C_t[TI][J], data_t C[I][J], int it) {
  store_C_beat: for (int t = 0; t < TI * (J / LANES); t++) {
#pragma HLS PIPELINE II=1
    const int r = t / (J / LANES), b = t % (J / LANES);
    store_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      C[it * TI + r][b * LANES + l] = C_t[r][b * LANES + l];
    }
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  tile_i: for (int it = 0; it < NT; it++) {
#pragma HLS DATAFLOW
    data_t A_t[TI][K];
#pragma HLS ARRAY_PARTITION variable=A_t cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=A_t cyclic factor=16 dim=2
    data_t C_t[TI][J];
#pragma HLS ARRAY_PARTITION variable=C_t cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=C_t cyclic factor=16 dim=2
    load_A(A, A_t, it);
    compute(A_t, B, C_t, it);
    store_C(C_t, C, it);
  }
}

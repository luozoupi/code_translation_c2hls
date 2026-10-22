#include "kernel.h"

// ---------------------------------------------------------------------------
// Configuration (no hls_stream / hls_vector dependency)
//   Output tile       : MM_TI x MM_TJ = 256 x 256 (16 tiles)
//   PE array          : MM_PI x MM_PJ = 32 x 32 outputs per cycle, 2 k per cycle
//   Sub-blocks / tile : MM_NSB = 64 (acc recurrence distance = 64 iterations)
//   A row-block reused across the 4 j-tiles of the same i-block
// ---------------------------------------------------------------------------
#define MM_LANES 16
#define MM_PI 32
#define MM_PJ 32
#define MM_TI 256
#define MM_TJ 256
#define MM_SBI (MM_TI / MM_PI)            // 8
#define MM_SBJ (MM_TJ / MM_PJ)            // 8
#define MM_NSB (MM_SBI * MM_SBJ)          // 64
#define MM_KB (K / MM_LANES)              // 64 beats per row
#define MM_KH (K / 2)                     // 512 k-pairs
#define MM_NTI (I / MM_TI)                // 4
#define MM_NTJ (J / MM_TJ)                // 4
#define MM_ITER_TILE (MM_KH * MM_NSB)     // 32768
#define MM_CB (MM_TJ / MM_LANES)          // 16 beats per output row

static void load_A_block(data_t A[I][K],
                         data_t Abuf[MM_SBI][MM_PI][MM_KB][MM_LANES], int ti) {
load_A_row:
  for (int row = 0; row < MM_TI; ++row) {
  load_A_beat:
    for (int kb = 0; kb < MM_KB; ++kb) {
#pragma HLS PIPELINE II=1
      const int gi = ti * MM_TI + row;
    load_A_lane:
      for (int l = 0; l < MM_LANES; ++l) {
#pragma HLS UNROLL
        Abuf[row >> 5][row & 31][kb][l] = A[gi][kb * MM_LANES + l];
      }
    }
  }
}

static void load_B_block(data_t B[J][K],
                         data_t Bbuf[MM_SBJ][MM_PJ][MM_KB][MM_LANES], int tj) {
load_B_row:
  for (int row = 0; row < MM_TJ; ++row) {
  load_B_beat:
    for (int kb = 0; kb < MM_KB; ++kb) {
#pragma HLS PIPELINE II=1
      const int gj = tj * MM_TJ + row;
    load_B_lane:
      for (int l = 0; l < MM_LANES; ++l) {
#pragma HLS UNROLL
        Bbuf[row >> 5][row & 31][kb][l] = B[gj][kb * MM_LANES + l];
      }
    }
  }
}

static void compute_tile(data_t Abuf[MM_SBI][MM_PI][MM_KB][MM_LANES],
                         data_t Bbuf[MM_SBJ][MM_PJ][MM_KB][MM_LANES],
                         data_t acc[MM_NSB][MM_PI][MM_PJ]) {
compute_main:
  for (int it = 0; it < MM_ITER_TILE; ++it) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
    const int kk = it >> 6;               // k-pair (0..511)
    const int s = it & (MM_NSB - 1);      // sub-block (0..63)
    const int sbi = s >> 3;
    const int sbj = s & 7;
    const int kb_r = kk >> 3;
    const int l0 = (kk & 7) << 1;
    const bool first = (kk == 0);

    data_t a0[MM_PI], a1[MM_PI], b0[MM_PJ], b1[MM_PJ];
#pragma HLS ARRAY_PARTITION variable=a0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=a1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=b0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=b1 complete dim=1
  read_a:
    for (int r = 0; r < MM_PI; ++r) {
#pragma HLS UNROLL
      a0[r] = Abuf[sbi][r][kb_r][l0];
      a1[r] = Abuf[sbi][r][kb_r][l0 + 1];
    }
  read_b:
    for (int c = 0; c < MM_PJ; ++c) {
#pragma HLS UNROLL
      b0[c] = Bbuf[sbj][c][kb_r][l0];
      b1[c] = Bbuf[sbj][c][kb_r][l0 + 1];
    }
  pe_row:
    for (int r = 0; r < MM_PI; ++r) {
#pragma HLS UNROLL
    pe_col:
      for (int c = 0; c < MM_PJ; ++c) {
#pragma HLS UNROLL
        data_t m0 = a0[r] * b0[c];
        data_t m1 = a1[r] * b1[c];
        data_t ps = m0 + m1;
#pragma HLS BIND_OP variable=ps op=fadd impl=fabric
        data_t old = first ? (data_t)0 : acc[s][r][c];
        acc[s][r][c] = old + ps;
      }
    }
  }
}

static void store_tile(data_t C[I][J], data_t acc[MM_NSB][MM_PI][MM_PJ],
                       int ti, int tj) {
store_row:
  for (int row = 0; row < MM_TI; ++row) {
  store_beat:
    for (int cb = 0; cb < MM_CB; ++cb) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C inter false
      const int gi = ti * MM_TI + row;
      const int gj = tj * MM_TJ + cb * MM_LANES;
      const int ds = (row >> 5) * MM_SBJ + (cb >> 1);
      const int dr = row & 31;
      const int dc = (cb & 1) * MM_LANES;
    store_lane:
      for (int l = 0; l < MM_LANES; ++l) {
#pragma HLS UNROLL
        C[gi][gj + l] = C[gi][gj + l] + acc[ds][dr][dc + l];
      }
    }
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static data_t Abuf[MM_SBI][MM_PI][MM_KB][MM_LANES];
#pragma HLS ARRAY_PARTITION variable=Abuf complete dim=2
#pragma HLS ARRAY_PARTITION variable=Abuf complete dim=4
#pragma HLS BIND_STORAGE variable=Abuf type=ram_s2p impl=bram
  static data_t Bbuf[MM_SBJ][MM_PJ][MM_KB][MM_LANES];
#pragma HLS ARRAY_PARTITION variable=Bbuf complete dim=2
#pragma HLS ARRAY_PARTITION variable=Bbuf complete dim=4
#pragma HLS BIND_STORAGE variable=Bbuf type=ram_s2p impl=bram
  static data_t acc[MM_NSB][MM_PI][MM_PJ];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=3
#pragma HLS BIND_STORAGE variable=acc type=ram_s2p impl=bram

tile_i_loop:
  for (int ti = 0; ti < MM_NTI; ++ti) {
    load_A_block(A, Abuf, ti);
  tile_j_loop:
    for (int tj = 0; tj < MM_NTJ; ++tj) {
      load_B_block(B, Bbuf, tj);
      compute_tile(Abuf, Bbuf, acc);
      store_tile(C, acc, ti, tj);
    }
  }
}
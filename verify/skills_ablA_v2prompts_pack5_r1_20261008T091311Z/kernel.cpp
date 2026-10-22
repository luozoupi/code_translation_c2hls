#include "kernel.h"

// Output-stationary GEMM: C[i][j] += sum_k A[i][k] * B[j][k]
//   PE array : PE_I x PE_J = 103 x 16 = 1648 fp32 MACs per cycle
//   B        : fully resident on chip (URAM), skew-banked over 16 banks
//   A, C     : row block (103 rows) staged on chip; C preloaded into the
//              accumulators (original C + sum order), written back once.
#define PE_I      103
#define PE_J      16
#define NBLK      ((I + PE_I - 1) / PE_I)
#define JT        (J / PE_J)
#define KC        (K / 16)
#define JC        (J / 16)
#define STAGE_IT  (K * JT)
#define BDEPTH    (JT * K)

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static data_t Bm[PE_J][BDEPTH];
#pragma HLS ARRAY_PARTITION variable=Bm complete dim=1
#pragma HLS BIND_STORAGE variable=Bm type=ram_2p impl=uram

  static data_t A_loc[PE_I][K];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=16 dim=2
#pragma HLS BIND_STORAGE variable=A_loc type=ram_2p impl=lutram

  static data_t Cb[PE_I][J];
#pragma HLS ARRAY_PARTITION variable=Cb complete dim=1
#pragma HLS ARRAY_PARTITION variable=Cb cyclic factor=16 dim=2
#pragma HLS BIND_STORAGE variable=Cb type=ram_2p impl=lutram

  // ---------------- load B once, skew-banked: (j,k) -> bank (j%16 + k)%16
load_b_row:
  for (int j = 0; j < J; ++j) {
  load_b_beat:
    for (int c = 0; c < KC; ++c) {
#pragma HLS PIPELINE II=1
      const int pjb  = j & 15;
      const int base = (j >> 4) * K + c * 16;
      data_t bv[16];
#pragma HLS ARRAY_PARTITION variable=bv complete dim=1
    load_b_lane:
      for (int l = 0; l < 16; ++l) {
#pragma HLS UNROLL
        bv[l] = B[j][c * 16 + l];
      }
    write_b_bank:
      for (int bk = 0; bk < PE_J; ++bk) {
#pragma HLS UNROLL
        const int l = (bk - pjb) & 15;
        Bm[bk][base + l] = bv[l];
      }
    }
  }

block_loop:
  for (int b = 0; b < NBLK; ++b) {
    // ---------------- load A row block and C row block (separate bundles, II=1)
  load_ac_row:
    for (int r = 0; r < PE_I; ++r) {
      const int row = b * PE_I + r;
      if (row < I) {
      load_ac_beat:
        for (int c = 0; c < KC; ++c) {
#pragma HLS PIPELINE II=1
        load_ac_lane:
          for (int l = 0; l < 16; ++l) {
#pragma HLS UNROLL
            A_loc[r][c * 16 + l] = A[row][c * 16 + l];
            Cb[r][c * 16 + l]    = C[row][c * 16 + l];
          }
        }
      } else {
      pad_ac_beat:
        for (int c = 0; c < KC; ++c) {
#pragma HLS PIPELINE II=1
        pad_ac_lane:
          for (int l = 0; l < 16; ++l) {
#pragma HLS UNROLL
            A_loc[r][c * 16 + l] = (data_t)0;
            Cb[r][c * 16 + l]    = (data_t)0;
          }
        }
      }
    }

    // ---------------- compute: 103 x 16 MACs per cycle, acc reuse distance 64
  compute_loop:
    for (int it = 0; it < STAGE_IT; ++it) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=Cb inter distance=64 true
      const int k  = it >> 6;
      const int jj = it & (JT - 1);

      data_t a_op[PE_I];
#pragma HLS ARRAY_PARTITION variable=a_op complete dim=1
    rd_a:
      for (int pi = 0; pi < PE_I; ++pi) {
#pragma HLS UNROLL
        a_op[pi] = A_loc[pi][k];
      }

      data_t braw[PE_J];
      data_t b_op[PE_J];
#pragma HLS ARRAY_PARTITION variable=braw complete dim=1
#pragma HLS ARRAY_PARTITION variable=b_op complete dim=1
      const int baddr = jj * K + k;
    rd_b:
      for (int bk = 0; bk < PE_J; ++bk) {
#pragma HLS UNROLL
        braw[bk] = Bm[bk][baddr];
      }
    rot_b:
      for (int pj = 0; pj < PE_J; ++pj) {
#pragma HLS UNROLL
        b_op[pj] = braw[(pj + k) & 15];
      }

    pe_row:
      for (int pi = 0; pi < PE_I; ++pi) {
#pragma HLS UNROLL
      pe_col:
        for (int pj = 0; pj < PE_J; ++pj) {
#pragma HLS UNROLL
          const int jc = jj * PE_J + pj;
          Cb[pi][jc] = Cb[pi][jc] + a_op[pi] * b_op[pj];
        }
      }
    }

    // ---------------- store: write-only, 16 floats per beat, II=1
  store_c_row:
    for (int r = 0; r < PE_I; ++r) {
      const int row = b * PE_I + r;
      if (row < I) {
      store_c_beat:
        for (int c = 0; c < JC; ++c) {
#pragma HLS PIPELINE II=1
        store_c_lane:
          for (int l = 0; l < 16; ++l) {
#pragma HLS UNROLL
            C[row][c * 16 + l] = Cb[r][c * 16 + l];
          }
        }
      }
    }
  }
}
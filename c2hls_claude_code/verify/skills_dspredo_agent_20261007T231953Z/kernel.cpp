#include "kernel.h"

// Output-stationary PE array: PI_ x PJ_ parallel float MACs per cycle.
// Each output accumulates in the exact sequential k order (p0 + p1 + ... + p1023),
// the fadd recurrence is hidden by interleaving NP independent outputs per PE
// (dependence distance NP >> pipeline depth).
#define PI_ 27                       // row PEs (non power of 2: rows padded to 1026)
#define PJ_ 64                       // col PEs
#define NTI 38                       // 27*38 = 1026 >= I
#define NTJ 2
#define NP (NTI * NTJ)               // 76 outputs interleaved per PE
#define TJ (PJ_ * NTJ)               // 128 cols per tile (all rows in one tile)
#define LANES 16                     // 512-bit / 32-bit
#define KW (K / LANES)               // 64 words per row
#define TJW (TJ / LANES)             // 8 words per tile row

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Full A and B staged on chip, banked by row (PE index), reshaped to 512-bit words.
  static data_t A_loc[PI_][NTI][KW][LANES];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_RESHAPE variable=A_loc complete dim=4
#pragma HLS BIND_STORAGE variable=A_loc type=ram_1p impl=uram
  static data_t B_loc[PJ_][J / PJ_][KW][LANES];
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=1
#pragma HLS ARRAY_RESHAPE variable=B_loc complete dim=4
#pragma HLS BIND_STORAGE variable=B_loc type=ram_1p impl=bram

  static data_t acc[PI_][PJ_][NP];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS BIND_STORAGE variable=acc type=ram_s2p impl=lutram

load_ab:
  for (int idx = 0; idx < I * KW; idx++) {
#pragma HLS PIPELINE II=1
    int r = idx / KW;
    int w = idx % KW;
  load_ab_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      A_loc[r % PI_][r / PI_][w][l] = A[r][w * LANES + l];
      B_loc[r % PJ_][r / PJ_][w][l] = B[r][w * LANES + l];
    }
  }

tile_j:
  for (int jt = 0; jt < J / TJ; jt++) {
    const int j0 = jt * TJ;

  compute_k:
    for (int k = 0; k < K; k++) {
    compute_pos:
      for (int pos = 0; pos < NP; pos++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
        int ti = pos / NTJ;
        int tj = pos % NTJ;
        int kw = k / LANES;
        int kl = k % LANES;
        data_t a[PI_];
#pragma HLS ARRAY_PARTITION variable=a complete
        data_t b[PJ_];
#pragma HLS ARRAY_PARTITION variable=b complete
      read_a:
        for (int pi = 0; pi < PI_; pi++) {
#pragma HLS UNROLL
          a[pi] = A_loc[pi][ti][kw][kl];
        }
      read_b:
        for (int pj = 0; pj < PJ_; pj++) {
#pragma HLS UNROLL
          b[pj] = B_loc[pj][jt * NTJ + tj][kw][kl];
        }
      pe_i:
        for (int pi = 0; pi < PI_; pi++) {
#pragma HLS UNROLL
        pe_j:
          for (int pj = 0; pj < PJ_; pj++) {
#pragma HLS UNROLL
            data_t prod = a[pi] * b[pj];
            data_t old = (k == 0) ? (data_t)0 : acc[pi][pj][pos];
            acc[pi][pj][pos] = old + prod;
          }
        }
      }
    }

  store_c:
    for (int idx = 0; idx < I * TJW; idx++) {
#pragma HLS PIPELINE II=1
      int r = idx / TJW;
      int w = idx % TJW;
      int pi = r % PI_;
      int ti = r / PI_;
    store_c_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        int col = w * LANES + l;
        int pos = ti * NTJ + col / PJ_;
        data_t cin = C[r][j0 + col];
        C[r][j0 + col] = cin + acc[pi][col % PJ_][pos];
      }
    }
  }
}
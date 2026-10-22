#include "kernel.h"

// ---------------------------------------------------------------------------
// Parallel MAC array: PI x PJ x UK = 27 x 2 x 32 = 1728 fp32 MACs per cycle
//   per lane (32-term dot product): 32 fmul*3 + (31 tree + 1 acc) fadd*2 = 160 DSP
//   54 lanes * 160 = 8640 DSP (~95.7% of 9024)
// B (1024x1024) fully staged on-chip, banked 2 (j) x 32 (k) = 64 banks.
// A/C streamed in 27-row tiles (38 tiles; last tile has 2 padded rows, guarded).
// ---------------------------------------------------------------------------
#define PI   27                     // rows of A/C per tile (spatial PEs)
#define PJ   2                      // parallel output columns
#define UK   32                     // parallel k terms (tree-reduced)
#define WL   16                     // 16 x 32b = 512-bit memory beat
#define NT   ((I + PI - 1) / PI)    // 38 row tiles
#define NJB  (J / PJ)               // 512 column blocks
#define NKC  (K / UK)               // 32 k chunks

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Full B on chip: bank (j%2, k%32), each bank 512 x 32 = 16384 words
  data_t b_loc[J][K];
#pragma HLS ARRAY_PARTITION variable=b_loc cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=b_loc cyclic factor=32 dim=2

  // A row tile: 27 rows, k banked by 32 (one bank per k-lane)
  data_t a_loc[PI][K];
#pragma HLS ARRAY_PARTITION variable=a_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=a_loc cyclic factor=32 dim=2

  // C accumulator tile: 27 rows, j banked by 16 (matches 512-bit load/store beat)
  data_t acc[PI][J];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=16 dim=2

  // ---------------- load all of B once ----------------
  load_b_row: for (int j = 0; j < J; j++) {
    load_b_col: for (int k0 = 0; k0 < K; k0 += WL) {
#pragma HLS PIPELINE II=1
      load_b_lane: for (int w = 0; w < WL; w++) {
#pragma HLS UNROLL
        b_loc[j][k0 + w] = B[j][k0 + w];
      }
    }
  }

  // ---------------- row tiles ----------------
  tile_loop: for (int t = 0; t < NT; t++) {
    const int i0 = t * PI;

    // load A rows and initial C rows (parallel on gmem0 / gmem2); J == K == 1024
    load_tile_row: for (int r = 0; r < PI; r++) {
      const int row = i0 + r;
      const bool valid = (row < I);
      const int row_c = valid ? row : (I - 1);
      load_tile_col: for (int k0 = 0; k0 < K; k0 += WL) {
#pragma HLS PIPELINE II=1
        load_tile_lane: for (int w = 0; w < WL; w++) {
#pragma HLS UNROLL
          data_t av = A[row_c][k0 + w];
          data_t cv = C[row_c][k0 + w];
          a_loc[r][k0 + w] = valid ? av : (data_t)0;
          acc[r][k0 + w]   = valid ? cv : (data_t)0;
        }
      }
    }

    // compute: each acc element revisited every NJB=512 cycles (>> add-tree depth)
    compute_kc: for (int kc = 0; kc < NKC; kc++) {
      compute_jb: for (int jb = 0; jb < NJB; jb++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
        compute_pi: for (int pi = 0; pi < PI; pi++) {
#pragma HLS UNROLL
          compute_pj: for (int pj = 0; pj < PJ; pj++) {
#pragma HLS UNROLL
            data_t p[UK];
#pragma HLS ARRAY_PARTITION variable=p complete dim=1
            mul_uk: for (int uk = 0; uk < UK; uk++) {
#pragma HLS UNROLL
              p[uk] = a_loc[pi][kc * UK + uk] * b_loc[jb * PJ + pj][kc * UK + uk];
            }
            tree_lvl: for (int s = UK / 2; s > 0; s >>= 1) {
#pragma HLS UNROLL
              tree_add: for (int u = 0; u < s; u++) {
#pragma HLS UNROLL
                p[u] = p[u] + p[u + s];
              }
            }
            acc[pi][jb * PJ + pj] = acc[pi][jb * PJ + pj] + p[0];
          }
        }
      }
    }

    // store finished C rows once
    store_tile_row: for (int r = 0; r < PI; r++) {
      const int row = i0 + r;
      if (row < I) {
        store_tile_col: for (int j0 = 0; j0 < J; j0 += WL) {
#pragma HLS PIPELINE II=1
          store_tile_lane: for (int w = 0; w < WL; w++) {
#pragma HLS UNROLL
            C[row][j0 + w] = acc[r][j0 + w];
          }
        }
      }
    }
  }
}
#include "kernel.h"

// Spatial datapath sizing (U280: 9024 DSP; fp32 MAC ~ 5 DSP)
//   TI rows x KU k-lanes = 52 x 32 = 1664 MACs/cycle  (~8.3K DSP)
#define TI     52
#define KU     32
#define LANES  16                       // 512-bit beat / 32-bit float
#define NUM_IT ((I + TI - 1) / TI)

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Full B resident on chip (URAM), banked on k for KU parallel reads.
  static data_t B_loc[J][K];
#pragma HLS BIND_STORAGE variable=B_loc type=ram_2p impl=uram
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=32 dim=2

  // A row-tile: one bank set per PE row, banked on k by KU.
  static data_t A_loc[TI][K];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=32 dim=2

  // C row-tile accumulators: per PE row, banked by LANES for 16-wide load/store.
  static data_t acc[TI][J];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=16 dim=2

  // ---------------- Load all of B once ----------------
  load_B_row: for (int j = 0; j < J; j++) {
    load_B_col: for (int k = 0; k < K; k += LANES) {
#pragma HLS PIPELINE II=1
      load_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        B_loc[j][k + l] = B[j][k + l];
      }
    }
  }

  // ---------------- Row tiles of A / C ----------------
  tile_i: for (int t = 0; t < NUM_IT; t++) {
    const int i0 = t * TI;

    // Load A tile and C tile (initial accumulator) together; rows past I
    // are clamped (garbage rows computed but never stored).
#if (K == J)
    load_AC_row: for (int ii = 0; ii < TI; ii++) {
      int row = i0 + ii;
      int row_c = (row < I) ? row : (I - 1);
      load_AC_col: for (int c = 0; c < K; c += LANES) {
#pragma HLS PIPELINE II=1
        load_AC_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          A_loc[ii][c + l] = A[row_c][c + l];
          acc[ii][c + l]   = (data_t)0;
        }
      }
    }
#else
    load_A_row: for (int ii = 0; ii < TI; ii++) {
      int row = i0 + ii;
      int row_c = (row < I) ? row : (I - 1);
      load_A_col: for (int c = 0; c < K; c += LANES) {
#pragma HLS PIPELINE II=1
        load_A_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          A_loc[ii][c + l] = A[row_c][c + l];
        }
      }
    }
    load_C_row: for (int ii = 0; ii < TI; ii++) {
      int row = i0 + ii;
      int row_c = (row < I) ? row : (I - 1);
      load_C_col: for (int c = 0; c < J; c += LANES) {
#pragma HLS PIPELINE II=1
        load_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          acc[ii][c + l] = (data_t)0;
        }
      }
    }
#endif

    // Compute: k-block outer, j inner (pipelined). Each acc[ii][jj] is
    // revisited only every J=1024 iterations, far beyond the fadd latency,
    // so the inter-iteration dependence on acc is not carried within the
    // pipeline depth.
    compute_kb: for (int kb = 0; kb < K / KU; kb++) {
      compute_j: for (int jj = 0; jj < J; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
        data_t bv[KU];
#pragma HLS ARRAY_PARTITION variable=bv complete dim=1
        load_bv: for (int kk = 0; kk < KU; kk++) {
#pragma HLS UNROLL
          bv[kk] = B_loc[jj][kb * KU + kk];
        }
        pe_row: for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
          data_t prod[KU];
#pragma HLS ARRAY_PARTITION variable=prod complete dim=1
          pe_mul: for (int kk = 0; kk < KU; kk++) {
#pragma HLS UNROLL
            prod[kk] = A_loc[ii][kb * KU + kk] * bv[kk];
          }
          pe_tree_lvl: for (int s = KU / 2; s > 0; s >>= 1) {
#pragma HLS UNROLL
            pe_tree_add: for (int q = 0; q < s; q++) {
#pragma HLS UNROLL
              prod[q] = prod[q] + prod[q + s];
            }
          }
          acc[ii][jj] = acc[ii][jj] + prod[0];
        }
      }
    }

    // Store valid rows of the C tile.
    store_C_row: for (int ii = 0; ii < TI; ii++) {
      int row = i0 + ii;
      if (row < I) {
        store_C_col: for (int c = 0; c < J; c += LANES) {
#pragma HLS PIPELINE II=1
          store_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            C[row][c + l] = acc[ii][c + l];
          }
        }
      }
    }
  }
}
#include "kernel.h"

#define TI 16
#define TJ 32

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_read_burst_length=256 max_write_burst_length=256 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  data_t A_tile[TI][K];
  data_t B_tile[TJ][K];
#pragma HLS ARRAY_PARTITION variable=B_tile dim=1 complete
  data_t C_loc[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=C_loc dim=2 complete

  tile_j: for (int j0 = 0; j0 < J; j0 += TJ) {
    // Load TJ full rows of B (contiguous region)
    load_B_row: for (int jj = 0; jj < TJ; jj++) {
      load_B_col: for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
        B_tile[jj][k] = B[j0 + jj][k];
      }
    }

    tile_i: for (int i0 = 0; i0 < I; i0 += TI) {
      // Load TI full rows of A (contiguous region)
      load_A_row: for (int ii = 0; ii < TI; ii++) {
        load_A_col: for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
          A_tile[ii][k] = A[i0 + ii][k];
        }
      }

      // Load the C tile (accumulate semantics preserved)
      load_C_row: for (int ii = 0; ii < TI; ii++) {
        load_C_col: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS PIPELINE II=1
          C_loc[ii][jj] = C[i0 + ii][j0 + jj];
        }
      }

      // Compute: k outer, ii pipelined, jj fully unrolled.
      // Each C_loc[ii][*] is revisited every TI iterations (> fadd latency).
      compute_k: for (int k = 0; k < K; k++) {
        compute_i: for (int ii = 0; ii < TI; ii++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_loc inter false
          data_t a = A_tile[ii][k];
          compute_j: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS UNROLL
            C_loc[ii][jj] = C_loc[ii][jj] + a * B_tile[jj][k];
          }
        }
      }

      // Store the C tile
      store_C_row: for (int ii = 0; ii < TI; ii++) {
        store_C_col: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS PIPELINE II=1
          C[i0 + ii][j0 + jj] = C_loc[ii][jj];
        }
      }
    }
  }
}
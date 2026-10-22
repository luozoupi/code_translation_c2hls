#include "kernel.h"

#define TI 128     // rows of A / C kept on-chip per i-tile (parallel MAC lanes)
#define TJ 16      // rows of B per j-tile (interleave distance for FP accumulation)
#define LANES 16   // 512-bit beat / 32-bit float

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static data_t A_tile[TI][K];
#pragma HLS ARRAY_PARTITION variable=A_tile complete dim=1
  static data_t C_tile[TI][J];
#pragma HLS ARRAY_PARTITION variable=C_tile complete dim=1
  static data_t B_tile[TJ][K];
#pragma HLS ARRAY_PARTITION variable=B_tile cyclic factor=LANES dim=2

  tile_i: for (int it = 0; it < I / TI; it++) {
    const int i0 = it * TI;

    load_A_row: for (int ii = 0; ii < TI; ii++) {
      load_A_col: for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
        A_tile[ii][k] = A[i0 + ii][k];
      }
    }

    load_C_row: for (int ii = 0; ii < TI; ii++) {
      load_C_col: for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
        C_tile[ii][j] = C[i0 + ii][j];
      }
    }

    tile_j: for (int jt = 0; jt < J / TJ; jt++) {
      const int j0 = jt * TJ;

      load_B_row: for (int jj = 0; jj < TJ; jj++) {
        load_B_beat: for (int k = 0; k < K; k += LANES) {
#pragma HLS PIPELINE II=1
          load_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            B_tile[jj][k + l] = B[j0 + jj][k + l];
          }
        }
      }

      // k outer, jj inner: each C_tile element is revisited every TJ cycles,
      // hiding the FP add latency while preserving the serial k summation order.
      compute_k: for (int k = 0; k < K; k++) {
        compute_jj: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_tile inter distance=16 true
          data_t b = B_tile[jj][k];
          compute_ii: for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
            C_tile[ii][j0 + jj] = C_tile[ii][j0 + jj] + A_tile[ii][k] * b;
          }
        }
      }
    }

    store_C_row: for (int ii = 0; ii < TI; ii++) {
      store_C_col: for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
        C[i0 + ii][j] = C_tile[ii][j];
      }
    }
  }
}
#include "kernel.h"

#define TJ 128
#define P 16

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_read_burst_length=256 num_read_outstanding=16 max_write_burst_length=256 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_read_burst_length=256 num_read_outstanding=16 max_write_burst_length=256 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_read_burst_length=256 num_read_outstanding=16 max_write_burst_length=256 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static data_t Bl[K][TJ];
#pragma HLS ARRAY_PARTITION variable=Bl cyclic factor=16 dim=2
  data_t Al[K];
  data_t acc[TJ];
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=16 dim=1

  tile_j: for (int jt = 0; jt < J; jt += TJ) {
    // Load B tile (rows jt..jt+TJ-1), transposed into Bl[k][jj]
    load_b_row: for (int jj = 0; jj < TJ; jj++) {
      load_b_col: for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
        Bl[k][jj] = B[jt + jj][k];
      }
    }

    row_i: for (int i = 0; i < I; i++) {
      load_a: for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
        Al[k] = A[i][k];
      }
      load_c: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS PIPELINE II=1
        acc[jj] = C[i][jt + jj];
      }

      compute_k: for (int k = 0; k < K; k++) {
        compute_jo: for (int jo = 0; jo < TJ / P; jo++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
          data_t a = Al[k];
          compute_lane: for (int p = 0; p < P; p++) {
#pragma HLS UNROLL
            int jj = jo * P + p;
            acc[jj] = acc[jj] + a * Bl[k][jj];
          }
        }
      }

      store_c: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS PIPELINE II=1
        C[i][jt + jj] = acc[jj];
      }
    }
  }
}
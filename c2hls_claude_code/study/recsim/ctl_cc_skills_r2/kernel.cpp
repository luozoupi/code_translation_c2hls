#include "kernel.h"

#define TI 64      // rows of A / C processed per i-block (parallel MAC lanes)
#define TJ 32      // rows of B per j-block (also RMW dependence distance)

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static data_t A_loc[TI][K];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
  static data_t C_loc[TI][J];
#pragma HLS ARRAY_PARTITION variable=C_loc complete dim=1
  static data_t B_loc[TJ][K];

  block_i: for (int ib = 0; ib < I / TI; ib++) {
    const int i0 = ib * TI;

    load_A: for (int idx = 0; idx < TI * K; idx++) {
#pragma HLS PIPELINE II=1
      int ti = idx / K;
      int k = idx % K;
      A_loc[ti][k] = A[i0 + ti][k];
    }

    load_C: for (int idx = 0; idx < TI * J; idx++) {
#pragma HLS PIPELINE II=1
      int ti = idx / J;
      int j = idx % J;
      C_loc[ti][j] = C[i0 + ti][j];
    }

    block_j: for (int jb = 0; jb < J / TJ; jb++) {
      const int j0 = jb * TJ;

      load_B: for (int idx = 0; idx < TJ * K; idx++) {
#pragma HLS PIPELINE II=1
        int tj = idx / K;
        int k = idx % K;
        B_loc[tj][k] = B[j0 + tj][k];
      }

      compute_k: for (int k = 0; k < K; k++) {
        compute_tj: for (int tj = 0; tj < TJ; tj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
#pragma HLS DEPENDENCE variable=C_loc inter true distance=32
          data_t b = B_loc[tj][k];
          compute_ti: for (int ti = 0; ti < TI; ti++) {
#pragma HLS UNROLL
            data_t prod = A_loc[ti][k] * b;
            C_loc[ti][j0 + tj] = C_loc[ti][j0 + tj] + prod;
          }
        }
      }
    }

    store_C: for (int idx = 0; idx < TI * J; idx++) {
#pragma HLS PIPELINE II=1
      int ti = idx / J;
      int j = idx % J;
      C[i0 + ti][j] = C_loc[ti][j];
    }
  }
}
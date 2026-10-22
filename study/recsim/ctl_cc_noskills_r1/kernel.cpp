#include "kernel.h"

#define TI 32
#define NP 8

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_read_burst_length=256 max_write_burst_length=256 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static data_t Abuf[TI][K];
#pragma HLS ARRAY_PARTITION variable=Abuf complete dim=1
  static data_t Cbuf[TI][J];
  data_t acc[TI][NP];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

tile_i:
  for (int i0 = 0; i0 < I; i0 += TI) {

  load_a_row:
    for (int ii = 0; ii < TI; ii++) {
    load_a_col:
      for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
        Abuf[ii][k] = A[i0 + ii][k];
      }
    }

  load_c_row:
    for (int ii = 0; ii < TI; ii++) {
    load_c_col:
      for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
        Cbuf[ii][j] = C[i0 + ii][j];
      }
    }

  compute_j:
    for (int j = 0; j < J; j++) {

    init_acc_i:
      for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
      init_acc_p:
        for (int p = 0; p < NP; p++) {
#pragma HLS UNROLL
          acc[ii][p] = 0.0f;
        }
      }

    compute_k:
      for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter distance=8 true
        data_t b = B[j][k];
        int p = k & (NP - 1);
      mac_ii:
        for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
          acc[ii][p] += Abuf[ii][k] * b;
        }
      }

    reduce_ii:
      for (int ii = 0; ii < TI; ii++) {
#pragma HLS PIPELINE II=1
        data_t s0 = acc[ii][0] + acc[ii][1];
        data_t s1 = acc[ii][2] + acc[ii][3];
        data_t s2 = acc[ii][4] + acc[ii][5];
        data_t s3 = acc[ii][6] + acc[ii][7];
        data_t t0 = s0 + s1;
        data_t t1 = s2 + s3;
        Cbuf[ii][j] = Cbuf[ii][j] + (t0 + t1);
      }
    }

  store_c_row:
    for (int ii = 0; ii < TI; ii++) {
    store_c_col:
      for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
        C[i0 + ii][j] = Cbuf[ii][j];
      }
    }
  }
}
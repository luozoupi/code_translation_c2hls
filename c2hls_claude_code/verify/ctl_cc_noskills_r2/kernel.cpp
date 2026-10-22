#include "kernel.h"

#define TI 32      // rows of A/C processed concurrently per tile
#define NL 8       // interleaved partial-sum lanes (> fadd latency)

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_read_burst_length=256 max_write_burst_length=256 num_read_outstanding=16 num_write_outstanding=16 latency=64
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_read_burst_length=256 max_write_burst_length=256 num_read_outstanding=16 num_write_outstanding=16 latency=64
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_read_burst_length=256 max_write_burst_length=256 num_read_outstanding=16 num_write_outstanding=16 latency=64
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  data_t a_buf[TI][K];
#pragma HLS ARRAY_PARTITION variable=a_buf complete dim=1
  data_t c_buf[TI][J];
  data_t acc[TI][NL];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

tile_i:
  for (int it = 0; it < I; it += TI) {

  load_a_row:
    for (int ii = 0; ii < TI; ii++) {
    load_a_col:
      for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
        a_buf[ii][k] = A[it + ii][k];
      }
    }

  load_c_row:
    for (int ii = 0; ii < TI; ii++) {
    load_c_col:
      for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
        c_buf[ii][j] = C[it + ii][j];
      }
    }

  loop_j:
    for (int j = 0; j < J; j++) {

    compute_k:
      for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
        data_t b = B[j][k];
        int l = k & (NL - 1);
      compute_ii:
        for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
          data_t prev = (k < NL) ? (data_t)0 : acc[ii][l];
          acc[ii][l] = prev + a_buf[ii][k] * b;
        }
      }

    reduce_ii:
      for (int ii = 0; ii < TI; ii++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=c_buf inter false
        data_t s0 = acc[ii][0] + acc[ii][1];
        data_t s1 = acc[ii][2] + acc[ii][3];
        data_t s2 = acc[ii][4] + acc[ii][5];
        data_t s3 = acc[ii][6] + acc[ii][7];
        data_t t0 = s0 + s1;
        data_t t1 = s2 + s3;
        c_buf[ii][j] = c_buf[ii][j] + (t0 + t1);
      }
    }

  store_c_row:
    for (int ii = 0; ii < TI; ii++) {
    store_c_col:
      for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
        C[it + ii][j] = c_buf[ii][j];
      }
    }
  }
}
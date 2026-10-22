#include "kernel.h"

#define JT 64   // B rows (j) held on chip per tile
#define IT 16   // A rows (i) per inner tile; >= fadd latency so II=1 holds

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static data_t Bbuf[JT][K];
#pragma HLS ARRAY_PARTITION variable=Bbuf dim=1 complete
  static data_t Abuf[IT][K];
  data_t Cbuf[IT][JT];
#pragma HLS ARRAY_PARTITION variable=Cbuf dim=2 complete

JT_LOOP:
  for (int jt = 0; jt < J; jt += JT) {
    // Load B tile
  LOAD_B:
    for (int jj = 0; jj < JT; jj++) {
      for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
        Bbuf[jj][k] = B[jt + jj][k];
      }
    }

  IT_LOOP:
    for (int it = 0; it < I; it += IT) {
      // Load A tile
    LOAD_A:
      for (int ii = 0; ii < IT; ii++) {
        for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
          Abuf[ii][k] = A[it + ii][k];
        }
      }
      // Load C tile (accumulate onto existing C)
    LOAD_C:
      for (int ii = 0; ii < IT; ii++) {
        for (int jj = 0; jj < JT; jj++) {
#pragma HLS PIPELINE II=1
          Cbuf[ii][jj] = C[it + ii][jt + jj];
        }
      }

      // Compute: per (i,j) the k-order is preserved exactly
    COMP_K:
      for (int k = 0; k < K; k++) {
      COMP_I:
        for (int ii = 0; ii < IT; ii++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=Cbuf inter false
          data_t a = Abuf[ii][k];
        COMP_J:
          for (int jj = 0; jj < JT; jj++) {
#pragma HLS UNROLL
            Cbuf[ii][jj] = Cbuf[ii][jj] + a * Bbuf[jj][k];
          }
        }
      }

      // Store C tile
    STORE_C:
      for (int ii = 0; ii < IT; ii++) {
        for (int jj = 0; jj < JT; jj++) {
#pragma HLS PIPELINE II=1
          C[it + ii][jt + jj] = Cbuf[ii][jj];
        }
      }
    }
  }
}
#include "kernel.h"

// C = A * B^T, I = J = K = 1024, fp32.
// Architecture: A staged on chip once (k-major banked), B streamed one
// j-tile (TN columns) at a time (also k-major banked).  Compute tiles are
// TM x TN output elements held entirely in registers, reduction over k.
#ifndef TM
#define TM 128
#endif
#ifndef TN
#define TN 64
#endif
#define VEC 16

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem0 max_widen_bitwidth = 512 max_read_burst_length = 64 depth = 1048576
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem1 max_widen_bitwidth = 512 max_read_burst_length = 64 depth = 1048576
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem2 max_widen_bitwidth = 512 max_write_burst_length = 64 depth = 1048576
#pragma HLS INTERFACE s_axilite port = return

  // A_p[k][i] = A[i][k]   (k-major so one k reads TM different banks)
  static data_t A_p[K][I];
#pragma HLS ARRAY_PARTITION variable = A_p dim = 1 cyclic factor = VEC
#pragma HLS ARRAY_PARTITION variable = A_p dim = 2 cyclic factor = TM

  // B_t[k][j] = B[j0+j][k]  (k-major, one j-tile resident)
  static data_t B_t[K][TN];
#pragma HLS ARRAY_PARTITION variable = B_t dim = 1 cyclic factor = VEC
#pragma HLS ARRAY_PARTITION variable = B_t dim = 2 cyclic factor = TN

  static data_t acc[TM][TN];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0

  // ---------------- stage all of A on chip ----------------
  load_a_i:
  for (int i = 0; i < I; i++) {
  load_a_k:
    for (int kk = 0; kk < K / VEC; kk++) {
#pragma HLS PIPELINE II = 1
      for (int v = 0; v < VEC; v++) {
#pragma HLS UNROLL
        A_p[kk * VEC + v][i] = A[i][kk * VEC + v];
      }
    }
  }

  for (int jb = 0; jb < J / TN; jb++) {
    const int j0 = jb * TN;

    // ---------------- stage one B tile ----------------
    load_b_j:
    for (int j2 = 0; j2 < TN; j2++) {
    load_b_k:
      for (int kk = 0; kk < K / VEC; kk++) {
#pragma HLS PIPELINE II = 1
        for (int v = 0; v < VEC; v++) {
#pragma HLS UNROLL
          B_t[kk * VEC + v][j2] = B[j0 + j2][kk * VEC + v];
        }
      }
    }

    for (int ib = 0; ib < I / TM; ib++) {
      const int i0 = ib * TM;

      init_acc:
      for (int i2 = 0; i2 < TM; i2++) {
        for (int j2 = 0; j2 < TN; j2++) {
#pragma HLS UNROLL
          acc[i2][j2] = 0;
        }
      }

      compute_k:
      for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II = 1
        for (int i2 = 0; i2 < TM; i2++) {
#pragma HLS UNROLL
          for (int j2 = 0; j2 < TN; j2++) {
#pragma HLS UNROLL
            acc[i2][j2] += A_p[k][i0 + i2] * B_t[k][j2];
          }
        }
      }

      store_i:
      for (int i2 = 0; i2 < TM; i2++) {
      store_j:
        for (int jj = 0; jj < TN / VEC; jj++) {
#pragma HLS PIPELINE II = 1
          for (int v = 0; v < VEC; v++) {
#pragma HLS UNROLL
            C[i0 + i2][j0 + jj * VEC + v] = acc[i2][jj * VEC + v];
          }
        }
      }
    }
  }
}

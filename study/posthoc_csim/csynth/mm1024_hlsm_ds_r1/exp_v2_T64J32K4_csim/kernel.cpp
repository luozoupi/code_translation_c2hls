#include "kernel.h"

/* ------------------------------------------------------------------
 * C[i][j] = sum_k A[i][k] * B[j][k]   (C = A * B^T), fp32, 1024^3
 *
 * Output-stationary tile: TI rows x TJ cols of C, with KL k-lanes
 * accumulated into KL independent register banks (named acc0..acc3).
 *
 * The whole B matrix (J x K, 4 MB) is staged on chip once, so A only
 * has to be streamed: each A row block is loaded once and reused for
 * every j tile.
 * ------------------------------------------------------------------ */
#ifndef TI
#define TI 64
#endif
#ifndef TJ
#define TJ 32
#endif
#ifndef KL
#define KL 4
#endif

#define JN (J / TJ)     /* number of j tiles */
#define IN (I / TI)     /* number of i tiles */
#define LD 16           /* m_axi lanes per beat (512 bit / fp32) */
#define SC (TJ < LD ? TJ : LD)

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmemA max_widen_bitwidth=512 num_read_outstanding=32 num_write_outstanding=2 max_read_burst_length=64 depth=1048576
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmemB max_widen_bitwidth=512 num_read_outstanding=32 num_write_outstanding=2 max_read_burst_length=64 depth=1048576
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmemC max_widen_bitwidth=512 num_read_outstanding=2 num_write_outstanding=32 max_write_burst_length=64 depth=1048576
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  /* full B staged on chip: B_loc[jt][j][k] */
  static data_t B_loc[JN][TJ][K];
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=2
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=LD dim=3

  /* current A row block */
  static data_t A_loc[TI][K];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=LD dim=2

  /* accumulators: KL independent banks */
  static data_t acc0[TI][TJ];
  static data_t acc1[TI][TJ];
  static data_t acc2[TI][TJ];
  static data_t acc3[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=acc0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=acc1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0
#pragma HLS ARRAY_PARTITION variable=acc3 complete dim=0

  /* ---------------- stage B once ---------------- */
loadB_j:
  for (int jt = 0; jt < JN; jt++) {
  loadB_jj:
    for (int j = 0; j < TJ; j++) {
    loadB_k:
      for (int k = 0; k < K; k += LD) {
#pragma HLS PIPELINE II=1
        for (int l = 0; l < LD; l++) {
#pragma HLS UNROLL
          B_loc[jt][j][k + l] = B[jt * TJ + j][k + l];
        }
      }
    }
  }

  /* ---------------- main loop over i tiles ---------------- */
main_i:
  for (int it = 0; it < IN; it++) {

  loadA_i:
    for (int i = 0; i < TI; i++) {
    loadA_k:
      for (int k = 0; k < K; k += LD) {
#pragma HLS PIPELINE II=1
        for (int l = 0; l < LD; l++) {
#pragma HLS UNROLL
          A_loc[i][k + l] = A[it * TI + i][k + l];
        }
      }
    }

  main_j:
    for (int jt = 0; jt < JN; jt++) {

      /* clear accumulators */
    clr_i:
      for (int i = 0; i < TI; i++) {
#pragma HLS UNROLL
      clr_j:
        for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
          acc0[i][j] = 0;
          acc1[i][j] = 0;
          acc2[i][j] = 0;
          acc3[i][j] = 0;
        }
      }

      /* ---- k reduction: 4 independent banks ---- */
    cmp_k:
      for (int k = 0; k < K; k += KL) {
#pragma HLS PIPELINE II=1
      cmp_i:
        for (int i = 0; i < TI; i++) {
#pragma HLS UNROLL
          data_t a0 = A_loc[i][k + 0];
          data_t a1 = A_loc[i][k + 1];
          data_t a2 = A_loc[i][k + 2];
          data_t a3 = A_loc[i][k + 3];
        cmp_j:
          for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
            acc0[i][j] += a0 * B_loc[jt][j][k + 0];
            acc1[i][j] += a1 * B_loc[jt][j][k + 1];
            acc2[i][j] += a2 * B_loc[jt][j][k + 2];
            acc3[i][j] += a3 * B_loc[jt][j][k + 3];
          }
        }
      }

      /* ---- store C tile: SC lanes per beat ---- */
    st_i:
      for (int i = 0; i < TI; i++) {
      st_j:
        for (int j = 0; j < TJ; j += SC) {
#pragma HLS PIPELINE II=1
          for (int l = 0; l < SC; l++) {
#pragma HLS UNROLL
            C[it * TI + i][jt * TJ + j + l] =
                (acc0[i][j + l] + acc1[i][j + l]) + (acc2[i][j + l] + acc3[i][j + l]);
          }
        }
      }
    }
  }
}

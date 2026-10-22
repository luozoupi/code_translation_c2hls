#include "kernel.h"

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  /* ------------------------------------------------------------------
   * Blocked / tiled GEMM with on-chip staging.
   *   C[i][j] += sum_k A[i][k] * B[j][k]      (K is the reduction dim)
   *
   * Working set per output tile (32 x 32):
   *   A_loc[32][64]  =  8 KB
   *   B_loc[32][64]  =  8 KB
   *   C_loc[32*32]   =  4 KB
   * Hot compute loop reads/writes ONLY local memory (no m_axi RMW).
   * ------------------------------------------------------------------ */
  const int TI = 32;   /* rows of the output tile           */
  const int TJ = 32;   /* cols of the output tile           */
  const int TK = 64;   /* k-panel staged on chip            */
  const int KU = 8;    /* k-unroll inside the hot pipeline  */
  const int LU = 4;    /* load-unroll (contiguous vector)   */

  data_t A_loc[TI][TK];
  data_t B_loc[TJ][TK];
  data_t C_loc[TI * TJ];

  /* Bank the reduction (k) dimension so the KU unrolled taps of
     A_loc[i][kk..kk+KU-1] and B_loc[j][kk..kk+KU-1] come from
     distinct BRAM banks -> no port conflict at II=1. */
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=KU dim=2
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=KU dim=2

  tile_i: for (int i0 = 0; i0 < I; i0 += TI) {
    tile_j: for (int j0 = 0; j0 < J; j0 += TJ) {

      /* Stage the (possibly non-zero) initial C tile once, so the
         accumulation below preserves the reference semantics
         C_out = C_in + A*B^T while never touching m_axi in the
         hot loop. */
      init_c: for (int ij = 0; ij < TI * TJ; ij++) {
#pragma HLS PIPELINE II=1
        const int i = ij / TJ;
        const int j = ij % TJ;
        C_loc[ij] = C[i0 + i][j0 + j];
      }

      panel_k: for (int k0 = 0; k0 < K; k0 += TK) {

        /* ---- bulk load of the A k-panel (unit stride in k) ---- */
        load_a: for (int i = 0; i < TI; i++) {
          for (int k = 0; k < TK; k += LU) {
#pragma HLS PIPELINE II=1
            for (int u = 0; u < LU; u++) {
#pragma HLS UNROLL
              A_loc[i][k + u] = A[i0 + i][k0 + k + u];
            }
          }
        }

        /* ---- bulk load of the B k-panel (unit stride in k) ---- */
        load_b: for (int j = 0; j < TJ; j++) {
          for (int k = 0; k < TK; k += LU) {
#pragma HLS PIPELINE II=1
            for (int u = 0; u < LU; u++) {
#pragma HLS UNROLL
              B_loc[j][k + u] = B[j0 + j][k0 + k + u];
            }
          }
        }

        /* ---- local-memory compute, one flattened (i,j) pipeline ---- */
        chunk_k: for (int kk = 0; kk < TK; kk += KU) {
          compute_ij: for (int ij = 0; ij < TI * TJ; ij++) {
#pragma HLS PIPELINE II=1
            const int i = ij / TJ;
            const int j = ij % TJ;
            data_t acc = C_loc[ij];
            for (int u = 0; u < KU; u++) {
#pragma HLS UNROLL
              acc += A_loc[i][kk + u] * B_loc[j][kk + u];
            }
            C_loc[ij] = acc;
          }
        }
      }

      /* ---- single bulk store of the finished tile ---- */
      store_c: for (int ij = 0; ij < TI * TJ; ij++) {
#pragma HLS PIPELINE II=1
        const int i = ij / TJ;
        const int j = ij % TJ;
        C[i0 + i][j0 + j] = C_loc[ij];
      }
    }
  }
}
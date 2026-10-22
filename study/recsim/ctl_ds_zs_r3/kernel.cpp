#include "kernel.h"

/* ---------------------------------------------------------------------------
 * Blocked matrix multiplication
 *
 *        C[i][j] = C[i][j] + sum_{k} A[i][k] * B[j][k]
 *
 * A : I x K,  B : J x K,  C : I x J   (row-major arrays from the header)
 *
 * For the default configuration (I = J = K = 1024) the three operands are
 * 12 MByte in total, so they must remain in off-chip memory.  The kernel is
 * therefore tiled: small blocks of A, B and C are staged into fully
 * partitioned on-chip buffers (registers) and the innermost reduction loop is
 * completely unrolled so that a whole TK-wide dot product is computed every
 * clock cycle.
 *
 * The loop bounds in the default configuration are exact multiples of the
 * tile sizes below (1024 / 16 = 64).
 * ------------------------------------------------------------------------- */

#define TI 16 /* rows of A / C handled per tile            */
#define TJ 16 /* rows of B / columns of C handled per tile */
#define TK 16 /* reduction depth handled per tile          */

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {

  /* ---- Global memory interfaces -------------------------------------- */
#pragma HLS INTERFACE m_axi port = A offset = slave bundle = gmem0 max_read_burst_length = 64
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port = B offset = slave bundle = gmem1 max_read_burst_length = 64
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = gmem2 max_read_burst_length = 64 max_write_burst_length = 64
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle=control

  /* ---- On-chip staging buffers --------------------------------------- */
  data_t a_tile[TI][TK];
  data_t b_tile[TJ][TK];
  data_t c_tile[TI][TJ];

  /* Complete partitioning of the innermost dimension turns every row of a
     tile into an independent register file, so all TK elements of a row can
     be consumed in the same cycle. */
#pragma HLS ARRAY_PARTITION variable = a_tile complete dim = 2
#pragma HLS ARRAY_PARTITION variable = b_tile complete dim = 2
#pragma HLS ARRAY_PARTITION variable = c_tile complete dim = 2

  /* ---- Tile loops ---------------------------------------------------- */
  for (int ii = 0; ii < I; ii += TI) {

    for (int jj = 0; jj < J; jj += TJ) {

      /* Seed the accumulator tile with the current content of C so that
         the accumulation semantics of the reference code are preserved. */
      for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
          c_tile[i][j] = C[ii + i][jj + j];
        }
      }

      /* Reduction over the K dimension in chunks of TK. */
      for (int kk = 0; kk < K; kk += TK) {

        /* ---- load A tile (TI x TK, row dominant) --------------------- */
        for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II = 1
          for (int k = 0; k < TK; k++) {
#pragma HLS UNROLL
            a_tile[i][k] = A[ii + i][kk + k];
          }
        }

        /* ---- load B tile (TJ x TK, row dominant) --------------------- */
        for (int j = 0; j < TJ; j++) {
#pragma HLS PIPELINE II = 1
          for (int k = 0; k < TK; k++) {
#pragma HLS UNROLL
            b_tile[j][k] = B[jj + j][kk + k];
          }
        }

        /* ---- multiply / accumulate ----------------------------------- */
        for (int i = 0; i < TI; i++) {
          for (int j = 0; j < TJ; j++) {
#pragma HLS PIPELINE II = 1
            data_t acc = c_tile[i][j];
            for (int k = 0; k < TK; k++) {
#pragma HLS UNROLL
              acc += a_tile[i][k] * b_tile[j][k];
            }
            c_tile[i][j] = acc;
          }
        }
      }

      /* ---- write back the C tile ------------------------------------- */
      for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
          C[ii + i][jj + j] = c_tile[i][j];
        }
      }
    }
  }
}
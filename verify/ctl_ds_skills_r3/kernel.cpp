#include "kernel.h"

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_read_burst_length=256 num_read_outstanding=16 max_write_burst_length=256 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Working-set tiles (compile-time constants, divide I/J/K exactly).
  const int TM = 32;   // rows of C per tile
  const int TN = 32;   // cols of C per tile
  const int TK = 32;   // reduction depth per k-block

  // On-chip staging: all hot-loop traffic is local; no global RMW.
  data_t A_local[TM][TK];
  data_t B_local[TN][TK];
  data_t C_local[TM][TN];

  // Bank the arrays only on the dimension actually accessed in parallel.
  //  - compute reads B_local[j][k] for all j  -> complete on dim 1 (j)
  //  - compute RMW   C_local[i][j] for all j  -> complete on dim 2 (j)
  //  - A_local[i][k] is a single read per pipelined iteration -> left as RAM
#pragma HLS ARRAY_PARTITION variable=B_local complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_local complete dim=2

  tile_rows: for (int i0 = 0; i0 < I; i0 += TM) {
    tile_cols: for (int j0 = 0; j0 < J; j0 += TN) {

      // ---- load the C tile once (no per-element global RMW later) ----
      load_C: for (int i = 0; i < TM; i++) {
        load_C_col: for (int j = 0; j < TN; j++) {
#pragma HLS PIPELINE II=1
          C_local[i][j] = C[i0 + i][j0 + j];
        }
      }

      // ---- reduction over K in fixed TK blocks, all on local memory ----
      tile_k: for (int k0 = 0; k0 < K; k0 += TK) {

        load_A: for (int i = 0; i < TM; i++) {
          load_A_k: for (int k = 0; k < TK; k++) {
#pragma HLS PIPELINE II=1
            A_local[i][k] = A[i0 + i][k0 + k];
          }
        }

        load_B: for (int j = 0; j < TN; j++) {
          load_B_k: for (int k = 0; k < TK; k++) {
#pragma HLS PIPELINE II=1
            B_local[j][k] = B[j0 + j][k0 + k];
          }
        }

        // 32 independent MACs per pipelined iteration (no serial FP chain).
        compute_k: for (int k = 0; k < TK; k++) {
          compute_i: for (int i = 0; i < TM; i++) {
#pragma HLS PIPELINE II=1
            compute_j: for (int j = 0; j < TN; j++) {
#pragma HLS UNROLL
              C_local[i][j] += A_local[i][k] * B_local[j][k];
            }
          }
        }
      }

      // ---- single store pass, one write per output element ----
      store_C: for (int i = 0; i < TM; i++) {
        store_C_col: for (int j = 0; j < TN; j++) {
#pragma HLS PIPELINE II=1
          C[i0 + i][j0 + j] = C_local[i][j];
        }
      }
    }
  }
}
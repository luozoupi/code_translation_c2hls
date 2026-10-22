#include "kernel.h"

/* ------------------------------------------------------------------ */
/*  Tiled GEMM:  C[i][j] += sum_k A[i][k] * B[j][k]                    */
/*                                                                     */
/*  Structure:                                                         */
/*    block_i / block_j   : 64 x 64 output tiles (C staged in BRAM)    */
/*    load_C              : pipelined burst load of the C tile         */
/*    block_k  (TK = 8)   : K is consumed in 8-deep contiguous panels  */
/*      load_AB           : fused pipelined load of A-panel and        */
/*                          B-panel into fully banked local tiles      */
/*      compute_ij        : flattened (i,j) pipeline II=1, k fully     */
/*                          unrolled -> 8 MACs / cycle, all operands   */
/*                          from local BRAM (no m_axi in compute)      */
/*    store_C             : pipelined burst store of the C tile        */
/* ------------------------------------------------------------------ */

#define TI 64
#define TJ 64
#define TK 8

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 latency=64 num_read_outstanding=16 max_read_burst_length=64 max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 latency=64 num_read_outstanding=16 max_read_burst_length=64 max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 latency=64 num_write_outstanding=16 max_write_burst_length=64 max_widen_bitwidth=512
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  /* ---- on-chip staging buffers ---------------------------------- */
  data_t A_tile[TI][TK];
  data_t B_tile[TJ][TK];
  data_t C_tile[TI][TJ];

  /* k is the parallel (fully unrolled) reduction dimension for both
     A and B -> bank these two dims so 8 simultaneous reads are legal */
#pragma HLS ARRAY_PARTITION variable=A_tile complete dim=2
#pragma HLS ARRAY_PARTITION variable=B_tile complete dim=2

  /* ================================================================ */
  block_i: for (int i0 = 0; i0 < I; i0 += TI) {
    block_j: for (int j0 = 0; j0 < J; j0 += TJ) {

      /* -------- 1) stage the output tile (keeps the +C semantics) - */
      load_C: for (int i = 0; i < TI; ++i) {
        for (int j = 0; j < TJ; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
          C_tile[i][j] = C[i0 + i][j0 + j];
        }
      }

      /* -------- 2) sweep K in panels of TK ------------------------ */
      block_k: for (int k0 = 0; k0 < K; k0 += TK) {

        /* fused A/B panel load: one contiguous burst from gmem0 and
           one from gmem1 per pipelined iteration (2 elements/cycle)  */
        load_AB: for (int t = 0; t < TI; ++t) {
          for (int k = 0; k < TK; ++k) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
            A_tile[t][k] = A[i0 + t][k0 + k];
            B_tile[t][k] = B[j0 + t][k0 + k];
          }
        }

        /* local compute: (i,j) nest flattened into one pipeline,
           reduction over k completely unrolled from banked locals   */
        compute_ij: for (int i = 0; i < TI; ++i) {
          for (int j = 0; j < TJ; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
            data_t acc = C_tile[i][j];
            reduce_k: for (int k = 0; k < TK; ++k) {
#pragma HLS UNROLL
              acc += A_tile[i][k] * B_tile[j][k];
            }
            C_tile[i][j] = acc;
          }
        }
      }

      /* -------- 3) burst the finished tile back to gmem2 ---------- */
      store_C: for (int i = 0; i < TI; ++i) {
        for (int j = 0; j < TJ; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
          C[i0 + i][j0 + j] = C_tile[i][j];
        }
      }
    }
  }
}
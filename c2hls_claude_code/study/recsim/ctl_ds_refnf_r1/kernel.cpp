#include "kernel.h"

/* ------------------------------------------------------------------ *
 *  autosa_mm :  C[i][j] = sum_k A[i][k] * B[j][k]      (C = A * B^T)
 *  I = J = K = 1024 (compile-time constants from kernel.h)
 *
 *  Strategy
 *  --------
 *  - B is staged once into on-chip memory (B_loc) and stays resident.
 *  - A is streamed in MI-row tiles (A_tile) which are reused across all
 *    NJ-column tiles of C.
 *  - Per (i0,j0) tile the 32x32 output accumulators live in private
 *    registers (acc), so the innermost MAC loop never touches m_axi.
 *  - Every global load/store walks 16 floats (512-bit) per iteration,
 *    with local memory banked on the consecutive (k / j) dimension by
 *    the same factor (LANES) so the burst actually reaches II=1.
 *  - Row dimension banked 32-way so the 32 parallel row operands per
 *    cycle come from 32 distinct memories.
 * ------------------------------------------------------------------ */

#define MI    32              /* rows of C per tile                     */
#define NJ    32              /* cols of C per tile                     */
#define LANES 16              /* 512-bit / 32-bit float                 */
#define KB    (K / LANES)     /* k-blocks of LANES                      */

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  /* on-chip working set ------------------------------------------------ */
  data_t B_loc[J / NJ][NJ][K];   /* resident copy of B   (4 MB)          */
  data_t A_tile[MI][K];          /* streamed A row tile  (128 KB)        */
  data_t acc[MI][NJ];            /* private accumulators (registers)     */

  /* row banking for the 32 simultaneous row reads, plus a small
     consecutive-dim banking so 16-wide bursts write conflict-free      */
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=2
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=4 dim=3
#pragma HLS ARRAY_PARTITION variable=A_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_tile cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

  /* ---------------- stage the whole of B on chip ---------------- */
  load_B: for (int jt = 0; jt < J / NJ; ++jt) {
    for (int jj = 0; jj < NJ; ++jj) {
      for (int kb = 0; kb < KB; ++kb) {
#pragma HLS PIPELINE II=1
        for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
          B_loc[jt][jj][kb * LANES + l] = B[jt * NJ + jj][kb * LANES + l];
        }
      }
    }
  }

  /* ---------------- stream A row-tiles ---------------- */
  tile_i: for (int i0 = 0; i0 < I; i0 += MI) {

    load_A: for (int ii = 0; ii < MI; ++ii) {
      for (int kb = 0; kb < KB; ++kb) {
#pragma HLS PIPELINE II=1
        for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
          A_tile[ii][kb * LANES + l] = A[i0 + ii][kb * LANES + l];
        }
      }
    }

    /* ---------------- column tiles of C ---------------- */
    tile_j: for (int jt = 0; jt < J / NJ; ++jt) {

      init_acc: for (int ii = 0; ii < MI; ++ii) {
#pragma HLS UNROLL
        for (int jj = 0; jj < NJ; ++jj) {
#pragma HLS UNROLL
          acc[ii][jj] = (data_t)0;
        }
      }

      /* hot loop : 32x32 MACs per iteration, operands from on-chip only */
      mac_k: for (int k = 0; k < K; ++k) {
#pragma HLS PIPELINE II=1
        for (int ii = 0; ii < MI; ++ii) {
#pragma HLS UNROLL
          for (int jj = 0; jj < NJ; ++jj) {
#pragma HLS UNROLL
            acc[ii][jj] += A_tile[ii][k] * B_loc[jt][jj][k];
          }
        }
      }

      /* ---------------- write the C tile back, 512-bit beats ------- */
      store_C: for (int ii = 0; ii < MI; ++ii) {
        for (int jb = 0; jb < NJ / LANES; ++jb) {
#pragma HLS PIPELINE II=1
          for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
            C[i0 + ii][jt * NJ + jb * LANES + l] = acc[ii][jb * LANES + l];
          }
        }
      }
    }
  }
}
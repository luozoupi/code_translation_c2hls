#include "kernel.h"

#define TI    16          // rows of C per tile (fully unrolled PEs)
#define TJ    128         // cols of C per tile
#define JL    8           // column lanes per cycle (TI*JL = 128 MACs/cycle)
#define JG    (TJ / JL)   // column groups -> acc recurrence distance (16)
#define LANES 16          // 512-bit / 32-bit float

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  data_t A_loc[TI][K];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1

  data_t B_loc[TJ][K];
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=JL dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=LANES dim=2

  data_t acc[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=JL dim=2

  row_block: for (int i0 = 0; i0 < I; i0 += TI) {

    // Load TI rows of A once; reused across all column blocks.
    load_A_row: for (int ii = 0; ii < TI; ii++) {
      load_A_k: for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
        A_loc[ii][k] = A[i0 + ii][k];
      }
    }

    col_block: for (int j0 = 0; j0 < J; j0 += TJ) {

      // Load TJ rows of B (B is [J][K]), 16 floats per beat.
      load_B_row: for (int jj = 0; jj < TJ; jj++) {
        load_B_k: for (int k = 0; k < K; k += LANES) {
#pragma HLS PIPELINE II=1
          load_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            B_loc[jj][k + l] = B[j0 + jj][k + l];
          }
        }
      }

      // Load current C tile into local accumulators (preserves C += semantics).
      load_C_row: for (int ii = 0; ii < TI; ii++) {
        load_C_col: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS PIPELINE II=1
          acc[ii][jj] = C[i0 + ii][j0 + jj];
        }
      }

      // Compute: serial k order per element (bit-exact vs reference).
      // Each acc element is touched once every JG=16 cycles > fadd latency.
      compute_k: for (int k = 0; k < K; k++) {
        compute_jg: for (int jg = 0; jg < JG; jg++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
          compute_ii: for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
            data_t a = A_loc[ii][k];
            compute_jl: for (int jl = 0; jl < JL; jl++) {
#pragma HLS UNROLL
              int j = jg * JL + jl;
              acc[ii][j] = acc[ii][j] + a * B_loc[j][k];
            }
          }
        }
      }

      // Store C tile once.
      store_C_row: for (int ii = 0; ii < TI; ii++) {
        store_C_col: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS PIPELINE II=1
          C[i0 + ii][j0 + jj] = acc[ii][jj];
        }
      }
    }
  }
}
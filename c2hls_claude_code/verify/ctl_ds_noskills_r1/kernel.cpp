#include "kernel.h"

#define TJ 8

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A bundle=gmem0 latency=64 num_read_outstanding=16 max_read_burst_length=64
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B bundle=gmem1 latency=64 num_read_outstanding=16 max_read_burst_length=64
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=C bundle=gmem2 latency=64 num_write_outstanding=16 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Resident tile of B: TJ rows x K, kept on-chip for the whole j-block.
  data_t B_tile[TJ][K];
  // Register accumulators for the TJ columns produced from one row of A.
  data_t acc[TJ];

#pragma HLS ARRAY_PARTITION variable=B_tile dim=1 complete
#pragma HLS ARRAY_PARTITION variable=acc complete

  // Outer loop over j-blocks: B_tile stays resident while we sweep all rows of A.
  jblock_loop: for (int jb = 0; jb < J; jb += TJ) {
    // Stage the B tile (TJ rows, contiguous in k) into local BRAM.
    load_B_outer: for (int jj = 0; jj < TJ; jj++) {
      load_B_inner: for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
        B_tile[jj][k] = B[jb + jj][k];
      }
    }

    // Sweep every row of A against the resident B tile.
    i_loop: for (int i = 0; i < I; i++) {
      init_acc: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS UNROLL
        acc[jj] = (data_t)0;
      }

      // Pipelined reduction: 1 A read (broadcast) + TJ BRAM reads + TJ FMAs per cycle.
      k_loop: for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
        data_t a = A[i][k];
        mac_loop: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS UNROLL
          acc[jj] += a * B_tile[jj][k];
        }
      }

      store_C: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS UNROLL
        C[i][jb + jj] = acc[jj];
      }
    }
  }
}
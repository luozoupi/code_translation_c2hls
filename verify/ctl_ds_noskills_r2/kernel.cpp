#include "kernel.h"

#define TI 32
#define TJ 32
#define TK 32

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A bundle=gmem0 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B bundle=gmem1 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=C bundle=gmem2 max_write_burst_length=256 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  data_t A_tile[TI][TK];
  data_t B_tile[TJ][TK];
  data_t acc[TI][TJ];

  // B_tile banks along j so a whole k-column (all j) is readable in one cycle.
#pragma HLS array_partition variable=B_tile complete dim=1
  // acc banks along j so a whole i-row (all j) is read-modify-written in one cycle.
#pragma HLS array_partition variable=acc complete dim=2

  tile_i: for (int ii = 0; ii < I; ii += TI) {
    tile_j: for (int jj = 0; jj < J; jj += TJ) {

      /* ---- zero the accumulator tile ---- */
      init_i: for (int i = 0; i < TI; i++) {
#pragma HLS unroll
        init_j: for (int j = 0; j < TJ; j++) {
#pragma HLS unroll
          acc[i][j] = 0;
        }
      }

      tile_k: for (int kk = 0; kk < K; kk += TK) {

        /* ---- stage a TI x TK block of A and a TJ x TK block of B ----
           k is innermost -> contiguous 32-beat bursts on both ports,
           1 element/cycle/port => 2 loads per cycle.                       */
        load_t: for (int t = 0; t < TI; t++) {
          load_k: for (int k = 0; k < TK; k++) {
#pragma HLS pipeline II=1
            A_tile[t][k] = A[ii + t][kk + k];
            B_tile[t][k] = B[jj + t][kk + k];
          }
        }

        /* ---- rank-1 style update from local staging ----
           pipelined over i at II=1, fully unrolled over j => TJ MACs/cycle.
           A_tile[i][k] is a single scalar read (loop invariant over j);
           B_tile[j][k] and acc[i][j] are banked so all TJ lanes hit
           distinct memories in the same cycle.                              */
        comp_k: for (int k = 0; k < TK; k++) {
          comp_i: for (int i = 0; i < TI; i++) {
#pragma HLS pipeline II=1
            comp_j: for (int j = 0; j < TJ; j++) {
#pragma HLS unroll
              acc[i][j] += A_tile[i][k] * B_tile[j][k];
            }
          }
        }
      }

      /* ---- write back the finished output tile (contiguous in j) ---- */
      store_i: for (int i = 0; i < TI; i++) {
        store_j: for (int j = 0; j < TJ; j++) {
#pragma HLS pipeline II=1
          C[ii + i][jj + j] = acc[i][j];
        }
      }
    }
  }
}
#include "kernel.h"

// C[i][j] = sum_k A[i][k] * B[j][k]   (C = A * B^T, overwrite)
//
// Load-compute-store tiling:
//   * TI x TJ output tile, K consumed in chunks of TK.
//   * A and B tiles are staged in on-chip buffers; the hot compute loop touches
//     local buffers only.
//   * One fp32 accumulator per output.  The reduction over k is a single
//     carried dependency, so the k-loop lands at II = 4 (fp32 add latency);
//     the tile is sized so that all TI*TJ MACs are issued in those 4 cycles,
//     i.e. TI*TJ/4 MACs per cycle.
//   * A/B rows are fetched as 512-bit bursts (LANES = 16 floats per beat) from
//     two distinct m_axi bundles which are loaded concurrently.

#define TI 80
#define TJ 64
#define TK 64
#define LANES 16
#define NCHUNK (TK / LANES)

static data_t A_loc[TI][TK];
static data_t B_loc[TJ][TK];
static data_t acc[TI][TJ];

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port = A bundle = gmem0 depth = 1048576 max_widen_bitwidth = 512 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = B bundle = gmem1 depth = 1048576 max_widen_bitwidth = 512 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = C bundle = gmem2 depth = 1048576 max_widen_bitwidth = 512 max_write_burst_length = 64 num_write_outstanding = 16
#pragma HLS INTERFACE s_axilite port = return

#pragma HLS ARRAY_PARTITION variable = A_loc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = A_loc cyclic factor = LANES dim = 2
#pragma HLS ARRAY_PARTITION variable = B_loc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = B_loc cyclic factor = LANES dim = 2
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0

  for (int i0 = 0; i0 < I; i0 += TI) {
    for (int j0 = 0; j0 < J; j0 += TJ) {

      for (int i = 0; i < TI; i++) {
#pragma HLS UNROLL
        for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
          acc[i][j] = (data_t)0;
        }
      }

      for (int k0 = 0; k0 < K; k0 += TK) {
        // ---- stage A tile and B tile (both ports busy in the same loop) ----
        for (int t = 0; t < TI; t++) {
#pragma HLS PIPELINE II = 1
          int ig = i0 + t;
          if (ig >= I) ig = I - 1;
          for (int kc = 0; kc < NCHUNK; kc++) {
#pragma HLS UNROLL
            for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
              A_loc[t][kc * LANES + l] = A[ig][k0 + kc * LANES + l];
            }
          }
          if (t < TJ) {
            for (int kc = 0; kc < NCHUNK; kc++) {
#pragma HLS UNROLL
              for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
                B_loc[t][kc * LANES + l] = B[j0 + t][k0 + kc * LANES + l];
              }
            }
          }
        }
        // ---- compute: TI*TJ MACs per k step, fp32 accumulator per output ----
        for (int k = 0; k < TK; k++) {
#pragma HLS PIPELINE II = 1
          for (int i = 0; i < TI; i++) {
#pragma HLS UNROLL
            for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
              acc[i][j] += A_loc[i][k] * B_loc[j][k];
            }
          }
        }
      }

      // ---- store (overwrite C) ----
      for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II = 1
        int ig = i0 + i;
        if (ig >= I) ig = I - 1;
        if (i0 + i < I) {
          for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
            C[ig][j0 + j] = acc[i][j];
          }
        }
      }
    }
  }
}

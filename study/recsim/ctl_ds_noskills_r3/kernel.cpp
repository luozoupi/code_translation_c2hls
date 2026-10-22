#include "kernel.h"

// ---------------------------------------------------------------------------
// Tiled out-of-place matrix multiply:  C[i][j] = sum_k A[i][k] * B[j][k]
//   - A rows are streamed from global memory inside the pipelined MAC loop
//     (1 element / cycle == exactly the 32 MAC/cycle compute rate).
//   - B is staged in a fully-partitioned local tile (transposed access pattern
//     otherwise destroys AXI burst efficiency).
//   - The C tile is kept in a local, row-partitioned accumulator so the
//     partial sums never touch global memory until the whole k-reduction is
//     done.
// ---------------------------------------------------------------------------

#define TI 128   // rows of A / rows of C per tile   (I/TI = 8)
#define TJ 32    // cols of C / rows of B per tile   (J/TJ = 32)
#define TK 32    // k-chunk depth                    (K/TK = 32)

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port = A bundle = gmem0 latency = 64 num_read_outstanding = 16 max_read_burst_length = 256
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port = B bundle = gmem1 latency = 64 num_read_outstanding = 16 max_read_burst_length = 256
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port = C bundle = gmem2 latency = 64 num_write_outstanding = 16 max_write_burst_length = 256
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port = return bundle=control

  // Local tile buffers -------------------------------------------------------
  data_t B_local[TJ][TK];   // staged B tile:  row j -> 32 floats of A/B
  data_t C_local[TI][TJ];   // local C accumulator tile
  data_t acc[TJ];           // register accumulator row (one row of C tile)

#pragma HLS ARRAY_PARTITION variable = B_local dim = 1 complete   // 32 banks, 1 per j
#pragma HLS ARRAY_PARTITION variable = C_local dim = 2 complete   // 32 banks, whole row in 1 cycle
#pragma HLS ARRAY_PARTITION variable = acc dim = 1 complete       // registers

  int ii, jj, kk, ik, j;

  tile_ii: for (ii = 0; ii < I; ii += TI) {
    tile_jj: for (jj = 0; jj < J; jj += TJ) {

      // ---------------------------------------------------------------------
      // zero the local C tile (one row written per cycle, banks are free)
      // ---------------------------------------------------------------------
      init_c: for (int e = 0; e < TI * TJ; e++) {
#pragma HLS PIPELINE II = 1
        C_local[e / TJ][e % TJ] = (data_t)0;
      }

      // ---------------------------------------------------------------------
      // reduction over the K dimension
      // ---------------------------------------------------------------------
      red_kk: for (kk = 0; kk < K; kk += TK) {

        // stage the B tile: TJ rows x TK contiguous columns, 1 word / cycle
        load_b: for (int e = 0; e < TJ * TK; e++) {
#pragma HLS PIPELINE II = 1
          B_local[e / TK][e % TK] = B[jj + (e / TK)][kk + (e % TK)];
        }

        // -----------------------------------------------------------------
        // flattened (i,k) MAC loop: one A element / cycle, TJ MACs / cycle
        // -----------------------------------------------------------------
        mac_ik: for (ik = 0; ik < TI * TK; ik++) {
#pragma HLS PIPELINE II = 1
          int i2 = ik / TK;   // row inside the A tile
          int k2 = ik % TK;   // column inside the k chunk

          if (k2 == 0) {
          load_acc: for (j = 0; j < TJ; j++) {
#pragma HLS UNROLL
              acc[j] = C_local[i2][j];
            }
          }

          data_t a = A[ii + i2][kk + k2];   // streaming, sequential address

        mac_j: for (j = 0; j < TJ; j++) {
#pragma HLS UNROLL
            acc[j] += a * B_local[j][k2];
          }

          if (k2 == TK - 1) {
          store_acc: for (j = 0; j < TJ; j++) {
#pragma HLS UNROLL
              C_local[i2][j] = acc[j];
            }
          }
        }
      }

      // ---------------------------------------------------------------------
      // write the finished C tile back (burst-friendly, contiguous in j)
      // ---------------------------------------------------------------------
      store_c: for (int e = 0; e < TI * TJ; e++) {
#pragma HLS PIPELINE II = 1
        C[ii + (e / TJ)][jj + (e % TJ)] = C_local[e / TJ][e % TJ];
      }
    }
  }
}
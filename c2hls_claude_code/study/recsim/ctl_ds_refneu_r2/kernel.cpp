#include "kernel.h"

// ---- tiling / datapath parameters -------------------------------------
#define BI   128          // rows of A (and C) staged per tile
#define BJ   128          // cols of B (and C) staged per tile
#define TIF   32          // parallel output rows per MAC sub-block
#define TJF   32          // parallel output cols per MAC sub-block
#define KB     2          // independent FP accumulator banks (k += KB)
#define LDW   16          // 512-bit lane count (16 x fp32)

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // ---- on-chip tiles ---------------------------------------------------
  data_t A_loc[BI][K];      // staged A panel (rows i0..i0+BI-1, all K)
  data_t B_loc[BJ][K];      // staged B panel (rows j0..j0+BJ-1, all K)
  data_t C_loc[BI][BJ];     // staged C tile

  // dim1 banked by TIF so TIF rows are readable in one cycle;
  // dim2 banked by LDW so the 16-lane 512-bit load walk is conflict free.
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=TIF dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=LDW dim=2
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=TJF dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=LDW dim=2
  // C_loc banked on the row (column index) so a 32-wide sub-block row is
  // written/read in a single cycle.
#pragma HLS ARRAY_PARTITION variable=C_loc cyclic factor=TJF dim=2

  tile_i0: for (int i0 = 0; i0 < I; i0 += BI) {
    tile_j0: for (int j0 = 0; j0 < J; j0 += BJ) {

      // ================= LOAD STAGE =====================================
      // Fused A/B panel load: one 512-bit beat on gmem0 and one on gmem1
      // per II=1 iteration.
      load_i: for (int i = 0; i < BI; ++i) {
        load_k: for (int kk = 0; kk < K; kk += LDW) {
#pragma HLS PIPELINE II=1
          load_l: for (int l = 0; l < LDW; ++l) {
#pragma HLS UNROLL
            A_loc[i][kk + l] = A[i0 + i][kk + l];
            B_loc[i][kk + l] = B[j0 + i][kk + l];
          }
        }
      }

      // ================= COMPUTE STAGE ==================================
      compute_ib: for (int ib = 0; ib < BI; ib += TIF) {
        compute_jb: for (int jb = 0; jb < BJ; jb += TJF) {

          data_t acc0[TIF][TJF];
          data_t acc1[TIF][TJF];
#pragma HLS ARRAY_PARTITION variable=acc0 complete
#pragma HLS ARRAY_PARTITION variable=acc1 complete

          init_i: for (int ii = 0; ii < TIF; ++ii) {
#pragma HLS UNROLL
            init_j: for (int jj = 0; jj < TJF; ++jj) {
#pragma HLS UNROLL
              acc0[ii][jj] = (data_t)0;
              acc1[ii][jj] = (data_t)0;
            }
          }

          // k advanced by the FP recurrence distance; both named banks are
          // written in the same pipeline iteration (no switch(k%2)).
          compute_k: for (int k = 0; k < K; k += KB) {
#pragma HLS PIPELINE II=1
            mac_i: for (int ii = 0; ii < TIF; ++ii) {
#pragma HLS UNROLL
              mac_j: for (int jj = 0; jj < TJF; ++jj) {
#pragma HLS UNROLL
                acc0[ii][jj] += A_loc[ib + ii][k]     * B_loc[jb + jj][k];
                acc1[ii][jj] += A_loc[ib + ii][k + 1] * B_loc[jb + jj][k + 1];
              }
            }
          }

          // finalise the sub-block into the on-chip C tile (single write)
          wr_i: for (int ii = 0; ii < TIF; ++ii) {
#pragma HLS PIPELINE II=1
            wr_j: for (int jj = 0; jj < TJF; ++jj) {
#pragma HLS UNROLL
              C_loc[ib + ii][jb + jj] = acc0[ii][jj] + acc1[ii][jj];
            }
          }
        }
      }

      // ================= STORE STAGE ====================================
      store_i: for (int i = 0; i < BI; ++i) {
        store_j: for (int jj = 0; jj < BJ; jj += LDW) {
#pragma HLS PIPELINE II=1
          store_l: for (int l = 0; l < LDW; ++l) {
#pragma HLS UNROLL
            C[i0 + i][j0 + jj + l] = C_loc[i][jj + l];
          }
        }
      }

    }
  }
}
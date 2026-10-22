#include "kernel.h"

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // ---- tiling / datapath constants -------------------------------------
  const int LANES = 16;          // 512 bits / 32 bits per float
  const int PI    = 8;           // output rows per tile (spatial)
  const int PJ    = 32;          // output cols per sub-tile (spatial)
  const int BJ    = 128;         // B column-panel width (resident)
  const int NJ    = BJ / PJ;     // sub-blocks per panel  (= 4)
  const int K4    = K / 4;       // k-step-4 reduction iterations

  // ---- on-chip staging ------------------------------------------------
  data_t A_tile[PI][K];          // resident A rows
  data_t B_panel[BJ][K];         // resident B column panel
  data_t acc0[PI][PJ];
  data_t acc1[PI][PJ];
  data_t acc2[PI][PJ];
  data_t acc3[PI][PJ];

  // A_tile: complete on rows (parallel i reads) + 16-way on k (wide load + k-step-4)
#pragma HLS ARRAY_PARTITION variable=A_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_tile cyclic factor=LANES dim=2
  // B_panel: 32-way on rows (parallel j reads) + 16-way on k (wide load + k-step-4)
#pragma HLS ARRAY_PARTITION variable=B_panel cyclic factor=PJ dim=1
#pragma HLS ARRAY_PARTITION variable=B_panel cyclic factor=LANES dim=2
  // accumulators live in registers
#pragma HLS ARRAY_PARTITION variable=acc0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=acc1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0
#pragma HLS ARRAY_PARTITION variable=acc3 complete dim=0

  // =====================================================================
  // Outer column panel: B_panel stays resident while the whole A row
  // sweep runs, so B is read from DRAM exactly once over the kernel.
  // =====================================================================
  loop_jc: for (int jc = 0; jc < J; jc += BJ) {

    // ---- bulk load B column panel, 512-bit contiguous beats -----------
    load_B: for (int j = 0; j < BJ; j++) {
      load_B_k: for (int k = 0; k < K; k += LANES) {
#pragma HLS PIPELINE II=1
        load_B_l: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          B_panel[j][k + l] = B[jc + j][k + l];
        }
      }
    }

    // ---- sweep A row tiles -------------------------------------------
    loop_ic: for (int ic = 0; ic < I; ic += PI) {

      // bulk load A row tile, 512-bit contiguous beats
      load_A: for (int i = 0; i < PI; i++) {
        load_A_k: for (int k = 0; k < K; k += LANES) {
#pragma HLS PIPELINE II=1
          load_A_l: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            A_tile[i][k + l] = A[ic + i][k + l];
          }
        }
      }

      // ---- sub-blocks over the resident B panel ------------------------
      loop_js: for (int js = 0; js < NJ; js++) {

        init_acc: for (int i = 0; i < PI; i++) {
#pragma HLS UNROLL
          init_acc_j: for (int j = 0; j < PJ; j++) {
#pragma HLS UNROLL
            acc0[i][j] = (data_t)0;
            acc1[i][j] = (data_t)0;
            acc2[i][j] = (data_t)0;
            acc3[i][j] = (data_t)0;
          }
        }

        // ---- hot reduction: k advanced by 4, all four banks written ----
        // each accumulator bank is touched once per iteration; the loop body
        // carries PI*PJ*4 = 1024 independent MACs fed from banked locals.
        red_k: for (int k4 = 0; k4 < K4; k4++) {
#pragma HLS PIPELINE II=1
          red_i: for (int i = 0; i < PI; i++) {
#pragma HLS UNROLL
            red_j: for (int j = 0; j < PJ; j++) {
#pragma HLS UNROLL
              acc0[i][j] += A_tile[i][4*k4 + 0] * B_panel[js*PJ + j][4*k4 + 0];
              acc1[i][j] += A_tile[i][4*k4 + 1] * B_panel[js*PJ + j][4*k4 + 1];
              acc2[i][j] += A_tile[i][4*k4 + 2] * B_panel[js*PJ + j][4*k4 + 2];
              acc3[i][j] += A_tile[i][4*k4 + 3] * B_panel[js*PJ + j][4*k4 + 3];
            }
          }
        }

        // ---- tree-combine the four banks and store, 512-bit beats ------
        store_C: for (int i = 0; i < PI; i++) {
          store_C_j: for (int j = 0; j < PJ; j += LANES) {
#pragma HLS PIPELINE II=1
            store_C_l: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
              C[ic + i][jc + js*PJ + j + l] =
                  (acc0[i][j+l] + acc1[i][j+l]) + (acc2[i][j+l] + acc3[i][j+l]);
            }
          }
        }
      }
    }
  }
}
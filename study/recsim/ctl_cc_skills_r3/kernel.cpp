#include "kernel.h"

#define TI 32          // rows of A / C kept on-chip per tile (reuse factor for B rows)
#define P  16          // parallel MAC lanes along k
#define KB (K / P)     // k-blocks per dot product

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_read_burst_length=256 num_read_outstanding=16 latency=64
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_read_burst_length=256 num_read_outstanding=16 latency=64
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_read_burst_length=256 max_write_burst_length=256 num_read_outstanding=16 num_write_outstanding=16 latency=64
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  data_t A_loc[TI][K];
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=16 dim=2
  data_t C_loc[TI][J];
  data_t B_row[K];
#pragma HLS ARRAY_PARTITION variable=B_row cyclic factor=16 dim=1
  data_t acc[TI];

row_tile: for (int i0 = 0; i0 < I; i0 += TI) {

    // ---- load A tile and C tile (contiguous rows -> bursts) ----
  load_a_row: for (int r = 0; r < TI; r++) {
    load_a_col: for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
        A_loc[r][k] = A[i0 + r][k];
      }
    }
  load_c_row: for (int r = 0; r < TI; r++) {
    load_c_col: for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
        C_loc[r][j] = C[i0 + r][j];
      }
    }

    // ---- sweep over columns j: one B row reused for TI rows ----
  col_loop: for (int j = 0; j < J; j++) {

    load_b_row: for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
        B_row[k] = B[j][k];
      }

    init_acc: for (int ii = 0; ii < TI; ii++) {
#pragma HLS PIPELINE II=1
        acc[ii] = C_loc[ii][j];
      }

    compute_dot: for (int t = 0; t < KB * TI; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc type=inter direction=RAW distance=32 dependent=true
        int kb = t / TI;
        int ii = t % TI;
        int kbase = kb * P;

        data_t prod[P];
#pragma HLS ARRAY_PARTITION variable=prod complete dim=1
      mul_lanes: for (int p = 0; p < P; p++) {
#pragma HLS UNROLL
          prod[p] = A_loc[ii][kbase + p] * B_row[kbase + p];
        }
        data_t s8[8];
#pragma HLS ARRAY_PARTITION variable=s8 complete dim=1
      tree_l1: for (int p = 0; p < 8; p++) {
#pragma HLS UNROLL
          s8[p] = prod[2 * p] + prod[2 * p + 1];
        }
        data_t s4[4];
#pragma HLS ARRAY_PARTITION variable=s4 complete dim=1
      tree_l2: for (int p = 0; p < 4; p++) {
#pragma HLS UNROLL
          s4[p] = s8[2 * p] + s8[2 * p + 1];
        }
        data_t s2[2];
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1
      tree_l3: for (int p = 0; p < 2; p++) {
#pragma HLS UNROLL
          s2[p] = s4[2 * p] + s4[2 * p + 1];
        }
        data_t s = s2[0] + s2[1];
        acc[ii] = acc[ii] + s;
      }

    writeback_acc: for (int ii = 0; ii < TI; ii++) {
#pragma HLS PIPELINE II=1
        C_loc[ii][j] = acc[ii];
      }
    }

    // ---- store C tile once ----
  store_c_row: for (int r = 0; r < TI; r++) {
    store_c_col: for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
        C[i0 + r][j] = C_loc[r][j];
      }
    }
  }
}
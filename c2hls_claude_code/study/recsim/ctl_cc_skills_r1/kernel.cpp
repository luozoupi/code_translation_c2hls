#include "kernel.h"

#define TI 64          // rows of A/C processed per block (parallel MAC lanes)
#define TJ 16          // rows of B per tile (accumulator reuse distance)
#define NIB (I / TI)   // number of row blocks
#define NJT (J / TJ)   // number of B tiles per row block
#define TILE_N (TJ * K) // flattened iterations per tile

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static data_t Aloc[TI][K];
#pragma HLS ARRAY_PARTITION variable=Aloc complete dim=1
  static data_t Cloc[TI][J];
#pragma HLS ARRAY_PARTITION variable=Cloc complete dim=1
  static data_t B0[TJ][K];
  static data_t B1[TJ][K];
  data_t acc[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

  blk_i: for (int ib = 0; ib < NIB; ib++) {
    const int i0 = ib * TI;

    load_A_row: for (int ii = 0; ii < TI; ii++) {
      load_A_col: for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
        Aloc[ii][k] = A[i0 + ii][k];
      }
    }

    load_C_row: for (int ii = 0; ii < TI; ii++) {
      load_C_col: for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
        Cloc[ii][j] = C[i0 + ii][j];
      }
    }

    // prologue: B tile 0 into B0
    load_B0: for (int n = 0; n < TILE_N; n++) {
#pragma HLS PIPELINE II=1
      const int lj = n / K;
      const int lk = n % K;
      B0[lj][lk] = B[lj][lk];
    }

    tile_j: for (int t = 0; t < NJT; t++) {
      const int cur = t & 1;
      const int j0 = t * TJ;
      const int nt = (t + 1 < NJT) ? (t + 1) : t;  // next tile (clamped, harmless reload)
      const int nj0 = nt * TJ;

      init_acc: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS PIPELINE II=1
        init_acc_lane: for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
          acc[ii][jj] = Cloc[ii][j0 + jj];
        }
      }

      compute_and_prefetch: for (int n = 0; n < TILE_N; n++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
#pragma HLS DEPENDENCE variable=B0 inter false
#pragma HLS DEPENDENCE variable=B1 inter false
#pragma HLS DEPENDENCE variable=B0 intra false
#pragma HLS DEPENDENCE variable=B1 intra false
        // compute indices: k outer, jj inner (acc reuse distance = TJ)
        const int k  = n / TJ;
        const int jj = n % TJ;
        // prefetch indices: sequential global addresses for burst
        const int lj = n / K;
        const int lk = n % K;

        data_t nb = B[nj0 + lj][lk];
        data_t b0 = B0[jj][k];
        data_t b1 = B1[jj][k];
        data_t b = cur ? b1 : b0;
        if (cur) {
          B0[lj][lk] = nb;
        } else {
          B1[lj][lk] = nb;
        }

        mac_lane: for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
          acc[ii][jj] = acc[ii][jj] + Aloc[ii][k] * b;
        }
      }

      writeback_acc: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS PIPELINE II=1
        writeback_lane: for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
          Cloc[ii][j0 + jj] = acc[ii][jj];
        }
      }
    }

    store_C_row: for (int ii = 0; ii < TI; ii++) {
      store_C_col: for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
        C[i0 + ii][j] = Cloc[ii][j];
      }
    }
  }
}
#include "kernel.h"

// C[i][j] += sum_k A[i][k] * B[j][k]
// B is kept fully on chip (URAM, 16 k-banks).  Rows of A/C are processed in
// blocks of TI rows; per cycle the datapath does TI x LANES MACs:
//   for each 16-wide k chunk kc, for each j (pipelined II=1):
//     C_loc[ii][j] += tree_sum_l( A[i0+ii][16kc+l] * B[j][16kc+l] )   (ii unrolled)
// Consecutive pipeline iterations touch different C_loc[.][j] so the FP add
// recurrence distance is J=1024 iterations (no II penalty).

#define LANES 16
#define TI 104
#define NKC (K / LANES)            // 64 chunks of k
#define NJC (J / LANES)            // 64 beats per C row
#define NBLK ((I + TI - 1) / TI)   // row blocks

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static data_t B_loc[J][K];
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=16 dim=2
#pragma HLS BIND_STORAGE variable=B_loc type=ram_2p impl=uram

  static data_t A_loc[NKC][TI][LANES];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=2
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=3

  static data_t C_loc[TI][J];
#pragma HLS ARRAY_PARTITION variable=C_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_loc cyclic factor=16 dim=2

  // ---- load all of B once (one 512-bit beat per cycle) ----
  load_B: for (int t = 0; t < J * NKC; t++) {
#pragma HLS PIPELINE II=1
    int j = t / NKC;
    int c = t % NKC;
    load_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      B_loc[j][c * LANES + l] = B[j][c * LANES + l];
    }
  }

  row_block: for (int b = 0; b < NBLK; b++) {
    const int i0 = b * TI;

    // ---- load A rows and C rows of this block (gmem0 + gmem2 in parallel) ----
    load_AC: for (int t = 0; t < TI * NKC; t++) {
#pragma HLS PIPELINE II=1
      int ii = t / NKC;
      int c = t % NKC;
      int row = i0 + ii;
      bool valid = row < I;
      int rrow = valid ? row : (I - 1);
      load_AC_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        data_t av = A[rrow][c * LANES + l];
        data_t cv = C[rrow][c * LANES + l];
        A_loc[c][ii][l] = valid ? av : (data_t)0;
        C_loc[ii][c * LANES + l] = valid ? cv : (data_t)0;
      }
    }

    // ---- compute: TI x LANES MACs per cycle ----
    compute_kc: for (int kc = 0; kc < NKC; kc++) {
      compute_j: for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
#pragma HLS DEPENDENCE variable=C_loc inter false
        data_t bv[LANES];
#pragma HLS ARRAY_PARTITION variable=bv complete
        read_b: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          bv[l] = B_loc[j][kc * LANES + l];
        }
        pe_row: for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
          data_t p[LANES];
#pragma HLS ARRAY_PARTITION variable=p complete
          mul_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            p[l] = A_loc[kc][ii][l] * bv[l];
          }
          data_t s8[8];
#pragma HLS ARRAY_PARTITION variable=s8 complete
          tree8: for (int l = 0; l < 8; l++) {
#pragma HLS UNROLL
            s8[l] = p[2 * l] + p[2 * l + 1];
          }
          data_t s4[4];
#pragma HLS ARRAY_PARTITION variable=s4 complete
          tree4: for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
            s4[l] = s8[2 * l] + s8[2 * l + 1];
          }
          data_t s01 = s4[0] + s4[1];
          data_t s23 = s4[2] + s4[3];
          data_t sum = s01 + s23;
          C_loc[ii][j] = C_loc[ii][j] + sum;
        }
      }
    }

    // ---- store C rows of this block ----
    store_C: for (int t = 0; t < TI * NJC; t++) {
#pragma HLS PIPELINE II=1
      int ii = t / NJC;
      int c = t % NJC;
      int row = i0 + ii;
      if (row < I) {
        store_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          C[row][c * LANES + l] = C_loc[ii][c * LANES + l];
        }
      }
    }
  }
}
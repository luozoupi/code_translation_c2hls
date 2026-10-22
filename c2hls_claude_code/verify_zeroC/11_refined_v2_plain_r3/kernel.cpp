#include "kernel.h"

#define PI_BLK 26                         // A rows processed in parallel
#define KL     64                         // k lanes per cycle (adder tree width)
#define LANES  16                         // floats per 512-bit beat
#define NKC    (K / KL)                   // k chunks
#define NBLK   ((I + PI_BLK - 1) / PI_BLK)

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Full B resident on chip (URAM), banked along k for KL parallel reads.
  static data_t B_loc[J][K];
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=64 dim=2
#pragma HLS BIND_STORAGE variable=B_loc type=ram_2p impl=uram

  // A row-block tile: every (row, k-lane) is its own small bank.
  static data_t A_t[PI_BLK][K];
#pragma HLS ARRAY_PARTITION variable=A_t complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_t cyclic factor=64 dim=2
#pragma HLS BIND_STORAGE variable=A_t type=ram_2p impl=lutram

  // Accumulators for C rows of the block (preloaded with C).
  static data_t acc[PI_BLK][J];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=16 dim=2

  // ---------------- load full B once (512-bit beats) ----------------
load_B_row:
  for (int j = 0; j < J; j++) {
  load_B_beat:
    for (int kb = 0; kb < K / LANES; kb++) {
#pragma HLS PIPELINE II=1
    load_B_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        B_loc[j][kb * LANES + l] = B[j][kb * LANES + l];
      }
    }
  }

  // ---------------- process A/C row blocks ----------------
block_loop:
  for (int blk = 0; blk < NBLK; blk++) {
    const int i0 = blk * PI_BLK;

    // load A tile and C rows concurrently (separate bundles)
  load_AC_row:
    for (int pi = 0; pi < PI_BLK; pi++) {
      int row = i0 + pi;
      int row_c = (row < I) ? row : (I - 1);   // clamp tail; result discarded
    load_AC_beat:
      for (int kb = 0; kb < K / LANES; kb++) {
#pragma HLS PIPELINE II=1
      load_AC_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          A_t[pi][kb * LANES + l] = A[row_c][kb * LANES + l];
          acc[pi][kb * LANES + l] = (data_t)0;
        }
      }
    }

    // compute: (kc, j) flattened pipeline, 26 x 64 MACs per cycle
  compute_kc:
    for (int kc = 0; kc < NKC; kc++) {
    compute_j:
      for (int j = 0; j < J; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
        data_t bv[KL];
#pragma HLS ARRAY_PARTITION variable=bv complete
      read_b:
        for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
          bv[l] = B_loc[j][kc * KL + l];
        }
      pe_row:
        for (int pi = 0; pi < PI_BLK; pi++) {
#pragma HLS UNROLL
          data_t p[KL];
#pragma HLS ARRAY_PARTITION variable=p complete
          data_t s32[32];
#pragma HLS ARRAY_PARTITION variable=s32 complete
          data_t s16[16];
#pragma HLS ARRAY_PARTITION variable=s16 complete
          data_t s8[8];
#pragma HLS ARRAY_PARTITION variable=s8 complete
          data_t s4[4];
#pragma HLS ARRAY_PARTITION variable=s4 complete
          data_t s2[2];
#pragma HLS ARRAY_PARTITION variable=s2 complete
        mul_lane:
          for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
            p[l] = A_t[pi][kc * KL + l] * bv[l];
          }
        tree_32:
          for (int l = 0; l < 32; l++) {
#pragma HLS UNROLL
            s32[l] = p[2 * l] + p[2 * l + 1];
          }
        tree_16:
          for (int l = 0; l < 16; l++) {
#pragma HLS UNROLL
            s16[l] = s32[2 * l] + s32[2 * l + 1];
          }
        tree_8:
          for (int l = 0; l < 8; l++) {
#pragma HLS UNROLL
            s8[l] = s16[2 * l] + s16[2 * l + 1];
          }
        tree_4:
          for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
            s4[l] = s8[2 * l] + s8[2 * l + 1];
          }
        tree_2:
          for (int l = 0; l < 2; l++) {
#pragma HLS UNROLL
            s2[l] = s4[2 * l] + s4[2 * l + 1];
          }
          data_t s1 = s2[0] + s2[1];
          acc[pi][j] = acc[pi][j] + s1;
        }
      }
    }

    // store C rows (contiguous 512-bit beats)
  store_C_row:
    for (int pi = 0; pi < PI_BLK; pi++) {
      int row = i0 + pi;
      if (row < I) {
      store_C_beat:
        for (int jb = 0; jb < J / LANES; jb++) {
#pragma HLS PIPELINE II=1
        store_C_lane:
          for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            C[row][jb * LANES + l] = acc[pi][jb * LANES + l];
          }
        }
      }
    }
  }
}
#include "kernel.h"

#define PE_I 4      // rows of A/C per block
#define PE_J 16     // output columns per cycle
#define KL   16     // k-lanes per cycle (one 512-bit beat of floats)
#define LANES 16    // 512 / 32

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Full B resident on chip (URAM), banked 16 (j) x 16 (k)
  static data_t B_loc[J][K];
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=16 dim=2
#pragma HLS BIND_STORAGE variable=B_loc type=ram_2p impl=uram

  // Per-row-block A and C staging
  data_t A_loc[PE_I][K];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=16 dim=2
  data_t C_loc[PE_I][J];
#pragma HLS ARRAY_PARTITION variable=C_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_loc cyclic factor=16 dim=2

  // ---------------- Load all of B once ----------------
load_B_row: for (int j = 0; j < J; j++) {
  load_B_beat: for (int kk = 0; kk < K / LANES; kk++) {
#pragma HLS PIPELINE II=1
    load_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      B_loc[j][kk * LANES + l] = B[j][kk * LANES + l];
    }
  }
}

  // ---------------- Row blocks ----------------
row_block: for (int ib = 0; ib < I / PE_I; ib++) {
  const int i0 = ib * PE_I;

  // Load A rows and C rows (separate bundles, parallel beats)
load_AC_row: for (int pi = 0; pi < PE_I; pi++) {
  load_AC_beat: for (int kk = 0; kk < K / LANES; kk++) {
#pragma HLS PIPELINE II=1
      load_AC_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        A_loc[pi][kk * LANES + l] = A[i0 + pi][kk * LANES + l];
        C_loc[pi][kk * LANES + l] = C[i0 + pi][kk * LANES + l];
      }
    }
  }

  // Compute: flattened (kb, jg) pipeline, 4x16x16 MACs per cycle.
  // Same C_loc element revisited every J/PE_J = 64 iterations (> pipeline depth).
compute_k: for (int kb = 0; kb < K / KL; kb++) {
  compute_j: for (int jg = 0; jg < J / PE_J; jg++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_loc inter false
      compute_pi: for (int pi = 0; pi < PE_I; pi++) {
#pragma HLS UNROLL
        compute_pj: for (int pj = 0; pj < PE_J; pj++) {
#pragma HLS UNROLL
          const int j = jg * PE_J + pj;
          data_t p[KL];
#pragma HLS ARRAY_PARTITION variable=p complete
          mul_k: for (int kl = 0; kl < KL; kl++) {
#pragma HLS UNROLL
            p[kl] = A_loc[pi][kb * KL + kl] * B_loc[j][kb * KL + kl];
          }
          data_t s8[8];
#pragma HLS ARRAY_PARTITION variable=s8 complete
          tree_l1: for (int t = 0; t < 8; t++) {
#pragma HLS UNROLL
            s8[t] = p[2 * t] + p[2 * t + 1];
          }
          data_t s4[4];
#pragma HLS ARRAY_PARTITION variable=s4 complete
          tree_l2: for (int t = 0; t < 4; t++) {
#pragma HLS UNROLL
            s4[t] = s8[2 * t] + s8[2 * t + 1];
          }
          data_t s2a = s4[0] + s4[1];
          data_t s2b = s4[2] + s4[3];
          data_t sum = s2a + s2b;
          C_loc[pi][j] = C_loc[pi][j] + sum;
        }
      }
    }
  }

  // Store C rows back once
store_C_row: for (int pi = 0; pi < PE_I; pi++) {
  store_C_beat: for (int jj = 0; jj < J / LANES; jj++) {
#pragma HLS PIPELINE II=1
      store_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        C[i0 + pi][jj * LANES + l] = C_loc[pi][jj * LANES + l];
      }
    }
  }
}
}
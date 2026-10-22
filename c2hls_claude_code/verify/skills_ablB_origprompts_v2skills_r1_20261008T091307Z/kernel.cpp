#include "kernel.h"

// Parallel datapath sizing (U280, fp32 MAC ~5 DSP):
//   P_I x P_J outputs per cycle, each with a KU-wide k dot-product tree
//   => P_I*P_J*KU = 7*16*16 = 1792 MACs/cycle
//   DSP ~ 1792*3 (fmul) + 112*15*2 (tree fadd) = 8736; final accumulate fadd on fabric
#define P_I   7
#define P_J   16
#define KU    16
#define LANES 16
#define TII   21                       // ii blocks per tile
#define TI    (TII * P_I)              // 147 rows of A/C per tile
#define NT    ((I + TI - 1) / TI)      // 7 tiles (1029 rows, last 5 guarded)
#define JB    (J / P_J)                // 64
#define KC    (K / KU)                 // 64

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Full B resident on chip: B_loc[pj][jj][k] holds B[jj*P_J+pj][k]
  static data_t B_loc[P_J][JB][K];
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=16 dim=3
#pragma HLS BIND_STORAGE variable=B_loc type=ram_2p impl=uram

  // A tile: A_loc[pi][ii][k] holds A[t*TI + ii*P_I + pi][k]
  static data_t A_loc[P_I][TII][K];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=16 dim=3

  // C tile accumulator: C_loc[pi][ii][j] holds C[t*TI + ii*P_I + pi][j]
  static data_t C_loc[P_I][TII][J];
#pragma HLS ARRAY_PARTITION variable=C_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_loc cyclic factor=16 dim=3

  // ---------------- Load all of B once ----------------
  load_B_jj: for (int jj = 0; jj < JB; jj++) {
    load_B_pj: for (int pj = 0; pj < P_J; pj++) {
      load_B_k: for (int c = 0; c < K; c += LANES) {
#pragma HLS PIPELINE II=1
        load_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          B_loc[pj][jj][c + l] = B[jj * P_J + pj][c + l];
        }
      }
    }
  }

  // ---------------- Row tiles of A / C ----------------
  tile_loop: for (int t = 0; t < NT; t++) {
    const int row0 = t * TI;

    // Load A tile and C tile together (separate AXI bundles)
    load_AC_ii: for (int ii = 0; ii < TII; ii++) {
      load_AC_pi: for (int pi = 0; pi < P_I; pi++) {
        load_AC_k: for (int c = 0; c < K; c += LANES) {
#pragma HLS PIPELINE II=1
          int gi = row0 + ii * P_I + pi;
          load_AC_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            if (gi < I) {
              A_loc[pi][ii][c + l] = A[gi][c + l];
              C_loc[pi][ii][c + l] = C[gi][c + l];
            } else {
              A_loc[pi][ii][c + l] = 0;
              C_loc[pi][ii][c + l] = 0;
            }
          }
        }
      }
    }

    // Compute: k-chunk outer; (ii, jj) inner pipelined -> each C_loc element
    // is revisited only every TII*JB = 1344 iterations: no FP recurrence on the pipeline.
    compute_kc: for (int kc = 0; kc < KC; kc++) {
      compute_ii: for (int ii = 0; ii < TII; ii++) {
        compute_jj: for (int jj = 0; jj < JB; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_loc inter false
          pe_i: for (int pi = 0; pi < P_I; pi++) {
#pragma HLS UNROLL
            pe_j: for (int pj = 0; pj < P_J; pj++) {
#pragma HLS UNROLL
              data_t prod[KU];
#pragma HLS ARRAY_PARTITION variable=prod complete
              mul_k: for (int u = 0; u < KU; u++) {
#pragma HLS UNROLL
                prod[u] = A_loc[pi][ii][kc * KU + u] * B_loc[pj][jj][kc * KU + u];
              }
              data_t s8[8];
#pragma HLS ARRAY_PARTITION variable=s8 complete
              tree_l1: for (int u = 0; u < 8; u++) {
#pragma HLS UNROLL
                s8[u] = prod[2 * u] + prod[2 * u + 1];
              }
              data_t s4[4];
#pragma HLS ARRAY_PARTITION variable=s4 complete
              tree_l2: for (int u = 0; u < 4; u++) {
#pragma HLS UNROLL
                s4[u] = s8[2 * u] + s8[2 * u + 1];
              }
              data_t s2[2];
#pragma HLS ARRAY_PARTITION variable=s2 complete
              tree_l3: for (int u = 0; u < 2; u++) {
#pragma HLS UNROLL
                s2[u] = s4[2 * u] + s4[2 * u + 1];
              }
              data_t psum = s2[0] + s2[1];
              int jcol = jj * P_J + pj;
              data_t acc_new = C_loc[pi][ii][jcol] + psum;
#pragma HLS BIND_OP variable=acc_new op=fadd impl=fabric
              C_loc[pi][ii][jcol] = acc_new;
            }
          }
        }
      }
    }

    // Store C tile (guard partial last tile)
    store_C_ii: for (int ii = 0; ii < TII; ii++) {
      store_C_pi: for (int pi = 0; pi < P_I; pi++) {
        store_C_j: for (int c = 0; c < J; c += LANES) {
#pragma HLS PIPELINE II=1
          int gi = row0 + ii * P_I + pi;
          store_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            if (gi < I) {
              C[gi][c + l] = C_loc[pi][ii][c + l];
            }
          }
        }
      }
    }
  }
}
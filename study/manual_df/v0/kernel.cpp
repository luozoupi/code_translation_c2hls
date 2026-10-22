// v0: ctl_cc_refnf_r2 unchanged except C = A*B^T (no C load; accumulators start at 0). Serial B / tiles as before.
#include "kernel.h"

// ---------------------------------------------------------------------------
// Tiling / parallelism parameters
//   PI x PJ outputs per cycle, each a KL-wide dot-product slice
//   => PI*PJ*KL = 4*16*16 = 1024 fp32 MACs per cycle
// ---------------------------------------------------------------------------
#define TI     64              // rows of A / C kept on chip per tile
#define PI     4               // parallel output rows
#define PJ     16              // parallel output cols
#define KL     16              // k lanes per cycle (adder tree)
#define LANES  16              // floats per 512-bit AXI beat
#define MAXW   ((K > J) ? K : J)

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Whole B resident on chip (URAM), banked for PJ rows x KL k-lanes
  static data_t B_loc[J][K];
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=16 dim=2
#pragma HLS BIND_STORAGE variable=B_loc type=ram_2p impl=uram

  // Row tile of A, banked for PI rows x KL k-lanes
  static data_t A_loc[TI][K];
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=16 dim=2

  // Row tile of C (accumulator), banked for PI rows x PJ cols
  static data_t C_loc[TI][J];
#pragma HLS ARRAY_PARTITION variable=C_loc cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=C_loc cyclic factor=16 dim=2

  // ---------------- load all of B once (512-bit beats) ----------------
  load_B_row: for (int j = 0; j < J; j++) {
    load_B_beat: for (int kb = 0; kb < K / LANES; kb++) {
#pragma HLS PIPELINE II=1
      load_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        B_loc[j][kb * LANES + l] = B[j][kb * LANES + l];
      }
    }
  }

  // ---------------- process row tiles of A / C ----------------
  tile_i: for (int it = 0; it < I / TI; it++) {
    const int i0 = it * TI;

    // fused load of A tile (gmem0) and C tile (gmem2)
    load_AC_row: for (int r = 0; r < TI; r++) {
      load_AC_beat: for (int b = 0; b < MAXW / LANES; b++) {
#pragma HLS PIPELINE II=1
        load_AC_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          const int idx = b * LANES + l;
          if (idx < K) A_loc[r][idx] = A[i0 + r][idx];
        }
      }
    }

    // compute: k-chunk outermost so each C_loc element is revisited
    // only every (TI/PI)*(J/PJ) = 1024 cycles -> no FP-add recurrence stall
    compute_kb: for (int kb = 0; kb < K / KL; kb++) {
      compute_ii: for (int ii = 0; ii < TI / PI; ii++) {
        compute_jb: for (int jb = 0; jb < J / PJ; jb++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_loc type=inter direction=RAW dependent=true distance=1024
          pe_row: for (int p = 0; p < PI; p++) {
#pragma HLS UNROLL
            pe_col: for (int q = 0; q < PJ; q++) {
#pragma HLS UNROLL
              const int ri = ii * PI + p;
              const int cj = jb * PJ + q;
              data_t prod[KL];
#pragma HLS ARRAY_PARTITION variable=prod complete dim=1
              mac_lane: for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
                prod[l] = A_loc[ri][kb * KL + l] * B_loc[cj][kb * KL + l];
              }
              tree_l8: for (int l = 0; l < 8; l++) {
#pragma HLS UNROLL
                prod[l] = prod[l] + prod[l + 8];
              }
              tree_l4: for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
                prod[l] = prod[l] + prod[l + 4];
              }
              tree_l2: for (int l = 0; l < 2; l++) {
#pragma HLS UNROLL
                prod[l] = prod[l] + prod[l + 2];
              }
              data_t s = prod[0] + prod[1];
              C_loc[ri][cj] = (kb == 0 ? (data_t)0 : C_loc[ri][cj]) + s;   // C = A*B^T: no C read
            }
          }
        }
      }
    }

    // store C tile (512-bit beats)
    store_C_row: for (int r = 0; r < TI; r++) {
      store_C_beat: for (int b = 0; b < J / LANES; b++) {
#pragma HLS PIPELINE II=1
        store_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          C[i0 + r][b * LANES + l] = C_loc[r][b * LANES + l];
        }
      }
    }
  }
}
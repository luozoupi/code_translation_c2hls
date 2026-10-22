#include "kernel.h"

// ---------------- PE array / tiling parameters ----------------
#define P_I   7                   // spatial PE rows (output rows per cycle)
#define P_J   16                  // spatial PE cols (output cols per cycle)
#define KL    16                  // k-lanes per PE (adder tree width)
#define KC    (K / KL)            // 64 k-chunks
#define RB    74                  // row blocks per row tile
#define TI    (P_I * RB)          // 518 rows per row tile
#define CB    16                  // col blocks per col tile
#define TJ    (P_J * CB)          // 256 cols per col tile
#define NTI   ((I + TI - 1) / TI) // 2 row tiles (last one partial: 506 rows)
#define NTJ   (J / TJ)            // 4 col tiles
#define ACCD  (RB * CB)           // 1184 accumulators per PE bank (dependence distance)

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // A row tile: [pe_row][row_block][k_chunk][k_lane]
  data_t A_loc[P_I][RB][KC][KL];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=4

  // B col tile: [pe_col][col_block][k_chunk][k_lane]
  data_t B_loc[P_J][CB][KC][KL];
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=4

  // C accumulators: [pe_row][pe_col][row_block*CB + col_block]
  data_t acc[P_I][P_J][ACCD];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

  tile_i: for (int ti = 0; ti < NTI; ti++) {
    const int i0 = ti * TI;
    const int rows = (I - i0 < TI) ? (I - i0) : TI;

    // ---------------- load A row tile (contiguous rows, 16 floats/cycle) ----------------
    {
      int kc = 0, pi = 0, rb = 0, il = 0;
      load_A: for (int t = 0; t < rows * KC; t++) {
#pragma HLS LOOP_TRIPCOUNT min=TI*KC max=TI*KC avg=TI*KC
#pragma HLS PIPELINE II=1
        load_A_lane: for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
          A_loc[pi][rb][kc][l] = A[i0 + il][kc * KL + l];
        }
        if (kc == KC - 1) {
          kc = 0;
          il++;
          if (pi == P_I - 1) { pi = 0; rb++; } else { pi++; }
        } else {
          kc++;
        }
      }
    }

    tile_j: for (int tj = 0; tj < NTJ; tj++) {
      const int j0 = tj * TJ;

      // ---------------- load B col tile (contiguous rows of B, 16 floats/cycle) ----------------
      load_B: for (int t = 0; t < TJ * KC; t++) {
#pragma HLS PIPELINE II=1
        const int jl = t / KC;      // power of two -> shift
        const int kc = t % KC;
        const int pj = jl % P_J;
        const int cb = jl / P_J;
        load_B_lane: for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
          B_loc[pj][cb][kc][l] = B[j0 + jl][kc * KL + l];
        }
      }

      // ---------------- load C tile into accumulators (16 floats/cycle) ----------------
      {
        int cb = 0, pi = 0, rb = 0, il = 0;
        load_C: for (int t = 0; t < rows * CB; t++) {
#pragma HLS LOOP_TRIPCOUNT min=TI*CB max=TI*CB avg=TI*CB
#pragma HLS PIPELINE II=1
          load_C_lane: for (int l = 0; l < P_J; l++) {
#pragma HLS UNROLL
            acc[pi][l][rb * CB + cb] = C[i0 + il][j0 + cb * P_J + l];
          }
          if (cb == CB - 1) {
            cb = 0;
            il++;
            if (pi == P_I - 1) { pi = 0; rb++; } else { pi++; }
          } else {
            cb++;
          }
        }
      }

      // ---------------- compute: 7x16 outputs x 16 k-lanes = 1792 MACs per cycle ----------------
      compute_k: for (int kc = 0; kc < KC; kc++) {
        compute_rb: for (int rb = 0; rb < RB; rb++) {
          compute_cb: for (int cb = 0; cb < CB; cb++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
            // same acc address recurs only every RB*CB = 1184 cycles
            const int idx = rb * CB + cb;
            pe_row: for (int pi = 0; pi < P_I; pi++) {
#pragma HLS UNROLL
              pe_col: for (int pj = 0; pj < P_J; pj++) {
#pragma HLS UNROLL
                data_t p16[KL];
#pragma HLS ARRAY_PARTITION variable=p16 complete dim=1
                mac_lane: for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
                  p16[l] = A_loc[pi][rb][kc][l] * B_loc[pj][cb][kc][l];
                }
                data_t p8[8];
#pragma HLS ARRAY_PARTITION variable=p8 complete dim=1
                tree_l1: for (int x = 0; x < 8; x++) {
#pragma HLS UNROLL
                  p8[x] = p16[2 * x] + p16[2 * x + 1];
                }
                data_t p4[4];
#pragma HLS ARRAY_PARTITION variable=p4 complete dim=1
                tree_l2: for (int x = 0; x < 4; x++) {
#pragma HLS UNROLL
                  p4[x] = p8[2 * x] + p8[2 * x + 1];
                }
                data_t p2[2];
#pragma HLS ARRAY_PARTITION variable=p2 complete dim=1
                tree_l3: for (int x = 0; x < 2; x++) {
#pragma HLS UNROLL
                  p2[x] = p4[2 * x] + p4[2 * x + 1];
                }
                const data_t s = p2[0] + p2[1];
                // final accumulate on LUT fabric to keep DSP strictly below device cap
                data_t nv = acc[pi][pj][idx] + s;
#pragma HLS BIND_OP variable=nv op=fadd impl=fabric
                acc[pi][pj][idx] = nv;
              }
            }
          }
        }
      }

      // ---------------- store C tile (16 floats/cycle) ----------------
      {
        int cb = 0, pi = 0, rb = 0, il = 0;
        store_C: for (int t = 0; t < rows * CB; t++) {
#pragma HLS LOOP_TRIPCOUNT min=TI*CB max=TI*CB avg=TI*CB
#pragma HLS PIPELINE II=1
          store_C_lane: for (int l = 0; l < P_J; l++) {
#pragma HLS UNROLL
            C[i0 + il][j0 + cb * P_J + l] = acc[pi][l][rb * CB + cb];
          }
          if (cb == CB - 1) {
            cb = 0;
            il++;
            if (pi == P_I - 1) { pi = 0; rb++; } else { pi++; }
          } else {
            cb++;
          }
        }
      }
    }
  }
}
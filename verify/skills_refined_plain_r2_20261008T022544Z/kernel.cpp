#include "kernel.h"

// ---------------------------------------------------------------------------
// Output-stationary tiled GEMM:  C[i][j] += sum_k A[i][k] * B[j][k]
//   Spatial PE array : PE_I x PE_J = 52 x 32 = 1664 fp32 MACs per cycle
//   Time-multiplexed : TT = 16 output columns per PE (hides FP add latency)
//   Column tile      : TJ = PE_J * TT = 512 ; B slab kept on-chip per j-tile
//   Row tile         : PE_I = 52 rows (last tile padded / guarded)
// Column mapping inside a j-tile: jj = t * PE_J + pj
// ---------------------------------------------------------------------------
#define PE_I   52
#define PE_J   32
#define TT     16
#define TJ     (PE_J * TT)                 // 512
#define LANES  16                          // 512-bit / 32-bit float
#define KV     (K / LANES)                 // 64 beats per row of A/B
#define CV     (TJ / LANES)                // 32 beats per C row segment
#define NUM_IT ((I + PE_I - 1) / PE_I)     // 20 row tiles
#define NUM_JT (J / TJ)                    // 2 column tiles

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // A row tile: one bank per PE row, k banked by LANES for 16-wide loads
  data_t A_loc[PE_I][K];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=16 dim=2

  // B column slab: B_loc[pj][t][k] holds row (j0 + t*PE_J + pj)
  data_t B_loc[PE_J][TT][K];
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=16 dim=3

  // Accumulators: one 16-deep LUTRAM per PE
  data_t acc[PE_I][PE_J][TT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram

  j_tile: for (int jt = 0; jt < NUM_JT; jt++) {
    const int j0 = jt * TJ;

    // ---- load B slab once per column tile (reused across all row tiles)
    load_B_row: for (int r = 0; r < TJ; r++) {
      load_B_k: for (int kv = 0; kv < KV; kv++) {
#pragma HLS PIPELINE II=1
        load_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          B_loc[r % PE_J][r / PE_J][kv * LANES + l] = B[j0 + r][kv * LANES + l];
        }
      }
    }

    i_tile: for (int it = 0; it < NUM_IT; it++) {
      const int i0 = it * PE_I;

      // ---- load A row tile (pad rows beyond I with zeros)
      load_A_row: for (int r = 0; r < PE_I; r++) {
        load_A_k: for (int kv = 0; kv < KV; kv++) {
#pragma HLS PIPELINE II=1
          const int row = i0 + r;
          load_A_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            data_t v = 0;
            if (row < I) v = A[row][kv * LANES + l];
            A_loc[r][kv * LANES + l] = v;
          }
        }
      }

      // ---- load C block into accumulators (C is accumulated into)
      load_C_row: for (int r = 0; r < PE_I; r++) {
        load_C_col: for (int cv = 0; cv < CV; cv++) {
#pragma HLS PIPELINE II=1
          const int row = i0 + r;
          load_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            const int jj = cv * LANES + l;
            data_t v = 0;
            if (row < I) v = C[row][j0 + jj];
            acc[r][jj % PE_J][jj / PE_J] = v;
          }
        }
      }

      // ---- compute: 1664 MACs per cycle, acc reuse distance = TT iterations
      compute_kt: for (int kt = 0; kt < K * TT; kt++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
        const int k = kt >> 4;
        const int t = kt & (TT - 1);
        compute_pi: for (int pi = 0; pi < PE_I; pi++) {
#pragma HLS UNROLL
          const data_t a = A_loc[pi][k];
          compute_pj: for (int pj = 0; pj < PE_J; pj++) {
#pragma HLS UNROLL
            acc[pi][pj][t] = acc[pi][pj][t] + a * B_loc[pj][t][k];
          }
        }
      }

      // ---- store C block once
      store_C_row: for (int r = 0; r < PE_I; r++) {
        store_C_col: for (int cv = 0; cv < CV; cv++) {
#pragma HLS PIPELINE II=1
          const int row = i0 + r;
          store_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            const int jj = cv * LANES + l;
            if (row < I) C[row][j0 + jj] = acc[r][jj % PE_J][jj / PE_J];
          }
        }
      }
    }
  }
}
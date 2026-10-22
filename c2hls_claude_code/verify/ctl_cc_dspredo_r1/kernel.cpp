#include "kernel.h"

// ---------------- tiling / parallelism parameters ----------------
#define TI     512                    // rows of A / C kept on chip per i-block
#define TJ     208                    // parallel output columns (j lanes) = 13 x 16-float beats
#define UK     8                      // k-unroll (multipliers per j lane)
#define LANES  16                     // 512-bit beat = 16 floats
#define KB     (K / LANES)            // beats per row of A/B (64)
#define NKC    (K / UK)               // k chunks (128)
#define NIT    (I / TI)               // i-blocks (2)
#define NJT    ((J + TJ - 1) / TJ)    // j-tiles (5; last one = 96 cols)
#define CB     (TJ / LANES)           // beats per C tile row (13)

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // A row-block: [row][k/16][k%16], 16 banks on the inner k dim
  static data_t A_loc[TI][KB][LANES];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=3

  // B tile: one private bank set per j lane, 16 k-banks each (LUTRAM)
  static data_t B_loc[TJ][KB][LANES];
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=3
#pragma HLS BIND_STORAGE variable=B_loc type=ram_2p impl=lutram

  // C accumulators: one bank per j lane
  static data_t acc[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

  tile_i: for (int it = 0; it < NIT; it++) {
    const int i0 = it * TI;

    // ---- load A row-block (contiguous, widened) ----
    load_A: for (int t = 0; t < TI * KB; t++) {
#pragma HLS PIPELINE II=1
      const int r  = t / KB;
      const int kb = t % KB;
      load_A_lane: for (int u = 0; u < LANES; u++) {
#pragma HLS UNROLL
        A_loc[r][kb][u] = A[i0 + r][kb * LANES + u];
      }
    }

    tile_j: for (int jt = 0; jt < NJT; jt++) {
      const int j0    = jt * TJ;
      const int ncols = (J - j0 < TJ) ? (J - j0) : TJ;   // 208 or 96 (tail)
      const int nb    = ncols / LANES;                    // 13 or 6

      // ---- load B tile rows j0..j0+ncols-1 (contiguous, widened) ----
      load_B: for (int t = 0; t < ncols * KB; t++) {
#pragma HLS LOOP_TRIPCOUNT min=6144 avg=13312 max=13312
#pragma HLS PIPELINE II=1
        const int r  = t / KB;
        const int kb = t % KB;
        load_B_lane: for (int u = 0; u < LANES; u++) {
#pragma HLS UNROLL
          B_loc[r][kb][u] = B[j0 + r][kb * LANES + u];
        }
      }

      // ---- load C tile into accumulators (aligned 16-float beats) ----
      load_C_row: for (int i = 0; i < TI; i++) {
        load_C_beat: for (int b = 0; b < CB; b++) {
#pragma HLS PIPELINE II=1
          if (b < nb) {
            load_C_lane: for (int u = 0; u < LANES; u++) {
#pragma HLS UNROLL
              acc[i][b * LANES + u] = C[i0 + i][j0 + b * LANES + u];
            }
          }
        }
      }

      // ---- compute: 208 j-lanes x 8 k-lanes = 1664 MACs per cycle ----
      compute_kc: for (int kc = 0; kc < NKC; kc++) {
        compute_i: for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
          const int kh  = kc >> 1;          // which 16-wide k group
          const int sel = (kc & 1) * UK;    // lower/upper half of the group
          data_t a_reg[UK];
#pragma HLS ARRAY_PARTITION variable=a_reg complete dim=1
          read_a: for (int u = 0; u < UK; u++) {
#pragma HLS UNROLL
            a_reg[u] = A_loc[i][kh][sel + u];
          }
          compute_jj: for (int jj = 0; jj < TJ; jj++) {
#pragma HLS UNROLL
            data_t s[UK];
#pragma HLS ARRAY_PARTITION variable=s complete dim=1
            mul_k: for (int u = 0; u < UK; u++) {
#pragma HLS UNROLL
              s[u] = a_reg[u] * B_loc[jj][kh][sel + u];
            }
            tree_level: for (int w = UK / 2; w > 0; w >>= 1) {
#pragma HLS UNROLL
              tree_add: for (int u = 0; u < w; u++) {
#pragma HLS UNROLL
                s[u] = s[u] + s[u + w];
              }
            }
            acc[i][jj] = acc[i][jj] + s[0];
          }
        }
      }

      // ---- store C tile (aligned 16-float beats) ----
      store_C_row: for (int i = 0; i < TI; i++) {
        store_C_beat: for (int b = 0; b < CB; b++) {
#pragma HLS PIPELINE II=1
          if (b < nb) {
            store_C_lane: for (int u = 0; u < LANES; u++) {
#pragma HLS UNROLL
              C[i0 + i][j0 + b * LANES + u] = acc[i][b * LANES + u];
            }
          }
        }
      }
    }
  }
}
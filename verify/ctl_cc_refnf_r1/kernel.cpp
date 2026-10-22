#include "kernel.h"

// ---------------------------------------------------------------------------
// Tiling parameters
//   Output tile: TI x TJ = 512 x 256 held on chip (acc)
//   Parallel MAC array: PI x PJ = 32 x 32 = 1024 fp32 MACs / cycle
//   Sub-blocks per tile: NSB = (TI/PI) * (TJ/PJ) = 16 * 8 = 128
//   -> each acc element is revisited every 128 pipeline iterations, so the
//      fp-add recurrence never limits II.
// ---------------------------------------------------------------------------
#define PI 32
#define PJ 32
#define TI 512
#define TJ 256
#define SI (TI / PI)          // 16
#define SJ (TJ / PJ)          // 8
#define NSB (SI * SJ)         // 128
#define LANES 16              // 512-bit beat / 32-bit float
#define KB (K / LANES)        // 64 beats per row
#define CB (TJ / LANES)       // 16 beats per C tile row

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // A strip for current i-tile: row r -> A_t[r % PI][r / PI][k]
  static data_t A_t[PI][SI][K];
#pragma HLS ARRAY_PARTITION variable=A_t complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_t cyclic factor=16 dim=3

  // B strip for current j-tile: row r -> B_t[r % PJ][r / PJ][k]
  static data_t B_t[PJ][SJ][K];
#pragma HLS ARRAY_PARTITION variable=B_t complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_t cyclic factor=16 dim=3

  // Output accumulators: element (r, c) -> acc[r % PI][c % PJ][(r/PI)*SJ + c/PJ]
  static data_t acc[PI][PJ][NSB];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram

  tile_i: for (int it = 0; it < I / TI; it++) {
    const int i0 = it * TI;

    // ---- load A strip (reused across all j-tiles) ----
    load_A_row: for (int r = 0; r < TI; r++) {
      load_A_beat: for (int kb = 0; kb < KB; kb++) {
#pragma HLS PIPELINE II=1
        load_A_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          A_t[r % PI][r / PI][kb * LANES + l] = A[i0 + r][kb * LANES + l];
        }
      }
    }

    tile_j: for (int jt = 0; jt < J / TJ; jt++) {
      const int j0 = jt * TJ;

      // ---- fused load: B strip (gmem1) || C tile into acc (gmem2) ----
      load_BC: for (int idx = 0; idx < TJ * KB; idx++) {
#pragma HLS PIPELINE II=1
        const int br = idx / KB;
        const int bkb = idx % KB;
        load_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          B_t[br % PJ][br / PJ][bkb * LANES + l] = B[j0 + br][bkb * LANES + l];
        }
        if (idx < TI * CB) {
          const int cr = idx / CB;
          const int ccb = idx % CB;
          load_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            const int c = ccb * LANES + l;
            acc[cr % PI][c % PJ][(cr / PI) * SJ + c / PJ] = C[i0 + cr][j0 + c];
          }
        }
      }

      // ---- compute: 32x32 MAC array, k outer, sub-block inner ----
      compute_k: for (int k = 0; k < K; k++) {
        compute_sb: for (int sb = 0; sb < NSB; sb++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter true distance=128
          const int si = sb / SJ;
          const int sj = sb % SJ;
          data_t a_reg[PI];
#pragma HLS ARRAY_PARTITION variable=a_reg complete
          data_t b_reg[PJ];
#pragma HLS ARRAY_PARTITION variable=b_reg complete
          fetch_a: for (int pi = 0; pi < PI; pi++) {
#pragma HLS UNROLL
            a_reg[pi] = A_t[pi][si][k];
          }
          fetch_b: for (int pj = 0; pj < PJ; pj++) {
#pragma HLS UNROLL
            b_reg[pj] = B_t[pj][sj][k];
          }
          mac_i: for (int pi = 0; pi < PI; pi++) {
#pragma HLS UNROLL
            mac_j: for (int pj = 0; pj < PJ; pj++) {
#pragma HLS UNROLL
              acc[pi][pj][sb] = acc[pi][pj][sb] + a_reg[pi] * b_reg[pj];
            }
          }
        }
      }

      // ---- store C tile ----
      store_C: for (int idx = 0; idx < TI * CB; idx++) {
#pragma HLS PIPELINE II=1
        const int cr = idx / CB;
        const int ccb = idx % CB;
        store_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          const int c = ccb * LANES + l;
          C[i0 + cr][j0 + c] = acc[cr % PI][c % PJ][(cr / PI) * SJ + c / PJ];
        }
      }
    }
  }
}
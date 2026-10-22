#include "kernel.h"

// ---------------------------------------------------------------------------
// Parallel datapath sizing (U280: 9024 DSP, fp32 MAC ~= 3 (fmul) + 2 (fadd) DSP)
//   PI x PJ = 26 x 64 = 1664 MACs/cycle  -> ~8320 DSP (~92%)
// Output tile: TI x TJ = (PI*SI) x (PJ*SJ) = 104 x 256, NS = SI*SJ = 16 sub-blocks
// Each acc entry is revisited every NS=16 cycles -> hides the FP add latency.
// ---------------------------------------------------------------------------
#define PI     26
#define PJ     64
#define SI     4
#define SJ     4
#define NS     (SI * SJ)          // 16
#define TI     (PI * SI)          // 104
#define TJ     (PJ * SJ)          // 256
#define LANES  16                 // 512-bit beat / 32-bit float
#define KW     (K / LANES)        // 64 words per row
#define NIT    ((I + TI - 1) / TI) // 10 (last tile partially valid)
#define NJT    (J / TJ)           // 4
#define JB     (J / PJ)           // 16 row-blocks of B

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // B fully staged on chip: row j -> bank (j % PJ), block (j / PJ)
  data_t B_loc[PJ][JB * KW][LANES];
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=3

  // A row-tile: row (si*PI + pi) -> bank pi, block si
  data_t A_loc[PI][SI * KW][LANES];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=3

  // Output-tile accumulators: (pi, pj) banks, NS-deep each
  data_t acc[PI][PJ][NS];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram

  // ---------------- Load all of B once (16 floats / cycle) ----------------
  load_B_row: for (int r = 0; r < J; ++r) {
    load_B_word: for (int kw = 0; kw < KW; ++kw) {
#pragma HLS PIPELINE II=1
      load_B_lane: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
        B_loc[r % PJ][(r / PJ) * KW + kw][l] = B[r][kw * LANES + l];
      }
    }
  }

  tile_i: for (int it = 0; it < NIT; ++it) {
    const int ibase = it * TI;

    // ---------------- Load A row-tile (reused across all j-tiles) ----------------
    load_A_si: for (int si = 0; si < SI; ++si) {
      load_A_pi: for (int pi = 0; pi < PI; ++pi) {
        load_A_word: for (int kw = 0; kw < KW; ++kw) {
#pragma HLS PIPELINE II=1
          const int i = ibase + si * PI + pi;
          load_A_lane: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
            data_t v = 0;
            if (i < I) v = A[i][kw * LANES + l];
            A_loc[pi][si * KW + kw][l] = v;
          }
        }
      }
    }

    tile_j: for (int jt = 0; jt < NJT; ++jt) {
      const int jbase = jt * TJ;

      // ---------------- Load C tile into accumulators ----------------
      load_C_si: for (int si = 0; si < SI; ++si) {
        load_C_pi: for (int pi = 0; pi < PI; ++pi) {
          load_C_word: for (int w = 0; w < TJ / LANES; ++w) {
#pragma HLS PIPELINE II=1
            const int i  = ibase + si * PI + pi;
            const int sj = w >> 2;
            const int c  = w & 3;
            const int s  = si * SJ + sj;
            load_C_lane: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
              data_t v = 0;
              if (i < I) v = C[i][jbase + w * LANES + l];
              acc[pi][c * LANES + l][s] = v;
            }
          }
        }
      }

      // ---------------- Compute: 1664 MACs per cycle ----------------
      compute_t: for (int t = 0; t < K * NS; ++t) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter RAW distance=16 true
        const int k  = t >> 4;
        const int s  = t & (NS - 1);
        const int si = s >> 2;
        const int sj = s & 3;
        const int kw = k >> 4;
        const int kl = k & (LANES - 1);

        data_t a[PI];
#pragma HLS ARRAY_PARTITION variable=a complete dim=1
        data_t b[PJ];
#pragma HLS ARRAY_PARTITION variable=b complete dim=1

        read_a: for (int pi = 0; pi < PI; ++pi) {
#pragma HLS UNROLL
          a[pi] = A_loc[pi][si * KW + kw][kl];
        }
        read_b: for (int pj = 0; pj < PJ; ++pj) {
#pragma HLS UNROLL
          b[pj] = B_loc[pj][(jt * SJ + sj) * KW + kw][kl];
        }
        mac_i: for (int pi = 0; pi < PI; ++pi) {
#pragma HLS UNROLL
          mac_j: for (int pj = 0; pj < PJ; ++pj) {
#pragma HLS UNROLL
            acc[pi][pj][s] = acc[pi][pj][s] + a[pi] * b[pj];
          }
        }
      }

      // ---------------- Store C tile ----------------
      store_C_si: for (int si = 0; si < SI; ++si) {
        store_C_pi: for (int pi = 0; pi < PI; ++pi) {
          store_C_word: for (int w = 0; w < TJ / LANES; ++w) {
#pragma HLS PIPELINE II=1
            const int i  = ibase + si * PI + pi;
            const int sj = w >> 2;
            const int c  = w & 3;
            const int s  = si * SJ + sj;
            if (i < I) {
              store_C_lane: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
                C[i][jbase + w * LANES + l] = acc[pi][c * LANES + l][s];
              }
            }
          }
        }
      }
    }
  }
}
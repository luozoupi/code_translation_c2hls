#include "kernel.h"

#define LANES 16                 // floats per 512-bit beat
#define PI    16                 // PE rows
#define PJ    32                 // PE cols   (PI*PJ = 512 fp32 MACs)
#define TI    512                // rows of A resident on chip
#define TJ    128                // cols of C per tile (rows of B)
#define SI    (TI / PI)          // 32
#define SJ    (TJ / PJ)          // 4
#define NSUB  (SI * SJ)          // 128 sub-blocks -> accumulator reuse distance
#define KB    (K / LANES)        // 64 beats per row

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // A row block: a_buf[row % PI][k % LANES][(row / PI) * KB + k / LANES]
  static data_t a_buf[PI][LANES][SI * KB];
#pragma HLS ARRAY_PARTITION variable=a_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=a_buf complete dim=2
#pragma HLS BIND_STORAGE variable=a_buf type=ram_2p impl=uram

  // B tile: b_buf[col % PJ][k % LANES][(col / PJ) * KB + k / LANES]
  static data_t b_buf[PJ][LANES][SJ * KB];
#pragma HLS ARRAY_PARTITION variable=b_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=b_buf complete dim=2
#pragma HLS BIND_STORAGE variable=b_buf type=ram_2p impl=bram

  // C tile accumulators: acc[i%PI][j%PJ][(i/PI)*SJ + (j/PJ)]
  static data_t acc[PI][PJ][NSUB];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

  row_block: for (int ib = 0; ib < I / TI; ib++) {
    const int i0 = ib * TI;

    // ---- load A row block once, reused for all column tiles ----
    load_A: for (int t = 0; t < TI * KB; t++) {
#pragma HLS PIPELINE II=1
      const int r  = t >> 6;          // / KB
      const int kb = t & (KB - 1);
      load_A_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        a_buf[r & (PI - 1)][l][(r >> 4) * KB + kb] = A[i0 + r][kb * LANES + l];
      }
    }

    col_tile: for (int jt = 0; jt < J / TJ; jt++) {
      const int j0 = jt * TJ;

      // ---- load B tile (TJ rows of B) ----
      load_B: for (int t = 0; t < TJ * KB; t++) {
#pragma HLS PIPELINE II=1
        const int r  = t >> 6;
        const int kb = t & (KB - 1);
        load_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          b_buf[r & (PJ - 1)][l][(r >> 5) * KB + kb] = B[j0 + r][kb * LANES + l];
        }
      }

      // ---- load C tile into accumulators ----
      load_C: for (int t = 0; t < TI * (TJ / LANES); t++) {
#pragma HLS PIPELINE II=1
        const int ci = t >> 3;          // / (TJ/LANES)
        const int cb = t & 7;
        load_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          const int j = cb * LANES + l;
          acc[ci & (PI - 1)][j & (PJ - 1)][(ci >> 4) * SJ + (j >> 5)] = C[i0 + ci][j0 + j];
        }
      }

      // ---- compute: k outer, sub-blocks inner (reuse distance NSUB) ----
      compute_k: for (int k = 0; k < K; k++) {
        compute_s: for (int s = 0; s < NSUB; s++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
          const int si = s >> 2;        // / SJ
          const int sj = s & (SJ - 1);
          const int kb = k >> 4;
          const int ln = k & (LANES - 1);

          data_t av[PI];
#pragma HLS ARRAY_PARTITION variable=av complete
          data_t bv[PJ];
#pragma HLS ARRAY_PARTITION variable=bv complete

          fetch_a: for (int p = 0; p < PI; p++) {
#pragma HLS UNROLL
            av[p] = a_buf[p][ln][si * KB + kb];
          }

          fetch_b: for (int q = 0; q < PJ; q++) {
#pragma HLS UNROLL
            bv[q] = b_buf[q][ln][sj * KB + kb];
          }

          mac_i: for (int p = 0; p < PI; p++) {
#pragma HLS UNROLL
            mac_j: for (int q = 0; q < PJ; q++) {
#pragma HLS UNROLL
              acc[p][q][s] = acc[p][q][s] + av[p] * bv[q];
            }
          }
        }
      }

      // ---- store C tile ----
      store_C: for (int t = 0; t < TI * (TJ / LANES); t++) {
#pragma HLS PIPELINE II=1
        const int ci = t >> 3;
        const int cb = t & 7;
        store_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          const int j = cb * LANES + l;
          C[i0 + ci][j0 + j] = acc[ci & (PI - 1)][j & (PJ - 1)][(ci >> 4) * SJ + (j >> 5)];
        }
      }
    }
  }
}
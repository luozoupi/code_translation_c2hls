#include "kernel.h"

#define LANES 16                 // floats per 512-bit beat
#define PI    22                 // PE rows
#define PJ    64                 // PE cols   (PI*PJ = 1408 fp32 MACs)
#define SI    47                 // row sub-blocks: PI*SI = 1034 >= I
#define TI    (PI * SI)          // 1034 rows of A resident on chip (all of A + pad)
#define TJ    256                // cols of C per tile (rows of B)
#define SJ    (TJ / PJ)          // 4
#define NSUB  (SI * SJ)          // 188 sub-blocks -> accumulator reuse distance
#define KB    (K / LANES)        // 64 beats per row
#define CB    (TJ / LANES)       // 16 beats per C tile row
#define PAD_ROWS (TI - I)        // 10 padded rows
#define PAD_SB   (I / PI)        // 46: sub-block of first pad row
#define PAD_P0   (I % PI)        // 12: PE row of first pad row

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // All of A resident: a_buf[row % PI][k % LANES][(row / PI) * KB + k / LANES]
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

  // ---- load all of A once (sequential rows -> 512-bit bursts) ----
  {
    int p  = 0;
    int sb = 0;
    load_A: for (int t = 0; t < I * KB; t++) {
#pragma HLS PIPELINE II=1
      const int r  = t >> 6;          // / KB
      const int kb = t & (KB - 1);
      load_A_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        a_buf[p][l][sb * KB + kb] = A[r][kb * LANES + l];
      }
      if (kb == KB - 1) {
        if (p == PI - 1) { p = 0; sb++; }
        else             { p++; }
      }
    }
  }

  // ---- zero the padded rows (rows I..TI-1) ----
  pad_A: for (int t = 0; t < PAD_ROWS * KB; t++) {
#pragma HLS PIPELINE II=1
    const int pr = PAD_P0 + (t >> 6);
    const int kb = t & (KB - 1);
    pad_A_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      a_buf[pr][l][PAD_SB * KB + kb] = 0.0f;
    }
  }

  col_tile: for (int jt = 0; jt < J / TJ; jt++) {
    const int j0 = jt * TJ;

    // ---- load B tile (TJ rows of B) ----
    load_B: for (int t = 0; t < TJ * KB; t++) {
#pragma HLS PIPELINE II=1
      const int r  = t >> 6;          // / KB
      const int kb = t & (KB - 1);
      load_B_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        b_buf[r & (PJ - 1)][l][(r >> 6) * KB + kb] = B[j0 + r][kb * LANES + l];
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
        const bool first = (k == 0);

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
            const data_t prev = first ? (data_t)0.0f : acc[p][q][s];
            acc[p][q][s] = prev + av[p] * bv[q];
          }
        }
      }
    }

    // ---- store C tile: C = C + acc (read & write channels in one II=1 loop) ----
    {
      int pc = 0;
      int sc = 0;
      store_C: for (int t = 0; t < I * CB; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C inter false
        const int ci = t >> 4;
        const int cb = t & (CB - 1);
        store_C_lane: for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          const int j = cb * LANES + l;
          const data_t cv = C[ci][j0 + j];
          C[ci][j0 + j] = cv + acc[pc][j & (PJ - 1)][sc * SJ + (j >> 6)];
        }
        if (cb == CB - 1) {
          if (pc == PI - 1) { pc = 0; sc++; }
          else              { pc++; }
        }
      }
    }
  }
}
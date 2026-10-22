#include "kernel.h"

/* ------------------------------------------------------------------
 * v4: single fp32 accumulator chain (KL = 1).
 *
 * Measured facts that drive this design (Vitis HLS 2023.2, xcu280):
 *   - the pipelined fp32 MAC loop always lands on II = 4, whatever the
 *     number of accumulator banks (1, 2, 4, 8, 16 all measured at 4);
 *   - so physical multipliers M = P/4 for P MAC slots in the body, and
 *     DSP = 3*M exactly (fmul_32ns_32ns_32_4_max_dsp_1 = 3 DSP);
 *   - LUT ~= 585 per multiplier (78 mul + 338 add + ~170 mux/reg).
 * With one accumulator bank there is no epilogue adder tree at all:
 * the C store is a plain register -> global copy at 1 beat/cycle, and
 * the summation order is exactly the golden's sequential order.
 *
 * Window DSP 6317..7219  =>  P in 8423..9625.  LUT caps P near 8900.
 * TI*TJ = 67*128 = 8576 -> M = 2144 -> DSP = 6432.
 * ------------------------------------------------------------------ */
#define TI 67
#define TJ 128
#define JN (J / TJ)              /* 8  (128 divides 1024) */
#define IN ((I + TI - 1) / TI)   /* 16 (1024/67 -> last tile partial) */
#define LD 16                    /* 512-bit beat = 16 fp32 */
#define SC 16

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmemA max_widen_bitwidth=512 num_read_outstanding=32 num_write_outstanding=2 max_read_burst_length=64 depth=1048576
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmemB max_widen_bitwidth=512 num_read_outstanding=32 num_write_outstanding=2 max_read_burst_length=64 depth=1048576
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmemC max_widen_bitwidth=512 num_read_outstanding=2 num_write_outstanding=32 max_write_burst_length=64 depth=1048576
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  /* whole B resident on chip (4 MB) */
  static data_t B_loc[JN][TJ][K];
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=2
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=LD dim=3

  /* current A row block */
  static data_t A_loc[TI][K];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=LD dim=2

  /* one accumulator per output of the tile */
  static data_t acc[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0

  /* ---------------- stage B once ---------------- */
loadB_j:
  for (int jt = 0; jt < JN; jt++) {
  loadB_jj:
    for (int j = 0; j < TJ; j++) {
    loadB_k:
      for (int k = 0; k < K; k += LD) {
#pragma HLS PIPELINE II=1
        for (int l = 0; l < LD; l++) {
#pragma HLS UNROLL
          B_loc[jt][j][k + l] = B[jt * TJ + j][k + l];
        }
      }
    }
  }

  /* ---------------- main loop over i tiles ---------------- */
main_i:
  for (int it = 0; it < IN; it++) {

  loadA_i:
    for (int i = 0; i < TI; i++) {
      int gi = it * TI + i;
      if (gi >= I) gi = I - 1;     /* clamp: recompute a dummy row */
    loadA_k:
      for (int k = 0; k < K; k += LD) {
#pragma HLS PIPELINE II=1
        for (int l = 0; l < LD; l++) {
#pragma HLS UNROLL
          A_loc[i][k + l] = A[gi][k + l];
        }
      }
    }

  main_j:
    for (int jt = 0; jt < JN; jt++) {

    clr_i:
      for (int i = 0; i < TI; i++) {
#pragma HLS UNROLL
      clr_j:
        for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
          acc[i][j] = 0;
        }
      }

      /* ---- k reduction, single chain ---- */
    cmp_k:
      for (int k = 0; k < K; k++) {
#pragma HLS PIPELINE II=1
      cmp_i:
        for (int i = 0; i < TI; i++) {
#pragma HLS UNROLL
          data_t a = A_loc[i][k];
        cmp_j:
          for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
            acc[i][j] += a * B_loc[jt][j][k];
          }
        }
      }

      /* ---- store C tile ---- */
    st_i:
      for (int i = 0; i < TI; i++) {
        if (it * TI + i < I) {
        st_j:
          for (int j = 0; j < TJ; j += SC) {
#pragma HLS PIPELINE II=1
            for (int l = 0; l < SC; l++) {
#pragma HLS UNROLL
              C[it * TI + i][jt * TJ + j + l] = acc[i][j + l];
            }
          }
        }
      }
    }
  }
}

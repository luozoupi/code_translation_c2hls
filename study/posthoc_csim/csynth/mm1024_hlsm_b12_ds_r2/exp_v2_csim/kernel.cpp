#include "kernel.h"

/* C[i][j] = sum_k A[i][k]*B[j][k], fp32, I=J=K=1024.
 *
 * Measured Vitis HLS facts this design is built on (probe256 csynth):
 *   - fp32 multiply = 3 DSP + 78 LUT   (fmul_32ns_32ns_32_4_max_dsp_1)
 *   - fp32 add      = 0 DSP + 338 LUT  (fadd_32ns_32ns_32_5_no_dsp_1)
 *   - a k-pipelined `acc += a*b` chain is a true fp recurrence of distance 1
 *     and the scheduler resolves it at II=4, then shares every operator 4:1.
 * So steady-state throughput is body_MACs/4 MACs per cycle and DSP is
 * 3 * body_MACs/4.
 *
 * Array: TI x TJ output tile, KL=4 k-lanes (one accumulator bank per lane).
 *   body MACs = TI*TJ*KL = 34*64*4 = 8704 -> 2176 MAC units -> 6528 DSP
 *   throughput = 2176 MACs/cycle
 *
 * Local memory: all of B staged in BRAM (row cyclic TJ, k cyclic LANES),
 * per-i-block A staged in BRAM (rows complete, k cyclic LANES), so every
 * operand read in the pipelined body hits its own bank.
 */

#define TI 34
#define TJ 64
#define KL 4
#define LANES 16                /* 512-bit / 32-bit float */
#define NIB 31                  /* ceil(I/TI) = ceil(1024/34) */
#define NJB (J / TJ)            /* 16 */
#define NACC (TI * TJ * KL)     /* 8704 */
#define KSTEP (K / KL)          /* 256 */
#define KVEC (K / LANES)        /* 64 */
#define JV (TJ / LANES)         /* 4 store beats per C row */

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmemA max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmemB max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmemC max_widen_bitwidth=512 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=return

  static data_t A_loc[TI][K];
  static data_t B_loc[J][K];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=LANES dim=2
#pragma HLS BIND_STORAGE variable=A_loc type=ram_s2p impl=bram
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=TJ dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=LANES dim=2
#pragma HLS BIND_STORAGE variable=B_loc type=ram_s2p impl=bram

  /* ---- stage all of B on chip (one 512-bit beat per cycle) ---- */
load_b:
  for (int r = 0; r < J; r++) {
    for (int kv = 0; kv < KVEC; kv++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL
      for (int l = 0; l < LANES; l++)
        B_loc[r][kv * LANES + l] = B[r][kv * LANES + l];
    }
  }

  for (int i0 = 0; i0 < NIB; i0++) {
    /* ---- stage this i block of A, zero padded past I ---- */
  load_a:
    for (int i = 0; i < TI; i++) {
      for (int kv = 0; kv < KVEC; kv++) {
#pragma HLS PIPELINE II=1
        if (i0 * TI + i < I) {
#pragma HLS UNROLL
          for (int l = 0; l < LANES; l++)
            A_loc[i][kv * LANES + l] = A[i0 * TI + i][kv * LANES + l];
        } else {
#pragma HLS UNROLL
          for (int l = 0; l < LANES; l++)
            A_loc[i][kv * LANES + l] = (data_t)0;
        }
      }
    }

    for (int j0 = 0; j0 < NJB; j0++) {
      data_t acc[NACC];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
      for (int t = 0; t < NACC; t++) {
#pragma HLS UNROLL
        acc[t] = (data_t)0;
      }
      /* KL lanes per output: each bank is written once per iteration, so the
       * fp add recurrence resolves at II=4 while all four lanes stay busy. */
      for (int k0 = 0; k0 < KSTEP; k0++) {
#pragma HLS PIPELINE II=1
        for (int i = 0; i < TI; i++) {
#pragma HLS UNROLL
          for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
            for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
              acc[(i * TJ + j) * KL + l] +=
                  A_loc[i][k0 * KL + l] * B_loc[j0 * TJ + j][k0 * KL + l];
            }
          }
        }
      }
      /* ---- store 16 outputs (one 512-bit beat) per cycle ---- */
      for (int t = 0; t < TI * JV; t++) {
#pragma HLS PIPELINE II=1
        int i = t / JV;
        int jv = t % JV;
        if (i0 * TI + i < I) {
#pragma HLS UNROLL
          for (int l = 0; l < LANES; l++) {
            int jj = jv * LANES + l;
            C[i0 * TI + i][j0 * TJ + jj] =
                (acc[(i * TJ + jj) * KL + 0] + acc[(i * TJ + jj) * KL + 1]) +
                (acc[(i * TJ + jj) * KL + 2] + acc[(i * TJ + jj) * KL + 3]);
          }
        }
      }
    }
  }
}

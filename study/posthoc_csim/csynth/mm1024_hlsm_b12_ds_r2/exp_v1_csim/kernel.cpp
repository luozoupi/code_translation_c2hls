#include "kernel.h"

/* C[i][j] = sum_k A[i][k]*B[j][k], fp32, I=J=K=1024.
 *
 * Architecture: full on-chip staging of A and B (URAM), broadcast MAC array
 * TI x TJ outputs with KL=4 independent k-lanes (4 accumulator banks so the
 * fp32 add recurrence of distance 4 does not force II=4).
 *
 *   MACs      = TI*TJ*KL = 20*16*4 = 1280  (fp32 MAC ~5 DSP -> ~6400 DSP)
 *   local A   = A_loc[NIB*TI][K]  partitioned cyclic 20 (rows) x cyclic 16 (k)
 *   local B   = B_loc[J][K]       partitioned cyclic 16 (rows) x cyclic 16 (k)
 *
 * Phases: one bulk load of A and B (512-bit per port, one A and one B beat per
 * cycle), then for every (i0,j0) output tile a K-pipelined MAC loop (II=1)
 * and a 512-bit store of the 16-wide row results.
 */

#define TI 20                   /* output rows per tile (A rows)      */
#define TJ 16                   /* output cols per tile (B rows)      */
#define KL 4                    /* k lanes = fp32 acc recurrence dist */
#define LANES 16                /* 512-bit / 32-bit floats            */
#define NIB 52                  /* ceil(I/TI) = ceil(1024/20)         */
#define NJB (J / TJ)            /* 64                                 */
#define AROWS (NIB * TI)        /* 1040                               */
#define NACC (TI * TJ * KL)     /* 1280                               */
#define KSTEP (K / KL)          /* 256                                */
#define KVEC (K / LANES)        /* 64                                 */

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmemA max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmemB max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmemC max_widen_bitwidth=512 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=return

  data_t A_loc[AROWS][K];
  data_t B_loc[J][K];
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=TI dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=LANES dim=2
#pragma HLS BIND_STORAGE variable=A_loc type=ram_s2p impl=uram
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=TJ dim=1
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=LANES dim=2
#pragma HLS BIND_STORAGE variable=B_loc type=ram_s2p impl=uram

  /* ---- bulk stage A and B into on-chip memory, 16 floats/cycle each ---- */
load_ab:
  for (int r = 0; r < AROWS; r++) {
    for (int kv = 0; kv < KVEC; kv++) {
#pragma HLS PIPELINE II=1
      if (r < I) {
#pragma HLS UNROLL
        for (int l = 0; l < LANES; l++)
          A_loc[r][kv * LANES + l] = A[r][kv * LANES + l];
      } else {
#pragma HLS UNROLL
        for (int l = 0; l < LANES; l++)
          A_loc[r][kv * LANES + l] = (data_t)0;
      }
      if (r < J) {
#pragma HLS UNROLL
        for (int l = 0; l < LANES; l++)
          B_loc[r][kv * LANES + l] = B[r][kv * LANES + l];
      }
    }
  }

  /* ---- compute all output tiles, then store each finished tile ---- */
  for (int i0 = 0; i0 < NIB; i0++) {
    for (int j0 = 0; j0 < NJB; j0++) {
      data_t acc[NACC];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
      for (int t = 0; t < NACC; t++) {
#pragma HLS UNROLL
        acc[t] = (data_t)0;
      }
      for (int k0 = 0; k0 < KSTEP; k0++) {
#pragma HLS PIPELINE II=1
        for (int i = 0; i < TI; i++) {
#pragma HLS UNROLL
          for (int j = 0; j < TJ; j++) {
#pragma HLS UNROLL
            for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
              acc[(i * TJ + j) * KL + l] +=
                  A_loc[i0 * TI + i][k0 * KL + l] *
                  B_loc[j0 * TJ + j][k0 * KL + l];
            }
          }
        }
      }
      for (int i = 0; i < TI; i++) {
#pragma HLS PIPELINE II=1
        if (i0 * TI + i < I) {
#pragma HLS UNROLL
          for (int j = 0; j < TJ; j++) {
            C[i0 * TI + i][j0 * TJ + j] =
                (acc[(i * TJ + j) * KL + 0] + acc[(i * TJ + j) * KL + 1]) +
                (acc[(i * TJ + j) * KL + 2] + acc[(i * TJ + j) * KL + 3]);
          }
        }
      }
    }
  }
}

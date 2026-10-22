#include "kernel.h"

// ---------------------------------------------------------------------------
// C = A * B^T  (I = J = K = 1024, fp32),  output-stationary GEMM.
//
// Architecture
//   * The whole output tile C[I][PADJ] is kept on chip (C_tile) for the whole
//     computation, so every C element is written exactly once to the m_axi port.
//   * A and B are streamed from global memory in K blocks of KBLK = 64
//     columns, 16 floats (512 bit) per cycle per port (A on gmem0, B on gmem1).
//   * The hot loop is one flattened pipelined nest: every cycle it consumes
//     TK = 4 consecutive k values for a PI x PJ = 17 x 20 group of outputs,
//     i.e. PI*PJ*TK = 1360 parallel multiply-adds.
//   * Each output accumulator lives in the C_tile banked memory (one element
//     per bank per cycle), so the k reduction is a memory read-modify-write
//     with no float recurrence in the pipeline -> II = 1.
//   * The accumulation order is strictly sequential in k (k = 0..1023), i.e.
//     the same order as the reference loop, so rounding matches the golden
//     model.
// ---------------------------------------------------------------------------

#define PI      17        // output rows per pipelined iteration
#define PJ      20        // output cols per pipelined iteration
#define TK      4         // k values consumed per output per iteration
#define KBLK    64        // K block staged in the local A/B buffers
#define LANES   32        // 1024 bit / 32 bit = floats per AXI beat

#define NGI     ((I + PI - 1) / PI)          // 61
#define NGJ     ((J + PJ - 1) / PJ)          // 52
#define PADI    (NGI * PI)                   // 1037
#define PADJ0   (NGJ * PJ)                   // 1040
#define PADJ    (((PADJ0 + 79) / 80) * 80)   // multiple of lcm(PJ,LANES)=80
#define NKB     ((K) / KBLK)                 // 16
#define NKT     ((KBLK) / TK)                // 16
#define NKL     ((KBLK) / LANES)             // 4

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A bundle=gmem0 depth=1048576 max_widen_bitwidth=1024 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=B bundle=gmem1 depth=1048576 max_widen_bitwidth=1024 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=C bundle=gmem2 depth=1048576 max_widen_bitwidth=1024 num_write_outstanding=32
#pragma HLS INTERFACE s_axilite port=return

  static data_t A_buf[PADI][KBLK];    // ~265 KB
  static data_t B_buf[PADJ][KBLK];    // ~266 KB
  static data_t C_tile[PADI][PADJ];   // ~4.3 MB on-chip accumulator

  // A_buf  : 17 row banks x 32 lane banks
  // B_buf  : 20 row banks x 32 lane banks
  // C_tile : 17 row banks x 32 col banks  (544 independent read/write ports)
  // The 32 col banks are what make both access patterns conflict free: the
  // compute loop touches 20 consecutive j per cycle (20 distinct banks) and
  // the store loop touches 32 consecutive j per cycle (32 distinct banks).
#pragma HLS ARRAY_PARTITION variable=A_buf  dim=1 type=cyclic factor=PI
#pragma HLS ARRAY_PARTITION variable=A_buf  dim=2 type=cyclic factor=LANES
#pragma HLS ARRAY_PARTITION variable=B_buf  dim=1 type=cyclic factor=PJ
#pragma HLS ARRAY_PARTITION variable=B_buf  dim=2 type=cyclic factor=LANES
#pragma HLS ARRAY_PARTITION variable=C_tile dim=1 type=cyclic factor=PI
#pragma HLS ARRAY_PARTITION variable=C_tile dim=2 type=cyclic factor=LANES

  // ------------------------------------------------------------------
  // zero the on-chip accumulator tile (garbage must not be accumulated)
  // ------------------------------------------------------------------
  init_c: for (int gi = 0; gi < NGI; gi++) {
#pragma HLS LOOP_FLATTEN
    for (int gj = 0; gj < NGJ; gj++) {
#pragma HLS PIPELINE II=1
      for (int ii = 0; ii < PI; ii++) {
#pragma HLS UNROLL
        for (int jj = 0; jj < PJ; jj++) {
#pragma HLS UNROLL
          C_tile[gi * PI + ii][gj * PJ + jj] = (data_t)0;
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // K blocked main loop
  // ------------------------------------------------------------------
  for (int kb = 0; kb < NKB; kb++) {

    // ---- stage A[0:I][kb*KBLK .. kb*KBLK+KBLK) and B likewise ----
    load_ab: for (int i = 0; i < I; i++) {
      for (int kc = 0; kc < NKL; kc++) {
#pragma HLS PIPELINE II=1
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          A_buf[i][kc * LANES + l] = A[i][kb * KBLK + kc * LANES + l];
          B_buf[i][kc * LANES + l] = B[i][kb * KBLK + kc * LANES + l];
        }
      }
    }

    // ---- accumulate over the K block ----
    for (int kt = 0; kt < NKT; kt++) {
      comp_i: for (int gi = 0; gi < NGI; gi++) {
#pragma HLS LOOP_FLATTEN
        comp_j: for (int gj = 0; gj < NGJ; gj++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_tile inter false
          for (int ii = 0; ii < PI; ii++) {
#pragma HLS UNROLL
            for (int jj = 0; jj < PJ; jj++) {
#pragma HLS UNROLL
              data_t acc = C_tile[gi * PI + ii][gj * PJ + jj];
              acc += A_buf[gi * PI + ii][kt * TK + 0] * B_buf[gj * PJ + jj][kt * TK + 0];
              acc += A_buf[gi * PI + ii][kt * TK + 1] * B_buf[gj * PJ + jj][kt * TK + 1];
              acc += A_buf[gi * PI + ii][kt * TK + 2] * B_buf[gj * PJ + jj][kt * TK + 2];
              acc += A_buf[gi * PI + ii][kt * TK + 3] * B_buf[gj * PJ + jj][kt * TK + 3];
              C_tile[gi * PI + ii][gj * PJ + jj] = acc;
            }
          }
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // write the accumulator tile out once (32 floats = 1024 bit per beat).
  // J is a multiple of LANES, so every burst is a full 512 bit beat and no
  // element-wise guard is needed here (a guard would stop the widening).
  // ------------------------------------------------------------------
  store_c: for (int i = 0; i < I; i++) {
    for (int jc = 0; jc < J / LANES; jc++) {
#pragma HLS PIPELINE II=1
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        C[i][jc * LANES + l] = C_tile[i][jc * LANES + l];
      }
    }
  }
}

#include "kernel.h"

// ---------------------------------------------------------------------------
// Parallelism: PJ output columns x KL reduction lanes = 1024 fp32 MACs / cycle
// Whole B resident on chip; C tiles are full rows (contiguous in memory).
// ---------------------------------------------------------------------------
#define TI    16                 // rows of C per i-tile
#define PJ    32                 // output columns per cycle (two 512-bit C beats)
#define KL    32                 // k lanes per cycle
#define LANES 16                 // floats per 512-bit beat
#define NIT   (I / TI)           // i-tiles
#define NJG   (J / PJ)           // column groups per row
#define KC    (K / KL)           // k chunks
#define KB    (K / LANES)        // beats per A/B row
#define JB    (J / LANES)        // beats per C row

// ---------------------------------------------------------------------------
// Load all of B once (contiguous, 512-bit beats)
// ---------------------------------------------------------------------------
static void load_B(data_t B[J][K], data_t Bl[J][K]) {
load_B_beat:
  for (int idx = 0; idx < J * KB; idx++) {
#pragma HLS PIPELINE II=1
    int j = idx / KB;
    int c = idx % KB;
  load_B_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      Bl[j][c * LANES + l] = B[j][c * LANES + l];
    }
  }
}

// ---------------------------------------------------------------------------
// Fused load: A row tile and C row tile in parallel (gmem0 + gmem2),
// one 512-bit beat of each per cycle. KB == JB here (K == J).
// ---------------------------------------------------------------------------
static void load_AC(data_t A[I][K], data_t C[I][J],
                    data_t Aloc[TI][K], data_t acc[TI][J], int it) {
load_AC_beat:
  for (int idx = 0; idx < TI * KB; idx++) {
#pragma HLS PIPELINE II=1
    int i = idx / KB;
    int c = idx % KB;
  load_A_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      Aloc[i][c * LANES + l] = A[it + i][c * LANES + l];
    }
  load_C_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      acc[i][c * LANES + l] = C[it + i][c * LANES + l];
    }
  }
}

// ---------------------------------------------------------------------------
// Store C full-row tile from local accumulator (contiguous, 512-bit beats)
// ---------------------------------------------------------------------------
static void store_C(data_t C[I][J], data_t acc[TI][J], int it) {
store_C_beat:
  for (int idx = 0; idx < TI * JB; idx++) {
#pragma HLS PIPELINE II=1
    int i = idx / JB;
    int c = idx % JB;
  store_C_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      C[it + i][c * LANES + l] = acc[i][c * LANES + l];
    }
  }
}

// ---------------------------------------------------------------------------
// Compute: 32 columns x 32 k-lanes per cycle. Loop order kc -> i -> jg gives
// accumulator reuse distance TI*NJG = 512 iterations (>> fadd latency).
// ---------------------------------------------------------------------------
static void compute(data_t Aloc[TI][K], data_t Bl[J][K], data_t acc[TI][J]) {
comp_kc:
  for (int kc = 0; kc < KC; kc++) {
  comp_row:
    for (int i = 0; i < TI; i++) {
    comp_jg:
      for (int jg = 0; jg < NJG; jg++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
        data_t a_reg[KL];
#pragma HLS ARRAY_PARTITION variable=a_reg complete dim=1
      comp_areg:
        for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
          a_reg[l] = Aloc[i][kc * KL + l];
        }
      comp_col:
        for (int jj = 0; jj < PJ; jj++) {
#pragma HLS UNROLL
          data_t prod[KL];
#pragma HLS ARRAY_PARTITION variable=prod complete dim=1
        comp_mul:
          for (int kk = 0; kk < KL; kk++) {
#pragma HLS UNROLL
            prod[kk] = a_reg[kk] * Bl[jg * PJ + jj][kc * KL + kk];
          }
        comp_tree_lvl:
          for (int s = KL / 2; s > 0; s >>= 1) {
#pragma HLS UNROLL
          comp_tree_add:
            for (int q = 0; q < KL / 2; q++) {
#pragma HLS UNROLL
              if (q < s) prod[q] = prod[q] + prod[q + s];
            }
          }
          data_t r = acc[i][jg * PJ + jj] + prod[0];
#pragma HLS BIND_OP variable=r op=fadd impl=fabric
          acc[i][jg * PJ + jj] = r;
        }
      }
    }
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static data_t Bl[J][K];
#pragma HLS ARRAY_PARTITION variable=Bl cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=Bl cyclic factor=32 dim=2
  static data_t Aloc[TI][K];
#pragma HLS ARRAY_PARTITION variable=Aloc cyclic factor=32 dim=2
  static data_t acc[TI][J];
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=32 dim=2

  load_B(B, Bl);

itile_loop:
  for (int itix = 0; itix < NIT; itix++) {
    int it = itix * TI;
    load_AC(A, C, Aloc, acc, it);
    compute(Aloc, Bl, acc);
    store_C(C, acc, it);
  }
}
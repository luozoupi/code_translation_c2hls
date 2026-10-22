#include "kernel.h"

// ---------------------------------------------------------------------------
// Parallelism: PJ=48 output columns x KL=32 reduction lanes = 1536 fp32 MACs/cycle
// DSP budget: per column 32 fmul(3) + 24 DSP fadd(2) = 144 -> 48*144 = 6912 DSP.
// Last 3 tree levels + accumulate add are fabric fadds.
// Whole B resident on chip (padded to NJG*PJ = 1056 rows).
// C store of tile t-1 is fused with A/C load of tile t (read+write channels).
// ---------------------------------------------------------------------------
#define TI    16                       // rows of C per i-tile
#define LANES 16                       // floats per 512-bit beat
#define QB    3                        // 512-bit beats per column group
#define PJ    (QB * LANES)             // 48 output columns per cycle
#define KL    32                       // k lanes per cycle
#define NIT   (I / TI)                 // 64 i-tiles
#define NJG   ((J + PJ - 1) / PJ)      // 22 column groups (padded)
#define KC    (K / KL)                 // 32 k chunks
#define KB    (K / LANES)              // 64 beats per A/B row
#define JB    (J / LANES)              // 64 beats per C row

// ---------------------------------------------------------------------------
// Load all of B once (contiguous, 512-bit beats). Row j -> (jg, jj).
// ---------------------------------------------------------------------------
static void load_B(data_t B[J][K], data_t Bl[NJG][PJ][K]) {
load_B_beat:
  for (int idx = 0; idx < J * KB; idx++) {
#pragma HLS PIPELINE II=1
    int j  = idx / KB;
    int c  = idx % KB;
    int jg = j / PJ;
    int jj = j % PJ;
  load_B_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      Bl[jg][jj][c * LANES + l] = B[j][c * LANES + l];
    }
  }
}

// ---------------------------------------------------------------------------
// First tile: load A row tile and C row tile (gmem0 + gmem2).
// ---------------------------------------------------------------------------
static void load_AC(data_t A[I][K], data_t C[I][J],
                    data_t Aloc[TI][K], data_t acc[TI][NJG][QB][LANES], int it) {
load_AC_beat:
  for (int idx = 0; idx < TI * KB; idx++) {
#pragma HLS PIPELINE II=1
    int i = idx / KB;
    int c = idx % KB;
    int jg = c / QB;
    int q  = c % QB;
  load_A_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      Aloc[i][c * LANES + l] = A[it + i][c * LANES + l];
    }
  load_C_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      acc[i][jg][q][l] = C[it + i][c * LANES + l];
    }
  }
}

// ---------------------------------------------------------------------------
// Fused transfer: store C rows of previous tile and load A/C rows of next
// tile in the same beat (C read + write channels concurrently). KB == JB.
// ---------------------------------------------------------------------------
static void xfer_AC(data_t A[I][K], data_t C[I][J],
                    data_t Aloc[TI][K], data_t acc[TI][NJG][QB][LANES],
                    int it_prev, int it_next) {
xfer_AC_beat:
  for (int idx = 0; idx < TI * KB; idx++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
#pragma HLS DEPENDENCE variable=C inter false
#pragma HLS DEPENDENCE variable=C intra false
    int i = idx / KB;
    int c = idx % KB;
    int jg = c / QB;
    int q  = c % QB;
  xfer_store_C_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      C[it_prev + i][c * LANES + l] = acc[i][jg][q][l];
    }
  xfer_load_A_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      Aloc[i][c * LANES + l] = A[it_next + i][c * LANES + l];
    }
  xfer_load_C_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      acc[i][jg][q][l] = C[it_next + i][c * LANES + l];
    }
  }
}

// ---------------------------------------------------------------------------
// Store C full-row tile from local accumulator (last tile)
// ---------------------------------------------------------------------------
static void store_C(data_t C[I][J], data_t acc[TI][NJG][QB][LANES], int it) {
store_C_beat:
  for (int idx = 0; idx < TI * JB; idx++) {
#pragma HLS PIPELINE II=1
    int i = idx / JB;
    int c = idx % JB;
    int jg = c / QB;
    int q  = c % QB;
  store_C_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      C[it + i][c * LANES + l] = acc[i][jg][q][l];
    }
  }
}

// ---------------------------------------------------------------------------
// Compute: 48 columns x 32 k-lanes per cycle. Loop order kc -> i -> jg gives
// accumulator reuse distance TI*NJG = 352 iterations (>> pipeline depth).
// Padded columns (j >= J) are computed but never stored.
// ---------------------------------------------------------------------------
static void compute(data_t Aloc[TI][K], data_t Bl[NJG][PJ][K],
                    data_t acc[TI][NJG][QB][LANES]) {
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
            prod[kk] = a_reg[kk] * Bl[jg][jj][kc * KL + kk];
          }
          // level 1 (16 DSP fadds)
          data_t p16[16];
#pragma HLS ARRAY_PARTITION variable=p16 complete dim=1
        comp_tree_l1:
          for (int q = 0; q < 16; q++) {
#pragma HLS UNROLL
            p16[q] = prod[q] + prod[q + 16];
          }
          // level 2 (8 DSP fadds)
          data_t p8[8];
#pragma HLS ARRAY_PARTITION variable=p8 complete dim=1
        comp_tree_l2:
          for (int q = 0; q < 8; q++) {
#pragma HLS UNROLL
            p8[q] = p16[q] + p16[q + 8];
          }
          // levels 3..5 + accumulate in fabric
          data_t s3_0 = p8[0] + p8[4];
#pragma HLS BIND_OP variable=s3_0 op=fadd impl=fabric
          data_t s3_1 = p8[1] + p8[5];
#pragma HLS BIND_OP variable=s3_1 op=fadd impl=fabric
          data_t s3_2 = p8[2] + p8[6];
#pragma HLS BIND_OP variable=s3_2 op=fadd impl=fabric
          data_t s3_3 = p8[3] + p8[7];
#pragma HLS BIND_OP variable=s3_3 op=fadd impl=fabric
          data_t s4_0 = s3_0 + s3_2;
#pragma HLS BIND_OP variable=s4_0 op=fadd impl=fabric
          data_t s4_1 = s3_1 + s3_3;
#pragma HLS BIND_OP variable=s4_1 op=fadd impl=fabric
          data_t s5 = s4_0 + s4_1;
#pragma HLS BIND_OP variable=s5 op=fadd impl=fabric
          data_t r = acc[i][jg][jj / LANES][jj % LANES] + s5;
#pragma HLS BIND_OP variable=r op=fadd impl=fabric
          acc[i][jg][jj / LANES][jj % LANES] = r;
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

  static data_t Bl[NJG][PJ][K];
#pragma HLS ARRAY_PARTITION variable=Bl complete dim=2
#pragma HLS ARRAY_PARTITION variable=Bl cyclic factor=32 dim=3
  static data_t Aloc[TI][K];
#pragma HLS ARRAY_PARTITION variable=Aloc cyclic factor=32 dim=2
  static data_t acc[TI][NJG][QB][LANES];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=3
#pragma HLS ARRAY_PARTITION variable=acc complete dim=4

  load_B(B, Bl);

  load_AC(A, C, Aloc, acc, 0);
  compute(Aloc, Bl, acc);

itile_loop:
  for (int itix = 1; itix < NIT; itix++) {
    int it = itix * TI;
    xfer_AC(A, C, Aloc, acc, it - TI, it);
    compute(Aloc, Bl, acc);
  }

  store_C(C, acc, (NIT - 1) * TI);
}
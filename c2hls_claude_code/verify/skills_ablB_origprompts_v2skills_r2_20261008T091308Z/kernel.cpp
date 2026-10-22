#include "kernel.h"

// ---------------------------------------------------------------------------
// Tiling / parallelism parameters (U280: 9024 DSP)
//   TI    : rows of A / C processed in parallel (spatial PEs)
//   LANES : k-lanes per cycle (one 512-bit beat of B)
//   PJ    : columns interleaved in the pipeline (FP-add recurrence distance)
//   MACs/cycle = TI*LANES = 1664  -> ~8320 DSP (fmul 3 + fadd 2 per MAC)
// ---------------------------------------------------------------------------
#define LANES  16
#define TI     104
#define PJ     16
#define KC     (K / LANES)              // 64 k-chunks per row
#define NG     (J / PJ)                 // 64 column groups
#define TG     (PJ * KC)                // 1024 cycles per column group
#define NUM_IB ((I + TI - 1) / TI)      // 10 row blocks (last one partial)

// Loads A/C row blocks, runs the 104x16 MAC array with B prefetched
// (ping-pong) directly from gmem1, stores C row blocks.
static void compute_all(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
  static data_t A_loc[TI][KC][LANES];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=3
#pragma HLS BIND_STORAGE variable=A_loc type=ram_1p impl=lutram

  static data_t C_loc[TI][J];
#pragma HLS ARRAY_PARTITION variable=C_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_loc cyclic factor=16 dim=2
#pragma HLS BIND_STORAGE variable=C_loc type=ram_s2p impl=bram

  static data_t Bb0[TG][LANES];
  static data_t Bb1[TG][LANES];
#pragma HLS ARRAY_PARTITION variable=Bb0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=Bb1 complete dim=2
#pragma HLS BIND_STORAGE variable=Bb0 type=ram_s2p impl=bram
#pragma HLS BIND_STORAGE variable=Bb1 type=ram_s2p impl=bram

  data_t *Af = &A[0][0];
  data_t *Bf = &B[0][0];
  data_t *Cf = &C[0][0];

row_blk:
  for (int ib = 0; ib < NUM_IB; ib++) {
    const int i0 = ib * TI;
    const int nrows = (I - i0 < TI) ? (I - i0) : TI;

    // ---- load A and C row block (A on gmem0, C on gmem2, in parallel) ----
  load_ac:
    for (int idx = 0; idx < nrows * KC; idx++) {
#pragma HLS LOOP_TRIPCOUNT min=TI*KC max=TI*KC avg=TI*KC
#pragma HLS PIPELINE II=1
      const int ii = idx / KC;
      const int kc = idx % KC;
      const int base = (i0 * KC + idx) * LANES;
    load_ac_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        A_loc[ii][kc][l] = Af[base + l];
        C_loc[ii][kc * LANES + l] = Cf[base + l];
      }
    }

    // ---- compute: one flattened pipeline; load B group g, compute group g-1 ----
  compute_tg:
    for (int t_all = 0; t_all < (NG + 1) * TG; t_all++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_loc type=inter direction=RAW distance=16 dependent=true
#pragma HLS DEPENDENCE variable=Bb0 type=inter dependent=false
#pragma HLS DEPENDENCE variable=Bb1 type=inter dependent=false
      const int g = t_all / TG;
      const int t = t_all % TG;
      const bool do_load = (g < NG);
      const bool do_comp = (g > 0);
      const bool wsel = (g & 1);
      const int cg = g - 1;
      const bool rsel = (cg & 1);

      // operands of the group being computed (address t = kc*PJ + js)
      data_t b[LANES];
#pragma HLS ARRAY_PARTITION variable=b complete
    rd_b_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        data_t r0 = Bb0[t][l];
        data_t r1 = Bb1[t][l];
        b[l] = rsel ? r1 : r0;
      }

      // prefetch next group from gmem1: beat (g*TG + t) -> row g*PJ + t/KC, chunk t%KC
      if (do_load) {
        const int js_w = t / KC;
        const int kc_w = t % KC;
        const int waddr = kc_w * PJ + js_w;
        const int rbase = (g * TG + t) * LANES;
      wr_b_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          data_t v = Bf[rbase + l];
          if (wsel) Bb1[waddr][l] = v;
          else      Bb0[waddr][l] = v;
        }
      }

      if (do_comp) {
        const int kc = t / PJ;
        const int js = t % PJ;
        const int col = cg * PJ + js;
      pe_row:
        for (int ii = 0; ii < TI; ii++) {
#pragma HLS UNROLL
          data_t p[LANES];
#pragma HLS ARRAY_PARTITION variable=p complete
        pe_mul:
          for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
            p[l] = A_loc[ii][kc][l] * b[l];
          }
          data_t s8[8];
#pragma HLS ARRAY_PARTITION variable=s8 complete
        tree_8:
          for (int l = 0; l < 8; l++) {
#pragma HLS UNROLL
            s8[l] = p[2 * l] + p[2 * l + 1];
          }
          data_t s4[4];
#pragma HLS ARRAY_PARTITION variable=s4 complete
        tree_4:
          for (int l = 0; l < 4; l++) {
#pragma HLS UNROLL
            s4[l] = s8[2 * l] + s8[2 * l + 1];
          }
          data_t s2[2];
#pragma HLS ARRAY_PARTITION variable=s2 complete
        tree_2:
          for (int l = 0; l < 2; l++) {
#pragma HLS UNROLL
            s2[l] = s4[2 * l] + s4[2 * l + 1];
          }
          data_t s = s2[0] + s2[1];
          if (ii < nrows) {
            C_loc[ii][col] = C_loc[ii][col] + s;
          }
        }
      }
    }

    // ---- store C row block ----
  store_c:
    for (int idx = 0; idx < nrows * KC; idx++) {
#pragma HLS LOOP_TRIPCOUNT min=TI*KC max=TI*KC avg=TI*KC
#pragma HLS PIPELINE II=1
      const int ii = idx / KC;
      const int kc = idx % KC;
      const int base = (i0 * KC + idx) * LANES;
    store_c_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        Cf[base + l] = C_loc[ii][kc * LANES + l];
      }
    }
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  compute_all(A, B, C);
}
#include "kernel.h"
#include <hls_stream.h>

// Output-row block processed per pass (parallel PE rows) and k-unroll per PE.
#define IB 26
#define KU 64
#define LW 16                       // floats per 512-bit beat
#define NKB (K / KU)                // k-chunks per row (16)
#define NIB ((I + IB - 1) / IB)     // row blocks (40)
#define QB (KU / LW)                // beats per row per k-chunk (4)
#define CHUNK_BEATS (IB * QB)       // A beats per k-chunk per block (104)
#define A_BLK_BEATS (IB * K / LW)   // A beats per block (1664)
#define C_BLK_BEATS (IB * J / LW)   // C beats per block (1664)
#define C_PER_KB (C_BLK_BEATS / NKB) // C beats drained per k-chunk (104)
#define T_FLAT (NKB * J)            // compute iterations per block

struct vec16 {
  data_t d[LW];
};

// One PE: 64-wide dot product with balanced adder tree (fully pipelined).
static data_t dot_ku(const data_t a[KU], const data_t b[KU]) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=a complete dim=1
#pragma HLS ARRAY_PARTITION variable=b complete dim=1
  data_t p[KU];
#pragma HLS ARRAY_PARTITION variable=p complete dim=1
  dot_mul: for (int u = 0; u < KU; u++) {
#pragma HLS UNROLL
    p[u] = a[u] * b[u];
  }
  dot_tree: for (int s = KU / 2; s > 0; s >>= 1) {
#pragma HLS UNROLL
    dot_tree_lvl: for (int u = 0; u < s; u++) {
#pragma HLS UNROLL
      p[u] = p[u] + p[u + s];
    }
  }
  return p[0];
}

static void load_B(data_t B[J][K], data_t B_loc[J][K]) {
  load_B_row: for (int j = 0; j < J; j++) {
    load_B_col: for (int k = 0; k < K; k += LW) {
#pragma HLS PIPELINE II=1
      load_B_lane: for (int u = 0; u < LW; u++) {
#pragma HLS UNROLL
        B_loc[j][k + u] = B[j][k + u];
      }
    }
  }
}

// Streams A in (block, k-chunk, row, beat) order.
static void load_A_task(data_t A[I][K], hls::stream<vec16> &sA) {
  load_A_blk: for (int ib = 0; ib < NIB; ib++) {
    load_A_kb: for (int kb = 0; kb < NKB; kb++) {
      load_A_beat: for (int b = 0; b < CHUNK_BEATS; b++) {
#pragma HLS PIPELINE II=1
        int ii = b / QB;
        int q = b % QB;
        int row = ib * IB + ii;
        if (row > I - 1) row = I - 1;
        vec16 v;
        load_A_lane: for (int u = 0; u < LW; u++) {
#pragma HLS UNROLL
          v.d[u] = A[row][kb * KU + q * LW + u];
        }
        sA.write(v);
      }
    }
  }
}

static void compute_task(const data_t B_loc[J][K], hls::stream<vec16> &sA,
                         hls::stream<vec16> &sC) {
  static data_t A_loc[IB][K];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=KU dim=2
#pragma HLS BIND_STORAGE variable=A_loc type=ram_2p impl=lutram

  static data_t C_loc[2][IB][J];
#pragma HLS ARRAY_PARTITION variable=C_loc complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_loc complete dim=2
#pragma HLS ARRAY_PARTITION variable=C_loc cyclic factor=LW dim=3

  data_t a_reg[IB][KU];
#pragma HLS ARRAY_PARTITION variable=a_reg complete dim=0

  // Prologue: first block of A.
  pre_A_kb: for (int kb = 0; kb < NKB; kb++) {
    pre_A_beat: for (int b = 0; b < CHUNK_BEATS; b++) {
#pragma HLS PIPELINE II=1
      int ii = b / QB;
      int q = b % QB;
      vec16 v = sA.read();
      pre_A_lane: for (int u = 0; u < LW; u++) {
#pragma HLS UNROLL
        A_loc[ii][kb * KU + q * LW + u] = v.d[u];
      }
    }
  }

  block_loop: for (int ib = 0; ib < NIB; ib++) {
    const int pc = ib & 1;        // current C buffer
    const int pp = pc ^ 1;        // previous C buffer (being drained)
    const bool has_next = (ib + 1 < NIB);
    const bool has_prev = (ib > 0);

    compute_flat: for (int t = 0; t < T_FLAT; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C_loc inter false
#pragma HLS DEPENDENCE variable=A_loc inter false
      const int kb = t / J;
      const int j = t % J;

      // Latch this k-chunk of A into registers at chunk start.
      latch_a_row: for (int ii = 0; ii < IB; ii++) {
#pragma HLS UNROLL
        latch_a_u: for (int u = 0; u < KU; u++) {
#pragma HLS UNROLL
          if (j == 0) a_reg[ii][u] = A_loc[ii][kb * KU + u];
        }
      }

      // Refill the (already latched) chunk kb with next block's A.
      if (has_next && j >= 512 && j < 512 + CHUNK_BEATS) {
        const int b = j - 512;
        const int ii = b / QB;
        const int q = b % QB;
        vec16 v = sA.read();
        refill_lane: for (int u = 0; u < LW; u++) {
#pragma HLS UNROLL
          A_loc[ii][kb * KU + q * LW + u] = v.d[u];
        }
      }

      // Drain previous block's C.
      if (has_prev && j >= 256 && j < 256 + C_PER_KB) {
        const int b = kb * C_PER_KB + (j - 256);
        const int row = b / (J / LW);
        const int cb = b % (J / LW);
        vec16 v;
        drain_lane: for (int u = 0; u < LW; u++) {
#pragma HLS UNROLL
          v.d[u] = C_loc[pp][row][cb * LW + u];
        }
        sC.write(v);
      }

      data_t bv[KU];
#pragma HLS ARRAY_PARTITION variable=bv complete dim=1
      gather_b: for (int u = 0; u < KU; u++) {
#pragma HLS UNROLL
        bv[u] = B_loc[j][kb * KU + u];
      }
      pe_row: for (int ii = 0; ii < IB; ii++) {
#pragma HLS UNROLL
        data_t s = dot_ku(a_reg[ii], bv);
        data_t prev = (kb == 0) ? (data_t)0 : C_loc[pc][ii][j];
        C_loc[pc][ii][j] = prev + s;
      }
    }
  }

  // Epilogue: drain last block's C.
  const int pl = (NIB - 1) & 1;
  post_C_beat: for (int b = 0; b < C_BLK_BEATS; b++) {
#pragma HLS PIPELINE II=1
    const int row = b / (J / LW);
    const int cb = b % (J / LW);
    vec16 v;
    post_C_lane: for (int u = 0; u < LW; u++) {
#pragma HLS UNROLL
      v.d[u] = C_loc[pl][row][cb * LW + u];
    }
    sC.write(v);
  }
}

static void store_C_task(hls::stream<vec16> &sC, data_t C[I][J]) {
  store_C_blk: for (int ib = 0; ib < NIB; ib++) {
    store_C_beat: for (int b = 0; b < C_BLK_BEATS; b++) {
#pragma HLS PIPELINE II=1
      const int row = ib * IB + b / (J / LW);
      const int cb = b % (J / LW);
      vec16 v = sC.read();
      if (row < I) {
        store_C_lane: for (int u = 0; u < LW; u++) {
#pragma HLS UNROLL
          C[row][cb * LW + u] = v.d[u];
        }
      }
    }
  }
}

static void mm_dataflow(data_t A[I][K], const data_t B_loc[J][K], data_t C[I][J]) {
#pragma HLS DATAFLOW
  hls::stream<vec16> sA("sA");
  hls::stream<vec16> sC("sC");
#pragma HLS STREAM variable=sA depth=256
#pragma HLS STREAM variable=sC depth=256
  load_A_task(A, sA);
  compute_task(B_loc, sA, sC);
  store_C_task(sC, C);
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_write_burst_length=256 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Full B held on-chip in URAM, banked on k so 64 k-values are read per cycle.
  static data_t B_loc[J][K];
#pragma HLS BIND_STORAGE variable=B_loc type=ram_2p impl=uram
#pragma HLS ARRAY_PARTITION variable=B_loc cyclic factor=KU dim=2

  load_B(B, B_loc);
  mm_dataflow(A, B_loc, C);
}

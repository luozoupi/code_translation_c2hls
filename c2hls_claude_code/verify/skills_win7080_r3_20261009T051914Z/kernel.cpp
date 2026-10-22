#include "kernel.h"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
#define LANES      16                    // floats per 512-bit beat
#define PI         5                     // parallel output rows
#define PJ         16                    // parallel output cols (== LANES)
#define KL         16                    // k lanes per dot product (== LANES)
#define TJ         256                   // output columns per j-tile
#define NJT        (J / TJ)              // 4 j-tiles
#define IB         ((I + PI - 1) / PI)   // 205 row blocks (1025 padded rows)
#define JB         (TJ / PJ)             // 16 column blocks per tile
#define CB         (TJ / LANES)          // 16 beats per C row segment
#define KB         (K / KL)              // 64 k-chunks
#define ACC_D      (IB * JB)             // 3280 accumulators per PE
#define A_BEATS    (I * K / LANES)       // 65536
#define B_BEATS    (TJ * K / LANES)      // 16384
#define C_BEATS    (I * TJ / LANES)      // 16384
#define MAC_ITERS  (KB * IB * JB)        // 209920 per tile
#define OUT_SPREAD 4                     // drain one beat every 4 MAC cycles

// ---------------------------------------------------------------------------
// 16-input balanced adder tree
// ---------------------------------------------------------------------------
static data_t tree16(const data_t x[KL]) {
#pragma HLS INLINE
  data_t s8[8], s4[4], s2[2];
#pragma HLS ARRAY_PARTITION variable=s8 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1
tree_l8:
  for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
    s8[i] = x[2 * i] + x[2 * i + 1];
  }
tree_l4:
  for (int i = 0; i < 4; i++) {
#pragma HLS UNROLL
    s4[i] = s8[2 * i] + s8[2 * i + 1];
  }
tree_l2:
  for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
    s2[i] = s4[2 * i] + s4[2 * i + 1];
  }
  return s2[0] + s2[1];
}

// ---------------------------------------------------------------------------
// Core: A resident, B/acc ping-pong, 5x16 outputs x 16 k-lanes per cycle
// B prefetch and C read-modify-write drain are overlapped with the MAC loop.
// ---------------------------------------------------------------------------
static void core(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
  // A: row i = ib*PI + q  ->  Abuf[q][ib][k]
  static data_t Abuf[PI][IB][K];
#pragma HLS ARRAY_PARTITION variable=Abuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=Abuf cyclic factor=16 dim=3
#pragma HLS BIND_STORAGE variable=Abuf type=ram_2p impl=uram

  // B tile: tile row r = jb*PJ + pj  ->  Bb[pj][jb][k]
  static data_t Bb0[PJ][JB][K];
  static data_t Bb1[PJ][JB][K];
#pragma HLS ARRAY_PARTITION variable=Bb0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=Bb0 cyclic factor=16 dim=3
#pragma HLS ARRAY_PARTITION variable=Bb1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=Bb1 cyclic factor=16 dim=3
#pragma HLS BIND_STORAGE variable=Bb0 type=ram_s2p impl=bram
#pragma HLS BIND_STORAGE variable=Bb1 type=ram_s2p impl=bram

  // acc: output (ib*PI+q, jt*TJ + jb*PJ + pj) -> acc[q][pj][ib*JB + jb]
  static data_t acc0[PI][PJ][ACC_D];
  static data_t acc1[PI][PJ][ACC_D];
#pragma HLS ARRAY_PARTITION variable=acc0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=acc1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc1 complete dim=2
#pragma HLS BIND_STORAGE variable=acc0 type=ram_s2p impl=bram
#pragma HLS BIND_STORAGE variable=acc1 type=ram_s2p impl=bram

  // Zero the padded A row(s) (rows >= I in the last row block)
pad_A_rows:
  for (int q = I % PI; q < PI; q++) {
  pad_A_k:
    for (int kb = 0; kb < KB; kb++) {
#pragma HLS PIPELINE II=1
    pad_A_lane:
      for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
        Abuf[q][IB - 1][kb * KL + l] = 0;
      }
    }
  }

  // Prologue: load all of A and B tile 0 (512-bit beats)
  {
    int pa = 0, ia = 0;
  load_A_B0:
    for (int t = 0; t < A_BEATS; t++) {
#pragma HLS PIPELINE II=1
      int ra = t / KB;
      int kb = t % KB;
    load_A_lane:
      for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
        Abuf[pa][ia][kb * KL + l] = A[ra][kb * KL + l];
      }
      if (t < B_BEATS) {
        int r = t / KB;
      load_B0_lane:
        for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
          Bb0[r % PJ][r / PJ][kb * KL + l] = B[r][kb * KL + l];
        }
      }
      if (kb == KB - 1) {
        if (pa == PI - 1) { pa = 0; ia++; } else { pa++; }
      }
    }
  }

  // Main: compute tile jt, prefetch B tile jt+1, drain acc of tile jt-1 into C
tiles:
  for (int jt = 0; jt < NJT; jt++) {
    const int p = jt & 1;
    const bool b_pref = (jt < NJT - 1);
    const bool o_drain = (jt > 0);
    const int col_o = (jt - 1) * TJ;
    int ib = 0, kc = 0;
    int cbo = 0, pco = 0, ico = 0;
  mac:
    for (int s = 0; s < MAC_ITERS; s++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc0 type=inter dependent=false
#pragma HLS DEPENDENCE variable=acc1 type=inter dependent=false
#pragma HLS DEPENDENCE variable=Bb0 type=inter dependent=false
#pragma HLS DEPENDENCE variable=Bb1 type=inter dependent=false
#pragma HLS DEPENDENCE variable=C type=inter dependent=false
      const int jb = s % JB;
      const int addr_c = ib * JB + jb;
      const int addr_o = ico * JB + cbo;

      // ---- operand fetch ----
      data_t a[PI][KL];
#pragma HLS ARRAY_PARTITION variable=a complete dim=0
      data_t b[PJ][KL];
#pragma HLS ARRAY_PARTITION variable=b complete dim=0
    fetch_a_q:
      for (int q = 0; q < PI; q++) {
#pragma HLS UNROLL
      fetch_a_l:
        for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
          a[q][l] = Abuf[q][ib][kc * KL + l];
        }
      }
    fetch_b_j:
      for (int pj = 0; pj < PJ; pj++) {
#pragma HLS UNROLL
      fetch_b_l:
        for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
          b[pj][l] = p ? Bb1[pj][jb][kc * KL + l] : Bb0[pj][jb][kc * KL + l];
        }
      }

      // ---- prefetch next B tile into the idle buffer ----
      if (b_pref && s < B_BEATS) {
        int r = s / KB;
        int kb = s % KB;
        int gr = (jt + 1) * TJ + r;
      pref_b_lane:
        for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
          data_t bv = B[gr][kb * KL + l];
          if (p) Bb0[r % PJ][r / PJ][kb * KL + l] = bv;
          else   Bb1[r % PJ][r / PJ][kb * KL + l] = bv;
        }
      }

      // ---- MAC array + accumulator update + drain read ----
      const bool o_act = o_drain && ((s % OUT_SPREAD) == 0) &&
                         (s < C_BEATS * OUT_SPREAD);
      data_t ov[PJ];
#pragma HLS ARRAY_PARTITION variable=ov complete dim=1
    pe_q:
      for (int q = 0; q < PI; q++) {
#pragma HLS UNROLL
      pe_j:
        for (int pj = 0; pj < PJ; pj++) {
#pragma HLS UNROLL
          const int ra0 = p ? addr_o : addr_c;
          const int ra1 = p ? addr_c : addr_o;
          data_t r0 = acc0[q][pj][ra0];
          data_t r1 = acc1[q][pj][ra1];
          data_t old = p ? r1 : r0;
          data_t oth = p ? r0 : r1;
          if (q == pco) ov[pj] = oth;

          data_t prod[KL];
#pragma HLS ARRAY_PARTITION variable=prod complete dim=1
        pe_mul:
          for (int l = 0; l < KL; l++) {
#pragma HLS UNROLL
            prod[l] = a[q][l] * b[pj][l];
          }
          data_t dot = tree16(prod);
          data_t base = (kc == 0) ? (data_t)0 : old;
          data_t nv = base + dot;
          if (p) acc1[q][pj][addr_c] = nv;
          else   acc0[q][pj][addr_c] = nv;
        }
      }

      if (o_act) {
        int row = ico * PI + pco;
      drain_c_lane:
        for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
          C[row][col_o + cbo * LANES + l] = C[row][col_o + cbo * LANES + l] + ov[l];
        }
        if (cbo == CB - 1) {
          cbo = 0;
          if (pco == PI - 1) { pco = 0; ico++; } else { pco++; }
        } else {
          cbo++;
        }
      }

      // ---- compute counters: s = (kc*IB + ib)*JB + jb ----
      if (jb == JB - 1) {
        if (ib == IB - 1) { ib = 0; kc++; } else { ib++; }
      }
    }
  }

  // Epilogue: drain the last tile's accumulators into C
  {
    int pco = 0, ico = 0;
    const int col_l = (NJT - 1) * TJ;
  drain_last:
    for (int s = 0; s < C_BEATS; s++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=C type=inter dependent=false
      int cbo = s % CB;
      int row = ico * PI + pco;
    drain_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        data_t v = ((NJT - 1) & 1) ? acc1[pco][l][ico * JB + cbo]
                                   : acc0[pco][l][ico * JB + cbo];
        C[row][col_l + cbo * LANES + l] = C[row][col_l + cbo * LANES + l] + v;
      }
      if (cbo == CB - 1) {
        if (pco == PI - 1) { pco = 0; ico++; } else { pco++; }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Top
// ---------------------------------------------------------------------------
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  core(A, B, C);
}
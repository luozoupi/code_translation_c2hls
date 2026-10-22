#include "kernel.h"

// ---------------------------------------------------------------------------
// Output-stationary PE array: PE_I x PE_J = 32 x 52 = 1664 fp32 MACs / cycle
// (~5 DSP per MAC -> ~8320 DSP on xcu280).
// C row-block of TI=512 rows is kept on chip (Cbuf) and accumulated in place,
// preserving the original sequential k order per element (bit-exact).
// A / B use skewed banking: element (row,k) -> bank (row + k) mod NBANK, so a
// 16-float 512-bit beat writes 16 distinct banks (II=1 loads), and the compute
// reads all banks at one address and rotates back into row order.
// ---------------------------------------------------------------------------

#define PE_I    32                     // parallel rows of C (A banks)
#define PE_J    52                     // parallel cols of C (B banks)
#define NSI     16                     // row sub-blocks per row-block (dep distance)
#define TI      (PE_I * NSI)           // 512 rows of C per block
#define NIB     (I / TI)               // 2 row-blocks
#define NJB     ((J + PE_J - 1) / PE_J) // 20 column groups (1040 padded cols)
#define LANES   16                     // floats per 512-bit beat
#define KBEATS  (K / LANES)            // 64 beats per row of A/B
#define CBEATS  (J / LANES)            // 64 beats per row of C
#define A_DEPTH (NSI * K)              // 16384 per A bank
#define B_DEPTH (NJB * K)              // 20480 per B bank
#define C_DEPTH (NSI * NJB)            // 320 per C bank
#define COMP_TRIP (NJB * K * NSI)      // 327680 compute iterations per block

// ------------------------------- loaders -----------------------------------
static void load_A(data_t A[I][K], data_t A_loc[PE_I][A_DEPTH], int ib) {
load_A_beats:
  for (int t = 0; t < TI * KBEATS; t++) {
#pragma HLS PIPELINE II=1
    int r  = t >> 6;
    int k0 = (t & 63) << 4;
    data_t v[LANES];
#pragma HLS ARRAY_PARTITION variable=v complete dim=1
  read_A_lanes:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      v[l] = A[ib * TI + r][k0 + l];
    }
    int pi = r & (PE_I - 1);
    int si = r >> 5;
    int s  = (pi + k0) & (PE_I - 1);
  write_A_banks:
    for (int b = 0; b < PE_I; b++) {
#pragma HLS UNROLL
      int l = (b - s) & (PE_I - 1);
      if (l < LANES) {
        A_loc[b][si * K + k0 + l] = v[l & (LANES - 1)];
      }
    }
  }
}

static void load_B(data_t B[J][K], data_t B_loc[PE_J][B_DEPTH], int nb) {
load_B_beats:
  for (int t = 0; t < nb; t++) {
#pragma HLS LOOP_TRIPCOUNT min=0 avg=65536 max=65536
#pragma HLS PIPELINE II=1
    int j  = t >> 6;
    int k0 = (t & 63) << 4;
    data_t v[LANES];
#pragma HLS ARRAY_PARTITION variable=v complete dim=1
  read_B_lanes:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      v[l] = B[j][k0 + l];
    }
    int jb = j / PE_J;
    int pj = j - jb * PE_J;
    int s  = (pj + k0) % PE_J;
  write_B_banks:
    for (int b = 0; b < PE_J; b++) {
#pragma HLS UNROLL
      int l = b - s;
      if (l < 0) l += PE_J;
      if (l < LANES) {
        B_loc[b][jb * K + k0 + l] = v[l & (LANES - 1)];
      }
    }
  }
}

static void load_C(data_t C[I][J], data_t Cbuf[PE_I][PE_J][C_DEPTH], int ib) {
load_C_beats:
  for (int t = 0; t < TI * CBEATS; t++) {
#pragma HLS PIPELINE II=1
    int r  = t >> 6;
    int c0 = (t & 63) << 4;
    data_t v[LANES];
#pragma HLS ARRAY_PARTITION variable=v complete dim=1
  read_C_lanes:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      v[l] = C[ib * TI + r][c0 + l];
    }
    int pi = r & (PE_I - 1);
    int si = r >> 5;
    int q  = c0 / PE_J;
    int rr = c0 - q * PE_J;
  write_C_banks:
    for (int pj = 0; pj < PE_J; pj++) {
#pragma HLS UNROLL
      int l, jb;
      if (pj >= rr) { l = pj - rr;        jb = q; }
      else          { l = pj + PE_J - rr; jb = q + 1; }
      if (l < LANES) {
        Cbuf[pi][pj][si * NJB + jb] = v[l & (LANES - 1)];
      }
    }
  }
}

static void load_phase(data_t A[I][K], data_t B[J][K], data_t C[I][J],
                       data_t A_loc[PE_I][A_DEPTH], data_t B_loc[PE_J][B_DEPTH],
                       data_t Cbuf[PE_I][PE_J][C_DEPTH], int ib, int nb) {
#pragma HLS DATAFLOW
  load_A(A, A_loc, ib);
  load_B(B, B_loc, nb);
  load_C(C, Cbuf, ib);
}

// ------------------------------- compute -----------------------------------
static void compute(data_t A_loc[PE_I][A_DEPTH], data_t B_loc[PE_J][B_DEPTH],
                    data_t Cbuf[PE_I][PE_J][C_DEPTH]) {
compute_mac:
  for (int t = 0; t < COMP_TRIP; t++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=Cbuf type=inter direction=RAW dependent=true distance=16
    int si = t & (NSI - 1);
    int k  = (t >> 4) & (K - 1);
    int jb = t >> 14;
    int km = k % PE_J;
    int ka = k & (PE_I - 1);

    data_t va[PE_I], a[PE_I];
    data_t vb[PE_J], bv[PE_J];
#pragma HLS ARRAY_PARTITION variable=va complete dim=1
#pragma HLS ARRAY_PARTITION variable=a  complete dim=1
#pragma HLS ARRAY_PARTITION variable=vb complete dim=1
#pragma HLS ARRAY_PARTITION variable=bv complete dim=1

  read_A_bank:
    for (int b = 0; b < PE_I; b++) {
#pragma HLS UNROLL
      va[b] = A_loc[b][si * K + k];
    }
  rotate_A:
    for (int pi = 0; pi < PE_I; pi++) {
#pragma HLS UNROLL
      a[pi] = va[(pi + ka) & (PE_I - 1)];
    }
  read_B_bank:
    for (int b = 0; b < PE_J; b++) {
#pragma HLS UNROLL
      vb[b] = B_loc[b][jb * K + k];
    }
  rotate_B:
    for (int pj = 0; pj < PE_J; pj++) {
#pragma HLS UNROLL
      int idx = pj + km;
      if (idx >= PE_J) idx -= PE_J;
      bv[pj] = vb[idx];
    }

    int ca = si * NJB + jb;
  mac_rows:
    for (int pi = 0; pi < PE_I; pi++) {
#pragma HLS UNROLL
    mac_cols:
      for (int pj = 0; pj < PE_J; pj++) {
#pragma HLS UNROLL
        Cbuf[pi][pj][ca] = Cbuf[pi][pj][ca] + a[pi] * bv[pj];
      }
    }
  }
}

// -------------------------------- store ------------------------------------
static void store_C(data_t C[I][J], data_t Cbuf[PE_I][PE_J][C_DEPTH], int ib) {
store_C_beats:
  for (int t = 0; t < TI * CBEATS; t++) {
#pragma HLS PIPELINE II=1
    int r  = t >> 6;
    int c0 = (t & 63) << 4;
    int pi = r & (PE_I - 1);
    int si = r >> 5;
    int q  = c0 / PE_J;
    int rr = c0 - q * PE_J;
    data_t o[PE_J];
#pragma HLS ARRAY_PARTITION variable=o complete dim=1
  read_C_banks:
    for (int pj = 0; pj < PE_J; pj++) {
#pragma HLS UNROLL
      int l, jb;
      if (pj >= rr) { l = pj - rr;        jb = q; }
      else          { l = pj + PE_J - rr; jb = q + 1; }
      o[pj] = (l < LANES) ? Cbuf[pi][pj][si * NJB + jb] : (data_t)0;
    }
  write_C_lanes:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      int pjl = rr + l;
      if (pjl >= PE_J) pjl -= PE_J;
      C[ib * TI + r][c0 + l] = o[pjl];
    }
  }
}

// --------------------------------- top -------------------------------------
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  static data_t A_loc[PE_I][A_DEPTH];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=1
#pragma HLS BIND_STORAGE variable=A_loc type=ram_2p impl=uram

  static data_t B_loc[PE_J][B_DEPTH];
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=1
#pragma HLS BIND_STORAGE variable=B_loc type=ram_2p impl=uram

  static data_t Cbuf[PE_I][PE_J][C_DEPTH];
#pragma HLS ARRAY_PARTITION variable=Cbuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=Cbuf complete dim=2
#pragma HLS BIND_STORAGE variable=Cbuf type=ram_2p impl=bram

row_block:
  for (int ib = 0; ib < NIB; ib++) {
    int nb = (ib == 0) ? (J * KBEATS) : 0;   // B loaded once, stays resident
    load_phase(A, B, C, A_loc, B_loc, Cbuf, ib, nb);
    compute(A_loc, B_loc, Cbuf);
    store_C(C, Cbuf, ib);
  }
}
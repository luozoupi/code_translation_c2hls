#include "kernel.h"

// ---------------------------------------------------------------------------
// Tiled, dataflow-overlapped GEMM:  C[i][j] += sum_k A[i][k] * B[j][k]
//   Datapath per cycle: PE_I x PE_J = 13 x 2 = 26 output PEs, each a KU=64
//   wide dot product (adder tree) => 1664 fp32 MACs / cycle (~8.3K DSP).
//   Reduction order: k-chunk outer, (ib,jb) inner -> each output accumulator
//   is revisited every NIB*NJB = 20*128 = 2560 cycles (no FP recurrence).
//   TI=256 padded to TI_PAD=260 (=20*13); pad rows are zero / never stored.
//   B_buf mapped to URAM (unused otherwise) to relieve BRAM.
// ---------------------------------------------------------------------------

#define TI     256             // output tile rows (rows of A)
#define TJ     256             // output tile cols (rows of B)
#define PE_I   13              // spatial PE rows
#define PE_J   2               // spatial PE cols
#define KU     64              // k lanes per PE (adder-tree width)
#define LANES  16              // 512-bit beat / 32-bit float

#define NIB    ((TI + PE_I - 1) / PE_I)   // 20
#define TI_PAD (NIB * PE_I)               // 260
#define NJB    (TJ / PE_J)                // 128
#define NT_I   (I / TI)
#define NT_J   (J / TJ)
#define NT     (NT_I * NT_J)
#define NKC    (K / KU)

// ----------------------------- load stages ---------------------------------
static void load_A(data_t A[I][K], data_t A_buf[TI_PAD][K], int t) {
  const int i0 = (t / NT_J) * TI;
load_A_row:
  for (int i = 0; i < TI_PAD; i++) {
  load_A_beat:
    for (int kb = 0; kb < K / LANES; kb++) {
#pragma HLS PIPELINE II=1
    load_A_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        data_t v = 0;
        if (i < TI) v = A[i0 + i][kb * LANES + l];
        A_buf[i][kb * LANES + l] = v;
      }
    }
  }
}

static void load_B(data_t B[J][K], data_t B_buf[TJ][K], int t) {
  const int j0 = (t % NT_J) * TJ;
load_B_row:
  for (int j = 0; j < TJ; j++) {
  load_B_beat:
    for (int kb = 0; kb < K / LANES; kb++) {
#pragma HLS PIPELINE II=1
    load_B_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        B_buf[j][kb * LANES + l] = B[j0 + j][kb * LANES + l];
      }
    }
  }
}

static void load_C(data_t C[I][J], data_t Cin[TI_PAD][TJ], int t) {
  const int i0 = (t / NT_J) * TI;
  const int j0 = (t % NT_J) * TJ;
load_C_row:
  for (int i = 0; i < TI_PAD; i++) {
  load_C_beat:
    for (int jb = 0; jb < TJ / LANES; jb++) {
#pragma HLS PIPELINE II=1
    load_C_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        data_t v = 0;
        if (i < TI) v = C[i0 + i][j0 + jb * LANES + l];
        Cin[i][jb * LANES + l] = v;
      }
    }
  }
}

// ----------------------------- dot product ---------------------------------
static data_t dot_ku(const data_t a[KU], const data_t b[KU]) {
#pragma HLS INLINE
  data_t p[KU];
  data_t l1[KU / 2];
  data_t l2[KU / 4];
  data_t l3[KU / 8];
  data_t l4[KU / 16];
  data_t l5[KU / 32];
#pragma HLS ARRAY_PARTITION variable=p  complete dim=1
#pragma HLS ARRAY_PARTITION variable=l1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l3 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=l5 complete dim=1
dot_mul:
  for (int u = 0; u < KU; u++) {
#pragma HLS UNROLL
    p[u] = a[u] * b[u];
  }
dot_lvl1:
  for (int u = 0; u < KU / 2; u++) {
#pragma HLS UNROLL
    l1[u] = p[2 * u] + p[2 * u + 1];
  }
dot_lvl2:
  for (int u = 0; u < KU / 4; u++) {
#pragma HLS UNROLL
    l2[u] = l1[2 * u] + l1[2 * u + 1];
  }
dot_lvl3:
  for (int u = 0; u < KU / 8; u++) {
#pragma HLS UNROLL
    l3[u] = l2[2 * u] + l2[2 * u + 1];
  }
dot_lvl4:
  for (int u = 0; u < KU / 16; u++) {
#pragma HLS UNROLL
    l4[u] = l3[2 * u] + l3[2 * u + 1];
  }
dot_lvl5:
  for (int u = 0; u < KU / 32; u++) {
#pragma HLS UNROLL
    l5[u] = l4[2 * u] + l4[2 * u + 1];
  }
  return l5[0] + l5[1];
}

// ----------------------------- compute stage -------------------------------
static void compute_tile(data_t A_buf[TI_PAD][K], data_t B_buf[TJ][K],
                         data_t Cin[TI_PAD][TJ], data_t Cout[TI_PAD][TJ]) {
compute_kc:
  for (int kc = 0; kc < NKC; kc++) {
  compute_ib:
    for (int ib = 0; ib < NIB; ib++) {
    compute_jb:
      for (int jb = 0; jb < NJB; jb++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=Cout type=inter dependent=false
        data_t a_reg[PE_I][KU];
        data_t b_reg[PE_J][KU];
#pragma HLS ARRAY_PARTITION variable=a_reg complete dim=0
#pragma HLS ARRAY_PARTITION variable=b_reg complete dim=0
      fetch_a_row:
        for (int pi = 0; pi < PE_I; pi++) {
#pragma HLS UNROLL
        fetch_a_k:
          for (int u = 0; u < KU; u++) {
#pragma HLS UNROLL
            a_reg[pi][u] = A_buf[ib * PE_I + pi][kc * KU + u];
          }
        }
      fetch_b_row:
        for (int pj = 0; pj < PE_J; pj++) {
#pragma HLS UNROLL
        fetch_b_k:
          for (int u = 0; u < KU; u++) {
#pragma HLS UNROLL
            b_reg[pj][u] = B_buf[jb * PE_J + pj][kc * KU + u];
          }
        }
      pe_row:
        for (int pi = 0; pi < PE_I; pi++) {
#pragma HLS UNROLL
        pe_col:
          for (int pj = 0; pj < PE_J; pj++) {
#pragma HLS UNROLL
            const int ii = ib * PE_I + pi;
            const int jj = jb * PE_J + pj;
            data_t s = dot_ku(a_reg[pi], b_reg[pj]);
            data_t prev = (kc == 0) ? Cin[ii][jj] : Cout[ii][jj];
            Cout[ii][jj] = prev + s;
          }
        }
      }
    }
  }
}

// ----------------------------- store stage ---------------------------------
static void store_C(data_t C[I][J], data_t Cout[TI_PAD][TJ], int t) {
  const int i0 = (t / NT_J) * TI;
  const int j0 = (t % NT_J) * TJ;
store_C_row:
  for (int i = 0; i < TI; i++) {
  store_C_beat:
    for (int jb = 0; jb < TJ / LANES; jb++) {
#pragma HLS PIPELINE II=1
    store_C_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        C[i0 + i][j0 + jb * LANES + l] = Cout[i][jb * LANES + l];
      }
    }
  }
}

// ----------------------------- top -----------------------------------------
extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

tile_loop:
  for (int t = 0; t < NT; t++) {
#pragma HLS DATAFLOW
    data_t A_buf[TI_PAD][K];
    data_t B_buf[TJ][K];
    data_t Cin[TI_PAD][TJ];
    data_t Cout[TI_PAD][TJ];
#pragma HLS ARRAY_PARTITION variable=A_buf cyclic factor=13 dim=1
#pragma HLS ARRAY_PARTITION variable=A_buf cyclic factor=64 dim=2
#pragma HLS ARRAY_PARTITION variable=B_buf cyclic factor=2  dim=1
#pragma HLS ARRAY_PARTITION variable=B_buf cyclic factor=64 dim=2
#pragma HLS BIND_STORAGE variable=B_buf type=ram_2p impl=uram
#pragma HLS ARRAY_PARTITION variable=Cin  cyclic factor=13 dim=1
#pragma HLS ARRAY_PARTITION variable=Cin  cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=Cout cyclic factor=13 dim=1
#pragma HLS ARRAY_PARTITION variable=Cout cyclic factor=16 dim=2
    load_A(A, A_buf, t);
    load_B(B, B_buf, t);
    load_C(C, Cin, t);
    compute_tile(A_buf, B_buf, Cin, Cout);
    store_C(C, Cout, t);
  }
}
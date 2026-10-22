#include "kernel.h"

#define LANES   16
#define PE_I    32                  // MAC rows
#define PE_J    40                  // MAC cols  -> 1280 fp32 MACs (~6400 DSP)
#define RI      8                   // row sub-blocks per tile
#define RJ      2                   // col sub-blocks per tile
#define NB      (RI * RJ)           // 16 interleaved sub-blocks (> FP add latency)
#define TI      (RI * PE_I)         // 256
#define TJ      (RJ * PE_J)         // 80
#define CBT     (TJ / LANES)        // 5 beats per full tile row
#define CBL     ((J - (NJT - 1) * TJ) / LANES)  // 4 beats in last tile row
#define NIT     (I / TI)            // 4 row tiles
#define NJT     ((J + TJ - 1) / TJ) // 13 col tiles (last = 64 cols)
#define NIB     (I / PE_I)          // 32 row blocks
#define NJB     (NJT * RJ)          // 26 col blocks of 40
#define KB      (K / LANES)         // 64 beats per row

// One-time load of all of A and B into resident URAM (512-bit beats, II=1)
static void load_AB_all(data_t A[I][K], data_t B[J][K],
                        data_t A_res[PE_I][NIB][K],
                        data_t B_res[PE_J][NJB][K]) {
#pragma HLS INLINE off
load_AB_beat:
  for (int b = 0; b < I * KB; b++) {
#pragma HLS PIPELINE II=1
    const int r  = b / KB;
    const int kb = b % KB;
    const int p  = r % PE_I;
    const int ib = r / PE_I;
    const int q  = r % PE_J;
    const int jb = r / PE_J;
  load_AB_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      A_res[p][ib][kb * LANES + l] = A[r][kb * LANES + l];
      B_res[q][jb][kb * LANES + l] = B[r][kb * LANES + l];
    }
  }
}

// C tile load: flat 512-bit walk, row staged in registers, one
// constant-indexed write per acc bank per row (no lane bank conflicts)
template <int CB>
static void load_C_tile(data_t C[I][J], data_t acc[NB][PE_I][PE_J],
                        int ti, int tj) {
#pragma HLS INLINE
  data_t rowbuf[TJ];
#pragma HLS ARRAY_PARTITION variable=rowbuf complete dim=1
  int r = 0;
  int cb = 0;
load_C_beat:
  for (int b = 0; b < TI * CB; b++) {
#pragma HLS PIPELINE II=1
  load_C_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      rowbuf[cb * LANES + l] = C[ti + r][tj + cb * LANES + l];
    }
    if (cb == CB - 1) {
      const int rb = (r >> 5) * RJ;
      const int p  = r & (PE_I - 1);
    load_C_commit:
      for (int q = 0; q < PE_J; q++) {
#pragma HLS UNROLL
        acc[rb][p][q]     = rowbuf[q];
        acc[rb + 1][p][q] = rowbuf[PE_J + q];
      }
      cb = 0;
      r++;
    } else {
      cb++;
    }
  }
}

// C tile store: flat 512-bit walk, row read with constant bank indices
template <int CB>
static void store_C_tile(data_t C[I][J], data_t acc[NB][PE_I][PE_J],
                         int ti, int tj) {
#pragma HLS INLINE
  int r = 0;
  int cb = 0;
store_C_beat:
  for (int b = 0; b < TI * CB; b++) {
#pragma HLS PIPELINE II=1
    const int rb = (r >> 5) * RJ;
    const int p  = r & (PE_I - 1);
    data_t rowv[TJ];
#pragma HLS ARRAY_PARTITION variable=rowv complete dim=1
  store_C_gather:
    for (int q = 0; q < PE_J; q++) {
#pragma HLS UNROLL
      rowv[q]        = acc[rb][p][q];
      rowv[PE_J + q] = acc[rb + 1][p][q];
    }
  store_C_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      C[ti + r][tj + cb * LANES + l] = rowv[cb * LANES + l];
    }
    if (cb == CB - 1) {
      cb = 0;
      r++;
    } else {
      cb++;
    }
  }
}

// Compute all C tiles against resident A and B (32x40 MAC array)
static void compute_all(data_t A_res[PE_I][NIB][K],
                        data_t B_res[PE_J][NJB][K],
                        data_t C[I][J]) {
#pragma HLS INLINE off

  data_t acc[NB][PE_I][PE_J];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=3
#pragma HLS BIND_STORAGE variable=acc type=ram_s2p impl=lutram

tile_i_loop:
  for (int it = 0; it < NIT; it++) {
  tile_j_loop:
    for (int jt = 0; jt < NJT; jt++) {
      const int ti = it * TI;
      const int tj = jt * TJ;

      if (jt < NJT - 1) {
        load_C_tile<CBT>(C, acc, ti, tj);
      } else {
        load_C_tile<CBL>(C, acc, ti, tj);
      }

    mac_k:
      for (int k = 0; k < K; k++) {
      mac_blk:
        for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc type=inter dependent=false
          const int ri = b / RJ;
          const int rj = b % RJ;
          const int ib = it * RI + ri;
          const int jb = jt * RJ + rj;
          data_t av[PE_I];
          data_t bv[PE_J];
#pragma HLS ARRAY_PARTITION variable=av complete dim=1
#pragma HLS ARRAY_PARTITION variable=bv complete dim=1
        rd_a:
          for (int p = 0; p < PE_I; p++) {
#pragma HLS UNROLL
            av[p] = A_res[p][ib][k];
          }
        rd_b:
          for (int q = 0; q < PE_J; q++) {
#pragma HLS UNROLL
            bv[q] = B_res[q][jb][k];
          }
        mac_p:
          for (int p = 0; p < PE_I; p++) {
#pragma HLS UNROLL
          mac_q:
            for (int q = 0; q < PE_J; q++) {
#pragma HLS UNROLL
              acc[b][p][q] = acc[b][p][q] + av[p] * bv[q];
            }
          }
        }
      }

      if (jt < NJT - 1) {
        store_C_tile<CBT>(C, acc, ti, tj);
      } else {
        store_C_tile<CBL>(C, acc, ti, tj);
      }
    }
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 max_write_burst_length=64 num_read_outstanding=16 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Resident A: 32 row banks x 8 k banks -> 256 URAMs
  static data_t A_res[PE_I][NIB][K];
#pragma HLS ARRAY_PARTITION variable=A_res complete dim=1
#pragma HLS ARRAY_PARTITION variable=A_res cyclic factor=8 dim=3
#pragma HLS BIND_STORAGE variable=A_res type=ram_t2p impl=uram

  // Resident B: 40 row banks x 8 k banks -> 320 URAMs (rows >= J unused, never stored)
  static data_t B_res[PE_J][NJB][K];
#pragma HLS ARRAY_PARTITION variable=B_res complete dim=1
#pragma HLS ARRAY_PARTITION variable=B_res cyclic factor=8 dim=3
#pragma HLS BIND_STORAGE variable=B_res type=ram_t2p impl=uram

  load_AB_all(A, B, A_res, B_res);
  compute_all(A_res, B_res, C);
}
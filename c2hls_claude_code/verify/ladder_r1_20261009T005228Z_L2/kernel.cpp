#include "kernel.h"

#define LANES   16
#define TI      128
#define TJ      128
#define PE_I    32
#define PE_J    32
#define RI      (TI / PE_I)        // 4
#define RJ      (TJ / PE_J)        // 4
#define NB      (RI * RJ)          // 16 interleaved sub-blocks (> FP add latency)
#define NS      (I / TI)           // 8 A strips
#define NTJ     (J / TJ)           // 8 C tiles per strip
#define KB      (K / LANES)        // 64 beats per row

// One-time load of all of B into on-chip URAM (resident for whole kernel)
static void load_B_all(data_t B[J][K], data_t B_res[J][K]) {
#pragma HLS INLINE off
load_B_beat:
  for (int b = 0; b < J * KB; b++) {
#pragma HLS PIPELINE II=1
    const int r  = b / KB;
    const int kb = b % KB;
  load_B_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      B_res[r][kb * LANES + l] = B[r][kb * LANES + l];
    }
  }
}

// Load a 128 x K strip of A in 512-bit beats (flat loop, no conditional)
static void load_A_strip(data_t A[I][K], data_t A_loc[TI][K], int s) {
#pragma HLS INLINE off
  const int ti = s * TI;
load_A_beat:
  for (int b = 0; b < TI * KB; b++) {
#pragma HLS PIPELINE II=1
    const int r  = b / KB;
    const int kb = b % KB;
  load_A_lane:
    for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
      A_loc[r][kb * LANES + l] = A[ti + r][kb * LANES + l];
    }
  }
}

// Compute all 8 C tiles of one A strip against resident B (32x32 MAC array)
static void compute_strip(data_t A_loc[TI][K], data_t B_res[J][K],
                          data_t C[I][J], int s) {
#pragma HLS INLINE off
  const int ti = s * TI;

  data_t acc[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=PE_I dim=1
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=PE_J dim=2
#pragma HLS BIND_STORAGE variable=acc type=ram_s2p impl=lutram

tile_j_loop:
  for (int jt = 0; jt < NTJ; jt++) {
    const int tj = jt * TJ;

  load_C_beat:
    for (int b = 0; b < TI * (TJ / LANES); b++) {
#pragma HLS PIPELINE II=1
      const int r  = b / (TJ / LANES);
      const int cb = b % (TJ / LANES);
    load_C_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        acc[r][cb * LANES + l] = C[ti + r][tj + cb * LANES + l];
      }
    }

  mac_k:
    for (int k = 0; k < K; k++) {
    mac_blk:
      for (int b = 0; b < NB; b++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc type=inter dependent=false
        const int ri = b / RJ;
        const int rj = b % RJ;
        data_t av[PE_I];
        data_t bv[PE_J];
#pragma HLS ARRAY_PARTITION variable=av complete dim=1
#pragma HLS ARRAY_PARTITION variable=bv complete dim=1
      rd_a:
        for (int p = 0; p < PE_I; p++) {
#pragma HLS UNROLL
          av[p] = A_loc[ri * PE_I + p][k];
        }
      rd_b:
        for (int q = 0; q < PE_J; q++) {
#pragma HLS UNROLL
          bv[q] = B_res[tj + rj * PE_J + q][k];
        }
      mac_p:
        for (int p = 0; p < PE_I; p++) {
#pragma HLS UNROLL
        mac_q:
          for (int q = 0; q < PE_J; q++) {
#pragma HLS UNROLL
            acc[ri * PE_I + p][rj * PE_J + q] =
                acc[ri * PE_I + p][rj * PE_J + q] + av[p] * bv[q];
          }
        }
      }
    }

  store_C_beat:
    for (int b = 0; b < TI * (TJ / LANES); b++) {
#pragma HLS PIPELINE II=1
      const int r  = b / (TJ / LANES);
      const int cb = b % (TJ / LANES);
    store_C_lane:
      for (int l = 0; l < LANES; l++) {
#pragma HLS UNROLL
        C[ti + r][tj + cb * LANES + l] = acc[r][cb * LANES + l];
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

#pragma HLS ALLOCATION function instances=compute_strip limit=1
#pragma HLS ALLOCATION function instances=load_A_strip limit=1

  // Resident B: 32 row banks (compute width) x 8 k banks -> 256 URAMs of 4096 words
  static data_t B_res[J][K];
#pragma HLS ARRAY_PARTITION variable=B_res cyclic factor=PE_J dim=1
#pragma HLS ARRAY_PARTITION variable=B_res cyclic factor=8 dim=2
#pragma HLS BIND_STORAGE variable=B_res type=ram_t2p impl=uram

  // Ping-pong A strips: 32 row banks (compute) x 16 k banks (512-bit beat)
  data_t A_buf0[TI][K];
  data_t A_buf1[TI][K];
#pragma HLS ARRAY_PARTITION variable=A_buf0 cyclic factor=PE_I dim=1
#pragma HLS ARRAY_PARTITION variable=A_buf0 cyclic factor=LANES dim=2
#pragma HLS ARRAY_PARTITION variable=A_buf1 cyclic factor=PE_I dim=1
#pragma HLS ARRAY_PARTITION variable=A_buf1 cyclic factor=LANES dim=2
#pragma HLS BIND_STORAGE variable=A_buf0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=A_buf1 type=ram_2p impl=bram

  load_B_all(B, B_res);
  load_A_strip(A, A_buf0, 0);

  // Load strip s+1 (gmem0) overlaps compute of strip s (resident B + gmem2)
strip_loop:
  for (int s = 0; s < NS; s++) {
    const int nxt = (s + 1 < NS) ? (s + 1) : s;
    if ((s & 1) == 0) {
      load_A_strip(A, A_buf1, nxt);
      compute_strip(A_buf0, B_res, C, s);
    } else {
      load_A_strip(A, A_buf0, nxt);
      compute_strip(A_buf1, B_res, C, s);
    }
  }
}
#include "kernel.h"

// ---------------------------------------------------------------------------
// Output-stationary GEMM:  C[i][j] += sum_k A[i][k] * B[j][k]
//   Parallel datapath per cycle: PI_ rows x PJ_ cols x KL_ k-lanes
//   = 11 x 16 x 8 = 1408 fp32 MACs (~3 DSP fmul + ~2 DSP fadd each => ~7040 DSP)
//   i = ii*PI_ + pi  (rows padded to NII*PI_ = 1034, pad rows never stored)
//   j = jj*PJ_ + pj  (PJ_ = 16 == 512-bit lane count -> static lane banking)
//   k = kc*KL_ + l   (k stored as [kb][16]; kc even -> lanes 0..7, odd -> 8..15)
// Operands A and B are fully resident on chip (URAM); C is staged in the local
// accumulator once (512-bit beats), updated on chip, and written back once.
// ---------------------------------------------------------------------------

#define PI_   11
#define PJ_   16
#define KL_   8
#define LANES 16                       // 512 / 32
#define NII   ((I + PI_ - 1) / PI_)    // 94  -> 1034 padded rows
#define NJJ   (J / PJ_)                // 64
#define NKB   (K / LANES)              // 64 beats per row
#define NKC   (K / KL_)                // 128 k-chunks
#define NS    (NII * NJJ)              // 6016 output blocks

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=1048576 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // A resident: [ii][pi][kb][lane]  -> 11*16 = 176 banks, depth 94*64 = 6016
  static data_t A_loc[NII][PI_][NKB][LANES];
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=2
#pragma HLS ARRAY_PARTITION variable=A_loc complete dim=4
#pragma HLS BIND_STORAGE variable=A_loc type=ram_2p impl=uram

  // B resident: [jj][pj][kb][lane]  -> 16*16 = 256 banks, depth 64*64 = 4096
  static data_t B_loc[NJJ][PJ_][NKB][LANES];
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=2
#pragma HLS ARRAY_PARTITION variable=B_loc complete dim=4
#pragma HLS BIND_STORAGE variable=B_loc type=ram_2p impl=uram

  // Accumulator (initialised from C): [s = ii*NJJ + jj][pi][pj] -> 176 banks
  static data_t acc[NS][PI_][PJ_];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=3
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=bram

  // ---------------- Phase 1: load A, B, C (512-bit beats, 3 parallel ports) --
  int ld_pi = 0, ld_ii = 0;
  load_all: for (int it = 0; it < I * NKB; ++it) {
#pragma HLS PIPELINE II=1
    int r  = it / NKB;
    int kb = it % NKB;
    load_lane: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
      A_loc[ld_ii][ld_pi][kb][l]       = A[r][kb * LANES + l];
      B_loc[r / PJ_][r % PJ_][kb][l]   = B[r][kb * LANES + l];
      acc[ld_ii * NJJ + kb][ld_pi][l]  = (data_t)0;
    }
    if (kb == NKB - 1) {
      if (ld_pi == PI_ - 1) { ld_pi = 0; ld_ii++; }
      else                  { ld_pi++; }
    }
  }

  // ---------------- Phase 2: on-chip compute, 1408 MACs / cycle --------------
  int s = 0, ii = 0, jj = 0, kc = 0;
  compute: for (int it = 0; it < NKC * NS; ++it) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc inter false
    // acc[s] is revisited only every NS (=6016) iterations >> pipeline depth.
    int  kq = kc >> 1;
    bool hi = (kc & 1) != 0;

    data_t a[PI_][KL_];
#pragma HLS ARRAY_PARTITION variable=a complete dim=0
    data_t b[PJ_][KL_];
#pragma HLS ARRAY_PARTITION variable=b complete dim=0

    rd_a_row: for (int pi = 0; pi < PI_; ++pi) {
#pragma HLS UNROLL
      rd_a_lane: for (int l = 0; l < KL_; ++l) {
#pragma HLS UNROLL
        data_t v_lo = A_loc[ii][pi][kq][l];
        data_t v_hi = A_loc[ii][pi][kq][l + KL_];
        a[pi][l] = hi ? v_hi : v_lo;
      }
    }
    rd_b_col: for (int pj = 0; pj < PJ_; ++pj) {
#pragma HLS UNROLL
      rd_b_lane: for (int l = 0; l < KL_; ++l) {
#pragma HLS UNROLL
        data_t v_lo = B_loc[jj][pj][kq][l];
        data_t v_hi = B_loc[jj][pj][kq][l + KL_];
        b[pj][l] = hi ? v_hi : v_lo;
      }
    }

    mac_row: for (int pi = 0; pi < PI_; ++pi) {
#pragma HLS UNROLL
      mac_col: for (int pj = 0; pj < PJ_; ++pj) {
#pragma HLS UNROLL
        data_t p[KL_];
#pragma HLS ARRAY_PARTITION variable=p complete dim=1
        mac_mul: for (int l = 0; l < KL_; ++l) {
#pragma HLS UNROLL
          p[l] = a[pi][l] * b[pj][l];
        }
        data_t t0 = p[0] + p[1];
        data_t t1 = p[2] + p[3];
        data_t t2 = p[4] + p[5];
        data_t t3 = p[6] + p[7];
        data_t u0 = t0 + t1;
        data_t u1 = t2 + t3;
        data_t sum = u0 + u1;
        acc[s][pi][pj] = acc[s][pi][pj] + sum;
      }
    }

    // counters: jj innermost, then ii, then kc
    if (jj == NJJ - 1) {
      jj = 0;
      if (ii == NII - 1) { ii = 0; kc++; }
      else               { ii++; }
    } else {
      jj++;
    }
    s = (s == NS - 1) ? 0 : s + 1;
  }

  // ---------------- Phase 3: store C once (512-bit beats) --------------------
  int st_pi = 0, st_ii = 0;
  store_c: for (int it = 0; it < I * NKB; ++it) {
#pragma HLS PIPELINE II=1
    int r  = it / NKB;
    int kb = it % NKB;
    store_lane: for (int l = 0; l < LANES; ++l) {
#pragma HLS UNROLL
      C[r][kb * LANES + l] = acc[st_ii * NJJ + kb][st_pi][l];
    }
    if (kb == NKB - 1) {
      if (st_pi == PI_ - 1) { st_pi = 0; st_ii++; }
      else                  { st_pi++; }
    }
  }
}
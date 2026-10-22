#include "kernel.h"

// C[i][j] = sum_k A[i][k] * B[j][k]
// Tiling: an A block of TI rows stays on-chip while all B tiles (TJ rows) stream past.
// Compute array: UI x TJ = 8 x 64 = 512 parallel MACs per cycle.
// Ping-pong B tiles and accumulators: while tile t is computed, tile t+1 of B
// is fetched and the result of tile t-1 is written to C (same pipelined loop).
#define TI 1024
#define TJ 64
#define UI 16
#define NIT (I / TI)
#define NJT (J / TJ)
#define TILE_ITERS (K * (TI / UI))   // == TJ*K == 65536

// Prologue per A block: load A block + first B tile, and flush the final C tile
// of the previous A block.
static void prologue(data_t A[I][K], data_t B[J][K], data_t C[I][J],
                     data_t A_loc[TI][K], data_t B0[TJ][K], data_t accp[TI][TJ],
                     int i0, bool st, int si0, int sj0) {
prologue_loop:
  for (int t = 0; t < TI * K; t++) {
#pragma HLS PIPELINE II=1
    int ii = t / K, k = t % K;
    A_loc[ii][k] = A[i0 + ii][k];
    if (t < TJ * K) {
      B0[t / K][t % K] = B[t / K][t % K];
    }
    if (st && t < TI * TJ) {
      C[si0 + t / TJ][sj0 + t % TJ] = accp[t / TJ][t % TJ];
    }
  }
}

static void compute_tile(data_t A_loc[TI][K], data_t Bc[TJ][K], data_t Bn[TJ][K],
                         data_t accc[TI][TJ], data_t accp[TI][TJ],
                         data_t B[J][K], data_t C[I][J],
                         bool ld, int jn, bool st, int si0, int sj0) {
compute_k:
  for (int k = 0; k < K; k++) {
  compute_i:
    for (int i1 = 0; i1 < TI / UI; i1++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=accc inter false
      int t = k * (TI / UI) + i1;
      // prefetch next B tile (row-major, sequential)
      if (ld) {
        Bn[t / K][t % K] = B[jn + t / K][t % K];
      }
      // drain previous C tile (row-major, sequential)
      if (st && t < TI * TJ) {
        C[si0 + t / TJ][sj0 + t % TJ] = accp[t / TJ][t % TJ];
      }
      // UI x TJ MAC array
      for (int u = 0; u < UI; u++) {
#pragma HLS UNROLL
        data_t a = A_loc[i1 * UI + u][k];
        for (int jj = 0; jj < TJ; jj++) {
#pragma HLS UNROLL
          data_t prev = (k == 0) ? (data_t)0 : accc[i1 * UI + u][jj];
          accc[i1 * UI + u][jj] = prev + a * Bc[jj][k];
        }
      }
    }
  }
}

static void store_tile(data_t C[I][J], data_t accp[TI][TJ], int si0, int sj0) {
store_C:
  for (int t = 0; t < TI * TJ; t++) {
#pragma HLS PIPELINE II=1
    C[si0 + t / TJ][sj0 + t % TJ] = accp[t / TJ][t % TJ];
  }
}

extern "C" void autosa_mm(data_t A[I][K], data_t B[J][K], data_t C[I][J]) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=512 max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=512 max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=return

  static data_t A_loc[TI][K];
#pragma HLS ARRAY_PARTITION variable=A_loc cyclic factor=UI dim=1
#pragma HLS BIND_STORAGE variable=A_loc type=ram_2p impl=uram
  static data_t B0[TJ][K];
#pragma HLS ARRAY_PARTITION variable=B0 complete dim=1
  static data_t B1[TJ][K];
#pragma HLS ARRAY_PARTITION variable=B1 complete dim=1
  static data_t acc0[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=acc0 cyclic factor=UI dim=1
#pragma HLS ARRAY_PARTITION variable=acc0 complete dim=2
  static data_t acc1[TI][TJ];
#pragma HLS ARRAY_PARTITION variable=acc1 cyclic factor=UI dim=1
#pragma HLS ARRAY_PARTITION variable=acc1 complete dim=2

tile_i:
  for (int it = 0; it < NIT; it++) {
    int i0 = it * TI;
    // last tile of previous block lives in acc1
    prologue(A, B, C, A_loc, B0, acc1, i0, it > 0, i0 - TI, (NJT - 1) * TJ);
  tile_j:
    for (int jp = 0; jp < NJT / 2; jp++) {
      int je = 2 * jp, jo = 2 * jp + 1;
      // even tile: compute with B0/acc0, prefetch odd tile into B1, drain acc1 (tile je-1)
      compute_tile(A_loc, B0, B1, acc0, acc1, B, C,
                   true, jo * TJ, jp > 0, i0, (je - 1) * TJ);
      // odd tile: compute with B1/acc1, prefetch next even tile into B0, drain acc0 (tile je)
      compute_tile(A_loc, B1, B0, acc1, acc0, B, C,
                   jo + 1 < NJT, (jo + 1) * TJ, true, i0, je * TJ);
    }
  }
  store_tile(C, acc1, (NIT - 1) * TI, (NJT - 1) * TJ);
}

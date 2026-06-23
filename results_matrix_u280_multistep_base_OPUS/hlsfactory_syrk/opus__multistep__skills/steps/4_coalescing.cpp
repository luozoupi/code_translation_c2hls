#include "syrk.h"
#include <string.h>

#define TILE 256

static void load_tile(double C_row[N], double C_tile_1[TILE], double C_tile_2[TILE],
                      int jt, int tile_len, double beta, int flag)
{
  if (flag == 0) {
    for (int jj = 0; jj < tile_len; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      C_tile_1[jj] = C_row[jt + jj] * beta;
    }
  } else {
    for (int jj = 0; jj < tile_len; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      C_tile_2[jj] = C_row[jt + jj] * beta;
    }
  }
}

static void compute_tile(double A_local[N][M], double C_tile_1[TILE], double C_tile_2[TILE],
                         int i, int jt, int tile_len, int m, double alpha, int flag)
{
  if (flag == 0) {
    for (int k = 0; k < m; k++) {
      double a_ik = A_local[i][k];
      for (int jj = 0; jj < tile_len; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS DEPENDENCE variable=C_tile_1 inter false
        C_tile_1[jj] += alpha * a_ik * A_local[jt + jj][k];
      }
    }
  } else {
    for (int k = 0; k < m; k++) {
      double a_ik = A_local[i][k];
      for (int jj = 0; jj < tile_len; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
#pragma HLS DEPENDENCE variable=C_tile_2 inter false
        C_tile_2[jj] += alpha * a_ik * A_local[jt + jj][k];
      }
    }
  }
}

static void store_tile(double C_row[N], double C_tile_1[TILE], double C_tile_2[TILE],
                       int jt, int tile_len, int flag)
{
  if (flag == 0) {
    for (int jj = 0; jj < tile_len; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      C_row[jt + jj] = C_tile_1[jj];
    }
  } else {
    for (int jj = 0; jj < tile_len; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      C_row[jt + jj] = C_tile_2[jj];
    }
  }
}

// Burst-friendly row load: copies one contiguous row from global memory into a
// local buffer in a single pipelined burst (memory coalescing on the AXI bus).
template <int LEN>
static void burst_read_row(double dst[LEN], const double* src)
{
  for (int j = 0; j < LEN; j++) {
#pragma HLS PIPELINE II=1
    dst[j] = src[j];
  }
}

template <int LEN>
static void burst_write_row(double* dst, const double src[LEN])
{
  for (int j = 0; j < LEN; j++) {
#pragma HLS PIPELINE II=1
    dst[j] = src[j];
  }
}

void kernel_syrk(
		 double alpha,
		 double beta,
		 double C[ N + 0][N + 0],
		 double A[ N + 0][M + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int m = M;

  int i, k, jt;

  // Stage A locally via burst reads to enable reuse across the k/j loops.
  double A_local[N][M];
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=1

  // Load entire A matrix using row-wise contiguous bursts (coalesced access).
  for (i = 0; i < n; i++) {
    double a_row[M];
#pragma HLS ARRAY_PARTITION variable=a_row cyclic factor=8 dim=1
    burst_read_row<M>(a_row, &A[i][0]);
    for (k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
      A_local[i][k] = a_row[k];
    }
  }

  // Double-buffered tile storage.
  double C_tile_1[TILE];
#pragma HLS ARRAY_PARTITION variable=C_tile_1 cyclic factor=8 dim=1
  double C_tile_2[TILE];
#pragma HLS ARRAY_PARTITION variable=C_tile_2 cyclic factor=8 dim=1

  for (i = 0; i < n; i++) {
    // Stage current C row locally via a contiguous burst (coalesced access).
    double C_row[N];
#pragma HLS ARRAY_PARTITION variable=C_row cyclic factor=8 dim=1
    burst_read_row<N>(C_row, &C[i][0]);

    // Number of valid columns for this row (lower triangular part).
    int jmax = i + 1;
    int num_tiles = (jmax + TILE - 1) / TILE;

    int prev_tile_valid = 0;
    int prev_jt = 0;
    int prev_tile_len = 0;
    int prev_flag = 0;

    for (int t = 0; t < num_tiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=((N + TILE - 1) / TILE)

      jt = t * TILE;
      int tile_end = jt + TILE;
      if (tile_end > jmax) tile_end = jmax;
      int tile_len = tile_end - jt;

      int flag = t % 2;

      // ---- LOAD phase for current tile (into buffer[flag]) ----
      load_tile(C_row, C_tile_1, C_tile_2, jt, tile_len, beta, flag);

      // ---- STORE phase for previous tile (from buffer[prev_flag]) overlaps ----
      if (prev_tile_valid) {
        store_tile(C_row, C_tile_1, C_tile_2, prev_jt, prev_tile_len, prev_flag);
      }

      // ---- COMPUTE phase for current tile (on buffer[flag]) ----
      compute_tile(A_local, C_tile_1, C_tile_2, i, jt, tile_len, m, alpha, flag);

      prev_tile_valid = 1;
      prev_jt = jt;
      prev_tile_len = tile_len;
      prev_flag = flag;
    }

    // ---- Final STORE for the last processed tile ----
    if (prev_tile_valid) {
      store_tile(C_row, C_tile_1, C_tile_2, prev_jt, prev_tile_len, prev_flag);
    }

    // Write back the updated C row via a contiguous burst (coalesced access).
    burst_write_row<N>(&C[i][0], C_row);
  }

}
#include "syrk.h"

#define TILE 256

static void load_tile(double C[N][N], double C_tile_1[TILE], double C_tile_2[TILE],
                      int i, int jt, int tile_len, double beta, int flag)
{
  if (flag == 0) {
    for (int jj = 0; jj < tile_len; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      C_tile_1[jj] = C[i][jt + jj] * beta;
    }
  } else {
    for (int jj = 0; jj < tile_len; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      C_tile_2[jj] = C[i][jt + jj] * beta;
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

static void store_tile(double C[N][N], double C_tile_1[TILE], double C_tile_2[TILE],
                       int i, int jt, int tile_len, int flag)
{
  if (flag == 0) {
    for (int jj = 0; jj < tile_len; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      C[i][jt + jj] = C_tile_1[jj];
    }
  } else {
    for (int jj = 0; jj < tile_len; jj++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE
      C[i][jt + jj] = C_tile_2[jj];
    }
  }
}

void kernel_syrk( 
		 double alpha,
		 double beta,
		 double C[ N + 0][N + 0],
		 double A[ N + 0][M + 0])
{
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=alpha bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const int n = N;
    const int m = M;

  int i, k, jt;

  // Stage A locally to enable reuse across the k/j loops (load phase for A).
  double A_local[N][M];
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=A_local cyclic factor=8 dim=1

  for (i = 0; i < n; i++) {
    for (k = 0; k < m; k++) {
#pragma HLS PIPELINE II=1
      A_local[i][k] = A[i][k];
    }
  }

  // Double-buffered tile storage.
  double C_tile_1[TILE];
#pragma HLS ARRAY_PARTITION variable=C_tile_1 cyclic factor=8 dim=1
  double C_tile_2[TILE];
#pragma HLS ARRAY_PARTITION variable=C_tile_2 cyclic factor=8 dim=1

  for (i = 0; i < n; i++) {
    // Number of valid columns for this row (lower triangular part).
    int jmax = i + 1;
    int num_tiles = (jmax + TILE - 1) / TILE;

    // Pipeline the tile processing across load/compute/store of consecutive tiles.
    // We process tiles in a software-pipelined manner using ping-pong buffers.
    //
    // Schedule (per tile t, flag = t % 2):
    //   load(t)    -> buffer[flag]
    //   compute(t) -> buffer[flag]
    //   store(t)   -> buffer[flag]
    // Load of tile t+1 can overlap with store of tile t since they use opposite buffers.

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
      load_tile(C, C_tile_1, C_tile_2, i, jt, tile_len, beta, flag);

      // ---- STORE phase for previous tile (from buffer[prev_flag]) overlaps ----
      if (prev_tile_valid) {
        store_tile(C, C_tile_1, C_tile_2, i, prev_jt, prev_tile_len, prev_flag);
      }

      // ---- COMPUTE phase for current tile (on buffer[flag]) ----
      compute_tile(A_local, C_tile_1, C_tile_2, i, jt, tile_len, m, alpha, flag);

      // Remember current tile to store on the next iteration.
      prev_tile_valid = 1;
      prev_jt = jt;
      prev_tile_len = tile_len;
      prev_flag = flag;
    }

    // ---- Final STORE for the last processed tile ----
    if (prev_tile_valid) {
      store_tile(C, C_tile_1, C_tile_2, i, prev_jt, prev_tile_len, prev_flag);
    }
  }

}